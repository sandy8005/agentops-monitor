import json
import re
from llm import logged_llm_call, ModelOutputInvalid
from pydantic import ValidationError
from schemas import ParsedResume, _strict_float
from logging_config import get_logger

log = get_logger(__name__)


def _raw_number(v, default=0.0):
    """
    Extract just the numeric value from 24, "24", or "24 months", WITHOUT unit
    conversion. Used for the months field, where the unit is already known from
    the field name — so we convert months->years exactly ONCE at the call site,
    never twice (the old bug: "24 months" -> 2.0 -> /12 -> 0.17).
    """
    if v is None:
        return default
    if isinstance(v, (int, float)):
        # finite + representable, else ValueError (-> unusable model output)
        return _strict_float(v, field_name="months")
    m = re.search(r"\d+(\.\d+)?", str(v))
    return float(m.group()) if m else default


def _to_float_or_default(v, default=0.0):
    """
    For fields where a MISSING value has a sensible default (like a per-entry
    duration the LLM omitted): coerce recoverable values, fall back to default
    only for None/absent — but still FAIL on genuine garbage like 'about two',
    so bad LLM output surfaces instead of silently becoming 0.
    """
    if v is None:
        return default
    return _strict_float(v, field_name="years")


def _normalize_education(edu_list):
    """Accept whatever the LLM returned and coerce each entry to {degree, institution, year}."""
    out = []
    for e in (edu_list or []):
        if not isinstance(e, dict):
            continue
        out.append({
            "degree": str(e.get("degree") or e.get("qualification") or e.get("name") or ""),
            "institution": str(e.get("institution") or e.get("school") or e.get("university") or e.get("college") or ""),
            "year": str(e.get("year") or e.get("graduation_year") or e.get("end_year") or e.get("completed") or "")
        })
    return out


def _normalize_projects(proj_list):
    """Coerce each project to {name, tech, evidence}. Accepts tech as list or comma
    string. `evidence` is the model's verbatim quote for the project (may be "");
    it is consumed by ground_projects() and never stored past validation."""
    out = []
    for p in (proj_list or []):
        if not isinstance(p, dict):
            continue
        tech = p.get("tech") or p.get("technologies") or p.get("stack") or []
        if isinstance(tech, str):
            tech = [t.strip() for t in tech.split(",") if t.strip()]
        elif not isinstance(tech, list):
            tech = []
        out.append({
            "name": str(p.get("name") or p.get("title") or p.get("project") or ""),
            "tech": [str(t) for t in tech if str(t).strip()],
            "evidence": str(p.get("evidence") or ""),
        })
    return out


def _normalize_experience(exp_list):
    """
    Coerce each experience entry to {title, company, years}.
    Handles alternative keys (role/position, months, duration) universally.
    Garbage durations fail loudly (via _strict_float) rather than becoming 0.
    """
    out = []
    for x in (exp_list or []):
        if not isinstance(x, dict):
            continue
        title = x.get("title") or x.get("role") or x.get("position") or x.get("job_title") or ""
        company = x.get("company") or x.get("employer") or x.get("organization") or ""

        if x.get("years") is not None:
            years = _to_float_or_default(x.get("years"))
        elif x.get("months") is not None:
            # months field: unit is known, take RAW number, convert to years ONCE.
            years = round(_raw_number(x.get("months")) / 12.0, 2)
        elif x.get("duration") is not None:
            years = _to_float_or_default(x.get("duration"))
        else:
            years = 0.0

        out.append({
            "title": str(title),
            "company": str(company),
            "years": years
        })
    return out


def _reconcile_experience(parsed_dict):
    """
    Cross-check the LLM's stated years_experience against the sum of itemized role
    durations and FLAG a large divergence — but do NOT overwrite the stated total.
    A naive sum is not ground truth: it double-counts concurrent/overlapping roles
    (accurate reconciliation needs date-range unioning) and under-counts when only
    some roles are itemized. So we keep the model's stated total and surface the
    discrepancy for a human / downstream awareness instead of silently replacing it.
    """
    stated = parsed_dict.get("years_experience", 0.0) or 0.0
    summed = round(sum(e.get("years", 0.0) or 0.0 for e in parsed_dict.get("experience", [])), 2)

    parsed_dict["years_experience_stated"] = stated
    parsed_dict["years_experience_summed"] = summed

    TOLERANCE_YEARS = 1.0
    if summed > 0 and abs(stated - summed) > TOLERANCE_YEARS:
        # Divergence > 1yr: FLAG it, but keep the stated total (the sum is unreliable
        # without date-range unioning). Do not silently replace years_experience.
        parsed_dict["experience_discrepancy"] = {
            "stated": stated,
            "summed": summed,
            "note": ("stated total and summed itemized-role durations diverge by "
                     ">1 year; the sum may double-count concurrent roles or miss "
                     "un-itemized ones — keeping the stated total, flagged for review."),
        }
    else:
        parsed_dict["experience_discrepancy"] = None

    # years_experience is left as STATED (not replaced).
    return parsed_dict


# --- Resume sections ---------------------------------------------------------
# A line is a section heading only if, stripped of decoration ("##", "**", ":"),
# it is EXACTLY one of these names. Each heading opens a section of one KIND; only
# "employment" sections hold professional experience. Project, academic and
# volunteer timelines are tracked separately and never count as employment.
_SECTION_KINDS = (
    ("employment",
     r"(?:(?:relevant|professional|work|industry|career)\s+)?experience"
     r"|employment(?:\s+history)?|work\s+history|career(?:\s+history)?"
     r"|professional\s+background|internships?(?:\s+experience)?"
     r"|(?:(?:work|professional)\s+)?experience\s+and\s+internships?"),
    ("project",
     r"(?:(?:personal|academic|side|key|selected|technical|relevant|notable|"
     r"open[\s-]source)\s+)?projects?(?:\s+experience)?"),
    ("academic",
     r"education(?:al\s+background)?|academic\s+(?:background|experience)|academics"
     r"|research(?:\s+experience)?|teaching(?:\s+experience)?|coursework|thesis"
     r"|qualifications"),
    ("volunteer",
     r"volunteer(?:ing)?(?:\s+(?:experience|work|activities))?"
     r"|community\s+(?:service|involvement)"),
    ("other",
     r"(?:(?:technical|core|key)\s+)?skills(?:\s+summary)?|(?:professional\s+)?summary"
     r"|profile|objective|about(?:\s+me)?|certifications?(?:\s+and\s+licenses)?|licenses"
     r"|awards(?:\s+and\s+honou?rs)?|honou?rs|achievements|publications|languages"
     r"|interests|hobbies|activities|extracurricular(?:\s+activities)?"
     r"|leadership(?:\s+experience)?|references|contact(?:\s+information)?"),
)
_SECTION_RES = [(kind, re.compile(rx)) for kind, rx in _SECTION_KINDS]


def section_heading(line):
    """The section KIND a heading line opens ("employment", "project", "academic",
    "volunteer", "other"), or None when the line is not a recognized heading."""
    s = re.sub(r"^[\s#*=_\-•]+|[\s:*=_\-]+$", "", str(line or "")).lower()
    s = " ".join(re.sub(r"\s*&\s*", " and ", s).split())
    if not s or len(s) > 50:
        return None
    for kind, rx in _SECTION_RES:
        if rx.fullmatch(s):
            return kind
    return None


def resume_sections(resume_text):
    """[(kind, [lines])] in order. Text before the first recognized heading has
    kind None (name, contact, an unlabeled summary)."""
    sections = [(None, [])]
    for line in str(resume_text or "").splitlines():
        kind = section_heading(line)
        if kind is not None:
            sections.append((kind, []))
        else:
            sections[-1][1].append(line)
    return sections


# --- Years-of-experience grounding -----------------------------------------
# A model-stated total is accepted only up to what the resume TEXT supports: an
# explicit statement ("5 years of experience") or the union of VERIFIED employment
# records ("Jan 2019 - Mar 2022", "2020 - Present" under an experience heading).
# Unparseable dates only LOWER the ceiling, so a claim is never inflated here.
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH_RX = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
             r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_DATE_RX = (r"(?:(?P<{p}m>" + _MONTH_RX + r")\.?,?\s+(?P<{p}my>(?:19|20)\d\d)"
            r"|(?P<{p}nm>0?[1-9]|1[0-2])\s*/\s*(?P<{p}ny>(?:19|20)\d\d)"
            r"|(?P<{p}y>(?:19|20)\d\d))")
_RANGE_RE = re.compile(
    r"(?<![\w/])" + _DATE_RX.format(p="s")
    + r"\s*(?:-|–|—|to|until|till)\s*"
    + r"(?:" + _DATE_RX.format(p="e")
    + r"|(?P<present>present|current(?:ly)?|now|today|date|ongoing))(?![\w/])",
    re.IGNORECASE)
# A line that is plainly a degree entry (its dates are study, not work).
_DEGREE_LINE_RE = re.compile(
    r"\b(bachelor'?s?|master'?s?|b\.?\s?tech|m\.?\s?tech|b\.?\s?sc|m\.?\s?sc|ph\.?\s?d|"
    r"mba|degree|gpa|cgpa|graduated|coursework)\b", re.I)
# Slack for month rounding between the model's figure and the resume's ranges.
_YEARS_TOLERANCE = 0.5


def _month_index(m, prefix, end):
    """Months since year 0 for one side of a range. A year-only start is January
    and a year-only end is December (generous: this is a CEILING on the claim)."""
    if m.group(prefix + "m"):
        return int(m.group(prefix + "my")) * 12 + _MONTHS[m.group(prefix + "m")[:3].lower()] - 1 + end
    if m.group(prefix + "nm"):
        return int(m.group(prefix + "ny")) * 12 + int(m.group(prefix + "nm")) - 1 + end
    return int(m.group(prefix + "y")) * 12 + (12 if end else 0)


def _today_index(today):
    import datetime
    today = today or datetime.date.today()
    return today.year * 12 + today.month          # exclusive end of the current month


def _line_intervals(line, now):
    """[(start, end, match)] for each plausible date range on one line."""
    out = []
    for m in _RANGE_RE.finditer(line):
        start = _month_index(m, "s", 0)
        end = now if m.group("present") else _month_index(m, "e", 1)
        end = min(end, now)
        if start < 1950 * 12 or end <= start:
            continue
        out.append((start, end, m))
    return out


def _union_years(intervals):
    """Years covered by the UNION of (start, end) month intervals, or None.
    Overlapping/concurrent intervals are counted once."""
    if not intervals:
        return None
    intervals = sorted(intervals)
    total, (cur_s, cur_e) = 0, intervals[0]
    for st, en in intervals[1:]:
        if st <= cur_e:
            cur_e = max(cur_e, en)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = st, en
    total += cur_e - cur_s
    return round(total / 12.0, 2)


def _has_words(text):
    return re.search(r"[A-Za-z]{2,}", text or "") is not None


def _ym(idx):
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def employment_records(resume_text, today=None):
    """
    VERIFIED employment records: a date range on a line under an
    experience/employment heading, with an entry header (title and/or company)
    either on the same line or on the closest preceding non-bullet line of that
    section. Degree lines are skipped. Ranges under project, academic, volunteer
    or other headings — or before any heading — are never employment.

    Returns [{"header", "start": "YYYY-MM", "end": "YYYY-MM", "months"}].
    """
    now = _today_index(today)
    records = []
    for kind, lines in resume_sections(resume_text):
        if kind != "employment":
            continue
        header = ""
        for line in lines:
            s = line.strip()
            if not s:
                continue
            found = [] if _DEGREE_LINE_RE.search(s) else _line_intervals(s, now)
            if not found:
                if s[:1] not in "-*•" and _has_words(s):
                    header = s            # candidate header for a date line below
                continue
            rest = s
            for _st, _en, m in found:
                rest = rest.replace(m.group(0), " ")
            rest = " ".join(rest.split()).strip(" |,;()[]-–—•*")
            this_header = rest if _has_words(rest) else header
            header = ""
            if not this_header:
                continue                  # dates with no title/company: unverified
            for st, en, _m in found:
                records.append({"header": this_header[:120], "start": _ym(st),
                                "end": _ym(en - 1), "months": en - st})
    return records


def _record_interval(rec):
    sy, sm = (int(x) for x in rec["start"].split("-"))
    ey, em = (int(x) for x in rec["end"].split("-"))
    return sy * 12 + sm - 1, ey * 12 + em


def employment_years_from_dates(resume_text, today=None):
    """Total years covered by the UNION of VERIFIED employment records
    (overlapping/concurrent roles are counted once), or None if there are none."""
    return _union_years([_record_interval(r) for r in employment_records(resume_text, today)])


def non_employment_timelines(resume_text, today=None):
    """{kind: years} for date ranges OUTSIDE employment sections ("project",
    "academic", "volunteer", "other", "unsectioned"). Recorded for audit only —
    these are never professional experience."""
    now = _today_index(today)
    per_kind = {}
    for kind, lines in resume_sections(resume_text):
        if kind == "employment":
            continue
        for line in lines:
            per_kind.setdefault(kind or "unsectioned", []).extend(
                (st, en) for st, en, _m in _line_intervals(line.strip(), now))
    return {k: _union_years(v) for k, v in per_kind.items() if v}


def ground_years_experience(parsed_dict, resume_text, today=None):
    """
    Ground the model's stated years_experience against the resume text.

    Supported ceiling = max(largest explicit "N years of experience" statement,
    union of VERIFIED employment records — see employment_records). A stated total
    within _YEARS_TOLERANCE of the ceiling is kept. Above it, years_experience is
    CLAMPED to the ceiling; with no evidence at all it becomes unknown (None +
    experience_unknown, the same semantics as the rules parser: never 0, never
    "sufficient"). Either way the claim is recorded in `ungrounded_experience`
    (None when grounded).

    A stated total of 0 (or none) while the verified employment records prove a
    positive total is CORRECTED to the records' total — never above it — and
    recorded in `experience_corrected` (None otherwise).
    """
    from rule_resume_parser import extract_years
    stated = parsed_dict.get("years_experience") or 0.0
    explicit = extract_years(resume_text)
    records = employment_records(resume_text, today=today)
    ranged = _union_years([_record_interval(r) for r in records])
    evidence = [v for v in (explicit, ranged) if v is not None]
    ceiling = max(evidence) if evidence else None
    parsed_dict["employment_records"] = records
    parsed_dict["years_experience_evidence"] = {
        "explicit_statement": explicit, "date_ranges": ranged,
        "non_employment_ranges": non_employment_timelines(resume_text, today=today)}
    parsed_dict["ungrounded_experience"] = None
    parsed_dict["experience_corrected"] = None
    if stated <= 0:
        if ranged:
            parsed_dict["years_experience"] = ranged
            parsed_dict["experience_corrected"] = {
                "stated": float(stated), "corrected_to": ranged,
                "reason": "verified_employment_records"}
            _refresh_discrepancy(parsed_dict)
        return parsed_dict
    if ceiling is not None and stated <= ceiling + _YEARS_TOLERANCE:
        return parsed_dict
    if ceiling is None:
        parsed_dict["years_experience"] = None
        parsed_dict["experience_unknown"] = True
        reason = "no_resume_evidence"
    else:
        parsed_dict["years_experience"] = round(ceiling, 1)
        reason = "exceeds_resume_evidence"
    parsed_dict["ungrounded_experience"] = {"stated": stated, "supported": ceiling,
                                            "reason": reason}
    return parsed_dict


def _refresh_discrepancy(parsed_dict):
    """After a correction, re-judge stated-vs-summed against the CORRECTED total
    (the model's original figure stays in years_experience_stated)."""
    if "years_experience_summed" not in parsed_dict:
        return
    years = parsed_dict.get("years_experience") or 0.0
    summed = parsed_dict.get("years_experience_summed") or 0.0
    if summed > 0 and abs(years - summed) > 1.0:
        parsed_dict["experience_discrepancy"] = {
            "stated": years, "summed": summed,
            "note": ("verified employment total and summed itemized-role durations "
                     "diverge by >1 year — flagged for review.")}
    else:
        parsed_dict["experience_discrepancy"] = None


def ground_experience_entries(experience, resume_text):
    """
    Keep a model-itemized experience entry only if its title or company occurs
    (whole phrase, whitespace/case-normalized) in an EMPLOYMENT section of the
    resume. An entry named only under Projects, or nowhere, is not employment and
    must not influence scoring. Returns {"experience": [...],
    "ungrounded_experience_entries": [...]}.
    """
    emp_text = _norm_ws("\n".join(line for kind, lines in resume_sections(resume_text)
                                  if kind == "employment" for line in lines))

    def _in_emp(v):
        v = _norm_ws(v)
        return bool(v) and re.search(
            r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", emp_text) is not None

    kept, dropped = [], []
    for e in (experience or []):
        (kept if (_in_emp(e.get("title")) or _in_emp(e.get("company"))) else dropped).append(e)
    return {"experience": kept, "ungrounded_experience_entries": dropped}


def _norm_ws(text):
    return " ".join(str(text or "").lower().split())


def ground_skills(skills, skill_evidence, resume_text):
    """
    Make the parser's skill claims TRACEABLE to the resume text, deterministically.

    The LLM is asked for a verbatim evidence snippet per skill. Here we keep only
    snippets that really occur in the resume (whitespace/case-normalized), and mark a
    skill as grounded if it has verified evidence OR the skill term itself appears in
    the resume as a whole word/phrase. Anything else is an UNGROUNDED claim — a
    candidate extraction error or hallucination — surfaced for monitoring and kept
    out of the judge's "candidate skills".

    Returns {"skill_evidence": [...verified...], "grounded_skills": [...],
             "ungrounded_skills": [...]}.
    """
    from skills import affirmative_skill_in_text
    resume_norm = _norm_ws(resume_text)
    resume_tokens = set(re.findall(r"[a-z0-9\+\#\.]+", resume_norm))
    verified = {}
    for item in (skill_evidence or []):
        if not isinstance(item, dict):
            continue
        skill = str(item.get("skill") or "").strip()
        ev = str(item.get("evidence") or "").strip()
        # The snippet must (a) occur verbatim in the resume AND (b) itself
        # AFFIRMATIVELY mention the claimed skill. A real quote about something else
        # ("Python engineer" offered as evidence for "Kubernetes") grounds nothing (R10).
        if skill and ev and _norm_ws(ev) in resume_norm \
                and affirmative_skill_in_text(skill, ev):
            verified.setdefault(skill.lower(), {"skill": skill, "evidence": ev})

    from skills import skill_in_text, text_index
    idx_low, idx_tokens = text_index(resume_text)

    def _literal(skill):
        # Alias-aware ("K8s" in the resume grounds a parsed "Kubernetes"), using the
        # SAME canonical vocabulary the scorer uses — and only AFFIRMATIVE mentions:
        # "no Kubernetes experience" does not ground Kubernetes.
        if not _norm_ws(skill):
            return False
        return (skill_in_text(skill, idx_low, idx_tokens)
                and affirmative_skill_in_text(skill, resume_text))

    grounded, ungrounded = [], []
    for sk in skills:
        (grounded if (sk.lower() in verified or _literal(sk)) else ungrounded).append(sk)
    return {"skill_evidence": list(verified.values()),
            "grounded_skills": grounded, "ungrounded_skills": ungrounded}


# How far after a project's name its description may run (characters). A project
# block is the name line plus its bullets; a blank line ends it earlier.
_PROJECT_WINDOW_CHARS = 600


def _phrase_regex(phrase):
    words = [re.escape(w) for w in str(phrase or "").split()]
    if not words:
        return None
    return re.compile(r"(?<![a-z0-9])" + r"\s+".join(words) + r"(?![a-z0-9])", re.IGNORECASE)


def _project_windows(name, resume_text, other_names=()):
    """The resume text that follows each occurrence of the project's NAME — i.e.
    that project's own description. It ends at the first of: a blank line, a
    recognized section heading (section_heading), a line naming another project,
    or _PROJECT_WINDOW_CHARS. A resume without blank lines therefore cannot leak
    the next section's technologies into the project."""
    rx = _phrase_regex(name)
    if rx is None:
        return []
    others = [r for r in (_phrase_regex(n) for n in other_names
                          if _norm_ws(n) and _norm_ws(n) != _norm_ws(name)) if r]
    text = str(resume_text or "")
    out = []
    for m in rx.finditer(text):
        lines = text[m.start():m.start() + _PROJECT_WINDOW_CHARS].split("\n")
        kept = lines[:1]
        for line in lines[1:]:
            if (not line.strip() or section_heading(line) is not None
                    or any(o.search(line) for o in others)):
                break
            kept.append(line)
        out.append("\n".join(kept))
    return out


def _section_texts(resume_text):
    """Whitespace/case-normalized text of each section, headings excluded."""
    return [_norm_ws("\n".join(lines)) for _kind, lines in resume_sections(resume_text)]


def ground_projects(projects, resume_text):
    """
    Project technology feeds the scorer's project-relevance category, so it gets
    the SAME deterministic grounding as skills: a model-claimed technology is kept
    only if the resume itself affirmatively mentions it IN THAT PROJECT'S CONTEXT —

      * in the model's verbatim evidence quote, when that quote really occurs
        within ONE section of the resume (whitespace/case-normalized); or
      * in the text that follows the project's name in the resume (its own
        description, up to a blank line, the next section heading or the next
        project's name — see _project_windows).

    A project is kept only if its name occurs in the resume or its evidence quote
    is verified; otherwise the whole project is an ungrounded claim. Mentioning a
    technology ELSEWHERE in the resume does not ground it for a project ("Python"
    in the skills list is not evidence that "Analytics Project" used Python), and a
    negated / aspirational mention ("no Kubernetes", "learning AWS") grounds
    nothing.

    Returns {"projects": [{name, tech}], "ungrounded_project_claims":
             [{"project": name, "tech": [dropped...], "dropped_project": bool}]}.
    """
    from skills import affirmative_skill_in_text
    # An evidence quote must lie within ONE section: a quote that runs across a
    # heading ("... Python Experience Data Engineer ... AWS") is not one project's text.
    sections = _section_texts(resume_text)
    all_names = [str(p.get("name") or "").strip() for p in (projects or [])]
    kept, ungrounded = [], []
    for p in (projects or []):
        name = str(p.get("name") or "").strip()
        tech = [str(t).strip() for t in (p.get("tech") or []) if str(t).strip()]
        ev = str(p.get("evidence") or "").strip()
        ev_ok = bool(ev) and any(_norm_ws(ev) in sec for sec in sections)
        contexts = ([ev] if ev_ok else []) + _project_windows(name, resume_text, all_names)
        if not contexts:
            ungrounded.append({"project": name, "tech": tech, "dropped_project": True})
            continue
        good = [t for t in tech
                if any(affirmative_skill_in_text(t, ctx) for ctx in contexts)]
        bad = [t for t in tech if t not in good]
        kept.append({"name": name, "tech": good})
        if bad:
            ungrounded.append({"project": name, "tech": bad, "dropped_project": False})
    return {"projects": kept, "ungrounded_project_claims": ungrounded}


def ground_education(education, resume_text):
    """
    Keep an education entry only if its institution or degree text occurs in the
    resume (whitespace/case-normalized); a year is kept only if it occurs too.
    Entries the resume does not support are reported, not stored as facts.

    Returns {"education": [...], "ungrounded_education": [...]}.
    """
    resume_norm = _norm_ws(resume_text)

    def _in_resume(v):
        # Whole-word/phrase match on the normalized text: "MS" must not be grounded
        # by "systems", nor "2019" by "120190".
        v = _norm_ws(v)
        return bool(v) and re.search(
            r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", resume_norm) is not None

    kept, ungrounded = [], []
    for e in (education or []):
        degree, inst, year = e.get("degree") or "", e.get("institution") or "", e.get("year") or ""
        if not (_in_resume(inst) or _in_resume(degree)):
            ungrounded.append({"degree": degree, "institution": inst, "year": year})
            continue
        kept.append({"degree": degree if _in_resume(degree) else "",
                     "institution": inst if _in_resume(inst) else "",
                     "year": year if _in_resume(year) else ""})
    return {"education": kept, "ungrounded_education": ungrounded}


def parse_resume(resume_text, run_id, step_id, budget=None):
    from prompt_safety import wrap_untrusted, HARDENING_PREAMBLE, detect_injection
    # The resume is the most attacker-controlled input (a candidate uploads it).
    # Injection-looking patterns are recorded as a SECURITY WARNING, not a review
    # request: the graph deliberately continues after parse_resume (there is no
    # human-review route here), so writing needs_human_review would make the Monitor
    # claim "human review required" for a review that will never happen. The
    # structural defense (delimiting + hardening) below is what protects the prompt.
    try:
        from prompt_safety import apply_injection_policy
        apply_injection_policy(detect_injection(resume_text), step_id, source="resume",
                               run_id=run_id, review_allowed=False)
    except Exception as e:
        # Recording the signal must not break parsing on a transient hiccup — but a
        # database failure or a bug in the detector itself is not "no signal": it
        # propagates (the parse step fails visibly) instead of silently dropping a
        # security check. Logged by TYPE only.
        from router import must_propagate
        if must_propagate(e):
            raise
        log.warning("could not record resume security signal (%s)", type(e).__name__,
                    extra={"step_id": step_id})
    prompt = f"""
{HARDENING_PREAMBLE}

Extract structured information from this resume.

RESUME:
{wrap_untrusted(resume_text, "RESUME")}

Return ONLY valid JSON, no markdown fences, no explanation, in EXACTLY this shape
and using EXACTLY these key names:
{{
  "skills": ["skill1", "skill2"],
  "years_experience": <number>,
  "education": [
    {{"degree": "...", "institution": "...", "year": "YYYY"}}
  ],
  "projects": [
    {{"name": "...", "tech": ["..."], "evidence": "..."}}
  ],
  "experience": [
    {{"title": "...", "company": "...", "years": <number>}}
  ],
  "skill_evidence": [
    {{"skill": "FastAPI", "evidence": "built a FastAPI backend serving 2k req/s"}}
  ]
}}

RULES:
- education "year": a STRING like "2025".
- experience "title": the job title (key must be "title", not "role").
- experience "years": total years in that role as a NUMBER (key must be "years", not "months").
- "years_experience": total professional years as a number.
- "skill_evidence": for each skill, a SHORT phrase COPIED VERBATIM from the resume
  that shows it. Never paraphrase. Omit a skill here if no such text exists.
- projects "name": the project's name exactly as written in the resume.
- projects "tech": ONLY technologies the resume states were used in THAT project.
- projects "evidence": a SHORT phrase COPIED VERBATIM from the resume that describes
  the project and names its technologies. Never paraphrase.
- If a value is unknown, use an empty string "" or 0 — never omit a key.
"""
    raw = logged_llm_call(prompt, run_id, step_id, operation="parse_resume", budget=budget)
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()

    # --- Parse model output -----------------------------------------------
    # Fail LOUD on unparseable output rather than returning None: a None return
    # would let the router cache json.dumps(None) == "null" and poison the parse
    # cache; raising surfaces the failure so the run fails cleanly and nothing bad
    # is cached.
    # Every "the model's output is unusable" condition raises ModelOutputInvalid, a
    # degraded-MODEL signal the router may fall back on. Errors in OUR code below
    # (reconciliation, grounding) are deliberately NOT wrapped — they must surface.
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # The model occasionally wraps the object in prose despite instructions;
        # salvage the outermost {...} before giving up.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ModelOutputInvalid(
                f"parse_resume: LLM output was not valid JSON (step_id={step_id})"
            )
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError as e:
            raise ModelOutputInvalid(
                f"parse_resume: LLM output was not valid JSON (step_id={step_id})") from e

    if not isinstance(parsed, dict):
        raise ModelOutputInvalid(
            f"parse_resume: expected a JSON object, got "
            f"{type(parsed).__name__} (step_id={step_id})"
        )

    # --- Normalize each field into the canonical shape --------------------
    skills = parsed.get("skills") or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]
    elif isinstance(skills, list):
        skills = [str(s).strip() for s in skills if str(s).strip()]
    else:
        skills = []

    try:
        # _strict_float raises ValueError on garbage durations ("about two") — that
        # is unusable MODEL output, not a bug in this module.
        normalized = {
            "skills": skills,
            "years_experience": _to_float_or_default(parsed.get("years_experience")),
            "education": _normalize_education(parsed.get("education")),
            "projects": _normalize_projects(parsed.get("projects")),
            "experience": _normalize_experience(parsed.get("experience")),
        }
    except ValueError as e:
        raise ModelOutputInvalid(f"parse_resume: unusable value in model output: {e} "
                                 f"(step_id={step_id})") from e

    # --- Deterministic grounding of projects and education -----------------
    # Project tech is scored (project relevance), so a technology the model
    # attributes to a project must be supported by that project's resume text —
    # the same anti-hallucination rule as skills. Education is grounded too.
    project_grounding = ground_projects(normalized["projects"], resume_text)
    education_grounding = ground_education(normalized["education"], resume_text)
    # Itemized experience entries are kept only when an EMPLOYMENT section of the
    # resume names their title or company (a fabricated or project-only "role" is
    # dropped and recorded, never scored as professional experience).
    experience_grounding = ground_experience_entries(normalized["experience"], resume_text)
    normalized["projects"] = project_grounding["projects"]
    normalized["education"] = education_grounding["education"]
    normalized["experience"] = experience_grounding["experience"]

    # --- Schema validation (core fields) ----------------------------------
    try:
        validated = ParsedResume.model_validate(normalized).model_dump()
    except ValidationError as e:
        raise ModelOutputInvalid(
            f"parse_resume: model output failed schema validation ({e.error_count()} "
            f"error(s), step_id={step_id})") from e

    # --- Cross-check stated vs summed experience (adds audit metadata) ----
    validated = _reconcile_experience(validated)
    # --- Ground the stated total against the resume text: clamp to the evidence;
    #     a stated 0 is corrected UP only to the verified employment total ------
    validated = ground_years_experience(validated, resume_text)

    # --- Deterministic grounding of the parsed skills ----------------------
    validated.update(ground_skills(skills, parsed.get("skill_evidence"), resume_text))
    validated["ungrounded_project_claims"] = project_grounding["ungrounded_project_claims"]
    validated["ungrounded_education"] = education_grounding["ungrounded_education"]
    validated["ungrounded_experience_entries"] = experience_grounding[
        "ungrounded_experience_entries"]

    return validated