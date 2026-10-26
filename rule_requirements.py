"""
Rule-based requirement extraction — a NO-LLM fallback for extract_requirements.

Used when the LLM is unavailable (quota exhausted / budget spent). It pulls a
best-effort requirements dict from the job description using a skills vocabulary
and simple required-vs-preferred signal phrases. Cruder than the LLM, but it
means a job always gets SOME requirements and never drops out of the pipeline
just because Gemini quota ran out. Zero LLM calls.
"""
import bisect
import re

# A practical skills vocabulary. Multi-word entries are matched as phrases;
# single tokens are matched whole-word. Extend as needed.
SKILL_VOCAB = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "scala", "kotlin", "swift", "r", "sql", "nosql",
    "react", "angular", "vue", "node.js", "node", "django", "flask", "fastapi",
    "spring", "express", ".net", "rails",
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "git", "ci/cd", "linux",
    "machine learning", "ml", "deep learning", "ai", "nlp", "computer vision",
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
    "spark", "hadoop", "kafka", "airflow", "etl", "data engineering",
    "rest", "graphql", "microservices", "api", "agile", "scrum",
    "devops", "mlops", "security", "cybersecurity",
]

# Phrases that signal a REQUIRED skill vs a PREFERRED/nice-to-have one.
REQUIRED_SIGNALS = ["required", "must have", "must-have", "requirement",
                    "you have", "you must", "we require", "essential",
                    "minimum qualifications", "you'll need"]
PREFERRED_SIGNALS = ["preferred", "nice to have", "nice-to-have", "bonus",
                     "plus", "a plus", "desirable", "advantageous",
                     "good to have", "ideally"]


def _tokens(text):
    return set(re.findall(r"[a-z0-9\+\#\.]+", text.lower()))


def _skill_in(skill, text_lower, tokens):
    s = skill.lower()
    if " " in s or "." in s or "/" in s:
        return s in text_lower
    return s in tokens


# A count: digits or a spelled-out number (one-twenty), optionally followed by the
# same value in parentheses ("three (3)", "3 (three)").
_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()) if i}
_NUM = r"(?:\d{1,2}|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b"
_PAREN_NUM = r"(?:\s*\(\s*" + _NUM + r"\s*\))?"
_YEARS_RE = re.compile(
    r"\b(" + _NUM + r")" + _PAREN_NUM + r"\s*(?:\+|plus)?\s*"
    r"(?:(?:-|–|to)\s*(" + _NUM + r")" + _PAREN_NUM + r"\s*\+?\s*)?"
    r"(?:or more\s+)?(?:years?|yrs?)\b")


def _num_value(token):
    """int value of a _NUM token ("3" or "three")."""
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]


# A soft UPPER value right after a mention: "3-5 years of experience, ideally 5"
# or "(preferably 5 years)". It applies to the higher figure only, so the minimum
# stays the mention's low value (see years_requirement_mentions).
_SOFT_UPPER_RE = re.compile(
    r"[^.;,()]{0,60}?(?:,\s*|\s*\(\s*|\s+-\s+)(?:ideally|preferably)\s+(" + _NUM + r")"
    r"\s*\+?\s*(?:(?:years?|yrs?)\b)?\s*\)?")
# Sentence-level context that makes a "N years" mention NOT a minimum requirement.
_YEARS_NOT_REQUIRED = re.compile(
    r"\b(preferred|nice to have|nice-to-have|bonus|a plus|is a plus|ideally|preferably|"
    r"desirable|advantageous|would be great)\b")
_YEARS_CONDITIONAL = re.compile(
    r"\b(for (the )?(manager|management|lead|leadership|senior|principal|staff)\b|"
    r"if applying|for this level)")
_YEARS_NOT_EXPERIENCE = re.compile(
    r"\b(years? old|years? ago|past " + _NUM + r" years|last " + _NUM + r" years|founded|"
    r"in business|for over " + _NUM + r" years|we have been|our company|growing for|"
    r"contract length|" + _NUM + r"[- ]year (contract|term|warranty|commitment))")


# A mention that states the OVERALL requirement for the role, judged within the
# mention's own clause (commas/semicolons bound it), not the whole sentence.
_YEARS_OVERALL = re.compile(r"\b(overall|in total|total|professional|industry|work experience)\b")


def _clause_around(sentence, start, end):
    """The mention's own clause: bounded by commas/semicolons, by the neighbouring
    "N years" mentions (so "3 years python and 5 years of industry experience"
    keeps "industry" with the 5), and by 40 characters either side."""
    left = max(sentence.rfind(",", 0, start), sentence.rfind(";", 0, start)) + 1
    rights = [i for i in (sentence.find(",", end), sentence.find(";", end)) if i != -1]
    right = min(rights) if rights else len(sentence)
    for other in _YEARS_RE.finditer(sentence):
        if other.end() <= start:
            left = max(left, other.end())
        elif other.start() >= end:
            right = min(right, other.start())
            break
    return sentence[max(left, start - 40):min(right, end + 40)]


def _years_from_text(text_lower):
    """Minimum years of experience from a posting, as a DEGRADED fallback.

    Each "N years" / "N+ years" / "N-M years" mention is judged by its own
    SENTENCE, not by a small character window:
      * skipped when the sentence marks it as preferred / a bonus,
      * skipped when it is conditional on another role ("5 years ... for the
        manager track"),
      * skipped when it is not about experience (company age, "years ago",
        contract length).
    When a remaining mention is marked as the OVERALL requirement in its own
    clause ("5 years overall", "5+ years of professional experience", "total of
    6 years"), the smallest such mention is returned: "3 years Python, 5 years
    overall" is 5, not 3. Otherwise the SMALLEST mention is returned, since with
    no overall marker the lowest figure is the only safe minimum for the whole
    role. A range "3-5 years" counts as 3.
    Returns 0.0 when nothing qualifies (unknown, not "requires zero years").

    This cannot build required_any_of groups or read structure; the extraction is
    tagged requirements_method=rule_based wherever it is used, and matches that
    rest only on it can be excluded (require_verified_matches)."""
    mentions = years_requirement_mentions(text_lower)
    overall = [lo for lo, _hi, is_overall in mentions if is_overall]
    if overall:
        return min(overall)
    return min(lo for lo, _hi, _o in mentions) if mentions else 0.0


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?\n])\s+|\n+|•")
# An explicit "this is required" marker inside a mention's own clause.
_YEARS_REQUIRED_CLAUSE = re.compile(
    r"\b(required|requires?|requirement|must|mandatory|essential|minimum|at least|needed)\b")


_YEARS_CONJUNCTION = re.compile(r"\b(?:and|but|while|whereas|however|although)\b")


def _sentences_with_offsets(text):
    pos = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        yield pos, text[pos:m.start()]
        pos = m.end()
    yield pos, text[pos:]


_HAS_NUMBER_RE = re.compile(r"\d|\b" + _NUM)
_PAREN_GROUP_RE = re.compile(r"\([^()]*\)")
# A soft ADVERB modifies what FOLLOWS it ("ideally also have Go"), up to the end of
# its clause; it needs at least one word after it to modify something else.
_SOFT_ADVERB_AFTER_RE = re.compile(r"\b(?:ideally|preferably|optionally)\b(?=[^,;.()]*\w)[^,;.()]*")


def _scope_to_mention(sentence, start, end):
    """The sentence as seen by ONE "N years" mention (round 5, backlog A), with
    offsets kept (blanked text becomes spaces):
      * a parenthetical that contains another number and not this mention is
        about THAT number: "3+ years experience (5+ preferred)" -> the 3 does
        not inherit "preferred";
      * "ideally/preferably/optionally" AFTER the mention modifies what follows
        it, not the years: "3+ years of experience who would ideally also have
        Go" keeps the 3. A trailing bare "..., ideally." still applies.
    A parenthetical without a number ("5+ years (preferred)") and a soft word
    BEFORE the mention ("Ideally 5 years") are left alone, so they still mark it."""
    chars = list(sentence)

    def blank(a, b):
        chars[a:b] = " " * (b - a)

    for g in _PAREN_GROUP_RE.finditer(sentence):
        if (g.end() <= start or g.start() >= end) and _HAS_NUMBER_RE.search(g.group(0)):
            blank(g.start(), g.end())
    for adv in _SOFT_ADVERB_AFTER_RE.finditer(sentence, end):
        blank(adv.start(), adv.end())
    return "".join(chars)


def _years_mention_is_preferred(sentence, start, end, section):
    """Classify ONE "N years" mention by its own clause first (backlog C), so
    "3+ years Python required, while 5+ years Java preferred" keeps the 3:
      * a preferred signal in its clause            -> preferred (skipped)
      * "not required" right after it in its clause -> not a requirement (skipped)
      * an explicit required marker in its clause   -> required
      * under a "Requirements:" section heading     -> required
      * a preferred signal elsewhere in the sentence, or a "Nice to have:"
        section heading                             -> preferred (conservative)
      * otherwise                                   -> required."""
    clause = _clause_around(sentence, start, end)
    # The words before the mention belong to it only after the last conjunction:
    # "5 years of Java preferred and 3 years of Python required" -> the 3's clause
    # starts at "3 years", so the 5's "preferred" is not read as the 3's.
    lead = clause[:clause.find(sentence[start:end])]
    cut = list(_YEARS_CONJUNCTION.finditer(lead))
    if cut:
        clause = clause[cut[-1].end():]
    if _YEARS_NOT_REQUIRED.search(clause) or _PREF_SIGNAL_RE.search(clause):
        return True
    post = sentence[end:end + 60]
    stop = _POST_STOP_RE.search(post)
    if _POST_NEG_RE.search(post[:stop.start()] if stop else post):
        return True
    if _YEARS_REQUIRED_CLAUSE.search(clause) or section == "required":
        return False
    if _YEARS_NOT_REQUIRED.search(sentence):
        return True
    return section == "preferred"


# Clause boundaries for scoping a conditional ("for the manager track") to the
# years mention it belongs to.
_YEARS_SEGMENT_CUT = re.compile(r"[,;:]|\b(?:and|but|while|whereas|however|although)\b")


def _conditional_mention_starts(sentence):
    """Start offsets of the "N years" mentions in `sentence` that a conditional
    ("for the senior/lead/manager track", "if applying", "for this level")
    applies to (round 6, backlog A). The sentence is cut into clauses at
    commas/semicolons/colons and and/but/while/...:
      * a conditional in a clause that holds mentions applies to those mentions:
        "3 years Python required, 5 years for the manager track" skips only the 5;
      * a conditional clause without a mention ("For the senior track, 5 years",
        "5 years of Python and Django, for the lead role") applies to the nearest
        clause with a mention, when mentions sit on one side of it only;
      * with mentions on both sides it is ambiguous and applies to every mention
        in the sentence (the conservative, former sentence-level behavior)."""
    conds = list(_YEARS_CONDITIONAL.finditer(sentence))
    if not conds:
        return set()
    cuts = [c.start() for c in _YEARS_SEGMENT_CUT.finditer(sentence)]

    def seg(pos):
        return bisect.bisect_right(cuts, pos)

    by_seg = {}
    for m in _YEARS_RE.finditer(sentence):
        by_seg.setdefault(seg(m.start()), []).append(m.start())
    out = set()
    for c in conds:
        s = seg(c.start())
        if s in by_seg:
            out.update(by_seg[s])
            continue
        left = [k for k in by_seg if k < s]
        right = [k for k in by_seg if k > s]
        if left and right:
            for starts in by_seg.values():
                out.update(starts)
        elif left or right:
            out.update(by_seg[max(left) if left else min(right)])
    return out


def years_requirement_mentions(text_lower):
    """[(low, high, is_overall)] for every "N years" / "N yrs" mention that states a
    years REQUIREMENT. Non-experience SENTENCES are skipped as a whole; a
    conditional ("for the manager track") skips only the mention whose clause it
    belongs to (_conditional_mention_starts); whether a mention is preferred is
    judged within its own CLAUSE, then its sentence, then the posting section it
    sits under (_years_mention_is_preferred). `high` is the upper end of a range
    ("3-5 years" -> (3.0, 5.0)), else equal to `low`."""
    out = []
    contexts = _line_contexts(text_lower)
    for offset, sentence in _sentences_with_offsets(text_lower):
        if _YEARS_NOT_EXPERIENCE.search(sentence):
            continue
        conditional = _conditional_mention_starts(sentence)
        for m in _YEARS_RE.finditer(sentence):
            if m.start() in conditional:
                continue
            n = _num_value(m.group(1))
            if not 0 < n <= 30:
                continue
            section = None
            for line_start, line_ctx in contexts:
                if line_start > offset + m.start():
                    break
                section = line_ctx
            hi = _num_value(m.group(2)) if m.group(2) else n
            # "3-5 years, ideally 5": the soft word is about the HIGHER value only,
            # so it is blanked (offsets kept) before classifying the minimum.
            judged = sentence
            soft = _SOFT_UPPER_RE.match(sentence, m.end())
            if soft and _num_value(soft.group(1)) > n:
                kw = list(re.finditer(r"ideally|preferably", sentence[m.end():soft.start(1)]))
                a, b = m.end() + kw[-1].start(), soft.end()
                judged = sentence[:a] + " " * (b - a) + sentence[b:]
            judged = _scope_to_mention(judged, m.start(), m.end())
            if _years_mention_is_preferred(judged, m.start(), m.end(), section):
                continue
            is_overall = bool(_YEARS_OVERALL.search(_clause_around(sentence, m.start(), m.end())))
            out.append((float(n), float(max(hi, n)), is_overall))
    return out


# --- Requirement classification from the posting's own wording (backlog B) ----
# A skill's label (required / preferred) must be supported by what the posting
# says AROUND its mentions. Each mention is classified, most specific first:
#   negated     "Python is not required", "no prior Kubernetes experience"
#   preferred   a preferred signal in its clause, else its sentence
#   required    a required signal in its clause, else its sentence
#   <heading>   otherwise the context of the posting section it sits under
#               ("Requirements:" / "Nice to have:")
#   context     a bare mention ("our team uses Python")
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)|[;\n•|]")
_CLAUSE_END_RE = re.compile(r"[,:]|\b(?:but|while|whereas|however|although)\b")
_PREF_SIGNAL_RE = re.compile(
    r"\b(?:preferred|prefer|nice[\s-]to[\s-]haves?|good[\s-]to[\s-]have|bonus|a plus|"
    r"pluses|desirable|desired|advantageous|ideally|optional|would be (?:great|nice))\b")
_REQ_SIGNAL_RE = re.compile(
    r"\b(?:required|requires?|requirements?|must|must[\s-]haves?|mandatory|essential|"
    r"needed|needs?|qualifications|you have|you bring|you will need|you'll need|"
    r"proficien(?:t|cy)|strong|solid|expert(?:ise)?|hands-on|demonstrated|"
    r"experience (?:with|in|using))\b")
# Negation AFTER the mention, within its own clause: "Python is not required".
_POST_NEG_RE = re.compile(
    r"(?:\bnot\b|n't\b|\bnever\b)\s+(?:\w+\s+){0,2}?(?:required|needed|necessary|"
    r"mandatory|essential|expected|a\s+(?:requirement|must|prerequisite))\b")
_POST_STOP_RE = re.compile(r"[,.;:!?\n•|]|\b(?:and|but|or|while|whereas)\b")
# A posting section label: "Nice to have:", "Requirements", "What you'll bring:".
_SECTION_LABEL_RE = re.compile(r"^\s*[-*#•]*\s*([^:\n]{1,60}?)\s*:\s*(\S.*)?$")
_BARE_LABEL_RE = re.compile(
    r"(?:(?:minimum|basic|preferred|required|desired|additional)\s+)?"
    r"(?:qualifications|requirements|skills)|nice[\s-]to[\s-]haves?|bonus(?:\s+points)?|"
    r"must[\s-]haves?|responsibilities|what you(?:'ll| will) (?:do|bring|need)|benefits|"
    r"about (?:us|the role)")


def _label_context(label):
    if _PREF_SIGNAL_RE.search(label):
        return "preferred"
    if _REQ_SIGNAL_RE.search(label) or re.search(r"\b(skills|what you)\b", label):
        return "required"
    return "context"


def _line_contexts(text_lower):
    """[(line_start, context)] — the section context in force on each line."""
    out, ctx, pos = [], None, 0
    for line in text_lower.split("\n"):
        stripped = line.strip(" \t-*#•")
        m = _SECTION_LABEL_RE.match(line)
        if m and len(m.group(1).split()) <= 6:
            ctx = _label_context(m.group(1))
        elif stripped and len(stripped.split()) <= 5 and _BARE_LABEL_RE.fullmatch(stripped):
            ctx = _label_context(stripped)
        out.append((pos, ctx))
        pos += len(line) + 1
    return out


def _span(text, start, end, end_re, lo=0, hi=None):
    hi = len(text) if hi is None else hi
    s = lo
    for m in end_re.finditer(text, lo, start):
        s = m.end()
    e = end_re.search(text, end, hi)
    return s, (e.start() if e else hi)


# Negation BEFORE the mention, at most three words earlier in the same clause
# (a colon does not end it: "Not required: Java"): "no prior Kubernetes",
# "without Python", "don't need Go". Unlike resume negation, "zero"/"none"/"lack"
# are not cues here ("zero-downtime deploys with Kubernetes" requires Kubernetes).
_PRE_NEG_RE = re.compile(r"(?:\bno|\bnot|\bwithout|\bnever|n't)(?:\W+\w+){0,3}\W*$")
_PRE_CLAUSE_END_RE = re.compile(r"[.!?](?=\s|$)|[;,\n•|]|\b(?:but|however|while|whereas)\b")


def _mention_class(low, start, end, contexts):
    p0, _p1 = _span(low, start, end, _PRE_CLAUSE_END_RE)
    if _PRE_NEG_RE.search(low[p0:start]):
        return "negated"
    post = low[end:end + 60]
    stop = _POST_STOP_RE.search(post)
    if _POST_NEG_RE.search(post[:stop.start()] if stop else post):
        return "negated"
    s0, s1 = _span(low, start, end, _SENTENCE_END_RE)
    c0, c1 = _span(low, start, end, _CLAUSE_END_RE, s0, s1)
    for span in (low[c0:c1], low[s0:s1]):
        if _PREF_SIGNAL_RE.search(span):
            return "preferred"
        if _REQ_SIGNAL_RE.search(span):
            return "required"
    ctx = None
    for line_start, line_ctx in contexts:
        if line_start > start:
            break
        ctx = line_ctx
    return ctx or "context"


def mention_classes(skill, text):
    """The set of classes of every mention of `skill` (alias-aware, whole word)."""
    from skills import _allowed_variants, _variant_regex
    low = str(text or "").lower()
    contexts = _line_contexts(low)
    classes = set()
    for v in _allowed_variants(skill):
        for m in _variant_regex(v).finditer(low):
            classes.add(_mention_class(low, m.start(), m.end(), contexts))
    return classes


def classify_requirements(reqs, job, negation_only=False):
    """
    Verify each required / preferred / any-of skill's LABEL against the posting.

    A skill becomes UNCERTAIN — removed from the scored lists and recorded in
    `uncertain_requirements` as {"field", "skill", "reason"} — when:
      * any of its mentions is negated                    (reason "negated");
      * it is labeled required but the posting only marks it preferred
                                                          ("classified_preferred");
      * it is labeled preferred but the posting only marks it required
                                                          ("classified_required").
    With negation_only=True (the degraded rules extractor, whose labels are a
    keyword heuristic) only the negation check applies. A skill whose mentions
    carry no signal at all keeps its label, recorded as "context_only" in
    `requirement_classification`. An any-of group left with no members is dropped.
    Returns a NEW dict.
    """
    text = f"{job.get('title') or ''}\n{job.get('description') or ''}"
    out = dict(reqs)
    uncertain = list(reqs.get("uncertain_requirements") or [])
    classified = []

    def _check(field, label, skills):
        kept = []
        for s in skills or []:
            classes = mention_classes(s, text)
            reason = None
            if "negated" in classes:
                reason = "negated"
            elif not negation_only and label == "required" \
                    and "preferred" in classes and "required" not in classes:
                reason = "classified_preferred"
            elif not negation_only and label == "preferred" \
                    and "required" in classes and "preferred" not in classes:
                reason = "classified_required"
            if reason:
                uncertain.append({"field": field, "skill": s, "reason": reason})
                continue
            kept.append(s)
            classified.append({"field": field, "skill": s,
                               "classification": label if label in classes
                               else "context_only"})
        return kept

    out["required_skills"] = _check("required_skills", "required", reqs.get("required_skills"))
    out["preferred_skills"] = _check("preferred_skills", "preferred",
                                     reqs.get("preferred_skills"))
    groups = []
    for g in reqs.get("required_any_of") or []:
        kept = _check("required_any_of", "required", g)
        if kept:
            groups.append(kept)
    out["required_any_of"] = groups
    out["uncertain_requirements"] = uncertain
    out["requirement_classification"] = classified
    return out


def extract_requirements_rule_based(job):
    """
    Best-effort, LLM-free requirements from a job description.
    Returns the same shape as the LLM extractor:
      {required_skills, required_any_of, preferred_skills, min_years_experience, responsibilities}
    Heuristic: a matched skill mentioned near a PREFERRED signal goes to preferred;
    otherwise it's treated as required. Falls back to 'required' when ambiguous.
    """
    desc = (job.get("description") or "") + " " + (job.get("title") or "")
    text_lower = desc.lower()
    tokens = _tokens(text_lower)

    # Which skills appear at all
    present = [s for s in SKILL_VOCAB if _skill_in(s, text_lower, tokens)]

    required, preferred = [], []
    for skill in present:
        # Look at a window around the skill's first mention for a preferred-signal.
        idx = text_lower.find(skill.lower())
        window = text_lower[max(0, idx - 60): idx + 60]
        if any(sig in window for sig in PREFERRED_SIGNALS):
            preferred.append(skill)
        else:
            required.append(skill)

    # De-dupe, keep order
    def _uniq(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    out = {
        "required_skills": _uniq(required),
        "required_any_of": [],
        "preferred_skills": _uniq(preferred),
        "min_years_experience": _years_from_text(text_lower),
        "responsibilities": [],
        # Every skill above was FOUND in the posting text, so nothing is ungrounded;
        # same shape as the grounded LLM extraction (job_parser.ground_requirements).
        "ungrounded_requirements": [],
        # min_years_experience above is read from the posting itself.
        "ungrounded_min_years": None,
        # Provenance travels WITH the data, so downstream evaluation can always tell
        # a crude keyword extraction from an LLM extraction.
        "extraction_method": "rule_based",
    }
    # A negated mention ("Python is not required") is never a requirement, here
    # either; the keyword labels themselves are too crude to cross-check.
    return classify_requirements(out, job, negation_only=True)