"""
Canonical skill normalization — ONE vocabulary for the whole pipeline.

Every stage that compares skills goes through this module: resume grounding
(parser.ground_skills), requirements (normalize_requirements), scoring and
project matching (scorer.calculate_match_score).

canonical_skill("K8s") == canonical_skill("kubernetes") == "kubernetes", and
skill_variants("kubernetes") lists every surface form to look for in free text.
Extend SKILL_ALIAS_GROUPS; the FIRST entry of each group is the canonical name.
"""
import re

SKILL_ALIAS_GROUPS = [
    ["kubernetes", "k8s"],
    ["postgresql", "postgres", "psql"],
    ["javascript", "js", "ecmascript"],
    ["typescript", "ts"],
    ["machine learning", "ml"],
    ["artificial intelligence", "ai"],
    ["natural language processing", "nlp"],
    ["computer vision"],
    ["google cloud platform", "gcp", "google cloud"],
    ["amazon web services", "aws"],
    ["microsoft azure", "azure"],
    ["golang", "go"],
    ["node.js", "nodejs", "node"],
    ["react", "react.js", "reactjs"],
    ["vue", "vue.js", "vuejs"],
    ["next.js", "nextjs"],
    ["c#", "csharp", "c sharp"],
    ["c++", "cpp"],
    [".net", "dotnet"],
    ["scikit-learn", "sklearn", "scikit learn"],
    ["ci/cd", "cicd", "ci cd", "continuous integration"],
    ["mongodb", "mongo"],
    ["elasticsearch", "elastic search"],
    ["large language models", "llm", "llms"],
    ["amazon s3", "s3"],
    ["rest", "rest api", "restful", "rest apis"],
]

# Aliases that are also ordinary English words. They only match when they are the
# term actually asked for: "golang" does not match "ready to go", but "Go" matches "Go".
AMBIGUOUS_VARIANTS = {"go", "node", "rest"}


def _norm(text):
    return " ".join(str(text or "").strip().lower().split())


def _build_index(groups):
    canon, variants = {}, {}
    for group in groups:
        names = [_norm(g) for g in group if _norm(g)]
        if not names:
            continue
        head = names[0]
        for n in names:
            canon.setdefault(n, head)
        variants.setdefault(head, [])
        for n in names:
            if n not in variants[head]:
                variants[head].append(n)
    return canon, variants


_CANON, _VARIANTS = _build_index(SKILL_ALIAS_GROUPS)


def canonical_skill(skill):
    """Canonical (lower-case) name for a skill; unknown skills are just normalized."""
    s = _norm(skill)
    return _CANON.get(s, s)


def skill_variants(skill):
    """Every surface form that means the same skill (canonical first)."""
    c = canonical_skill(skill)
    return list(_VARIANTS.get(c, [c]))


_TOKEN_RE = re.compile(r"[a-z0-9\+\#\.]+")


def text_index(text):
    """(normalized text, token set) — the pair skill_in_text() looks things up in."""
    low = _norm(text)
    tokens = set(_TOKEN_RE.findall(low))
    tokens |= {t.rstrip(".") for t in tokens if t.endswith(".")}
    return low, tokens


def _variant_present(variant, low, tokens):
    if not variant:
        return False
    if " " in variant or "/" in variant or "-" in variant:
        return variant in low            # phrase match
    return variant in tokens             # whole-word match ('go' never hits 'django')


def _allowed_variants(skill, surface_forms=None):
    """Aliases that may be searched for `skill`. An ambiguous short alias ('go',
    'node', 'rest') is allowed only when it is the requested term itself OR one of
    the ORIGINAL surface forms the requirement was written with — canonicalization
    ('Go' -> 'golang') must not throw away the fact that the job said 'Go' (R11)."""
    asked = {_norm(skill)} | {_norm(f) for f in (surface_forms or []) if _norm(f)}
    return [v for v in skill_variants(skill)
            if not (v in AMBIGUOUS_VARIANTS and v not in asked)]


def skill_in_text(skill, low, tokens, surface_forms=None):
    """True if ANY allowed alias of `skill` occurs in the indexed text (whole word /
    phrase). See _allowed_variants for the ambiguous-alias rule."""
    for v in _allowed_variants(skill, surface_forms):
        if _variant_present(v, low, tokens):
            return True
    return False


# --- Negation-aware evidence (R08) -------------------------------------------
# "No Python experience" must not count as Python evidence. A mention is NEGATED
# when a negation cue appears shortly before it in the same clause. Clause
# boundaries: sentence punctuation, semicolons, newlines, and contrastive "but".
_NEG_CUE_RE = re.compile(
    r"(?<![a-z0-9\-])(no|not|never|without|lacking|lacks|lack|none|zero|"
    r"unfamiliar with|no prior|no professional)(?![a-z0-9\-])")
_CLAUSE_SPLIT_RE = re.compile(r"[.!?;\n\u2022|]|\bbut\b|\bhowever\b")
_MAX_CUE_DISTANCE_WORDS = 4


def _variant_regex(variant):
    parts = [re.escape(p) for p in variant.split(" ")]
    return re.compile(r"(?<![a-z0-9])" + r"\s+".join(parts) + r"(?![a-z0-9+#])")


def _mention_is_negated(raw_lower, start):
    """True if the mention starting at `start` is inside a negated phrase."""
    clause_start = 0
    for m in _CLAUSE_SPLIT_RE.finditer(raw_lower, 0, start):
        clause_start = m.end()
    prefix = raw_lower[clause_start:start]
    cues = list(_NEG_CUE_RE.finditer(prefix))
    if not cues:
        return False
    between = prefix[cues[-1].end():]
    return len(between.split()) <= _MAX_CUE_DISTANCE_WORDS


# Negation AFTER the mention (N05): "Python: none", "Python experience: none",
# "Python - no", "Python (none)", "Python: not yet", "Python — n/a".
_NEG_SUFFIX_RE = re.compile(
    r"^\s*(?:experience|skills?|knowledge|exposure|proficiency|level)?\s*"
    r"(?:[:=\-\u2013\u2014(]\s*)"
    r"(?:none|no|nil|n/?a|zero|0(?!\.\d|\s*\+|\d)|not\b|never\b|no experience)")
# Aspirational / uncertain mentions are NOT evidence of competency.
_UNCERTAIN_CUE_RE = re.compile(
    r"(?<![a-z0-9\-])(currently learning|learning|want to learn|plan to learn|planning to learn|"
    r"interested in|aspiring|hoping to|eager to learn|self-studying|studying)(?![a-z0-9\-])")


def _mention_state(raw_lower, start, end):
    """'negated' | 'uncertain' | 'affirmative' for one mention."""
    if _mention_is_negated(raw_lower, start):
        return "negated"
    # a ':' or '-' after the skill is part of the same clause; look ~40 chars ahead
    suffix = raw_lower[end:min(len(raw_lower), end + 40)]
    nl = suffix.find("\n")
    if nl >= 0:
        suffix = suffix[:nl]
    if _NEG_SUFFIX_RE.match(suffix):
        return "negated"
    clause_start = 0
    for m in _CLAUSE_SPLIT_RE.finditer(raw_lower, 0, start):
        clause_start = m.end()
    prefix = raw_lower[clause_start:start]
    cues = list(_UNCERTAIN_CUE_RE.finditer(prefix))
    if cues and len(prefix[cues[-1].end():].split()) <= _MAX_CUE_DISTANCE_WORDS:
        return "uncertain"
    return "affirmative"


def skill_mention_states(skill, raw_text, surface_forms=None):
    """{'affirmative': n, 'negated': n, 'uncertain': n} for mentions of `skill`."""
    raw_lower = str(raw_text or "").lower()
    out = {"affirmative": 0, "negated": 0, "uncertain": 0}
    for v in _allowed_variants(skill, surface_forms):
        for m in _variant_regex(v).finditer(raw_lower):
            out[_mention_state(raw_lower, m.start(), m.end())] += 1
    return out


def skill_mentions(skill, raw_text, surface_forms=None):
    """(affirmative_count, non_affirmative_count). Negated AND uncertain mentions are
    both non-affirmative: neither is evidence of the skill."""
    st = skill_mention_states(skill, raw_text, surface_forms)
    return st["affirmative"], st["negated"] + st["uncertain"]


def affirmative_skill_in_text(skill, raw_text, surface_forms=None):
    """True only if the text contains at least one NON-negated mention of the skill.
    'Experienced Java engineer. No Python experience.' -> Python is False."""
    aff, _neg = skill_mentions(skill, raw_text, surface_forms)
    return aff > 0


def skill_in_set(skill, canonical_set):
    return canonical_skill(skill) in canonical_set


def canonical_set(skills):
    return {canonical_skill(s) for s in (skills or []) if _norm(s)}


def _uniq(items):
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_requirements(requirements):
    """
    Canonicalize and DEDUPLICATE requirements so one competency is never counted
    twice: aliases collapse ("K8s" + "Kubernetes"), an any-of group containing a
    flat required skill is dropped ("Python" AND "Python or Java" is just "Python"),
    duplicate groups collapse, a one-member group becomes a flat skill, and a skill
    that is both required and preferred stays only in required. Returns a NEW dict.
    """
    req = dict(requirements or {})
    # Keep every ORIGINAL surface form per canonical skill (R11): 'Go' canonicalizes
    # to 'golang', but matching must still be allowed to look for the word 'go'.
    surface = {k: list(v) for k, v in (req.get("_surface_forms") or {}).items()}
    def _remember(raw):
        c = canonical_skill(raw)
        if c:
            forms = surface.setdefault(c, [])
            n = _norm(raw)
            if n and n not in forms:
                forms.append(n)
    for raw in list(req.get("required_skills") or []) + list(req.get("preferred_skills") or []):
        _remember(raw)
    for g in req.get("required_any_of") or []:
        for raw in (g or []):
            _remember(raw)
    req["_surface_forms"] = surface
    required = _uniq([canonical_skill(s) for s in req.get("required_skills") or []])
    groups, seen_groups = [], set()
    for g in req.get("required_any_of") or []:
        members = _uniq([canonical_skill(s) for s in (g or [])])
        if not members:
            continue
        if len(members) == 1:
            if members[0] not in required:
                required.append(members[0])
            continue
        key = frozenset(members)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        groups.append(members)
    required_set = set(required)
    groups = [g for g in groups if not (set(g) & required_set)]
    preferred = [s for s in _uniq([canonical_skill(s) for s in req.get("preferred_skills") or []])
                 if s not in required_set]
    req["required_skills"] = required
    req["required_any_of"] = groups
    req["preferred_skills"] = preferred
    req.setdefault("min_years_experience", 0.0)
    return req