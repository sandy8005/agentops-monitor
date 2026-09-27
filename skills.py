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


def skill_in_text(skill, low, tokens):
    """True if ANY alias of `skill` occurs in the indexed text (whole word / phrase).
    Ambiguous short aliases count only when they are the requested term itself."""
    asked = _norm(skill)
    for v in skill_variants(skill):
        if v in AMBIGUOUS_VARIANTS and v != asked:
            continue
        if _variant_present(v, low, tokens):
            return True
    return False


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
