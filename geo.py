"""
Structured country / region resolution for geographic eligibility.

The previous resolver put COUNTRIES inside REGION keyword lists ("germany" was a
spelling of "europe"), so a "Germany only" posting and a French candidate both
resolved to "europe" and matched. Here text resolves to a structured GeoScope:

    countries   ISO 3166-1 alpha-2 codes ("de", "fr", "us", ...)
    regions     region ids from a containment hierarchy (below)
    worldwide   the text opens the posting to everyone
    excluded    the text contains an exclusion ("except", "excluding", "not in",
                "outside"): the scope cannot be read positively -> callers treat
                the posting as unresolved instead of guessing

Hierarchy (child -> parent):

    world
     ├─ americas ─┬─ north_america  (us, ca, mx)
     │            └─ latam          (mx, br, ar, ...)       mx is in both
     ├─ emea ─────┬─ europe ── eu   (member states)
     │            ├─ middle_east
     │            └─ africa
     └─ apac

Matching is greedy longest-phrase over normalized words, so "latin america" never
also yields "america", and "south africa" never also yields "africa". US state
names and the two-letter state codes after a comma ("Austin, TX") resolve to the
US. Ambiguous names are deliberately NOT resolved: "georgia" (US state or country),
"ca" on its own (California or Canada), "in"/"or"/"me" outside the ", XX" form.

Eligibility (eligibility(posting_text, candidate_text)):
    posting worldwide                                    -> eligible
    either side unresolved, or the posting has excluded  -> unknown
    candidate COUNTRY covered by a posting country or by
      a posting region that contains it                  -> eligible
    candidate country not covered                        -> ineligible
    candidate REGION only (no country):
      the posting covers that region or an ancestor      -> eligible
      the posting covers something INSIDE that region    -> unknown (maybe)
      otherwise                                          -> ineligible
"""
import re
from dataclasses import dataclass, field

REGION_PARENT = {
    "americas": "world", "north_america": "americas", "latam": "americas",
    "emea": "world", "europe": "emea", "eu": "europe", "middle_east": "emea",
    "africa": "emea", "apac": "world",
}

REGION_NAMES = {
    "world": (),          # reached only via WORLDWIDE terms
    "americas": ("americas", "the americas"),
    "north_america": ("north america", "northern america"),
    "latam": ("latam", "latin america", "south america", "central america"),
    "emea": ("emea",),
    "europe": ("europe", "european", "eea", "european economic area", "cet", "cest"),
    "eu": ("eu", "european union"),
    "middle_east": ("middle east", "mena", "gcc"),
    "africa": ("africa", "african"),
    "apac": ("apac", "asia", "asia pacific", "asia-pacific", "oceania", "anz"),
}

WORLDWIDE = ("worldwide", "anywhere", "global", "globally", "anywhere in the world",
             "international", "any location", "everywhere", "world")
EXCLUSION_WORDS = ("except", "excluding", "exclude", "not in", "outside", "outside of",
                   "no ", "non-")

_EU = {"at", "be", "bg", "hr", "cy", "cz", "dk", "ee", "fi", "fr", "de", "gr", "hu", "ie",
       "it", "lv", "lt", "lu", "mt", "nl", "pl", "pt", "ro", "sk", "si", "es", "se"}

# code: (direct regions, names/aliases)
COUNTRIES = {
    "us": ({"north_america"}, ("us", "usa", "u.s.", "u.s.a.", "united states",
                               "united states of america", "america")),
    "ca": ({"north_america"}, ("canada",)),
    "mx": ({"north_america", "latam"}, ("mexico",)),
    "br": ({"latam"}, ("brazil", "brasil")),
    "ar": ({"latam"}, ("argentina",)),
    "co": ({"latam"}, ("colombia",)),
    "cl": ({"latam"}, ("chile",)),
    "pe": ({"latam"}, ("peru",)),
    "uy": ({"latam"}, ("uruguay",)),
    "cr": ({"latam"}, ("costa rica",)),
    "ec": ({"latam"}, ("ecuador",)),
    "gb": ({"europe"}, ("uk", "u.k.", "united kingdom", "great britain", "britain",
                        "england", "scotland", "wales", "northern ireland")),
    "ie": ({"europe"}, ("ireland",)),
    "de": ({"europe"}, ("germany", "deutschland")),
    "fr": ({"europe"}, ("france",)),
    "es": ({"europe"}, ("spain",)),
    "pt": ({"europe"}, ("portugal",)),
    "it": ({"europe"}, ("italy",)),
    "nl": ({"europe"}, ("netherlands", "the netherlands", "holland")),
    "be": ({"europe"}, ("belgium",)),
    "lu": ({"europe"}, ("luxembourg",)),
    "at": ({"europe"}, ("austria",)),
    "ch": ({"europe"}, ("switzerland",)),
    "dk": ({"europe"}, ("denmark",)),
    "se": ({"europe"}, ("sweden",)),
    "no": ({"europe"}, ("norway",)),
    "fi": ({"europe"}, ("finland",)),
    "is": ({"europe"}, ("iceland",)),
    "pl": ({"europe"}, ("poland",)),
    "cz": ({"europe"}, ("czech republic", "czechia")),
    "sk": ({"europe"}, ("slovakia",)),
    "hu": ({"europe"}, ("hungary",)),
    "ro": ({"europe"}, ("romania",)),
    "bg": ({"europe"}, ("bulgaria",)),
    "gr": ({"europe"}, ("greece",)),
    "hr": ({"europe"}, ("croatia",)),
    "si": ({"europe"}, ("slovenia",)),
    "rs": ({"europe"}, ("serbia",)),
    "ee": ({"europe"}, ("estonia",)),
    "lv": ({"europe"}, ("latvia",)),
    "lt": ({"europe"}, ("lithuania",)),
    "cy": ({"europe"}, ("cyprus",)),
    "mt": ({"europe"}, ("malta",)),
    "ua": ({"europe"}, ("ukraine",)),
    "tr": ({"europe", "middle_east"}, ("turkey", "turkiye")),
    "ae": ({"middle_east"}, ("uae", "united arab emirates", "dubai", "abu dhabi")),
    "sa": ({"middle_east"}, ("saudi arabia", "ksa")),
    "qa": ({"middle_east"}, ("qatar",)),
    "il": ({"middle_east"}, ("israel",)),
    "jo": ({"middle_east"}, ("jordan",)),
    "kw": ({"middle_east"}, ("kuwait",)),
    "bh": ({"middle_east"}, ("bahrain",)),
    "om": ({"middle_east"}, ("oman",)),
    "lb": ({"middle_east"}, ("lebanon",)),
    "eg": ({"africa", "middle_east"}, ("egypt",)),
    "ng": ({"africa"}, ("nigeria", "lagos")),
    "ke": ({"africa"}, ("kenya", "nairobi")),
    "za": ({"africa"}, ("south africa",)),
    "gh": ({"africa"}, ("ghana",)),
    "ma": ({"africa"}, ("morocco",)),
    "tn": ({"africa"}, ("tunisia",)),
    "et": ({"africa"}, ("ethiopia",)),
    "ug": ({"africa"}, ("uganda",)),
    "rw": ({"africa"}, ("rwanda",)),
    "in": ({"apac"}, ("india",)),
    "pk": ({"apac"}, ("pakistan",)),
    "bd": ({"apac"}, ("bangladesh",)),
    "lk": ({"apac"}, ("sri lanka",)),
    "jp": ({"apac"}, ("japan",)),
    "kr": ({"apac"}, ("south korea", "korea")),
    "cn": ({"apac"}, ("china",)),
    "tw": ({"apac"}, ("taiwan",)),
    "hk": ({"apac"}, ("hong kong",)),
    "sg": ({"apac"}, ("singapore",)),
    "my": ({"apac"}, ("malaysia",)),
    "th": ({"apac"}, ("thailand",)),
    "vn": ({"apac"}, ("vietnam", "viet nam")),
    "ph": ({"apac"}, ("philippines",)),
    "id": ({"apac"}, ("indonesia",)),
    "au": ({"apac"}, ("australia",)),
    "nz": ({"apac"}, ("new zealand",)),
}
for _c in _EU:
    COUNTRIES[_c][0].add("eu")

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin",
    "wy": "wyoming", "dc": "district of columbia",
}
# "georgia" is a US state AND a country: never resolved. "washington" alone is
# allowed (the state; "Washington, DC" is also US either way).
_AMBIGUOUS = {"georgia"}


@dataclass(frozen=True)
class GeoScope:
    countries: frozenset = field(default_factory=frozenset)
    regions: frozenset = field(default_factory=frozenset)
    worldwide: bool = False
    excluded: bool = False

    @property
    def resolved(self):
        return bool(self.countries or self.regions or self.worldwide)


def _ancestors(region):
    out = set()
    r = REGION_PARENT.get(region)
    while r:
        out.add(r)
        r = REGION_PARENT.get(r)
    return out


def region_closure(regions):
    """Regions plus all their ancestors."""
    out = set(regions)
    for r in regions:
        out |= _ancestors(r)
    return out


def _is_within(region, outer):
    return region == outer or outer in _ancestors(region)


def country_regions(code):
    return region_closure(COUNTRIES[code][0])


# phrase (tuple of words) -> ("country", code) | ("region", id) | ("world", None)
def _build_phrases():
    table = {}

    def add(phrase, value):
        key = tuple(phrase.split())
        table.setdefault(key, value)
    for code, (_regions, names) in COUNTRIES.items():
        for n in names:
            add(n, ("country", code))
    for state in US_STATES.values():
        add(state, ("country", "us"))
    for region, names in REGION_NAMES.items():
        for n in names:
            add(n, ("region", region))
    for w in WORLDWIDE:
        add(w, ("world", None))
    return table


_PHRASES = _build_phrases()
_MAX_WORDS = max(len(k) for k in _PHRASES)
_STATE_SUFFIX_RE = re.compile(r",\s*([A-Za-z]{2})\s*(?:[,;)/|]|$)")


def _words(text):
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*\.?|[a-z]+-[a-z]+", text.lower())


def resolve(text):
    """GeoScope for a free-text location / eligibility statement."""
    raw = str(text or "")
    low = " " + " ".join(raw.lower().split()) + " "
    excluded = any(f" {w}" in low for w in EXCLUSION_WORDS)
    words = [w.rstrip(".") if w.count(".") == 1 and w.endswith(".") else w
             for w in _words(raw)]
    countries, regions, world = set(), set(), False
    i = 0
    while i < len(words):
        hit = None
        for n in range(min(_MAX_WORDS, len(words) - i), 0, -1):
            key = tuple(words[i:i + n])
            if key in _PHRASES and not (n == 1 and key[0] in _AMBIGUOUS):
                hit = (n, _PHRASES[key])
                break
        if hit is None:
            i += 1
            continue
        n, (kind, value) = hit
        if kind == "world" and i + n < len(words) and words[i + n] in ("in", "within"):
            # "anywhere in the EU" is a scope, not worldwide: let what follows decide.
            i += n
            continue
        if kind == "country":
            countries.add(value)
        elif kind == "region":
            regions.add(value)
        else:
            world = True
        i += n
    # "Austin, TX" / "San Jose, CA": a two-letter US state code after a comma.
    for m in _STATE_SUFFIX_RE.finditer(raw):
        if m.group(1).lower() in US_STATES and m.group(1).isupper():
            countries.add("us")
    return GeoScope(frozenset(countries), frozenset(regions), world, excluded)


def _covers_country(posting, code):
    return code in posting.countries or bool(country_regions(code) & posting.regions)


def eligibility(posting_text, candidate_text):
    """'eligible' / 'ineligible' / 'unknown' — see the module docstring."""
    if not str(candidate_text or "").strip() or not str(posting_text or "").strip():
        return "unknown"
    p = resolve(posting_text)
    if p.excluded:
        return "unknown"
    if p.worldwide:
        return "eligible"
    if not p.resolved:
        return "unknown"
    c = resolve(candidate_text)
    if c.excluded or not (c.countries or c.regions):
        return "unknown"
    if c.countries:
        return ("eligible" if any(_covers_country(p, code) for code in c.countries)
                else "ineligible")
    # Candidate named only regions.
    for r in c.regions:
        if region_closure({r}) & p.regions:
            return "eligible"
    inside = any(_is_within(pr, r) for pr in p.regions for r in c.regions) or \
        any(country_regions(pc) & set(c.regions) for pc in p.countries)
    return "unknown" if inside else "ineligible"