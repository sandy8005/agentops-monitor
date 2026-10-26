"""Round 5, backlog B: clear remote location forms are recognized as a stated remote
mode; negated, hedged or mixed wording stays unknown. "Remote, OR" is a remote role
in Oregon (a US state code resolved via geo.py), not the word "or". "Worldwide" /
"Anywhere" alone state a SCOPE, not a work mode, so they stay unknown unless the
posting's own mode or text also says remote. Synthetic locations only."""
import pytest

from job_source import _location_says_remote, work_mode_status

REMOTE = [
    "Remote",
    "Remote-first",
    "Remote first",
    "Remote-only",
    "Fully remote",
    "100% remote",
    "Work from home",
    "Work-from-home",
    "Working from home (US)",
    "WFH",
    "WFH - Canada",
    "Telecommute",
    "Telecommuting, USA",
    "Remote, OR",            # Oregon
    "Remote, NO",            # Norway
    "Remote, CA",
    "Remote (OR)",
    "Remote / OR",
    "Remote, OR, US",
    "Remote - Worldwide",
    "Anywhere (Remote)",
    "Worldwide, fully remote",
]

UNKNOWN = [
    "Not remote",
    "Non-remote",
    "No remote work",
    "Remote-friendly",
    "Remote-possible",
    "Remote or onsite",
    "Remote, or onsite",     # lowercase "or" is the conjunction
    "Remote OR onsite",      # not a ", XX" place code
    "Hybrid-remote",
    "Hybrid remote",
    "Remote-first (hybrid optional)",
    "Work from home optional",
    "Work from home or office",
    "Partially remote",
    "Remoteville, TX",
    "Worldwide",
    "Anywhere",
    "Anywhere in the EU",
    "Portland, OR",
    "Homestead, FL",
    "",
    None,
]


@pytest.mark.parametrize("loc", REMOTE)
def test_clear_remote_forms(loc):
    assert _location_says_remote(loc) is True
    assert work_mode_status({"work_mode": None, "location": loc}, "remote") == "verified"


@pytest.mark.parametrize("loc", UNKNOWN)
def test_hedged_or_absent_forms_stay_unknown(loc):
    assert _location_says_remote(loc) is False
    assert work_mode_status({"work_mode": None, "location": loc}, "remote") == "unknown"


@pytest.mark.parametrize("loc", ["Worldwide", "Anywhere"])
def test_worldwide_counts_when_the_posting_states_remote(loc):
    assert work_mode_status({"work_mode": "remote", "location": loc}, "remote") == "verified"


def test_remote_location_is_a_mismatch_for_onsite_search():
    assert work_mode_status({"work_mode": None, "location": "Remote, OR"}, "onsite") == "mismatch"
