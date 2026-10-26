"""Round 6, backlog B: a lowercase "or" between PLACES is a place list, not hedged
wording. "Remote - US or Canada" used to state no work mode (the "or" read like
"remote or onsite"), while "Remote - US/Canada" was remote. Hedging stays when
"or" joins work modes. Synthetic location strings only."""
import pytest

from job_source import _location_says_remote, work_mode_status


@pytest.mark.parametrize("loc", [
    "Remote - US or Canada",
    "Remote (US or Canada)",
    "Remote - US, Canada or Mexico",
    "Remote - EMEA or APAC",
    "Remote (UK or Ireland)",
    "Remote - United States or Canada",
    "US or Canada (Remote)",
    "Remote - Germany or Austria",
    "Remote - US and Canada",
    "Remote - Europe or Latin America",
    # unchanged positives
    "Remote - US/Canada",
    "Remote - Portland, OR",
])
def test_place_list_is_remote(loc):
    assert _location_says_remote(loc) is True


@pytest.mark.parametrize("loc", [
    "Remote or onsite",
    "Remote or hybrid - US",
    "Remote or on-site (US or Canada)",
    "US or Canada, remote or office",
    "Remote - US or onsite",
    "Remote or in-office",
    "Onsite in US or remote",
    "Remote - US or Canada, hybrid",
    "Remote possible - US or Canada",
])
def test_work_mode_hedging_stays_unknown(loc):
    assert _location_says_remote(loc) is False


def test_place_list_verifies_a_remote_constraint():
    job = {"work_mode": None, "location": "Remote - US or Canada"}
    assert work_mode_status(job, "remote") == "verified"
