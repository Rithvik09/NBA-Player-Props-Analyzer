"""Which injury-report reasons count as a *medical* absence.

`RecoveryContext` builds two indexes from the same official reports, and they
deliberately disagree:

  injury_index   every row, filtered later on `status == 'Out'` alone. It
                 answers "who cannot play tonight", so a G League assignment,
                 a suspension or a personal absence belongs in it -- those
                 players are genuinely unavailable.

  medical_index  additionally requires the reason to be medical. It answers
                 "when was this player last hurt", feeding games_since_return
                 and missed_games_before_return, whose spec says "latest
                 confirmed *medical* Out game". A G League assignment is not an
                 injury spell and must not start one.

Those two filters must never be unified. These tests pin both halves.

The medical half was implemented as `'injury' in reason.lower()`, which works
for the NBA's usual "Injury/Illness - <part>; <detail>" text but silently
excludes category labels that are medical without carrying the word. Measured
against the 3,216 archived reports, all 361 rows reading only "Concussion
Protocol" were invisible -- 116 distinct player-days over 33 players, in every
season since 2022-23.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.historical_injuries import is_medical_absence, normalize_name
from tests.test_full_feature_audit import BOS, fixture

# PLAYER_ID 2544 in the shared fixture; the PDFs print "Last, First".
PLAYER = "James, LeBron"
PLAYER_ID = 2544
DAY = "2025-01-10"


def _report(reason: str, *, status: str = "Out", day: str = DAY,
            published: str = "2025-01-10T12:00:00-05:00") -> dict:
    return {
        "published_at": published,
        "rows": [{
            "team": "Boston Celtics",
            "game_date": day,
            "player": PLAYER,
            "status": status,
            "reason": reason,
        }],
    }


@pytest.fixture
def context(monkeypatch):
    """A RecoveryContext whose only injury reports are the ones a test names."""
    def _build(*reports: dict):
        monkeypatch.setattr("src.recovered_features.reports", lambda: list(reports))
        _logs, _raw, inputs = fixture()
        return inputs.recovery
    return _build


# ---------------------------------------------------------------------------
# The medical half: what starts an absence spell
# ---------------------------------------------------------------------------

MEDICAL_EXAMPLES = [
    "Injury/Illness - Left Knee; Soreness",   # the ordinary case
    "Injury/Illness - Illness",
    "Concussion Protocol",                     # medical, but has no "injury"
]


@pytest.mark.parametrize("reason", MEDICAL_EXAMPLES)
def test_a_medical_absence_is_indexed_as_one(context, reason):
    ctx = context(_report(reason))
    assert PLAYER_ID in ctx.medical_index, (
        f"{reason!r} did not register as a medical absence, so it can never "
        f"start an injury spell for games_since_return"
    )
    assert DAY in {day for _stamp, day in ctx.medical_index[PLAYER_ID]}


NON_MEDICAL_EXAMPLES = [
    "G League - Two-Way",
    "G League - On Assignment",
    "Not With Team",
    "Personal Reasons",
    "League Suspension",
    "Team Suspension",
    "Rest",
    "Trade Pending",
    "Coach's Decision",
    "Ineligible To Play",
]


@pytest.mark.parametrize("reason", NON_MEDICAL_EXAMPLES)
def test_a_non_medical_absence_never_starts_an_injury_spell(context, reason):
    """These are real absences but not injuries.

    Counting them would date a return-from-injury from a G League recall or a
    suspension, which is how `missed_games_before_return` becomes fiction.
    """
    ctx = context(_report(reason))
    assert PLAYER_ID not in ctx.medical_index, (
        f"{reason!r} was treated as a medical absence"
    )


def test_a_player_who_is_merely_questionable_is_not_absent(context):
    ctx = context(_report("Injury/Illness - Left Knee; Soreness", status="Questionable"))
    assert PLAYER_ID not in ctx.medical_index


# ---------------------------------------------------------------------------
# The availability half: who cannot play, for any reason
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reason", NON_MEDICAL_EXAMPLES + MEDICAL_EXAMPLES)
def test_team_availability_ignores_the_reason_entirely(context, reason):
    """The two filters must stay separate.

    `team_key_players_out` and `opp_injury_impact` count anyone who is Out: a
    suspended or G-League-assigned starter is just as absent as an injured one.
    Narrowing this to medical reasons would under-report absences on exactly
    the rosters where it matters most.
    """
    ctx = context(_report(reason))
    rows = ctx.injury_index[(normalize_name("Boston Celtics"), DAY)]
    assert rows, f"{reason!r} was dropped from the availability index"
    assert rows[0][1][0]["status"] == "Out"


# ---------------------------------------------------------------------------
# The predicate itself, apart from the index
# ---------------------------------------------------------------------------
#
# The tests above drive `is_medical_absence` through RecoveryContext, which is
# how it is actually used but also hides the edge shapes. These pin it directly.

@pytest.mark.parametrize("reason,expected", [
    ("Injury/Illness - Left Knee; Soreness", True),
    ("Concussion Protocol", True),
    ("CONCUSSION PROTOCOL", True),                 # teams vary the casing
    ("concussion protocol", True),
    # The archive carries 1,590 Out rows with this prefix; the real reason
    # follows it and must still be read.
    ("NOT YET SUBMITTED Injury/Illness - Left Knee; Surgery", True),
    ("G League - Two-Way", False),
    ("Health and Safety Protocols", False),
    ("Return to Competition Reconditioning", False),
    ("", False),
    (None, False),                                  # must not raise
])
def test_the_predicate_classifies_each_reason_shape(reason, expected):
    assert is_medical_absence(reason) is expected


@pytest.mark.parametrize("reason", [
    # Excluded labels are excluded only when they stand ALONE. Where a reason
    # also names an injury, the injury decides -- 127 archived rows look like
    # this, and reading the exclusion as absolute would drop them.
    "Injury/Illness - Right Ankle; Return to Competition Reconditioning",
    "Injury/Illness - N/a; Return to Competition Reconditioning/Back Soreness",
    "Injury/Illness - Left Knee; Soreness / G League Assignment",
])
def test_a_compound_reason_naming_an_injury_stays_medical(reason):
    assert is_medical_absence(reason) is True


@pytest.mark.parametrize("reason", [
    "Rest - Left Knee Injury Management",
    "Injury Management G League - Two-Way",
])
def test_known_leading_label_conflicts_are_left_classified_as_medical(reason):
    """Pinned as a known state, not endorsed.

    About 26 archived rows lead with Rest or a G League assignment yet name an
    injury, so the substring rule calls them medical. That predates this
    predicate and is deliberately unchanged: load-managing a named joint is
    arguably medical, and reclassifying it is a spec decision. This test exists
    so the behaviour is recorded rather than rediscovered as a surprise.
    """
    assert is_medical_absence(reason) is True
