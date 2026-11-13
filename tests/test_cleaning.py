"""Cleaning rules, tested on small hand-made match tables."""

import hashlib
import json

import pandas as pd
import pytest

from src.data import cleaning, mappings
from src.data.inspect_raw import MATCHES_COLUMNS


def raw_row(**overrides):
    row = {
        "id": "1", "season": "2007/08", "city": "Bangalore", "date": "2008-04-18",
        "match_type": "League", "player_of_match": "X", "venue": "M Chinnaswamy Stadium",
        "team1": "Royal Challengers Bangalore", "team2": "Kolkata Knight Riders",
        "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field",
        "winner": "Kolkata Knight Riders", "result": "runs", "result_margin": "140",
        "target_runs": "223", "target_overs": "20", "super_over": "N", "method": "NA",
        "umpire1": "A", "umpire2": "B",
    }
    row.update(overrides)
    return row


def frame(*rows):
    return pd.DataFrame(list(rows), columns=MATCHES_COLUMNS)


def test_team_aliases_merge_renamed_franchises_only():
    assert mappings.normalize_team("Delhi Daredevils") == "Delhi Capitals"
    assert mappings.normalize_team("Kings XI Punjab") == "Punjab Kings"
    assert mappings.normalize_team("Royal Challengers Bangalore") == "Royal Challengers Bengaluru"
    assert mappings.normalize_team("Rising Pune Supergiants") == "Rising Pune Supergiant"
    # Separate franchises stay separate.
    assert mappings.normalize_team("Deccan Chargers") != mappings.normalize_team("Sunrisers Hyderabad")
    assert mappings.normalize_team("Gujarat Lions") != mappings.normalize_team("Gujarat Titans")
    assert mappings.normalize_team("Pune Warriors") != mappings.normalize_team("Rising Pune Supergiant")


def test_unknown_names_raise():
    with pytest.raises(mappings.UnknownNameError):
        mappings.normalize_team("Mumbai Indianz")
    with pytest.raises(mappings.UnknownNameError):
        mappings.normalize_venue("Lord's")


def test_venue_aliases_and_whitespace():
    assert mappings.normalize_venue("Feroz Shah Kotla") == "Arun Jaitley Stadium, Delhi"
    assert mappings.normalize_venue("  Eden   Gardens ") == "Eden Gardens, Kolkata"
    # Canonical names map to themselves, so the API can accept them directly.
    for canonical in mappings.VENUES:
        assert mappings.normalize_venue(canonical) == canonical


def test_clean_applies_rules():
    out = cleaning.clean_matches(frame(raw_row()))
    row = out.iloc[0]
    assert row["season"] == 2008 and row["season_raw"] == "2007/08"
    assert row["team1"] == "Royal Challengers Bengaluru"
    assert row["team1_raw"] == "Royal Challengers Bangalore"
    assert row["toss_winner"] == "Royal Challengers Bengaluru"
    assert row["venue"] == "M Chinnaswamy Stadium, Bengaluru"
    assert row["city"] == "Bengaluru" and row["country"] == "India"
    assert row["team1_win"] == 0
    assert not row["is_playoff"]
    assert list(out.columns) == cleaning.CLEAN_COLUMNS


def test_missing_city_is_filled_from_venue():
    out = cleaning.clean_matches(frame(raw_row(city="NA", venue="Sharjah Cricket Stadium")))
    assert out.iloc[0]["city"] == "Sharjah" and out.iloc[0]["country"] == "UAE"


def test_no_result_matches_are_dropped():
    rows = frame(
        raw_row(id="1"),
        raw_row(id="2", winner="NA", result="no result", result_margin="NA"),
    )
    out = cleaning.clean_matches(rows)
    assert out["match_id"].tolist() == [1]


def test_tie_keeps_super_over_winner():
    out = cleaning.clean_matches(frame(raw_row(
        result="tie", result_margin="NA", super_over="Y", winner="Royal Challengers Bangalore",
    )))
    assert out.iloc[0]["team1_win"] == 1 and bool(out.iloc[0]["super_over"])


def test_output_sorted_by_date_then_id():
    rows = frame(
        raw_row(id="30", date="2008-04-20"),
        raw_row(id="20", date="2008-04-19"),
        raw_row(id="10", date="2008-04-20"),
    )
    assert cleaning.clean_matches(rows)["match_id"].tolist() == [20, 10, 30]


@pytest.mark.parametrize("override, message", [
    ({"winner": "Mumbai Indians"}, "winner outside the match-up"),
    ({"toss_winner": "Mumbai Indians"}, "toss winner outside the match-up"),
    ({"team2": "Royal Challengers Bengaluru"}, "a team plays itself"),
    ({"toss_decision": "bowl"}, "toss_decision"),
    ({"match_type": "Friendly"}, "Unknown match types"),
    ({"date": "18/04/2008"}, "invalid date"),
])
def test_invalid_rows_are_rejected(override, message):
    with pytest.raises(cleaning.CleaningError, match=message):
        cleaning.clean_matches(frame(raw_row(**override)))


def test_raw_frame_is_not_modified():
    raw = frame(raw_row())
    before = raw.copy()
    cleaning.clean_matches(raw)
    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.skipif(not cleaning.RAW_MATCHES.exists(), reason="raw matches.csv not downloaded")
def test_real_data_cleans_to_expected_shape():
    raw = pd.read_csv(cleaning.RAW_MATCHES, dtype=str, keep_default_na=False)
    out = cleaning.clean_matches(raw)
    assert len(out) == 1090
    assert set(out["team1"]) | set(out["team2"]) == set(mappings.CANONICAL_TEAMS)
    assert out["venue"].nunique() == 36
    assert out["season"].between(2008, 2024).all()
    active_2024 = set(out.loc[out["season"] == 2024, "team1"]) | set(out.loc[out["season"] == 2024, "team2"])
    assert active_2024 == set(mappings.ACTIVE_TEAMS)


@pytest.mark.skipif(not cleaning.MANIFEST_PATH.exists() or not cleaning.CLEAN_PATH.exists(),
                    reason="cleaning has not been run")
def test_manifest_hash_matches_output_file():
    manifest = json.loads(cleaning.MANIFEST_PATH.read_text(encoding="utf-8"))
    actual = hashlib.sha256(cleaning.CLEAN_PATH.read_bytes()).hexdigest().upper()
    assert manifest["output"]["sha256"] == actual
