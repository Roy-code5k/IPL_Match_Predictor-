"""The frontend's hard-coded lists must agree with src/data/mappings.py and the cleaned data."""

import re
from pathlib import Path

import pandas as pd
import pytest

from src.data import mappings
from src.features.builder import CLEAN_PATH

DATA_JS = Path(__file__).resolve().parents[1] / "app" / "frontend" / "src" / "data.js"


def js_block(name: str) -> str:
    text = DATA_JS.read_text(encoding="utf-8")
    match = re.search(rf"export const {name} = [\[{{](.*?)\n[\]}}]", text, re.S)
    assert match, f"{name} not found in data.js"
    return match.group(1)


def js_strings(block: str) -> list[str]:
    return [a or b for a, b in re.findall(r"'([^']*)'|\"([^\"]*)\"", block)]


def test_venues_are_the_canonical_grounds():
    venues = js_strings(js_block("venues"))
    assert venues == sorted(mappings.VENUES)


def test_teams_and_aliases_match_mappings():
    names = re.findall(r"name: '([^']+)'", js_block("teams"))
    assert sorted(names) == sorted(mappings.CANONICAL_TEAMS)
    pairs = dict(re.findall(r"'([^']+)': '([^']+)'", js_block("franchiseAliases")))
    assert pairs == mappings.TEAM_ALIASES


@pytest.mark.skipif(not CLEAN_PATH.exists(), reason="run python -m src.data.cleaning first")
def test_frontend_counts_match_clean_data():
    clean = pd.read_csv(CLEAN_PATH)
    for name, played, wins in re.findall(r"name: '([^']+)', played: (\d+), wins: (\d+)", js_block("teams")):
        in_match = (clean["team1"] == name) | (clean["team2"] == name)
        assert int(in_match.sum()) == int(played), name
        assert int((clean["winner"] == name).sum()) == int(wins), name
    counts = clean["venue"].value_counts()
    leading = re.findall(r"name: '([^']+)', matches: (\d+)", js_block("leadingVenues"))
    assert [(v, int(n)) for v, n in leading] == list(counts.head(len(leading)).items())
