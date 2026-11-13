"""Raw-data integrity checks. Skipped when the Kaggle CSVs are not downloaded."""

import pytest

from src.data import inspect_raw

pytestmark = pytest.mark.skipif(
    not all((inspect_raw.RAW_DIR / name).exists() for name in inspect_raw.EXPECTED_SHA256),
    reason="raw CSVs not present in data/raw/ (see docs/data_source.md)",
)


@pytest.mark.parametrize("name", sorted(inspect_raw.EXPECTED_SHA256))
def test_raw_file_hash_matches_record(name):
    assert inspect_raw.sha256(inspect_raw.RAW_DIR / name) == inspect_raw.EXPECTED_SHA256[name]


def test_raw_schemas():
    assert list(inspect_raw.load_raw("matches.csv").columns) == inspect_raw.MATCHES_COLUMNS
    assert list(inspect_raw.load_raw("deliveries.csv").columns) == inspect_raw.DELIVERIES_COLUMNS


def test_all_inspection_checks_pass():
    _, failures = inspect_raw.build_report()
    assert failures == []


def test_tidy_markdown_separates_blocks():
    text = "# Title\nParagraph\n| a | b |\n|---|---|\n| 1 | 2 |\nAfter"
    assert inspect_raw.tidy_markdown(text) == (
        "# Title\n\nParagraph\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nAfter\n"
    )
