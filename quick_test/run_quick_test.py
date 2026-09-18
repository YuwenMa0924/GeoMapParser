#!/usr/bin/env python3
"""Quick self-test for the GeoMapParser repository.

This test does not call any model or API and requires no dependencies beyond
the Python 3 standard library. It validates that the bundled example outputs
(data/example_outputs/*.json - the structured metadata extracted for the
100-map representative subset) are present, parseable and well-formed, and it
recomputes the field-presence rates over the subset as a consistency check.

Run from anywhere:

    python quick_test/run_quick_test.py

Expected result: a field-presence summary table followed by
"ALL CHECKS PASSED".
"""
import json
from pathlib import Path

FIELDS = [
    "title", "scale", "projection", "publication_agency",
    "publication_date", "data_source", "legend_data",
]
EXPECTED_FILES = 100

# Conservative lower bounds for the field-presence rates over the subset
# (looser than the published values; used only to detect corrupted data).
MIN_RATES = {
    "title": 0.90,
    "scale": 0.85,
    "projection": 0.80,
    "publication_agency": 0.85,
    "publication_date": 0.85,
    "data_source": 0.80,
}


def field_is_present(record, field):
    value = record.get(field)
    if field in ("publication_agency", "legend_data"):
        return isinstance(value, list) and len(value) > 0
    return value not in (None, "", [], {})


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "data" / "example_outputs"
    files = sorted(out_dir.glob("*.json"))
    assert len(files) == EXPECTED_FILES, (
        f"expected {EXPECTED_FILES} example JSON files in {out_dir}, found {len(files)}"
    )

    records, failed = [], []
    for fp in files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            failed.append((fp.name, str(exc)))
            continue
        if isinstance(obj, dict):
            records.append(obj)
        else:
            failed.append((fp.name, "not a JSON object"))
    assert not failed, f"unparseable files: {failed[:5]}"
    assert len(records) == EXPECTED_FILES

    counts = {f: 0 for f in FIELDS}
    for record in records:
        for field in FIELDS:
            counts[field] += int(field_is_present(record, field))

    print("Field-presence rates over the 100-map representative subset")
    print("(computed from the bundled example outputs)")
    print("-" * 58)
    for field in FIELDS:
        rate = counts[field] / EXPECTED_FILES
        print(f"  {field:22s} {counts[field]:3d}/{EXPECTED_FILES}  ({rate:6.1%})")
    print("-" * 58)

    for field, floor in MIN_RATES.items():
        rate = counts[field] / EXPECTED_FILES
        assert rate >= floor, (
            f"presence rate of '{field}' is {rate:.1%}, below the sanity floor {floor:.0%}"
        )
    print("Sanity thresholds satisfied.")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
