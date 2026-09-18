"""Small offline checks for portable evidence and malformed catalogue input."""
import unittest
from pathlib import Path

from probe_support import date_summary, source_id


class EvidenceTests(unittest.TestCase):
    def test_relative_id_is_independent_of_checkout_location(self):
        for base in (Path("/tmp/checkout-one"), Path("/tmp/checkout-two")):
            self.assertEqual(source_id(base / "scripts/metrics.py", [("repo", base)]),
                             "repo/scripts/metrics.py")

    def test_unknown_source_root_is_rejected_without_exposing_path(self):
        with self.assertRaisesRegex(ValueError, "outside the declared source roots"):
            source_id("/tmp/unregistered/file", [("repo", Path("/tmp/registered"))])

    def test_dates_count_missing_invalid_and_fractional_rows(self):
        summary = date_summary([" 1981/07/18 13:35:36 ", "2024-01-01T00:00:00.250Z",
                                "nonsense", None, "", 12345])
        self.assertEqual(summary, dict(valid_timestamp_rows=2, missing_timestamp_rows=2,
                                      invalid_timestamp_rows=2, pre_2002_picks=1,
                                      fractional_timestamp_rows=1))

    def test_all_missing_dates_are_not_reported_as_valid(self):
        self.assertEqual(date_summary([None, " "])["valid_timestamp_rows"], 0)


if __name__ == "__main__":
    unittest.main()
