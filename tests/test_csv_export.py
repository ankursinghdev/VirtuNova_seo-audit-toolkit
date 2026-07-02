"""Tests for write_csv_report function."""
import csv
import os
import tempfile
import pytest
from seo_audit_tool_extended import write_csv_report


class TestWriteCsvReport:
    def _make_report(self, pages=None):
        if pages is None:
            pages = {
                "https://example.com": {
                    "fetch": {"status": 200},
                    "analysis": {
                        "title": {"text": "Example", "length": 7},
                        "meta_description": {"text": "Desc", "length": 4},
                        "h1": {"count": 1},
                        "word_count": 200,
                        "images": {"total": 3, "missing_alt_count": 1},
                        "links": {"count": 10},
                        "canonical": "https://example.com",
                        "hreflangs": [{"hreflang": "en", "href": "https://example.com/en"}],
                        "json_ld": [{"@context": "https://schema.org", "@type": "WebPage"}],
                    },
                    "scores": {"score": 95, "reasons": ["Low word count (<100)"]},
                },
            }
        return {"pages": pages, "pagespeed": {}}

    def test_csv_file_created(self, tmp_path):
        csv_path = str(tmp_path / "output" / "report.csv")
        report = self._make_report()
        write_csv_report(report, csv_path)
        assert os.path.exists(csv_path)

    def test_csv_has_header(self, tmp_path):
        csv_path = str(tmp_path / "report.csv")
        report = self._make_report()
        write_csv_report(report, csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            assert "url" in fields
            assert "http_status" in fields
            assert "title" in fields
            assert "score" in fields

    def test_csv_row_data(self, tmp_path):
        csv_path = str(tmp_path / "report.csv")
        report = self._make_report()
        write_csv_report(report, csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            row = rows[0]
            assert row["url"] == "https://example.com"
            assert row["http_status"] == "200"
            assert row["title"] == "Example"
            assert row["h1_count"] == "1"

    def test_csv_empty_pages(self, tmp_path):
        csv_path = str(tmp_path / "report.csv")
        report = self._make_report(pages={})
        write_csv_report(report, csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 0

    def test_csv_missing_analysis(self, tmp_path):
        csv_path = str(tmp_path / "report.csv")
        pages = {
            "https://example.com": {
                "fetch": {"status": 200},
                "analysis": None,
                "scores": None,
            }
        }
        report = self._make_report(pages=pages)
        write_csv_report(report, csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1

    def test_csv_multiple_pages(self, tmp_path):
        csv_path = str(tmp_path / "report.csv")
        pages = {
            f"https://example.com/page{i}": {
                "fetch": {"status": 200},
                "analysis": {
                    "title": {"text": f"Page {i}", "length": 6},
                    "meta_description": {"text": "", "length": 0},
                    "h1": {"count": 1},
                    "word_count": 100,
                    "images": {"total": 0, "missing_alt_count": 0},
                    "links": {"count": 0},
                    "canonical": "",
                    "hreflangs": [],
                    "json_ld": [],
                },
                "scores": {"score": 90, "reasons": ["Missing meta description"]},
            }
            for i in range(5)
        }
        report = self._make_report(pages=pages)
        write_csv_report(report, csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 5
