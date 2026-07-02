"""Tests for generate_pdf_report function."""
import os
import pytest
from seo_audit_tool_extended import generate_pdf_report


class TestGeneratePdfReport:
    def _make_report_data(self):
        return {
            "site": "https://example.com",
            "generated_at": "2025-01-01 00:00:00 UTC",
            "pages": {
                "https://example.com": {
                    "fetch": {"status": 200},
                    "analysis": {
                        "title": {"text": "Example", "length": 7},
                        "meta_description": {"text": "Desc", "length": 4},
                        "h1": {"count": 1},
                    },
                    "scores": {"score": 95, "reasons": ["Low word count (<100)"]},
                },
                "https://example.com/about": {
                    "fetch": {"status": 200},
                    "analysis": {
                        "title": {"text": "About", "length": 5},
                        "meta_description": {"text": "", "length": 0},
                        "h1": {"count": 0},
                    },
                    "scores": {"score": 70, "reasons": ["Missing meta description", "Missing H1"]},
                },
            },
        }

    def test_pdf_created(self, tmp_path):
        output = str(tmp_path / "report.pdf")
        generate_pdf_report(self._make_report_data(), output_path=output)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

    def test_pdf_with_empty_pages(self, tmp_path):
        output = str(tmp_path / "report.pdf")
        data = {"site": "https://example.com", "generated_at": "now", "pages": {}}
        generate_pdf_report(data, output_path=output)
        assert os.path.exists(output)

    def test_pdf_with_offpage_data(self, tmp_path):
        output = str(tmp_path / "report.pdf")
        data = self._make_report_data()
        data["pages"]["https://example.com"]["offpage"] = {
            "domain_authority": 45,
            "page_authority": 30,
        }
        generate_pdf_report(data, output_path=output)
        assert os.path.exists(output)

    def test_pdf_creates_directory(self, tmp_path):
        output = str(tmp_path / "subdir" / "report.pdf")
        generate_pdf_report(self._make_report_data(), output_path=output)
        assert os.path.exists(output)

    def test_pdf_no_scores(self, tmp_path):
        output = str(tmp_path / "report.pdf")
        data = {
            "site": "https://example.com",
            "generated_at": "now",
            "pages": {
                "https://example.com": {
                    "fetch": {"status": 200},
                    "analysis": {},
                },
            },
        }
        generate_pdf_report(data, output_path=output)
        assert os.path.exists(output)
