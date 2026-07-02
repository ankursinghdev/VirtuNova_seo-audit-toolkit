"""Tests for score_page function."""
import pytest
from seo_audit_tool_extended import score_page


class TestScorePage:
    def test_perfect_page(self):
        analysis = {
            "title": {"text": "Good Title", "length": 10},
            "meta_description": {"text": "Description", "length": 11},
            "h1": {"count": 1, "texts": ["Heading"]},
            "word_count": 500,
        }
        fetch = {"status": 200}
        result = score_page(analysis, fetch)
        assert result["score"] == 100
        assert result["reasons"] == []

    def test_missing_title(self):
        analysis = {
            "title": {"text": "", "length": 0},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 1},
            "word_count": 200,
        }
        fetch = {"status": 200}
        result = score_page(analysis, fetch)
        assert result["score"] == 80
        assert "Missing title" in result["reasons"]

    def test_missing_meta_description(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "", "length": 0},
            "h1": {"count": 1},
            "word_count": 200,
        }
        fetch = {"status": 200}
        result = score_page(analysis, fetch)
        assert result["score"] == 90
        assert "Missing meta description" in result["reasons"]

    def test_missing_h1(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 0},
            "word_count": 200,
        }
        fetch = {"status": 200}
        result = score_page(analysis, fetch)
        assert result["score"] == 90
        assert "Missing H1" in result["reasons"]

    def test_low_word_count(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 1},
            "word_count": 50,
        }
        fetch = {"status": 200}
        result = score_page(analysis, fetch)
        assert result["score"] == 95
        assert "Low word count (<100)" in result["reasons"]

    def test_http_error_404(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 1},
            "word_count": 200,
        }
        fetch = {"status": 404}
        result = score_page(analysis, fetch)
        assert result["score"] == 0
        assert any("HTTP error" in r for r in result["reasons"])

    def test_http_error_500(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 1},
            "word_count": 200,
        }
        fetch = {"status": 500}
        result = score_page(analysis, fetch)
        assert result["score"] == 0

    def test_none_status(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 1},
            "word_count": 200,
        }
        fetch = {"status": None}
        result = score_page(analysis, fetch)
        assert result["score"] == 0

    def test_all_issues_combined(self):
        analysis = {
            "title": {"text": "", "length": 0},
            "meta_description": {"text": "", "length": 0},
            "h1": {"count": 0},
            "word_count": 10,
        }
        fetch = {"status": 200}
        result = score_page(analysis, fetch)
        expected = 100 - 20 - 10 - 10 - 5
        assert result["score"] == expected
        assert len(result["reasons"]) == 4

    def test_score_never_negative(self):
        analysis = {}
        fetch = {"status": 500}
        result = score_page(analysis, fetch)
        assert result["score"] >= 0

    def test_missing_analysis_keys(self):
        analysis = {}
        fetch = {"status": 200}
        result = score_page(analysis, fetch)
        assert result["score"] >= 0
