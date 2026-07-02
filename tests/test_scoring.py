"""Tests for compute_page_score function."""
import pytest
from seo_toolkit.scoring import compute_page_score


class TestComputePageScore:
    def test_perfect_page(self):
        analysis = {
            "title": {"text": "Good Title", "length": 10},
            "meta_description": {"text": "Description", "length": 11},
            "h1": {"count": 1, "texts": ["Heading"]},
            "canonical": "https://example.com",
            "word_count": 500,
        }
        fetch = {"status": 200}
        result = compute_page_score(analysis, fetch)
        assert result["score"] == 100
        assert result["reasons"] == []

    def test_missing_title(self):
        analysis = {
            "title": {"text": "", "length": 0},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 1},
            "canonical": "https://example.com",
            "word_count": 200,
        }
        fetch = {"status": 200}
        result = compute_page_score(analysis, fetch)
        assert result["score"] == 80
        assert "Missing title" in result["reasons"]

    def test_title_too_long(self):
        analysis = {
            "title": {"text": "A" * 65, "length": 65},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 1},
            "canonical": "https://example.com",
            "word_count": 200,
        }
        fetch = {"status": 200}
        result = compute_page_score(analysis, fetch)
        assert result["score"] == 95
        assert any("Title too long" in r for r in result["reasons"])

    def test_missing_meta_description(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "", "length": 0},
            "h1": {"count": 1},
            "canonical": "https://example.com",
            "word_count": 200,
        }
        fetch = {"status": 200}
        result = compute_page_score(analysis, fetch)
        assert result["score"] == 90
        assert "Missing meta description" in result["reasons"]

    def test_meta_description_too_long(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "D" * 165, "length": 165},
            "h1": {"count": 1},
            "canonical": "https://example.com",
            "word_count": 200,
        }
        fetch = {"status": 200}
        result = compute_page_score(analysis, fetch)
        assert result["score"] == 95
        assert any("Meta description too long" in r for r in result["reasons"])

    def test_missing_h1(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 0},
            "canonical": "https://example.com",
            "word_count": 200,
        }
        fetch = {"status": 200}
        result = compute_page_score(analysis, fetch)
        assert result["score"] == 90
        assert "Missing H1" in result["reasons"]

    def test_multiple_h1(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 3},
            "canonical": "https://example.com",
            "word_count": 200,
        }
        fetch = {"status": 200}
        result = compute_page_score(analysis, fetch)
        assert result["score"] == 95
        assert any("Multiple H1 tags" in r for r in result["reasons"])

    def test_missing_canonical(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 1},
            "word_count": 200,
        }
        fetch = {"status": 200}
        result = compute_page_score(analysis, fetch)
        assert result["score"] == 95
        assert "Missing canonical tag" in result["reasons"]

    def test_low_word_count(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 1},
            "canonical": "https://example.com",
            "word_count": 50,
        }
        fetch = {"status": 200}
        result = compute_page_score(analysis, fetch)
        assert result["score"] == 95
        assert "Low word count (<100)" in result["reasons"]

    def test_http_error_404(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 1},
            "canonical": "https://example.com",
            "word_count": 200,
        }
        fetch = {"status": 404}
        result = compute_page_score(analysis, fetch)
        assert result["score"] == 0
        assert any("HTTP error" in r for r in result["reasons"])

    def test_http_error_500(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 1},
            "canonical": "https://example.com",
            "word_count": 200,
        }
        fetch = {"status": 500}
        result = compute_page_score(analysis, fetch)
        assert result["score"] == 0

    def test_none_status(self):
        analysis = {
            "title": {"text": "Title", "length": 5},
            "meta_description": {"text": "Desc", "length": 4},
            "h1": {"count": 1},
            "canonical": "https://example.com",
            "word_count": 200,
        }
        fetch = {"status": None}
        result = compute_page_score(analysis, fetch)
        assert result["score"] == 0

    def test_all_issues_combined(self):
        analysis = {
            "title": {"text": "", "length": 0},
            "meta_description": {"text": "", "length": 0},
            "h1": {"count": 0},
            "word_count": 10,
        }
        fetch = {"status": 200}
        result = compute_page_score(analysis, fetch)
        # Missing title (-20), Missing meta desc (-10), Missing H1 (-10),
        # Missing canonical (-5), Low word count (-5)
        expected = 100 - 20 - 10 - 10 - 5 - 5
        assert result["score"] == expected
        assert len(result["reasons"]) == 5

    def test_score_never_negative(self):
        analysis = {}
        fetch = {"status": 500}
        result = compute_page_score(analysis, fetch)
        assert result["score"] >= 0

    def test_missing_analysis_keys(self):
        analysis = {}
        fetch = {"status": 200}
        result = compute_page_score(analysis, fetch)
        assert result["score"] >= 0
