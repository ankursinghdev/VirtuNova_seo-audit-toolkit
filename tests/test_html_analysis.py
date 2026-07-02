"""Tests for analyze_html_from_text, validate_json_ld, and canonical_chain_check."""
import json
import pytest
from seo_audit_tool_extended import (
    analyze_html_from_text,
    validate_json_ld,
    canonical_chain_check,
)

FULL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Test Page Title</title>
    <meta name="description" content="A test meta description for SEO.">
    <meta name="robots" content="index, follow">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="canonical" href="https://example.com/page">
    <link rel="alternate" hreflang="en" href="https://example.com/en/page">
    <link rel="alternate" hreflang="fr" href="https://example.com/fr/page">
    <script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "WebPage", "name": "Test"}
    </script>
</head>
<body>
    <h1>Main Heading</h1>
    <p>Some body text with enough words to be meaningful for word count testing purposes.</p>
    <img src="/img/a.png" alt="Photo A">
    <img src="/img/b.png">
    <a href="/about">About</a>
    <a href="/contact">Contact</a>
</body>
</html>"""

MINIMAL_HTML = """<!DOCTYPE html><html><head><title></title></head><body></body></html>"""

NO_BODY_HTML = """<!DOCTYPE html><html><head><title>No body</title></head></html>"""


class TestAnalyzeHtmlFromText:
    def test_title_extraction(self):
        result = analyze_html_from_text("https://example.com", FULL_HTML)
        assert result["title"]["text"] == "Test Page Title"
        assert result["title"]["length"] == len("Test Page Title")

    def test_meta_description(self):
        result = analyze_html_from_text("https://example.com", FULL_HTML)
        assert result["meta_description"]["text"] == "A test meta description for SEO."
        assert result["meta_description"]["length"] > 0

    def test_h1(self):
        result = analyze_html_from_text("https://example.com", FULL_HTML)
        assert result["h1"]["count"] == 1
        assert "Main Heading" in result["h1"]["texts"]

    def test_canonical(self):
        result = analyze_html_from_text("https://example.com", FULL_HTML)
        assert result["canonical"] == "https://example.com/page"

    def test_meta_robots(self):
        result = analyze_html_from_text("https://example.com", FULL_HTML)
        assert result["meta_robots"] == "index, follow"

    def test_viewport(self):
        result = analyze_html_from_text("https://example.com", FULL_HTML)
        assert "width=device-width" in result["viewport"]

    def test_json_ld(self):
        result = analyze_html_from_text("https://example.com", FULL_HTML)
        assert len(result["json_ld"]) == 1
        assert result["json_ld"][0]["@type"] == "WebPage"

    def test_images(self):
        result = analyze_html_from_text("https://example.com", FULL_HTML)
        assert result["images"]["total"] == 2
        assert result["images"]["missing_alt_count"] == 1

    def test_links(self):
        result = analyze_html_from_text("https://example.com", FULL_HTML)
        assert result["links"]["count"] == 2

    def test_word_count(self):
        result = analyze_html_from_text("https://example.com", FULL_HTML)
        assert result["word_count"] > 0

    def test_hreflangs(self):
        result = analyze_html_from_text("https://example.com", FULL_HTML)
        assert len(result["hreflangs"]) == 2
        langs = {h["hreflang"] for h in result["hreflangs"]}
        assert langs == {"en", "fr"}

    def test_empty_html_returns_empty_dict(self):
        result = analyze_html_from_text("https://example.com", "")
        assert result == {}

    def test_none_html_returns_empty_dict(self):
        result = analyze_html_from_text("https://example.com", None)
        assert result == {}

    def test_minimal_html(self):
        result = analyze_html_from_text("https://example.com", MINIMAL_HTML)
        assert result["title"]["text"] == ""
        assert result["title"]["length"] == 0
        assert result["meta_description"]["text"] == ""
        assert result["h1"]["count"] == 0
        assert result["word_count"] == 0

    def test_no_body(self):
        result = analyze_html_from_text("https://example.com", NO_BODY_HTML)
        assert result["word_count"] == 0

    def test_multiple_h1(self):
        html = "<html><head><title>T</title></head><body><h1>A</h1><h1>B</h1></body></html>"
        result = analyze_html_from_text("https://example.com", html)
        assert result["h1"]["count"] == 2

    def test_invalid_json_ld_stored_as_raw(self):
        html = """<html><head><title>T</title>
        <script type="application/ld+json">not valid json</script>
        </head><body></body></html>"""
        result = analyze_html_from_text("https://example.com", html)
        assert len(result["json_ld"]) == 1
        assert "raw" in result["json_ld"][0]


class TestValidateJsonLd:
    def test_valid_block(self):
        blocks = [{"@context": "https://schema.org", "@type": "WebPage"}]
        issues = validate_json_ld(blocks)
        assert issues == []

    def test_missing_context(self):
        blocks = [{"@type": "WebPage"}]
        issues = validate_json_ld(blocks)
        assert any(i["issue"] == "missing @context" for i in issues)

    def test_missing_type(self):
        blocks = [{"@context": "https://schema.org"}]
        issues = validate_json_ld(blocks)
        assert any(i["issue"] == "missing @type" for i in issues)

    def test_graph_allowed(self):
        blocks = [{"@context": "https://schema.org", "@graph": []}]
        issues = validate_json_ld(blocks)
        assert issues == []

    def test_not_a_dict(self):
        blocks = ["not-a-dict"]
        issues = validate_json_ld(blocks)
        assert any(i["issue"] == "not a dict" for i in issues)

    def test_empty_list(self):
        assert validate_json_ld([]) == []

    def test_multiple_blocks_mixed(self):
        blocks = [
            {"@context": "https://schema.org", "@type": "WebPage"},
            {"name": "missing everything"},
        ]
        issues = validate_json_ld(blocks)
        assert len(issues) == 2
        assert all(i["index"] == 1 for i in issues)


class TestCanonicalChainCheck:
    def test_no_chains(self):
        pages = {
            "https://a.com": {"analysis": {"canonical": "https://a.com"}},
        }
        chains = canonical_chain_check(pages)
        assert chains == []

    def test_self_referencing_canonical(self):
        pages = {
            "https://a.com/page": {"analysis": {"canonical": "https://a.com/page"}},
        }
        chains = canonical_chain_check(pages)
        assert chains == []

    def test_chain_detected(self):
        pages = {
            "https://a.com/old": {"analysis": {"canonical": "https://a.com/new"}},
            "https://a.com/new": {"analysis": {"canonical": "https://a.com/final"}},
            "https://a.com/final": {"analysis": {"canonical": "https://a.com/final"}},
        }
        chains = canonical_chain_check(pages)
        assert len(chains) >= 1
        long_chain = max(chains, key=len)
        assert len(long_chain) >= 3

    def test_no_canonical(self):
        pages = {
            "https://a.com": {"analysis": {}},
        }
        chains = canonical_chain_check(pages)
        assert chains == []

    def test_canonical_to_external_page(self):
        pages = {
            "https://a.com/page": {"analysis": {"canonical": "https://external.com/page"}},
        }
        chains = canonical_chain_check(pages)
        assert chains == []
