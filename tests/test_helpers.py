"""Tests for normalize_url and same_origin helper functions."""
import pytest
from seo_audit_tool_extended import normalize_url, same_origin


class TestNormalizeUrl:
    def test_absolute_url(self):
        result = normalize_url("https://example.com/page", "https://other.com/path")
        assert result == "https://other.com/path"

    def test_relative_url(self):
        result = normalize_url("https://example.com/page/", "sub")
        assert result == "https://example.com/page/sub"

    def test_relative_url_parent(self):
        result = normalize_url("https://example.com/a/b/", "../c")
        assert result == "https://example.com/a/c"

    def test_root_relative_url(self):
        result = normalize_url("https://example.com/a/b", "/c")
        assert result == "https://example.com/c"

    def test_empty_href_returns_none(self):
        assert normalize_url("https://example.com", "") is None

    def test_none_href_returns_none(self):
        assert normalize_url("https://example.com", None) is None

    def test_javascript_link_returns_none(self):
        assert normalize_url("https://example.com", "javascript:void(0)") is None

    def test_mailto_link_returns_none(self):
        assert normalize_url("https://example.com", "mailto:a@b.com") is None

    def test_hash_only_returns_none(self):
        assert normalize_url("https://example.com", "#section") is None

    def test_fragment_stripped(self):
        result = normalize_url("https://example.com", "/page#frag")
        assert "#" not in result
        assert result == "https://example.com/page"

    def test_whitespace_stripped(self):
        result = normalize_url("https://example.com", "  /page  ")
        assert result == "https://example.com/page"

    def test_query_string_preserved(self):
        result = normalize_url("https://example.com", "/page?q=1&r=2")
        assert result == "https://example.com/page?q=1&r=2"


class TestSameOrigin:
    def test_same_origin(self):
        assert same_origin("https://example.com/a", "https://example.com/b") is True

    def test_different_host(self):
        assert same_origin("https://a.com/x", "https://b.com/x") is False

    def test_different_scheme(self):
        assert same_origin("http://example.com", "https://example.com") is False

    def test_different_port(self):
        assert same_origin("https://example.com:80", "https://example.com:443") is False

    def test_same_with_paths(self):
        assert same_origin(
            "https://example.com/deep/path?q=1",
            "https://example.com/other",
        ) is True
