"""Tests for Moz authentication and metrics functions."""
import pytest
from seo_audit_tool_extended import create_moz_auth, fetch_moz_metrics


class TestCreateMozAuth:
    def test_returns_tuple(self):
        expires, sig = create_moz_auth("my_access_id", "my_secret")
        assert isinstance(expires, int)
        assert isinstance(sig, str)

    def test_expires_in_future(self):
        import time
        expires, _ = create_moz_auth("id", "secret")
        assert expires > int(time.time())

    def test_different_inputs_different_signatures(self):
        _, sig1 = create_moz_auth("id1", "secret1")
        _, sig2 = create_moz_auth("id2", "secret2")
        assert sig1 != sig2

    def test_consistent_for_same_second(self):
        _, sig1 = create_moz_auth("id", "secret")
        _, sig2 = create_moz_auth("id", "secret")
        assert sig1 == sig2


class TestFetchMozMetrics:
    def test_no_keys_returns_none_values(self):
        result = fetch_moz_metrics("https://example.com", None, None)
        assert result["domain_authority"] is None
        assert result["page_authority"] is None
        assert result["external_links"] is None

    def test_empty_keys_returns_none_values(self):
        result = fetch_moz_metrics("https://example.com", "", "")
        assert result["domain_authority"] is None
        assert result["page_authority"] is None

    def test_with_keys_returns_stub(self):
        result = fetch_moz_metrics("https://example.com", "access_id", "secret")
        assert "note" in result
        assert result["domain_authority"] is None
