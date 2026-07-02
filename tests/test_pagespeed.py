"""Tests for pagespeed_insights function."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from seo_audit_tool_extended import pagespeed_insights


class TestPagespeedInsights:
    @pytest.mark.asyncio
    async def test_no_api_key(self):
        result = await pagespeed_insights("https://example.com", None)
        assert result == {"error": "no_api_key"}

    @pytest.mark.asyncio
    async def test_empty_api_key(self):
        result = await pagespeed_insights("https://example.com", "")
        assert result == {"error": "no_api_key"}

    @pytest.mark.asyncio
    async def test_aiohttp_not_installed(self):
        with patch("seo_audit_tool_extended.aiohttp", None):
            result = await pagespeed_insights("https://example.com", "key123")
            assert result == {"error": "aiohttp-not-installed"}

    @pytest.mark.asyncio
    async def test_successful_api_call(self):
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"lighthouseResult": {"score": 90}})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        with patch("seo_audit_tool_extended.aiohttp", mock_aiohttp):
            result = await pagespeed_insights("https://example.com", "key123", strategy="desktop")
            assert result == {"lighthouseResult": {"score": 90}}

    @pytest.mark.asyncio
    async def test_api_call_exception(self):
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=Exception("connection failed"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        with patch("seo_audit_tool_extended.aiohttp", mock_aiohttp):
            result = await pagespeed_insights("https://example.com", "key123")
            assert "error" in result
