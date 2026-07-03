"""Tests for render_page_content and PlaywrightCrawler (mocked Playwright)."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from seo_audit_tool_extended import render_page_content, PlaywrightCrawler


class TestRenderPageContent:
    @pytest.mark.asyncio
    async def test_returns_error_when_browser_is_none(self):
        result = await render_page_content(None, "https://example.com")
        assert result["error"] == "playwright-not-installed"
        assert result["status"] is None

    @pytest.mark.asyncio
    async def test_returns_content_on_success(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.all_headers = AsyncMock(return_value={"content-type": "text/html"})

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=mock_resp)
        mock_page.content = AsyncMock(return_value="<html><body>Hello</body></html>")
        mock_page.title = AsyncMock(return_value="Test Title")
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        result = await render_page_content(mock_browser, "https://example.com")
        assert result["status"] == 200
        assert "Hello" in result["content"]
        assert result["title"] == "Test Title"
        mock_page.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(side_effect=Exception("launch failed"))

        result = await render_page_content(mock_browser, "https://example.com")
        assert "error" in result
        assert "launch failed" in result["error"]

    @pytest.mark.asyncio
    async def test_page_closed_on_mid_operation_failure(self):
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=AsyncMock(status=200))
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.content = AsyncMock(side_effect=Exception("content extraction failed"))
        mock_page.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        result = await render_page_content(mock_browser, "https://example.com")
        assert "error" in result
        assert "content extraction failed" in result["error"]
        mock_page.close.assert_awaited_once()


class TestPlaywrightCrawler:
    def test_init(self):
        crawler = PlaywrightCrawler("https://example.com", max_pages=10)
        assert crawler.seed == "https://example.com"
        assert crawler.max_pages == 10
        assert "https://example.com" in crawler.seen

    def test_default_max_pages(self):
        crawler = PlaywrightCrawler("https://example.com")
        assert crawler.max_pages == 100

    @pytest.mark.asyncio
    async def test_raises_without_playwright(self):
        with patch("seo_audit_tool_extended.async_playwright", None):
            crawler = PlaywrightCrawler("https://example.com", max_pages=1)
            with pytest.raises(RuntimeError, match="Playwright is required"):
                await crawler.run()
