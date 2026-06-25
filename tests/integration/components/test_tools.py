import json

import pytest

from daystrom.components.tools import web_fetch
from daystrom.permissions import WebFetchPermission


class TestWebFetchIntegration:
    """Integration tests for web_fetch tool with real network calls."""

    @pytest.fixture
    def permission(self):
        return WebFetchPermission(allowed=True)

    def test_fetch_json_from_httpbin(self, permission):
        """Test fetching JSON from httpbin.org."""
        result = web_fetch(
            "https://httpbin.org/json", format="json", permissions=permission
        )

        parsed = json.loads(result)
        assert "slideshow" in parsed

    def test_fetch_html_from_httpbin(self, permission):
        """Test fetching HTML content from httpbin.org."""
        result = web_fetch(
            "https://httpbin.org/html", format="html", permissions=permission
        )

        assert "<html>" in result or "<!DOCTYPE" in result.upper()
        assert "Herman Melville" in result

    def test_fetch_markdown_from_html(self, permission):
        """Test fetching HTML and converting to markdown."""
        result = web_fetch(
            "https://httpbin.org/html", format="markdown", permissions=permission
        )

        # Should have text content but no HTML tags
        assert "Herman Melville" in result
        assert "<html>" not in result
        assert "<body>" not in result

    def test_fetch_text_from_html(self, permission):
        """Test fetching HTML and converting to text."""
        result = web_fetch(
            "https://httpbin.org/html", format="text", permissions=permission
        )

        # Should have text content but no HTML tags
        assert "Herman Melville" in result
        assert "<html>" not in result

    def test_fetch_user_agent_endpoint(self, permission):
        """Test fetching from httpbin user-agent endpoint."""
        result = web_fetch(
            "https://httpbin.org/user-agent", format="json", permissions=permission
        )

        parsed = json.loads(result)
        assert "user-agent" in parsed

    def test_fetch_headers_endpoint(self, permission):
        """Test fetching from httpbin headers endpoint to verify accept header."""
        result = web_fetch(
            "https://httpbin.org/headers", format="json", permissions=permission
        )

        parsed = json.loads(result)
        assert "headers" in parsed
        # Should have Accept header set
        assert "Accept" in parsed["headers"]

    def test_fetch_status_404_raises(self, permission):
        """Test that 404 responses raise an error."""
        import httpx

        with pytest.raises(httpx.HTTPStatusError):
            web_fetch("https://httpbin.org/status/404", permissions=permission)

    def test_fetch_status_500_raises(self, permission):
        """Test that 500 responses raise an error."""
        import httpx

        with pytest.raises(httpx.HTTPStatusError):
            web_fetch("https://httpbin.org/status/500", permissions=permission)

    def test_redirects_are_followed(self, permission):
        """Test that redirect responses are followed automatically."""
        result = web_fetch(
            "https://httpbin.org/redirect-to?url=/html",
            format="text",
            permissions=permission,
        )

        assert "Herman Melville" in result
