from unittest.mock import MagicMock

import httpx
import pytest

from daystrom.components.tools import web_fetch
from daystrom.exceptions import ToolCallError
from daystrom.permissions import WebFetchPermission


class TestWebFetch:
    """Unit tests for web_fetch tool with mocked HTTP responses."""

    @pytest.fixture
    def permission(self):
        return WebFetchPermission(allowed=True)

    @pytest.fixture
    def mock_response(self):
        """Create a mock httpx response factory."""

        def _create_response(
            text: str = "",
            content_type: str = "text/plain",
            status_code: int = 200,
            json_data: dict | None = None,
        ):
            response = MagicMock(spec=httpx.Response)
            response.text = text
            response.headers = {"content-type": content_type}
            response.status_code = status_code
            response.raise_for_status = MagicMock()
            response.is_redirect = False

            if status_code >= 400:
                response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    message="Error",
                    request=MagicMock(),
                    response=response,
                )

            if json_data is not None:
                response.json = MagicMock(return_value=json_data)

            return response

        return _create_response

    def test_fetch_markdown_plain_text(self, mocker, mock_response, permission):
        """Test fetching plain text content with markdown format."""
        response = mock_response(text="Hello, World!", content_type="text/plain")
        mock_get = mocker.patch("httpx.get", return_value=response)

        result = web_fetch(
            "https://example.com", format="markdown", permissions=permission
        )

        assert result == "Hello, World!"
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args.kwargs
        assert "text/markdown" in call_kwargs["headers"]["accept"]

    def test_fetch_plain_text(self, mocker, mock_response, permission):
        """Test fetching content with text format."""
        response = mock_response(text="Plain text content", content_type="text/plain")
        mock_get = mocker.patch("httpx.get", return_value=response)

        result = web_fetch("https://example.com", format="text", permissions=permission)

        assert result == "Plain text content"
        call_kwargs = mock_get.call_args.kwargs
        assert "text/plain" in call_kwargs["headers"]["accept"]

    def test_fetch_markdown_converts_html(self, mocker, mock_response, permission):
        """Test that HTML content is converted to markdown when format is markdown."""
        html_content = "<h1>Title</h1><p>Paragraph text</p>"
        response = mock_response(text=html_content, content_type="text/html")
        mocker.patch("httpx.get", return_value=response)

        result = web_fetch(
            "https://example.com", format="markdown", permissions=permission
        )

        assert "Title" in result
        assert "Paragraph" in result
        assert "<h1>" not in result
        assert "<p>" not in result

    def test_fetch_text_converts_html(self, mocker, mock_response, permission):
        """Test that HTML tags are stripped when format is text."""
        html_content = "<h1>Title</h1><p>Paragraph</p><a href='#'>Link</a>"
        response = mock_response(text=html_content, content_type="text/html")
        mocker.patch("httpx.get", return_value=response)

        result = web_fetch("https://example.com", format="text", permissions=permission)

        assert "Title" in result
        assert "Paragraph" in result
        assert "<h1>" not in result
        assert "<p>" not in result

    def test_fetch_html_returns_raw(self, mocker, mock_response, permission):
        """Test fetching content with html format returns raw HTML."""
        html_content = "<html><body><h1>Title</h1></body></html>"
        response = mock_response(text=html_content, content_type="text/html")
        mock_get = mocker.patch("httpx.get", return_value=response)

        result = web_fetch("https://example.com", format="html", permissions=permission)

        assert result == html_content
        call_kwargs = mock_get.call_args.kwargs
        assert "text/html" in call_kwargs["headers"]["accept"]

    def test_fetch_json_returns_raw_text(self, mocker, mock_response, permission):
        """Test fetching JSON returns the response text when content-type is not application/json."""
        json_text = '{"key": "value"}'
        response = mock_response(text=json_text, content_type="text/plain")
        mock_get = mocker.patch("httpx.get", return_value=response)

        result = web_fetch("https://example.com", format="json", permissions=permission)

        assert result == json_text
        call_kwargs = mock_get.call_args.kwargs
        assert "application/json" in call_kwargs["headers"]["accept"]

    def test_fetch_json_parses_json_content_type(
        self, mocker, mock_response, permission
    ):
        """Test fetching JSON parses and re-serializes when content-type is application/json."""
        json_data = {"key": "value", "nested": {"a": 1}}
        response = mock_response(
            content_type="application/json",
            json_data=json_data,
        )
        mocker.patch("httpx.get", return_value=response)

        result = web_fetch("https://example.com", format="json", permissions=permission)

        # Should be valid JSON string
        import json

        parsed = json.loads(result)
        assert parsed == json_data

    def test_default_format_is_markdown(self, mocker, mock_response, permission):
        """Test that default format is markdown."""
        response = mock_response(text="Content", content_type="text/plain")
        mock_get = mocker.patch("httpx.get", return_value=response)

        result = web_fetch("https://example.com", permissions=permission)

        assert result == "Content"
        call_kwargs = mock_get.call_args.kwargs
        assert "text/markdown;q=1.0" in call_kwargs["headers"]["accept"]

    def test_unsupported_format_raises_error(self, mocker, mock_response, permission):
        """Test that unsupported format raises ToolCallError."""
        response = mock_response(text="Content")
        mocker.patch("httpx.get", return_value=response)

        with pytest.raises(ToolCallError):
            web_fetch(
                "https://example.com", format="invalid_format", permissions=permission
            )

    def test_http_error_propagates(self, mocker, mock_response, permission):
        """Test that HTTP errors are propagated."""
        response = mock_response(text="Not Found", status_code=404)
        mocker.patch("httpx.get", return_value=response)

        with pytest.raises(httpx.HTTPStatusError):
            web_fetch("https://example.com", permissions=permission)

    def test_timeout_is_set(self, mocker, mock_response, permission):
        """Test that timeout is set to 10 seconds."""
        response = mock_response(text="Content")
        mock_get = mocker.patch("httpx.get", return_value=response)

        web_fetch("https://example.com", permissions=permission)

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["timeout"] is not None

    def test_url_is_passed_correctly(self, mocker, mock_response, permission):
        """Test that URL is passed correctly to httpx.get."""
        response = mock_response(text="Content")
        mock_get = mocker.patch("httpx.get", return_value=response)

        web_fetch("https://example.com/path?query=value", permissions=permission)

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://example.com/path?query=value"

    def test_accept_header_markdown_format(self, mocker, mock_response, permission):
        """Test accept header priorities for markdown format."""
        response = mock_response(text="Content")
        mock_get = mocker.patch("httpx.get", return_value=response)

        web_fetch("https://example.com", format="markdown", permissions=permission)

        accept = mock_get.call_args.kwargs["headers"]["accept"]
        # markdown should have highest priority
        assert "text/markdown;q=1.0" in accept
        assert "text/x-markdown;q=0.9" in accept

    def test_accept_header_text_format(self, mocker, mock_response, permission):
        """Test accept header priorities for text format."""
        response = mock_response(text="Content")
        mock_get = mocker.patch("httpx.get", return_value=response)

        web_fetch("https://example.com", format="text", permissions=permission)

        accept = mock_get.call_args.kwargs["headers"]["accept"]
        # plain text should have highest priority
        assert "text/plain;q=1.0" in accept

    def test_accept_header_html_format(self, mocker, mock_response, permission):
        """Test accept header priorities for html format."""
        response = mock_response(text="Content")
        mock_get = mocker.patch("httpx.get", return_value=response)

        web_fetch("https://example.com", format="html", permissions=permission)

        accept = mock_get.call_args.kwargs["headers"]["accept"]
        # html should have highest priority
        assert "text/html;q=1.0" in accept

    def test_accept_header_json_format(self, mocker, mock_response, permission):
        """Test accept header priorities for json format."""
        response = mock_response(text="Content")
        mock_get = mocker.patch("httpx.get", return_value=response)

        web_fetch("https://example.com", format="json", permissions=permission)

        accept = mock_get.call_args.kwargs["headers"]["accept"]
        # json should have highest priority
        assert "application/json;q=1.0" in accept

    def test_redirects_are_followed(self, mocker, mock_response, permission):
        """Test that redirect responses are followed automatically."""
        response = mock_response(
            text="Redirect destination content",
            content_type="text/plain",
        )
        mock_get = mocker.patch("httpx.get", return_value=response)

        result = web_fetch("https://example.com", permissions=permission)

        assert result == "Redirect destination content"
        assert mock_get.call_args.kwargs["follow_redirects"] is True
