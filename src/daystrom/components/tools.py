import json
from pathlib import Path

from markdownify import markdownify as md

from daystrom.components.tool_util import tool
from daystrom.exceptions import ToolCallError
from daystrom.permissions import ReadPermission
from daystrom.skills import Skill, load_skill


@tool(type="default")
def skill(skill_name: str, *, permissions: ReadPermission, skills: dict[str, Skill]):
    """Activates a skill with a specified name.

    Args:
        skill_name (str): The skill to activate

    Returns:
        str: The fetched content as a string.
    """
    skill = skills.get(skill_name)
    if not skill:
        raise ToolCallError("skill", f"Unknown skill: {skill_name}")

    if not permissions.can_read(skill.directory):
        raise ToolCallError(
            "skill", f"Read permission denied for path: {skill.directory}"
        )

    return load_skill(skill)


@tool(type="default")
def read_file(
    path: Path | str, offset: int = 0, limit: int = 200, *, permissions: ReadPermission
) -> str:
    if not permissions.can_read(path):
        raise ToolCallError("read_file", f"Read permission denied for path: {path}")

    if offset < 0:
        raise ToolCallError("read_file", "offset must be greater than or equal to 0")
    if limit < 1:
        raise ToolCallError("read_file", "limit must be greater than or equal to 1")

    read_path = Path(path).expanduser()

    text = ""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolCallError(
            "read_file", f"File is not valid UTF-8 text: {path}"
        ) from exc

    all_lines = text.splitlines()
    end = offset + limit
    lines = all_lines[offset:end]

    output = [f"<path>{path}</path>", "<type>file</type>", "<content>"]
    output.extend(
        f"{line_number}: {line}"
        for line_number, line in enumerate(lines, start=offset + 1)
    )

    total_lines = len(all_lines)
    if end < total_lines:
        output.append(
            f"(Showing lines {offset}-{offset + len(lines) - 1} of {total_lines}. Use a larger offset to continue.)"
        )
    else:
        output.append(f"(End of file - total {total_lines} lines)")

    output.append("</content>")
    return "\n".join(output)


@tool(type="default")
def web_fetch(url: str, format: str = "markdown") -> str:
    """Fetches content from a given URL.

    Args:
        url (str): The URL to fetch content from.
        format (str, optional): The format of the content to fetch, text, html, json, or markdown. Default "markdown".

    Returns:
        str: The fetched content as a string.
    """
    import httpx

    accept_header = "*/*"
    match format:
        case "text":
            accept_header = (
                "text/plain;q=1.0, text/markdown;q=0.9, text/html;q=0.8, */*;q=0.1"
            )
        case "html":
            accept_header = "text/html;q=1.0, application/xhtml+xml;q=0.9, text/plain;q=0.8, text/markdown;q=0.7, */*;q=0.1"
        case "json":
            accept_header = "application/json;q=1.0, text/markdown;q=0.9, text/x-markdown;q=0.8, text/plain;q=0.7, text/html;q=0.6, */*;q=0.1"
        case "markdown":
            accept_header = "text/markdown;q=1.0, text/x-markdown;q=0.9, text/plain;q=0.8, text/html;q=0.7, */*;q=0.1"
        case _:
            raise ToolCallError("web_fetch", f"Unsupported format: {format}")

    headers = {
        "accept": accept_header,
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    }
    response = httpx.get(url, timeout=10.0, headers=headers, follow_redirects=True)

    response.raise_for_status()
    response_type = response.headers.get("content-type") or ""

    match format:
        case "text":
            ans = response.text
            if "text/html" in response_type:
                ans = md(ans, convert=[])
            return ans
        case "html":
            return response.text
        case "json":
            ans = response.text
            if "application/json" in response_type:
                ans = json.dumps(response.json())
            return ans
        case "markdown":
            ans = response.text
            if "text/html" in response_type:
                ans = md(ans)
            return ans
        case _:
            raise ToolCallError("web_fetch", f"Unsupported format: {format}")
