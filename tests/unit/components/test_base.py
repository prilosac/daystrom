import re

import pytest

from daystrom import Provider
from daystrom.components import (
    CUSTOM_TOOLS,
    DEFAULT_TOOLS,
    LLM,
    Context,
    LLMResponse,
    Message,
    Tool,
    ToolCall,
    tool,
)


def test_tool_decorator_basic():
    """Test @tool decorator creates Tool with correct name and description."""

    @tool
    def my_test_tool(x: str) -> str:
        """Short description.

        This is the long description.
        """
        return x

    assert "my_test_tool" in CUSTOM_TOOLS
    t = CUSTOM_TOOLS["my_test_tool"]
    assert t.name == "my_test_tool"
    assert t.description == "This is the long description."
    assert t.callable is not None

    # Cleanup
    del CUSTOM_TOOLS["my_test_tool"]


def test_tool_decorator_short_description_fallback():
    """Test @tool decorator uses short description when no long description."""

    @tool
    def short_desc_tool() -> str:
        """Only a short description."""
        return "ok"

    t = CUSTOM_TOOLS["short_desc_tool"]
    assert t.description == "Only a short description."

    # Cleanup
    del CUSTOM_TOOLS["short_desc_tool"]


def test_tool_decorator_extracts_params():
    """Test @tool decorator extracts parameter info correctly."""

    @tool
    def param_tool(required_param: str, optional_param: int = 10) -> str:
        """A tool with params.

        Args:
            required_param: A required string parameter.
            optional_param: An optional integer parameter.
        """
        return f"{required_param}-{optional_param}"

    t = CUSTOM_TOOLS["param_tool"]

    assert "required_param" in t.params
    assert t.params["required_param"]["type"] == str
    assert t.params["required_param"]["required"] is True

    assert "optional_param" in t.params
    assert t.params["optional_param"]["type"] == int
    assert t.params["optional_param"]["required"] is False

    # Cleanup
    del CUSTOM_TOOLS["param_tool"]


def test_tool_decorator_rejects_custom_restricted_keyword_only_params():
    with pytest.raises(TypeError, match="'permissions' is reserved"):

        @tool
        def permissions_tool(query: str, *, permissions: object) -> str:
            return query

    with pytest.raises(TypeError, match="'skills' is reserved"):

        @tool
        def skills_tool(query: str, *, skills: dict) -> str:
            return query


def test_tool_decorator_hides_default_restricted_keyword_only_params():
    @tool(type="default")
    def keyword_only_tool(
        query: str, *, limit: int, permissions: object, skills: dict
    ) -> str:
        return query

    t = DEFAULT_TOOLS["keyword_only_tool"]

    assert "query" in t.params
    assert "limit" in t.params
    assert t.params["limit"]["type"] == int
    assert "permissions" not in t.params
    assert "skills" not in t.params

    del DEFAULT_TOOLS["keyword_only_tool"]


def test_tool_decorator_list_param():
    """Test @tool decorator handles list type parameters with items."""

    @tool
    def list_tool(tags: list[str]) -> str:
        """A tool with list param."""
        return ",".join(tags)

    t = CUSTOM_TOOLS["list_tool"]

    assert t.params["tags"]["type"] == list[str]
    assert "items" in t.params["tags"]
    assert t.params["tags"]["items"]["type"] == str

    # Cleanup
    del CUSTOM_TOOLS["list_tool"]


def test_tool_decorator_rejects_args():
    """Test @tool decorator raises TypeError for *args."""
    with pytest.raises(TypeError, match="\\*args is not supported"):

        @tool
        def bad_tool(*args) -> str:
            """Bad tool."""
            return str(args)


def test_tool_decorator_rejects_kwargs():
    """Test @tool decorator raises TypeError for **kwargs."""
    with pytest.raises(TypeError, match="\\*\\*kwargs is not supported"):

        @tool
        def bad_tool(**kwargs) -> str:
            """Bad tool."""
            return str(kwargs)


def test_tool_decorator_rejects_multi_type_tuple():
    """Test @tool decorator raises TypeError for multi-type tuple."""
    with pytest.raises(TypeError, match="Only single-type iterables"):

        @tool
        def bad_tool(data: tuple[str, int]) -> str:
            """Bad tool."""
            return str(data)


def test_tool_decorator_callable_still_works():
    """Test decorated function still works as expected."""

    @tool
    def working_tool(a: str, b: str = "default") -> str:
        """A working tool."""
        return f"{a}-{b}"

    # Direct call should work
    result = working_tool("hello", "world")
    assert result == "hello-world"

    # Default argument should work
    result = working_tool("hello")
    assert result == "hello-default"

    # Cleanup
    del CUSTOM_TOOLS["working_tool"]


def test_tool_direct_creation():
    """Test creating a Tool directly without the decorator."""

    def my_func(x: str) -> str:
        return x.upper()

    t = Tool(
        callable=my_func,
        name="direct_tool",
        display_name="Direct Tool",
        description="A directly created tool.",
        params={"x": {"type": str, "description": "Input string", "required": True}},
    )

    assert t.name == "direct_tool"
    assert t.display_name == "Direct Tool"
    assert t.description == "A directly created tool."
    assert t.params == {
        "x": {"type": str, "description": "Input string", "required": True}
    }
    assert t.callable is my_func

    # Test the call method works
    assert t.call("hello") == "HELLO"


def test_tool_str_method():
    """Test Tool __str__ method returns expected format."""

    def dummy_func() -> None:
        pass

    t = Tool(
        callable=dummy_func,
        name="str_test_tool",
        description="Test description.",
        params={"arg1": {"type": str, "required": True}},
    )

    result = str(t)
    assert (
        result
        == "Tool(name=str_test_tool, description=Test description., params={'arg1': {'type': <class 'str'>, 'required': True}})"
    )


class TestMessage:
    def test_init(self):
        msg = Message(role="user", text="Hello")
        assert msg.text == "Hello"
        assert msg.role == "user"

    def test_str(self):
        tool = Tool(
            callable=lambda x: x,
            name="sample_tool",
            description="A sample tool",
            params={"x": {"type": str, "description": "Input", "required": True}},
        )
        tool_call = ToolCall(
            tool=tool,
            tool_call_id="call_123",
            args=[],
            kwargs={"x": "val"},
        )
        msg = Message(
            role="user", text="Hello", tool_call_id="call_123", tool_calls=[tool_call]
        )
        result = str(msg)
        # Use regex to match the Tool object with any memory address
        expected_pattern = re.compile(
            r"""user: Hello
    Tool Call ID: call_123
    Tool Calls:
    ToolCall\(tool=<daystrom\.components\.tool_util\.Tool object at 0x[0-9a-f]+>, tool_call_id='call_123', args=\[\], kwargs=\{'x': 'val'\}\)"""
        )
        assert expected_pattern.match(result)

    def test_str_with_different_roles(self):
        assert str(Message(role="assistant", text="Hi")) == "assistant: Hi"
        assert str(Message(role="system", text="Be helpful")) == "system: Be helpful"


class ConcreteLLM(LLM):
    """Minimal concrete LLM subclass for testing."""

    def invoke(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(text="", tool_calls=[])

    def track_usage(self, *args, **kwargs):
        pass


class TestLLM:
    def test_total_cost_both_costs(self):
        """Test total_cost with both input and output costs."""
        llm = ConcreteLLM(provider=Provider.OPENROUTER, model="test-model")
        llm.input_cost = 2.5
        llm.output_cost = 10.0
        llm.input_tokens = 1000
        llm.output_tokens = 500

        # (1000 / 1_000_000) * 2.5 + (500 / 1_000_000) * 10.0
        # = 0.0025 + 0.005 = 0.0075
        assert llm.total_cost == 0.0075

    def test_total_cost_input_only(self):
        """Test total_cost with only input cost set."""
        llm = ConcreteLLM(provider=Provider.OPENROUTER, model="test-model")
        llm.input_cost = 2.5
        llm.output_cost = None
        llm.input_tokens = 1000

        # (1000 / 1_000_000) * 2.5 = 0.0025
        assert llm.total_cost == 0.0025

    def test_total_cost_output_only(self):
        """Test total_cost with only output cost set."""
        llm = ConcreteLLM(provider=Provider.OPENROUTER, model="test-model")
        llm.input_cost = None
        llm.output_cost = 10.0
        llm.output_tokens = 500

        # (500 / 1_000_000) * 10.0 = 0.005
        assert llm.total_cost == 0.005

    def test_total_cost_both_none(self):
        """Test total_cost when both costs are None."""
        llm = ConcreteLLM(provider=Provider.OPENROUTER, model="test-model")
        llm.input_cost = None
        llm.output_cost = None

        assert llm.total_cost == 0.0

    def test_total_cost_zero_tokens(self):
        """Test total_cost with valid costs but zero tokens."""
        llm = ConcreteLLM(provider=Provider.OPENROUTER, model="test-model")
        llm.input_cost = 2.5
        llm.output_cost = 10.0
        llm.input_tokens = 0
        llm.output_tokens = 0

        assert llm.total_cost == 0.0


class TestContext:
    def test_init_empty(self):
        ctx = Context()
        assert ctx.messages == []

    def test_add_message(self):
        ctx = Context()
        ctx.add_message("user", "Hello")

        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == "user"
        assert ctx.messages[0].text == "Hello"

    def test_add_multiple_messages(self):
        ctx = Context()
        ctx.add_message("user", "Hello")
        ctx.add_message("assistant", "Hi there")
        ctx.add_message("user", "How are you?")

        assert len(ctx.messages) == 3
        assert ctx.messages[0].role == "user"
        assert ctx.messages[1].role == "assistant"
        assert ctx.messages[2].role == "user"

    def test_add_message_preserves_order(self):
        ctx = Context()
        ctx.add_message("system", "You are helpful")
        ctx.add_message("user", "First")
        ctx.add_message("assistant", "Response")
        ctx.add_message("user", "Second")

        roles = [msg.role for msg in ctx.messages]
        texts = [msg.text for msg in ctx.messages]

        assert roles == ["system", "user", "assistant", "user"]
        assert texts == ["You are helpful", "First", "Response", "Second"]

    def test_add_message_with_empty_text(self):
        ctx = Context()
        ctx.add_message("user", "")

        assert len(ctx.messages) == 1
        assert ctx.messages[0].text == ""

    def test_add_message_with_multiline_text(self):
        ctx = Context()
        multiline = "Line 1\nLine 2\nLine 3"
        ctx.add_message("user", multiline)

        assert ctx.messages[0].text == multiline
