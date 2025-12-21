import pytest

from daystrom.context import Context, Message


class TestMessage:
    def test_init(self):
        msg = Message(text="Hello", role="user")
        assert msg.text == "Hello"
        assert msg.role == "user"

    def test_str(self):
        msg = Message(text="Hello", role="user")
        assert str(msg) == "user: Hello"

    def test_str_with_different_roles(self):
        assert str(Message(text="Hi", role="assistant")) == "assistant: Hi"
        assert str(Message(text="Be helpful", role="system")) == "system: Be helpful"


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
