from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from daystrom import Provider
from daystrom.components import Context, LLMResponse, LLMStreamEvent, Tool
from daystrom.components.openai import OpenAIChatCompletions
from daystrom.exceptions import InvalidComponentError


class FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def make_chunk(delta=None, finish_reason=None, usage=None):
    choices = []
    if delta is not None or finish_reason is not None:
        choices.append(
            {
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        )

    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": choices,
            "usage": usage,
        }
    )


@pytest.fixture
def context():
    context = Context()
    context.add_message("user", "Say hello")
    return context


@pytest.fixture
def client():
    return OpenAIChatCompletions(
        provider=Provider.OPENROUTER,
        model="test-model",
        api_key="key",
        tools={
            "get_weather": Tool(
                callable=lambda location: f"Weather in {location}",
                name="get_weather",
                description="Get weather for a location",
                params={
                    "location": {
                        "type": str,
                        "description": "City name",
                        "required": True,
                    }
                },
            )
        },
    )


def test_invoke_stream_yields_text_delta_events_and_final_response(client, context):
    chunks = [
        make_chunk(delta={"content": "Hel"}),
        make_chunk(delta={"content": "lo"}),
        make_chunk(usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}),
    ]
    client.client.chat.completions.create = MagicMock(return_value=iter(chunks))

    events = list(client.invoke_stream(context))

    assert [event.type for event in events] == [
        "text_delta",
        "text_delta",
        "message_done",
    ]
    assert all(isinstance(event, LLMStreamEvent) for event in events)
    assert "".join(event.text for event in events if event.type == "text_delta") == "Hello"
    assert events[-1].response == LLMResponse(text="Hello", tool_calls=[])
    assert client.input_tokens == 2
    assert client.output_tokens == 3

    client.client.chat.completions.create.assert_called_once()
    call_kwargs = client.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is True
    assert call_kwargs["stream_options"] == {"include_usage": True}


def test_invoke_returns_final_stream_response(client, context):
    chunks = [make_chunk(delta={"content": "Hello sync"})]
    client.client.chat.completions.create = MagicMock(return_value=iter(chunks))

    response = client.invoke(context)

    assert response == LLMResponse(text="Hello sync", tool_calls=[])


def test_invoke_stream_accumulates_tool_calls_until_message_done(client, context):
    chunks = [
        make_chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Par',
                        },
                    }
                ]
            }
        ),
        make_chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {
                            "arguments": 'is"}',
                        },
                    }
                ]
            },
            finish_reason="tool_calls",
        ),
    ]
    client.client.chat.completions.create = MagicMock(return_value=iter(chunks))

    events = list(client.invoke_stream(context))

    assert [event.type for event in events] == ["message_done"]
    final_response = events[-1].response
    assert isinstance(final_response, LLMResponse)
    assert final_response.text == ""
    assert len(final_response.tool_calls) == 1
    tool_call = final_response.tool_calls[0]
    assert tool_call.tool is client.tools["get_weather"]
    assert tool_call.tool_call_id == "call_weather"
    assert tool_call.args == []
    assert tool_call.kwargs == {"location": "Paris"}


def test_invoke_stream_accumulates_multiple_tool_calls_by_index(client, context):
    chunks = [
        make_chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Par',
                        },
                    },
                    {
                        "index": 1,
                        "id": "call_weather_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Lon',
                        },
                    },
                ]
            }
        ),
        make_chunk(
            delta={
                "tool_calls": [
                    {"index": 1, "function": {"arguments": 'don"}'}},
                    {"index": 0, "function": {"arguments": 'is"}'}},
                ]
            },
            finish_reason="tool_calls",
        ),
    ]
    client.client.chat.completions.create = MagicMock(return_value=iter(chunks))

    events = list(client.invoke_stream(context))

    final_response = events[-1].response
    assert isinstance(final_response, LLMResponse)
    assert [tool.tool_call_id for tool in final_response.tool_calls] == [
        "call_weather_1",
        "call_weather_2",
    ]
    assert [tool.kwargs for tool in final_response.tool_calls] == [
        {"location": "Paris"},
        {"location": "London"},
    ]


def test_invoke_stream_raises_for_malformed_tool_call_delta(client, context):
    chunks = [
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_weather",
                                type="function",
                                function=None,
                            )
                        ],
                    )
                )
            ],
        )
    ]
    client.client.chat.completions.create = MagicMock(return_value=iter(chunks))

    with pytest.raises(InvalidComponentError):
        list(client.invoke_stream(context))


@pytest.mark.parametrize(
    "tool_call_delta, error_fragment",
    [
        (
            {
                "index": 0,
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            },
            "missing tool call id",
        ),
        (
            {
                "index": 0,
                "id": "call_unknown",
                "type": "function",
                "function": {"name": "unknown_tool", "arguments": "{}"},
            },
            "unknown tool 'unknown_tool'",
        ),
        (
            {
                "index": 0,
                "id": "call_bad_json",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location":'},
            },
            "malformed JSON",
        ),
        (
            {
                "index": 0,
                "id": "call_non_object",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "[]"},
            },
            "expected JSON object",
        ),
    ],
)
def test_invoke_stream_raises_invalid_component_error_for_bad_tool_calls(
    client, context, tool_call_delta, error_fragment
):
    chunks = [make_chunk(delta={"tool_calls": [tool_call_delta]})]
    client.client.chat.completions.create = MagicMock(return_value=iter(chunks))

    with pytest.raises(InvalidComponentError, match=error_fragment):
        list(client.invoke_stream(context))


@pytest.mark.asyncio
async def test_ainvoke_stream_yields_matching_event_shape(client, context, mocker):
    chunks = [
        make_chunk(delta={"content": "Hi"}),
        make_chunk(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    ]
    fake_completions = SimpleNamespace(
        create=AsyncMock(return_value=FakeAsyncStream(chunks))
    )
    fake_async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=fake_completions)
    )
    async_openai = mocker.patch(
        "daystrom.components.openai.chat_completions.AsyncOpenAI",
        return_value=fake_async_client,
        create=True,
    )

    events = [event async for event in client.ainvoke_stream(context)]

    assert [event.type for event in events] == ["text_delta", "message_done"]
    assert events[0].text == "Hi"
    assert events[-1].response == LLMResponse(text="Hi", tool_calls=[])
    assert client.input_tokens == 1
    assert client.output_tokens == 1
    async_openai.assert_called_once_with(
        base_url=client.provider.value.base_url,
        api_key="key",
    )
    fake_completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_ainvoke_returns_final_stream_response(client, context, mocker):
    chunks = [make_chunk(delta={"content": "Hello async"})]
    fake_completions = SimpleNamespace(
        create=AsyncMock(return_value=FakeAsyncStream(chunks))
    )
    fake_async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=fake_completions)
    )
    mocker.patch(
        "daystrom.components.openai.chat_completions.AsyncOpenAI",
        return_value=fake_async_client,
        create=True,
    )

    response = await client.ainvoke(context)

    assert response == LLMResponse(text="Hello async", tool_calls=[])
