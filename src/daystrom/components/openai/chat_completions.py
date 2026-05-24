import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, get_origin

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import FunctionDefinition

from daystrom import Provider
from daystrom.components import LLM, Context, LLMResponse, LLMStreamEvent, Tool, ToolCall
from daystrom.exceptions import InvalidComponentError


@dataclass
class _StreamingToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""
    type: str = "function"


class OpenAIChatCompletions(LLM):
    def __init__(
        self,
        model: str,
        provider: Provider | None = None,
        api_key: str | None = None,
        tools: dict[str, Tool] | None = None,
    ):
        super().__init__(
            model=model,
            tools=tools or {},
            provider=provider or Provider.OPENAI,
        )
        self.api_key = api_key or self.provider.value.get_api_key()
        self.client = OpenAI(
            base_url=self.provider.value.base_url,
            api_key=self.api_key,
        )
        self._async_client: AsyncOpenAI | None = None

    def invoke(self, context: Context | None = None) -> LLMResponse:
        response = LLMResponse(text="", tool_calls=[])
        for event in self.invoke_stream(context):
            if event.type == "message_done" and event.response is not None:
                response = event.response
        return response

    def invoke_stream(self, context: Context | None = None) -> Iterator[LLMStreamEvent]:
        stream = self.client.chat.completions.create(
            **self._get_completion_kwargs(context),
            stream=True,
            stream_options={"include_usage": True},
        )

        text_chunks: list[str] = []
        tool_calls: dict[int, _StreamingToolCall] = {}
        for chunk in stream:
            yield from self._handle_stream_chunk(chunk, text_chunks, tool_calls)

        yield LLMStreamEvent(
            type="message_done",
            response=self._build_response_from_stream(text_chunks, tool_calls),
        )

    async def ainvoke(self, context: Context | None = None) -> LLMResponse:
        response = LLMResponse(text="", tool_calls=[])
        async for event in self.ainvoke_stream(context):
            if event.type == "message_done" and event.response is not None:
                response = event.response
        return response

    async def ainvoke_stream(
        self, context: Context | None = None
    ) -> AsyncIterator[LLMStreamEvent]:
        stream = await self.async_client.chat.completions.create(
            **self._get_completion_kwargs(context),
            stream=True,
            stream_options={"include_usage": True},
        )

        text_chunks: list[str] = []
        tool_calls: dict[int, _StreamingToolCall] = {}
        async for chunk in stream:
            for event in self._handle_stream_chunk(chunk, text_chunks, tool_calls):
                yield event

        yield LLMStreamEvent(
            type="message_done",
            response=self._build_response_from_stream(text_chunks, tool_calls),
        )

    @property
    def async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self.provider.value.base_url,
                api_key=self.api_key,
            )
        return self._async_client

    def _get_completion_kwargs(self, context: Context | None = None) -> dict[str, Any]:
        return {
            "model": self.model,
            "tools": self._get_tool_context(),
            "messages": self._get_prompt_context(context),
        }

    def _handle_stream_chunk(
        self,
        chunk: Any,
        text_chunks: list[str],
        tool_calls: dict[int, _StreamingToolCall],
    ) -> list[LLMStreamEvent]:
        self.track_usage(getattr(chunk, "usage", None))
        events: list[LLMStreamEvent] = []

        for choice in getattr(chunk, "choices", []) or []:
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                text_chunks.append(content)
                events.append(LLMStreamEvent(type="text_delta", text=content))

            for tool_call_delta in getattr(delta, "tool_calls", None) or []:
                self._accumulate_tool_call_delta(tool_call_delta, tool_calls)

        return events

    def _accumulate_tool_call_delta(
        self,
        tool_call_delta: Any,
        tool_calls: dict[int, _StreamingToolCall],
    ) -> None:
        index = getattr(tool_call_delta, "index", None)
        if index is None:
            index = len(tool_calls)

        tool_call = tool_calls.setdefault(index, _StreamingToolCall())

        tool_call_id = getattr(tool_call_delta, "id", None)
        if tool_call_id:
            tool_call.id = tool_call_id

        tool_call_type = getattr(tool_call_delta, "type", None)
        if tool_call_type:
            tool_call.type = tool_call_type

        function = getattr(tool_call_delta, "function", None)
        if function is None:
            return

        name = getattr(function, "name", None)
        if name:
            tool_call.name = name

        arguments = getattr(function, "arguments", None)
        if arguments:
            tool_call.arguments += arguments

    def _build_response_from_stream(
        self,
        text_chunks: list[str],
        streamed_tool_calls: dict[int, _StreamingToolCall],
    ) -> LLMResponse:
        tool_calls = [
            self._build_tool_call(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                tool_type=tool_call.type,
            )
            for _, tool_call in sorted(streamed_tool_calls.items())
        ]
        return LLMResponse(text="".join(text_chunks), tool_calls=tool_calls)

    def _build_tool_call(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: str,
        tool_type: str | None = "function",
    ) -> ToolCall:
        if tool_type != "function" or not tool_name:
            raise InvalidComponentError(
                self.__class__.__name__,
                "Found unsupported tool call - missing 'function' attribute",
            )
        if not tool_call_id:
            raise InvalidComponentError(
                self.__class__.__name__,
                f"Found invalid tool call for '{tool_name}' - missing tool call id",
            )
        if tool_name not in self.tools:
            raise InvalidComponentError(
                self.__class__.__name__,
                f"Found invalid tool call - unknown tool '{tool_name}'",
            )

        try:
            kwargs = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            raise InvalidComponentError(
                self.__class__.__name__,
                f"Found invalid tool call arguments for '{tool_name}' - malformed JSON",
            ) from exc
        if not isinstance(kwargs, dict):
            raise InvalidComponentError(
                self.__class__.__name__,
                f"Found invalid tool call arguments for '{tool_name}' - expected JSON object",
            )

        return ToolCall(
            tool=self.tools[tool_name],
            tool_call_id=tool_call_id,
            args=[],
            kwargs=kwargs,
        )

    def track_usage(self, usage):
        if usage:
            self.output_tokens += usage.completion_tokens
            self.input_tokens += usage.prompt_tokens

    def _get_prompt_context(
        self, context: Context | None = None
    ) -> list[ChatCompletionMessageParam]:
        """
        Returns the messages in the context formatted for OpenAI API
        """
        if not context:
            return []

        fmt_messages = []
        for msg in context.messages:
            match msg.role:
                case "user":
                    fmt_messages.append(
                        ChatCompletionUserMessageParam(role="user", content=msg.text)
                    )
                case "assistant":
                    tool_calls = []
                    for tool_call in msg.tool_calls:
                        # ChatCompletionMessageToolCallUnionParam
                        tool_calls.append(
                            {
                                "function": {
                                    "name": tool_call.tool.name,
                                    "arguments": json.dumps(tool_call.kwargs),
                                },
                                "type": "function",
                                "id": tool_call.tool_call_id,
                            }
                        )
                    if tool_calls:
                        fmt_messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=msg.text,
                                tool_calls=tool_calls,
                            )
                        )
                    else:
                        fmt_messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant", content=msg.text
                            )
                        )
                case "system":
                    fmt_messages.append(
                        ChatCompletionDeveloperMessageParam(
                            role="developer", content=msg.text
                        )
                    )
                case "tool":
                    fmt_messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool", content=msg.text, tool_call_id=msg.tool_call_id
                        )
                    )
                case _:
                    raise ValueError(
                        f"Unsupported message role: {msg.role} for {self.__class__.__name__}"
                    )

        return fmt_messages

    def _get_tool_context(self) -> list[ChatCompletionToolUnionParam]:
        tool_schemas = []

        for tool in self.tools.values():
            function = FunctionDefinition(
                name=tool.name,
                description=tool.description,
                parameters=self._format_tool_params(tool),
            )
            tool_schema = ChatCompletionToolParam(function=function, type="function")
            tool_schemas.append(tool_schema)
        return tool_schemas

    def _format_tool_params(self, tool: Tool) -> dict:
        params = {"type": "object", "properties": {}}
        required_params = []
        type_map = {
            dict: "object",
            list: "array",
            tuple: "array",
            str: "string",
            int: "integer",
            float: "number",
            None: "null",
            bool: "boolean",
        }

        for pname, pinfo in tool.params.items():
            params["properties"][pname] = {
                "type": type_map[get_origin(pinfo["type"]) or pinfo["type"]],
                "description": pinfo["description"],
            }

            param_items = pinfo.get("items")
            if param_items is not None:
                param_type = param_items["type"]
                params["properties"][pname]["items"] = {
                    "type": type_map[get_origin(param_type) or param_type],
                }

            if pinfo["required"]:
                required_params.append(pname)

        if required_params:
            params["required"] = required_params

        return params
