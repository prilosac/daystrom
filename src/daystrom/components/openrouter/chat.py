import os

from openrouter import OpenRouter
from openrouter.components import AssistantMessage
from openrouter.components import Message as OpenRouterMessage  # , ToolResponseMessage
from openrouter.components import SystemMessage, UserMessage

from daystrom import Provider
from daystrom.components import LLM, Context, LLMResponse


class OpenRouterChat(LLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
    ):
        super().__init__(model=model, provider=Provider.OPENROUTER)
        self.client = OpenRouter(api_key=api_key or self.provider.value.get_api_key())

    def invoke(self, context: Context | None = None) -> LLMResponse:
        response = LLMResponse(text="".join(self.invoke_stream(context)), tool_calls=[])
        return response

    def invoke_stream(self, context: Context | None = None):
        messages = self._get_prompt_context(context)
        res = self.client.chat.send(messages=messages, model=self.model, stream=True)

        response_content = ""
        for event in res:
            self.track_usage(event.usage)
            if event.choices and isinstance(event.choices[0].delta.content, str):
                content_chunk = event.choices[0].delta.content
                response_content += content_chunk
                yield content_chunk

    async def ainvoke(self, context: Context | None = None) -> LLMResponse:
        response = LLMResponse(
            text="".join([chunk async for chunk in self.ainvoke_stream(context)]),
            tool_calls=[],
        )
        return response

    async def ainvoke_stream(self, context: Context | None = None):
        messages = self._get_prompt_context(context)
        res = await self.client.chat.send_async(
            messages=messages, model=self.model, stream=True
        )

        response_content = ""
        async for event in res:
            self.track_usage(event.usage)
            if event.choices and isinstance(event.choices[0].delta.content, str):
                content_chunk = event.choices[0].delta.content
                response_content += content_chunk
                yield content_chunk

    def track_usage(self, usage):
        if usage:
            self.output_tokens += usage.completion_tokens
            self.input_tokens += usage.prompt_tokens

    def _get_prompt_context(self, context: Context | None) -> list[OpenRouterMessage]:
        """
        Returns the messages in the context formatted for OpenRouter API
        """
        if not context:
            return []

        fmt_messages = []
        for msg in context.messages:
            match msg.role:
                case "user":
                    fmt_messages.append(UserMessage(role=msg.role, content=msg.text))
                case "assistant":
                    fmt_messages.append(
                        AssistantMessage(role=msg.role, content=msg.text)
                    )
                case "system":
                    fmt_messages.append(SystemMessage(role=msg.role, content=msg.text))

        return fmt_messages
