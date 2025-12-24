import os

from openrouter import OpenRouter
from openrouter.components import AssistantMessage
from openrouter.components import Message as OpenRouterMessage  # , ToolResponseMessage
from openrouter.components import SystemMessage, UserMessage

from daystrom import Context
from daystrom.components import LLM


class OpenRouterChat(LLM[str]):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        context: Context | None = None,
    ):
        self.client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY") or api_key)
        if context:
            self.context = context
        else:
            self.context = Context()
        self.model = model
        super().__init__()

    def invoke(self, prompt) -> str:
        return "".join(self.invoke_stream(prompt))

    def invoke_stream(self, prompt):
        self.context.add_message("user", prompt)
        messages = self._get_prompt_context()
        res = self.client.chat.send(messages=messages, model=self.model, stream=True)

        response_content = ""
        for event in res:
            if isinstance(event.choices[0].delta.content, str):
                content_chunk = event.choices[0].delta.content
                response_content += content_chunk
                yield content_chunk
        self.context.add_message("assistant", response_content)

    async def ainvoke(self, prompt) -> str:
        return "".join([chunk async for chunk in self.ainvoke_stream(prompt)])

    async def ainvoke_stream(self, prompt):
        self.context.add_message("user", prompt)
        messages = self._get_prompt_context()
        res = await self.client.chat.send_async(
            messages=messages, model=self.model, stream=True
        )

        response_content = ""
        async for event in res:
            if isinstance(event.choices[0].delta.content, str):
                content_chunk = event.choices[0].delta.content
                response_content += content_chunk
                yield content_chunk
        self.context.add_message("assistant", response_content)

    def _get_prompt_context(self) -> list[OpenRouterMessage]:
        """
        Returns the messages in the context formatted for OpenRouter API
        """
        fmt_messages = []
        for msg in self.context.messages:
            match msg.role:
                case "user":
                    fmt_messages.append(UserMessage(content=msg.text))
                case "assistant":
                    fmt_messages.append(AssistantMessage(content=msg.text))
                case "system":
                    fmt_messages.append(SystemMessage(content=msg.text))

        return fmt_messages
