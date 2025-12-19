from openrouter import OpenRouter
from openrouter.components import AssistantMessage
from openrouter.components import Message as OpenRouterMessage  # , ToolResponseMessage
from openrouter.components import SystemMessage, UserMessage

from daystrom.context import Context


class OpenRouterChat:
    def __init__(
        self,
        api_key,
        model="anthropic/claude-haiku-4.5",
        context: Context | None = None,
    ):
        self.client = OpenRouter(api_key=api_key)
        if context:
            self.context = context
        else:
            self.context = Context()
        self.model = model
        super().__init__()

    def invoke(self, prompt):
        return "".join(self.invoke_stream(prompt))

    def invoke_stream(self, prompt):
        self.context.add_message("user", prompt)
        messages = self.get_prompt_context()
        res = self.client.chat.send(messages=messages, model=self.model, stream=True)

        response_content = ""
        for event in res:
            if isinstance(event.choices[0].delta.content, str):
                content_chunk = event.choices[0].delta.content
                response_content += content_chunk
                yield content_chunk
        self.context.add_message("assistant", response_content)

    async def ainvoke(self, prompt):
        return "".join([chunk async for chunk in self.ainvoke_stream(prompt)])

    async def ainvoke_stream(self, prompt):
        self.context.add_message("user", prompt)
        messages = self.get_prompt_context()
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

    def get_prompt_context(self) -> list[OpenRouterMessage]:
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
