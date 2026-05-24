import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from daystrom import Provider
from daystrom.components.tool_util import CUSTOM_TOOLS, DEFAULT_TOOLS, Tool
from daystrom.exceptions import ToolCallError

ComponentResponseT = TypeVar("ComponentResponseT")

log = logging.getLogger(__name__)


class Component(Generic[ComponentResponseT], ABC):
    @abstractmethod
    def invoke(self, *args, **kwargs) -> ComponentResponseT | None:
        pass  # pragma: no cover


@dataclass
class ToolCall:
    tool: Tool
    tool_call_id: str
    args: list
    kwargs: dict


class Message:
    def __init__(
        self,
        role: str,
        text: str,
        tool_call_id: str = "",
        tool_calls: list[ToolCall] | None = None,
    ):
        self.role = role
        self.text = text
        self.tool_call_id = tool_call_id
        self.tool_calls: list[ToolCall] = tool_calls or []

    def __str__(self):
        parts = []
        parts.append(f"{self.role}: {self.text}")
        if self.tool_call_id:
            parts.append(f"    Tool Call ID: {self.tool_call_id}")
        if self.tool_calls:
            parts.append("    Tool Calls:")
            for tool in self.tool_calls:
                parts.append(f"    {str(tool)}")

        return "\n".join(parts)


class Context:
    def __init__(self, messages: list | None = None):
        self.messages: list[Message] = messages or []

    def add_message(
        self,
        role: str,
        text: str,
        tool_call_id: str = "",
        tool_calls: list[ToolCall] | None = None,
    ):
        self.messages.append(
            Message(
                text=text, role=role, tool_call_id=tool_call_id, tool_calls=tool_calls
            )
        )

    def print_feed(self):
        for message in self.messages:
            print(message)


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]


@dataclass
class LLMStreamEvent:
    type: Literal["text_delta", "message_done"]
    text: str = ""
    response: LLMResponse | None = None


@dataclass
class AgentResponse:
    text: str


class LLM(Component[LLMResponse]):
    tools: dict[str, Tool]
    provider: Provider
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float | None
    output_cost: float | None
    context_limit: int | None
    output_limit: int | None

    def __init__(
        self,
        provider: Provider,
        model: str,
        tools: dict[str, Tool] | None = None,
    ):
        self.provider = provider
        self.model = model

        self.tools = tools or {}
        self.input_tokens: int = 0
        self.output_tokens: int = 0

        self.input_cost = None
        self.output_cost = None
        self.context_limit = None
        self.output_limit = None
        model_metadata = provider.value.models.get(self.model)
        if model_metadata:
            self.input_cost = model_metadata.input_cost
            self.output_cost = model_metadata.output_cost
            self.context_limit = model_metadata.context_limit
            self.output_limit = model_metadata.output_limit

    @abstractmethod
    def invoke(self, *args, **kwargs) -> LLMResponse:
        pass  # pragma: no cover

    @abstractmethod
    def track_usage(self, *args, **kwargs):
        pass  # pragma: no cover

    @property
    def total_cost(self) -> float:
        """Calculate total cost based on token usage and costs.

        Returns 0.0 if cost information is not available.
        """
        if self.input_cost is None and self.output_cost is None:
            return 0.0
        # Costs are per million tokens
        input_total = (self.input_tokens / 1_000_000) * (self.input_cost or 0.0)
        output_total = (self.output_tokens / 1_000_000) * (self.output_cost or 0.0)
        return input_total + output_total


class Agent(Component[AgentResponse]):
    llm: LLM
    context: Context
    max_loops: int
    tools: dict

    def __init__(
        self,
        llm: LLM,
        context: Context | None = None,
        tools: dict[str, Tool] | None = None,
        max_loops: int = 30,
    ):
        # import tools here if they haven't been
        # already so the agent has access
        if not DEFAULT_TOOLS:
            import daystrom.components.tools

        self.llm = llm

        if context:
            self.context = context
        else:
            self.context = Context()

        self.max_loops = max_loops
        if tools is None:
            tools = DEFAULT_TOOLS.copy()
            tools.update(CUSTOM_TOOLS)
        self.tools = tools
        self.llm.tools = self.tools

    def invoke(self, prompt, *args, **kwargs) -> AgentResponse:
        loop = 0
        self.context.add_message("user", prompt)

        res = None
        while loop < self.max_loops:
            loop += 1
            res = self.llm.invoke(self.context)
            self.context.add_message(
                "assistant", text=res.text, tool_calls=res.tool_calls
            )

            # if no tools were called, the agent loop is done
            if not res.tool_calls:
                break

            for tool_call in res.tool_calls:
                try:
                    tool_res = tool_call.tool.call(*tool_call.args, **tool_call.kwargs)
                    self.context.add_message(
                        "tool", text=tool_res, tool_call_id=tool_call.tool_call_id
                    )
                except ToolCallError as e:
                    self.context.add_message(
                        "tool",
                        text=f"Tool call failed! Error: {e.message}",
                        tool_call_id=tool_call.tool_call_id,
                    )
                    log.exception(f"Tool call failed: {tool_call.tool.name}")
                except Exception as e:
                    self.context.add_message(
                        "tool",
                        text=f"Tool call failed! Error: {e}",
                        tool_call_id=tool_call.tool_call_id,
                    )
                    log.exception(f"Tool call failed: {tool_call.tool.name}")

        agent_res = AgentResponse(text=(res.text if res else ""))
        return agent_res
