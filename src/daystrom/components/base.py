import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, get_origin

from docstring_parser import parse

ComponentResponseT = TypeVar("ComponentResponseT")


class Component(Generic[ComponentResponseT], ABC):
    @abstractmethod
    def invoke(self, *args, **kwargs) -> ComponentResponseT:
        pass


class Tool:
    def __init__(
        self,
        callable,
        name: str = "",
        display_name: str = "",
        description: str = "",
        params: dict = {},
    ):
        self.callable = callable
        self.name = name or callable.__name__
        self.display_name = display_name or self.name.replace("_", " ").title()
        self.description = description or callable.__doc__ or ""
        self.params = params

    def call(self, *args, **kwargs):
        return self.callable(*args, **kwargs)


@dataclass
class ToolCall:
    tool: Tool
    args: list
    kwargs: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]


class LLM(Component[LLMResponse]):
    @abstractmethod
    def invoke(self, *args, **kwargs) -> LLMResponse:
        pass


DEFAULT_TOOLS = []


# this is a decorator to be @tool above each tool fuction
def tool(func):
    docstring = parse(func.__doc__ or "")
    inspect_params = inspect.signature(func).parameters

    func_params = {}
    required_func_params = []

    for idx, (name, param) in enumerate(inspect_params.items()):
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError("*args is not supported in tool parameters.")
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            raise TypeError("**kwargs is not supported in tool parameters.")

        if param.default == inspect._empty:
            required_func_params.append(name)

        func_params[name] = {
            "type": param.annotation,
            "description": docstring.params[idx].description,
        }

        if get_origin(param.annotation) in (list, tuple):
            if len(param.annotation.__args__) > 1:
                raise TypeError(
                    "Only single-type iterables are allowed as parameters to tool calls."
                )
            else:
                func_params[name]["items"] = {"type": param.annotation.__args__[0]}

    tool_desc = docstring.long_description or docstring.short_description or ""

    new_tool = Tool(func, name=func.__name__, description=tool_desc, params=func_params)

    DEFAULT_TOOLS.append(new_tool)

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class Agent(Component[ComponentResponseT]):
    def __init__(
        self, llm: LLM, tools: list[Tool] = DEFAULT_TOOLS, max_loops: int = 30
    ):
        self.llm = llm
        self.tools = tools
        self.max_loops = 30

    def invoke(self, prompt, *args, **kwargs) -> ComponentResponseT:
        loop = 0
        while loop < self.max_loops:
            loop += 1
            res = self.llm.invoke()

            if not res.tool_calls:
                break

            for tool_call in res.tool_calls:
                tool_res = tool_call.tool.call(*tool_call.args, **tool_call.kwargs)
        return self.llm.invoke(prompt)


class Agent:
    _llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY"),
    )
    _max_iterations = 10

    def __init__(
        self,
        model,
        tool_mapping=TOOL_MAPPING,
        tool_schemas=TOOL_SCHEMAS,
        messages=[],
        prompt=None,
        system_prompt=None,
    ):
        self.model = model
        self.tool_mapping = tool_mapping
        self.tool_schemas = tool_schemas

        if not (messages or prompt):
            raise ValueError(
                "Either prompt or messages is required to initialize the agent."
            )
        if messages and prompt:
            raise ValueError(
                "Only one of prompt or messages can be provided to initialize the agent."
            )
        if messages:
            self.messages = messages
        elif prompt:
            self.messages = [{"role": "user", "content": prompt}]

        if system_prompt:
            self.messages.insert(0, {"role": "system", "content": system_prompt})

    def call_llm(self):
        response = self._llm_client.chat.completions.create(
            model=self.model,
            tools=self.tool_schemas,
            messages=self.messages,
        )
        self.messages.append(response.choices[0].message.model_dump())
        return response

    def call_tool(self, tool_response):
        log.info(
            f"Calling tool: {tool_response.function.name} with args: {tool_response.function.arguments}"
        )
        tool_name = tool_response.function.name
        tool_args = json.loads(tool_response.function.arguments)

        tool_result = str(self.tool_mapping[tool_name](**tool_args))
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_response.id,
                "content": tool_result,
            }
        )
        return tool_result

    def run_loop(self):
        iteration = 0
        while iteration < self._max_iterations:
            iteration += 1
            response = self.call_llm()
            tool_responses = response.choices[0].message.tool_calls

            if tool_responses is not None:
                for tool_response in tool_responses:
                    self.call_tool(tool_response)
            else:
                break

        return self.messages[-1]["content"]
