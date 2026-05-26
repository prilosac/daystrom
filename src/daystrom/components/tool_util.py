import functools
import inspect
import logging
from typing import get_origin

from docstring_parser import parse

log = logging.getLogger(__name__)

DEFAULT_TOOLS = {}
CUSTOM_TOOLS = {}


class Tool:
    def __init__(
        self,
        callable,
        name: str = "",
        display_name: str = "",
        description: str = "",
        params: dict | None = None,
    ):
        self.callable = callable
        self.name = name or callable.__name__
        self.display_name = display_name or self.name.replace("_", " ").title()
        self.description = description or callable.__doc__ or ""
        self.params = params or {}

    def __str__(self):
        return f"Tool(name={self.name}, description={self.description}, params={self.params})"

    def call(self, *args, **kwargs):
        return self.callable(*args, **kwargs)


# this is a decorator to be @tool above each tool function
def tool(func=None, *, type="custom"):
    def tool_dec(func):
        docstring = parse(func.__doc__ or "")
        inspect_params = inspect.signature(func).parameters

        func_params = {}

        for idx, (name, param) in enumerate(inspect_params.items()):
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TypeError("*args is not supported in tool parameters.")
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise TypeError("**kwargs is not supported in tool parameters.")
            # pattern here is that keyword only arguments are recognized and
            # injected by the agent, not for the LLM to be concerned with
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                continue

            required = False
            if param.default is inspect.Parameter.empty:
                required = True

            description = ""
            if len(docstring.params) >= idx + 1:
                description = docstring.params[idx].description

            func_params[name] = {
                "type": param.annotation,
                "description": description,
                "required": required,
            }

            if get_origin(param.annotation) in (list, tuple):
                if len(param.annotation.__args__) > 1:
                    raise TypeError(
                        "Only single-type iterables are allowed as parameters to tool calls."
                    )
                else:
                    func_params[name]["items"] = {"type": param.annotation.__args__[0]}

        tool_desc = docstring.long_description or docstring.short_description or ""

        new_tool = Tool(
            func, name=func.__name__, description=tool_desc, params=func_params
        )

        match type:
            case "default":
                DEFAULT_TOOLS[new_tool.name] = new_tool
            case "custom":
                CUSTOM_TOOLS[new_tool.name] = new_tool
            case _:
                raise ValueError(f"Unsupported tool type: {type}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    if func is None:
        return tool_dec

    return tool_dec(func)
