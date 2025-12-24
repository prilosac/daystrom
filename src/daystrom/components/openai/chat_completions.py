import os

from openai import OpenAI
from openai.types.chat import (  # ChatCompletionDeveloperMessageParam, # should probably use this one, it replaces system_message on some newer models apparently; ChatCompletionFunctionMessageParam,; ChatCompletionToolMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from daystrom import Context
from daystrom.components import LLM, LLMResponse
from daystrom.exceptions import InvalidComponentError


class OpenAIChatCompletions(LLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        url: str | None = None,
        context: Context | None = None,
    ):
        self.client = OpenAI(
            base_url=url, api_key=os.getenv("OPENROUTER_API_KEY") or api_key
        )
        if context:
            self.context = context
        else:
            self.context = Context()
        self.model = model
        super().__init__()

    def invoke(self, prompt) -> LLMResponse:
        self.context.add_message("user", prompt)
        messages = self._get_prompt_context()
        completion = self.client.chat.completions.create(
            messages=messages, model=self.model
        )
        completion_text = completion.choices[0].message.content or ""
        self.context.add_message("assistant", completion_text)
        response = LLMResponse(text=completion_text, tool_calls=[])
        return response

    def _get_prompt_context(self) -> list[ChatCompletionMessageParam]:
        """
        Returns the messages in the context formatted for OpenRouter API
        """
        fmt_messages = []
        for msg in self.context.messages:
            match msg.role:
                case "user":
                    fmt_messages.append(
                        ChatCompletionUserMessageParam(role="user", content=msg.text)
                    )
                case "assistant":
                    fmt_messages.append(
                        ChatCompletionAssistantMessageParam(
                            role="assistant", content=msg.text
                        )
                    )
                case "system":
                    fmt_messages.append(
                        ChatCompletionSystemMessageParam(
                            role="system", content=msg.text
                        )
                    )

        return fmt_messages


# this is a decorator to be @tool above each tool fuction
def tool(func):
    docstring = parse(func.__doc__ or "")
    inspect_params = inspect.signature(func).parameters
    if len(docstring.params) != len(inspect_params):
        raise ValueError(
            f"Type hints do not align with docstrings: num_args: (type hint) {len(inspect_params)} vs. {len(docstring.params)} (docstring)"
        )

    func_params = {}
    required_func_params = []
    py_to_json_type_map = {
        "list": "array",
        "tuple": "array",
        "int": "integer",
        "str": "string",
    }
    # py_to_json_type_map = {
    #    "dict": "object",
    #    "list": "array",
    #    "tuple": "array",
    #    "str": "string",
    #    "int": "integer",
    #    "float": "number",
    #    "None": "null",
    #    #True   true
    #    #False   false
    #    #int, float, int- & float-derived Enums     number
    # }
    for idx, (name, param) in enumerate(inspect_params.items()):
        if param.default == inspect._empty:
            required_func_params.append(name)

        param_type_name = py_to_json_type_map.get(
            param.annotation.__name__, param.annotation.__name__
        )
        if param.annotation == inspect._empty:
            param_type_name = "string"

        if name != docstring.params[idx].arg_name:
            raise TypeError(
                f"Type hints do not align with docstrings: arg_name: (type hint) {name} vs. {docstring.params[idx].arg_name} (docstring)"
            )

        func_params[name] = {
            "type": param_type_name,
            "description": docstring.params[idx].description,
        }

        if get_origin(param.annotation) in (list, tuple):
            if len(param.annotation.__args__) > 1:
                raise TypeError(
                    "Only single-type iterables are allowed as parameters to tool calls."
                )
            else:
                func_params[name]["items"] = {
                    "type": py_to_json_type_map.get(
                        param.annotation.__args__[0].__name__,
                        param.annotation.__args__[0].__name__,
                    )
                }

    tool_desc = docstring.long_description or docstring.short_description or ""
    tool_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": tool_desc,
            "parameters": {
                "type": "object",
                "properties": func_params,
                "required": required_func_params,
            },
        },
    }

    TOOL_SCHEMAS.append(tool_schema)
    TOOL_MAPPING[func.__name__] = func

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
