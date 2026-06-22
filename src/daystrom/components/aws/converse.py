import os
from dataclasses import asdict, dataclass
from typing import Literal, get_origin

import boto3
from botocore.config import Config

from daystrom import Provider
from daystrom.components import LLM, Context, LLMResponse, Tool, ToolCall
from daystrom.exceptions import InvalidComponentError


@dataclass
class AwsAccessKeyAuth:
    kind: Literal["access_key"] = "access_key"
    config: Config = Config(signature_version="v4")
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str | None = None
    region_name: str = ""

    def get_auth_dict(self) -> dict:
        auth_dict = asdict(self)
        auth_dict = {
            k: v for k, v in auth_dict.items() if k != "kind" and v is not None
        }
        return auth_dict


@dataclass
class AwsBearerTokenAuth:
    kind: Literal["bearer_token"] = "bearer_token"
    aws_bearer_token_bedrock: str = ""
    region_name: str = ""

    def get_auth_dict(self) -> dict:
        auth_dict = asdict(self)
        # boto3.client doesn't accept aws_bearer_token_bedrock as a direct
        # arg like it does access and secret keys, so we have to omit it
        # here and let it pull it from the env
        auth_dict = {
            k: v
            for k, v in auth_dict.items()
            if k != "kind" and k != "aws_bearer_token_bedrock" and v is not None
        }
        return auth_dict


@dataclass
class AwsFailedAuth:
    kind: Literal["failed"] = "failed"
    message: str = ""


class BedrockConverse(LLM):
    def __init__(
        self,
        model: str,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        aws_bearer_token_bedrock: str | None = None,
        tools: dict[str, Tool] | None = None,
    ):
        super().__init__(
            model=model,
            tools=tools or {},
            provider=Provider.AWS_BEDROCK,
        )

        auth_obj = self.get_auth(
            region_name,
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
            aws_bearer_token_bedrock,
        )
        if auth_obj.kind == "failed":
            raise InvalidComponentError(self.__class__.__name__, auth_obj.message)
        self.client = boto3.client(
            "bedrock-runtime",
            **auth_obj.get_auth_dict(),
        )

    def invoke(self, context: Context | None = None) -> LLMResponse:
        system_messages, messages = self._get_prompt_context(context)

        kwargs = {
            "modelId": self.model,
            "messages": messages,
        }
        if system_messages:
            kwargs["system"] = system_messages

        tool_config = self._get_tool_context()
        if tool_config:
            kwargs["toolConfig"] = tool_config

        response = self.client.converse(**kwargs)
        self.track_usage(response.get("usage"))

        # Parse response content blocks
        content_blocks = response["output"]["message"].get("content", [])

        text = ""
        tool_calls = []
        for block in content_blocks:
            if "text" in block:
                text += block["text"]
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        tool=self.tools[tool_use["name"]],
                        tool_call_id=tool_use["toolUseId"],
                        args=[],
                        kwargs=tool_use.get("input", {}),
                    )
                )

        return LLMResponse(text=text, tool_calls=tool_calls)

    def track_usage(self, usage):
        if usage:
            self.input_tokens += usage.get("inputTokens", 0)
            self.output_tokens += usage.get("outputTokens", 0)

    def _get_prompt_context(
        self, context: Context | None = None
    ) -> tuple[list[dict], list[dict]]:
        """
        Returns (system_messages, conversation_messages) formatted for the
        Bedrock Converse API.

        System messages are separated out because Converse takes them as a
        top-level parameter rather than inline in the messages array.

        Consecutive tool-result messages are grouped into a single 'user'
        message with multiple toolResult content blocks, since Converse
        requires alternating user/assistant roles.
        """
        if not context:
            return [], []

        system_messages: list[dict] = []
        messages: list[dict] = []

        i = 0
        while i < len(context.messages):
            msg = context.messages[i]

            match msg.role:
                case "system":
                    system_messages.append({"text": msg.text})
                    i += 1
                case "user":
                    messages.append({"role": "user", "content": [{"text": msg.text}]})
                    i += 1
                case "assistant":
                    content: list[dict] = []
                    if msg.text:
                        content.append({"text": msg.text})
                    for tool_call in msg.tool_calls:
                        content.append(
                            {
                                "toolUse": {
                                    "toolUseId": tool_call.tool_call_id,
                                    "name": tool_call.tool.name,
                                    "input": tool_call.kwargs,
                                }
                            }
                        )
                    messages.append({"role": "assistant", "content": content})
                    i += 1
                case "tool":
                    # Group consecutive tool messages into one user message
                    tool_results: list[dict] = []
                    while (
                        i < len(context.messages) and context.messages[i].role == "tool"
                    ):
                        tool_msg = context.messages[i]
                        tool_results.append(
                            {
                                "toolResult": {
                                    "toolUseId": tool_msg.tool_call_id,
                                    "content": [{"text": tool_msg.text}],
                                }
                            }
                        )
                        i += 1
                    messages.append({"role": "user", "content": tool_results})
                case _:
                    raise ValueError(
                        f"Unsupported message role: {msg.role} for {self.__class__.__name__}"
                    )

        return system_messages, messages

    def _get_tool_context(self) -> dict:
        if not self.tools:
            return {}

        tools = []
        for tool in self.tools.values():
            tool_spec = {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description or tool.name,
                    "inputSchema": {"json": self._format_tool_params(tool)},
                }
            }
            tools.append(tool_spec)

        return {"tools": tools}

    def _format_tool_params(self, tool: Tool) -> dict:
        params: dict = {"type": "object", "properties": {}}
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

    def get_auth(
        self,
        region,
        aws_access_key_id,
        aws_secret_access_key,
        aws_session_token,
        aws_bearer_token_bedrock,
    ) -> AwsAccessKeyAuth | AwsBearerTokenAuth | AwsFailedAuth:
        region = (
            region
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-west-2"
        )

        bearer_token = aws_bearer_token_bedrock or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        if bearer_token:
            return AwsBearerTokenAuth(
                aws_bearer_token_bedrock=bearer_token,
                region_name=region,
            )
        access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            return AwsAccessKeyAuth(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                aws_session_token=aws_session_token or os.getenv("AWS_SESSION_TOKEN"),
                region_name=region,
            )
        return AwsFailedAuth(
            message="One of the following must be provided, neither was found:\n\n1. AWS_BEARER_TOKEN_BEDROCK\n2. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY"
        )
