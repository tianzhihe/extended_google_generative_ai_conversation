from __future__ import annotations

"""Conversation support for the Extended Google Generative AI Conversation integration."""

"""
Revised version of the Google Generative AI Conversation integration
with a structure similar to Code 2's function-call handling approach.
"""


import codecs
from collections.abc import Callable
from typing import Any, Literal, cast

import yaml

from google.genai.errors import APIError
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Content,
    FunctionDeclaration,
    GenerateContentConfig,
    HarmCategory,
    Part,
    SafetySetting,
    Schema,
    Tool,
)

from voluptuous_openapi import convert

from homeassistant.components import assist_pipeline, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import chat_session
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent
from homeassistant.helpers import llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_CHAT_MODEL,
    CONF_DANGEROUS_BLOCK_THRESHOLD,
    CONF_HARASSMENT_BLOCK_THRESHOLD,
    CONF_HATE_BLOCK_THRESHOLD,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_SEXUAL_BLOCK_THRESHOLD,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_HARM_BLOCK_THRESHOLD,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_K,
    RECOMMENDED_TOP_P,
    CONF_FUNCTIONS,
    DEFAULT_CONF_FUNCTIONS,
    CONF_USE_TOOLS,
    DEFAULT_USE_TOOLS,
)

from .exceptions import (
    FunctionLoadFailed,
    FunctionNotFound,
    InvalidFunction,
    ParseArgumentsFailed,
    TokenLengthExceededError,
)

from .helpers import (
    get_function_executor
)

MAX_TOOL_ITERATIONS = 10

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = GoogleGenerativeAIConversationEntity(config_entry)
    async_add_entities([agent])

SUPPORTED_SCHEMA_KEYS = {
    "min_items",
    "example",
    "property_ordering",
    "pattern",
    "minimum",
    "default",
    "any_of",
    "max_length",
    "title",
    "min_properties",
    "min_length",
    "max_items",
    "maximum",
    "nullable",
    "max_properties",
    "type",
    "description",
    "enum",
    "format",
    "items",
    "properties",
    "required",
}

def _camel_to_snake(name: str) -> str:
    """Convert camel case to snake case."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")

def _format_schema(schema: dict[str, Any]) -> Schema:
    """Format the schema to be compatible with Gemini API."""
    if subschemas := schema.get("allOf"):
        for subschema in subschemas:
            if "type" in subschema:
                return _format_schema(subschema)
        return _format_schema(subschemas[0])

    result = {}
    for key, val in schema.items():
        key = _camel_to_snake(key)
        if key not in SUPPORTED_SCHEMA_KEYS:
            continue
        if key == "any_of":
            val = [_format_schema(subschema) for subschema in val]
        elif key == "type":
            val = val.upper()
        elif key == "format":
            if schema.get("type") == "string" and val not in ("enum", "date-time"):
                continue
            if schema.get("type") == "number" and val not in ("float", "double"):
                continue
            if schema.get("type") == "integer" and val not in ("int32", "int64"):
                continue
            if schema.get("type") not in ("string", "number", "integer"):
                continue
        elif key == "items":
            val = _format_schema(val)
        elif key == "properties":
            val = {k: _format_schema(v) for k, v in val.items()}
        result[key] = val

    if result.get("enum") and result.get("type") != "STRING":
        result["type"] = "STRING"
        result["enum"] = [str(item) for item in result["enum"]]

    if result.get("type") == "OBJECT" and not result.get("properties"):
        result["properties"] = {"json": {"type": "STRING"}}
        result["required"] = []
    return cast(Schema, result)

def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> Tool:
    """Format tool specification."""
    if tool.parameters.schema:
        parameters = _format_schema(
            convert(tool.parameters, custom_serializer=custom_serializer)
        )
    else:
        parameters = None

    return Tool(
        function_declarations=[
            FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=parameters,
            )
        ]
    )

def _escape_decode(value: Any) -> Any:
    """Recursively call codecs.escape_decode on all values."""
    if isinstance(value, str):
        return codecs.escape_decode(bytes(value, "utf-8"))[0].decode("utf-8")
    if isinstance(value, list):
        return [_escape_decode(item) for item in value]
    if isinstance(value, dict):
        return {k: _escape_decode(v) for k, v in value.items()}
    return value

def _create_google_tool_response_content(
    content: list[conversation.ToolResultContent],
) -> Content:
    """Create a Google tool response content."""
    return Content(
        parts=[
            Part.from_function_response(
                name=tool_result.tool_name,
                response=tool_result.tool_result
            )
            for tool_result in content
        ]
    )

def _convert_content(
    content: conversation.UserContent
    | conversation.AssistantContent
    | conversation.SystemContent,
) -> Content:
    """Convert HA content to Google content."""
    if content.role != "assistant" or not content.tool_calls:  # type: ignore[union-attr]
        role = "model" if content.role == "assistant" else content.role
        return Content(
            role=role,
            parts=["""Conversation support for Extended Google Generative AI Conversation integration."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from google.genai import ChatSession
from google.genai.types import Content, Part, FunctionDeclaration, Tool
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm

# Added imports
from homeassistant.util import ulid
import voluptuous as vol

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_TOP_K,
    CONF_FUNCTIONS,
    CONF_MAX_FUNCTION_CALLS,
    DEFAULT_MAX_FUNCTION_CALLS,
    DOMAIN,
    LOGGER,
)
from .exceptions import (
    FunctionLoadFailed,
    FunctionNotFound,
    InvalidFunction,
    ParseArgumentsFailed,
)
from .helpers import get_function_executor

class GoogleGenerativeAIConversationAgent(conversation.AbstractConversationAgent):
    """Google Gemini conversation agent with OpenAI-style function calling."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.client = entry.runtime_data
        self.history: dict[str, list[dict]] = {}

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a conversation input."""
        conversation_id = user_input.conversation_id or ulid.ulid()
        messages = self.history.get(conversation_id, [])

        # Initialize conversation
        if not messages:
            messages = [self._create_system_message()]
            self.history[conversation_id] = messages

        messages.append({"role": "user", "content": user_input.text})

        try:
            response = await self._execute_conversation(
                user_input=user_input,
                messages=messages,
                conversation_id=conversation_id,
                n_requests=0
            )
        except HomeAssistantError as err:
            LOGGER.error("Conversation error: %s", err)
            response = intent.IntentResponse(language=user_input.language)
            response.async_set_error(str(err))
            return conversation.ConversationResult(
                response=response, conversation_id=conversation_id
            )

        return conversation.ConversationResult(
            response=response, conversation_id=conversation_id
        )

    async def _execute_conversation(
        self,
        user_input: conversation.ConversationInput,
        messages: list[dict],
        conversation_id: str,
        n_requests: int
    ) -> intent.IntentResponse:
        """Execute conversation with function calling support."""
        if n_requests >= self.entry.options.get(CONF_MAX_FUNCTION_CALLS, DEFAULT_MAX_FUNCTION_CALLS):
            raise HomeAssistantError("Maximum function calls reached")

        # Generate Gemini-compatible contents
        contents = [
            Content(role=msg["role"], parts=[Part.from_text(msg["content"])]
            for msg in messages if msg["role"] in ("user", "system")
        ]

        # Add tool definitions
        tools = self._load_tools()
        chat = self.client.chat.create(
            model=self.entry.options[CONF_CHAT_MODEL],
            contents=contents,
            tools=tools,
            config=self._create_generation_config()
        )

        try:
            response = await chat.send_message()
        except APIError as err:
            raise HomeAssistantError(f"API error: {err}") from err

        # Process response
        response_message = self._parse_response(response)
        messages.append(response_message)

        # Handle function calls
        if "function_call" in response_message:
            result = await self._execute_function_call(
                response_message["function_call"],
                user_input,
                messages,
                conversation_id,
                n_requests
            )
            messages.append(result)
            return await self._execute_conversation(
                user_input, messages, conversation_id, n_requests + 1
            )

        return self._create_response(response_message)

    def _load_tools(self) -> list[Tool]:
        """Load and format tools for Gemini API."""
        tools = []
        for func_spec in self._get_functions():
            tools.append(Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name=func_spec["name"],
                        description=func_spec["description"],
                        parameters=self._format_parameters(func_spec["parameters"])
                    )
                ]
            ))
        return tools

    def _get_functions(self) -> list[dict]:
        """Load functions from configuration."""
        try:
            functions = self.entry.options.get(CONF_FUNCTIONS, [])
            return [self._validate_function(f) for f in functions]
        except (vol.Invalid, ValueError) as err:
            raise FunctionLoadFailed(f"Invalid function configuration: {err}") from err

    def _validate_function(self, func_config: dict) -> dict:
        """Validate and format function configuration."""
        function_executor = get_function_executor(func_config["type"])
        return function_executor.validate_config(func_config)

    def _format_parameters(self, parameters: dict) -> dict:
        """Convert parameters to Gemini format."""
        # Simplified parameter formatting compared to original
        return {
            "type": "OBJECT",
            "properties": {
                k: {"type": v["type"].upper(), "description": v.get("description", "")}
                for k, v in parameters["properties"].items()
            },
            "required": parameters.get("required", [])
        }

    async def _execute_function_call(
        self,
        function_call: dict,
        user_input: conversation.ConversationInput,
        messages: list[dict],
        conversation_id: str,
        n_requests: int
    ) -> dict:
        """Execute a function call and return result."""
        func_name = function_call["name"]
        try:
            args = json.loads(function_call["arguments"])
        except json.JSONDecodeError as err:
            raise ParseArgumentsFailed(function_call["arguments"]) from err

        # Find and execute function
        func = next(
            (f for f in self._get_functions() if f["name"] == func_name),
            None
        )
        if not func:
            raise FunctionNotFound(func_name)

        executor = get_function_executor(func["type"])
        result = await executor.execute(self.hass, func, args, user_input)

        return {
            "role": "function",
            "name": func_name,
            "content": json.dumps(result),
        }

    def _create_system_message(self) -> dict:
        """Create system message from prompt template."""
        prompt = self.entry.options.get(CONF_PROMPT, "")
        return {
            "role": "system",
            "content": prompt.format(ha_name=self.hass.config.location_name)
        }

    def _create_generation_config(self) -> dict:
        """Create generation configuration."""
        return {
            "temperature": self.entry.options.get(CONF_TEMPERATURE, 0.5),
            "top_p": self.entry.options.get(CONF_TOP_P, 0.95),
            "top_k": self.entry.options.get(CONF_TOP_K, 40),
            "max_output_tokens": self.entry.options.get(CONF_MAX_TOKENS, 2048)
        }

    def _parse_response(self, response) -> dict:
        """Parse Gemini response to message format."""
        if not response.candidates:
            raise HomeAssistantError("No response candidates found")

        parts = response.candidates[0].content.parts
        return {
            "role": "assistant",
            "content": " ".join(p.text for p in parts if p.text),
            "function_call": self._extract_function_call(parts)
        }

    def _extract_function_call(self, parts: list[Part]) -> dict | None:
        """Extract function call from response parts."""
        for part in parts:
            if part.function_call:
                return {
                    "name": part.function_call.name,
                    "arguments": json.dumps(part.function_call.args)
                }
        return None

    def _create_response(self, message: dict) -> intent.IntentResponse:
        """Create Home Assistant response."""
        response = intent.IntentResponse()
        response.async_set_speech(message["content"])
        return response
