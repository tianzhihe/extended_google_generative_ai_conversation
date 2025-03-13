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
            parts=[
                Part.from_text(text=content.content if content.content else ""),
            ],
        )

    assert type(content) is conversation.AssistantContent
    parts: list[Part] = []
    if content.content:
        parts.append(Part.from_text(text=content.content))

    if content.tool_calls:
        parts.extend(
            [
                Part.from_function_call(
                    name=tool_call.tool_name,
                    args=_escape_decode(tool_call.tool_args),
                )
                for tool_call in content.tool_calls
            ]
        )

    return Content(role="model", parts=parts)

class GoogleGenerativeAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Google Generative AI conversation agent with refactored function-calling structure."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._genai_client = entry.runtime_data
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Google",
            model="Generative AI",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    def get_functions(self):
        """Load user-defined or default functions."""
        try:
            function_str = self.entry.options.get(CONF_FUNCTIONS)
            result = yaml.safe_load(function_str) if function_str else DEFAULT_CONF_FUNCTIONS
            if result:
                for setting in result:
                    func_exec = get_function_executor(setting["function"]["type"])
                    setting["function"] = func_exec.to_arguments(setting["function"])
            return result
        except (InvalidFunction, FunctionNotFound) as err:
            raise err
        except Exception as err:
            raise FunctionLoadFailed() from err

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence from the user."""
        with (
            chat_session.async_get_chat_session(
                self.hass, user_input.conversation_id
            ) as session,
            conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            return await self._async_handle_message(user_input, chat_log)

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle the AI request using a loop that checks for function calls."""
        options = self.entry.options
        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools: list[Tool | Callable[..., Any]] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        # Add user-defined functions as "tools"
        user_functions = self.get_functions() or []
        if user_functions:
            # Each item in user_functions is { "spec": {...}, "function": {...} } from load
            # but for the Google client, we only pass the "spec" as a Tool
            # We'll assume "spec" has the structure for a single function
            # or we can generate a Tool from it if needed
            for fn_def in user_functions:
                # Some direct mapping from "spec" to Tool is possible
                if "name" in fn_def["spec"]:
                    # Minimal usage: create a Tool with the same name
                    # For advanced usage, you'd adapt parameters
                    fn_tool = Tool(
                        function_declarations=[
                            FunctionDeclaration(
                                name=fn_def["spec"]["name"],
                                description=fn_def["spec"].get("description", ""),
                                parameters=None,
                            )
                        ]
                    )
                    if tools is None:
                        tools = []
                    tools.append(fn_tool)

        model_name = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        supports_system_instruction = (
            "gemini-1.0" not in model_name and "gemini-pro" not in model_name
        )

        prompt_content = cast(
            conversation.SystemContent,
            chat_log.content[0],
        )
        if prompt_content.content:
            prompt = prompt_content.content
        else:
            raise HomeAssistantError("Invalid prompt content")

        messages: list[Content] = []
        tool_results: list[conversation.ToolResultContent] = []

        # Convert chat_log content except the final user request
        for chat_content in chat_log.content[1:-1]:
            if chat_content.role == "tool_result":
                tool_results.append(cast(conversation.ToolResultContent, chat_content))
                continue
            if tool_results:
                messages.append(_create_google_tool_response_content(tool_results))
                tool_results.clear()
            messages.append(_convert_content(chat_content))

        if tool_results:
            messages.append(_create_google_tool_response_content(tool_results))

        # Configuration for Generative AI calls
        generate_config = GenerateContentConfig(
            temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            top_k=options.get(CONF_TOP_K, RECOMMENDED_TOP_K),
            top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            max_output_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            safety_settings=[
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=options.get(CONF_HATE_BLOCK_THRESHOLD, RECOMMENDED_HARM_BLOCK_THRESHOLD),
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=options.get(CONF_HARASSMENT_BLOCK_THRESHOLD, RECOMMENDED_HARM_BLOCK_THRESHOLD),
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=options.get(CONF_DANGEROUS_BLOCK_THRESHOLD, RECOMMENDED_HARM_BLOCK_THRESHOLD),
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=options.get(CONF_SEXUAL_BLOCK_THRESHOLD, RECOMMENDED_HARM_BLOCK_THRESHOLD),
                ),
            ],
            tools=tools or None,
            system_instruction=prompt if supports_system_instruction else None,
            automatic_function_calling=AutomaticFunctionCallingConfig(
                disable=True,
                maximum_remote_calls=None
            ),
        )

        # For models that don't allow system_instruction, prepend the prompt as a user message
        if not supports_system_instruction:
            messages = [
                Content(role="user", parts=[Part.from_text(text=prompt)]),
                Content(role="model", parts=[Part.from_text(text="Ok")]),
                *messages,
            ]

        # Begin conversation with new message from user
        chat_request: str | Content = user_input.text

        for _ in range(MAX_TOOL_ITERATIONS):
            # Query the model with user text or tool responses
            resp_content, found_tool_calls = await self._query_model(
                messages=messages,
                message_input=chat_request,
                model=model_name,
                config=generate_config,
            )

            # If a safety block or other error occurred, raise
            if resp_content is None:
                raise HomeAssistantError("Content blocked by safety settings or error occurred.")

            # If no tool calls returned, break out
            if not found_tool_calls:
                # Add final assistant content to chat log
                await chat_log.async_add_assistant_message(resp_content)
                break

            # Otherwise, handle tool calls
            # We store assistant text plus calls in the chat log, gather tool results, and send them back
            combined_response = []
            for item in await chat_log.async_add_assistant_content(
                conversation.AssistantContent(
                    agent_id=user_input.agent_id,
                    content=resp_content,
                    tool_calls=[
                        llm.ToolInput(tool_name=tool_call.tool_name, tool_args=_escape_decode(tool_call.tool_args))
                        for tool_call in found_tool_calls
                    ],
                )
            ):
                combined_response.append(item)

            # Execute each found tool call, gather results
            tool_result_contents: list[conversation.ToolResultContent] = []
            for call in found_tool_calls:
                result = await self._execute_function_call(call)
                tool_result_contents.append(
                    conversation.ToolResultContent(
                        tool_name=call.tool_name,
                        tool_result=result
                    )
                )

            # Prepare next iteration's message input from the function call results
            chat_request = _create_google_tool_response_content(tool_result_contents)

        # Build final HA conversation response
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(resp_content or "")
        return conversation.ConversationResult(
            response=response, conversation_id=chat_log.conversation_id
        )

    async def _query_model(
        self,
        messages: list[Content],
        message_input: str | Content,
        model: str,
        config: GenerateContentConfig,
    ) -> tuple[str | None, list[llm.ToolInput] | None]:
        """
        Send a single query to the Google Generative AI model.
        Returns a tuple of (assistant_text, list_of_tool_calls).
        If the response is blocked or error occurs, returns (None, None).
        """
        # Create chat object
        chat = self._genai_client.aio.chats.create(
            model=model,
            history=messages,
            config=config,
        )

        try:
            chat_response = await chat.send_message(message=message_input)
        except (APIError, ValueError) as err:
            LOGGER.error("Error sending message: %s", err)
            raise HomeAssistantError(
                f"Sorry, I had a problem talking to Google Generative AI: {err}"
            ) from err

        if chat_response.prompt_feedback:
            # If the model has blocked the prompt for any reason
            return None, None

        response_parts = chat_response.candidates[0].content.parts
        if not response_parts:
            return None, None

        assistant_text = " ".join(part.text.strip() for part in response_parts if part.text)
        # Identify any function/tool calls
        tool_calls: list[llm.ToolInput] = []
        for part in response_parts:
            if part.function_call:
                tool_call = part.function_call
                tool_calls.append(
                    llm.ToolInput(
                        tool_name=tool_call.name,
                        tool_args=_escape_decode(tool_call.args),
                    )
                )

        return assistant_text, tool_calls or None

    async def _execute_function_call(self, tool_call: llm.ToolInput) -> str:
        """Locate and execute a user-defined function."""
        # Convert the JSON string to an object, then run the matching function
        user_functions = self.get_functions() or []
        fn_name = tool_call.tool_name
        for fn_def in user_functions:
            if fn_def["spec"]["name"] == fn_name:
                func_exec = get_function_executor(fn_def["function"]["type"])
                try:
                    args = conversation.json_loads(tool_call.tool_args)  # or `json.loads`
                except ValueError as err:
                    raise ParseArgumentsFailed(tool_call.tool_args) from err
                result = await func_exec.execute(
                    self.hass,
                    fn_def["function"],
                    args,
                    None,  # user_input not always needed
                    None,  # exposed_entities not always needed
                )
                return str(result)
        raise FunctionNotFound(fn_name)

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        await hass.config_entries.async_reload(entry.entry_id)

