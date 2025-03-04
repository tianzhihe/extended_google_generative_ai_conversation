"""Google Generative AI Conversation integration with function calling."""

from __future__ import annotations

import codecs
from collections.abc import Callable
from typing import Any, Literal, cast

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
from homeassistant.helpers import chat_session, device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

# ---------------------------------------------------------------------
# IMPORT your constants, including function-calling toggles:
# (Adjust the import to match your actual project structure.)
# ---------------------------------------------------------------------
from .const import (
    DOMAIN,
    LOGGER,
    EVENT_CONVERSATION_FINISHED,
    CONF_PROMPT,
    CONF_USE_FUNCTIONS,
    CONF_FUNCTIONS,
    DEFAULT_FUNCTIONS,
    DEFAULT_USE_FUNCTIONS,
    CONF_MAX_TOKENS,
    RECOMMENDED_MAX_TOKENS,
    CONF_TEMPERATURE,
    RECOMMENDED_TEMPERATURE,
    CONF_TOP_K,
    RECOMMENDED_TOP_K,
    CONF_TOP_P,
    RECOMMENDED_TOP_P,
    CONF_CHAT_MODEL,
    RECOMMENDED_CHAT_MODEL,
    CONF_HARASSMENT_BLOCK_THRESHOLD,
    CONF_HATE_BLOCK_THRESHOLD,
    CONF_DANGEROUS_BLOCK_THRESHOLD,
    CONF_SEXUAL_BLOCK_THRESHOLD,
    RECOMMENDED_HARM_BLOCK_THRESHOLD,
    TIMEOUT_MILLIS,
    # etc.
)

MAX_TOOL_ITERATIONS = 10  # How many times we allow back-and-forth calls

# ---------------------------------------------------------------------
# RE-USE THE helpers.py logic or embed it here directly.
# The snippet below shows a simplified example referencing your
# actual helpers.py classes and constants.
# ---------------------------------------------------------------------
import os
import re
import sqlite3
import time
import yaml
import voluptuous as vol

from homeassistant.helpers.template import Template
from homeassistant.exceptions import ServiceNotFound

from .helpers import (
    FUNCTION_EXECUTORS,
    get_function_executor,
    NativeFunctionExecutor,
    HomeAssistantError,
    EntityNotFound,
    EntityNotExposed,
    FunctionNotFound,
    InvalidFunction,
    # etc...
)

# You might want a helper function to load a user’s function definitions from
# your config entry or from an OptionsFlow. The example below is simplistic:
def load_functions_from_config(
    hass: HomeAssistant, entry: ConfigEntry
) -> list[dict[str, Any]]:
    """Load function definitions (schemas) from the user's config entry."""
    raw = entry.options.get(CONF_FUNCTIONS, DEFAULT_FUNCTIONS)
    try:
        data = yaml.safe_load(raw) if isinstance(raw, str) else []
        if not data:
            return []
        return data
    except (yaml.YAMLError, TypeError):
        LOGGER.warning("Failed to parse function definitions from config.")
        return []


# ---------------------------------------------------------------------
# The existing code from the Google integration is below, updated to:
#  1) Parse the user’s function definitions
#  2) Build Tools for them
#  3) If the AI calls a function, run it through the appropriate executor
# ---------------------------------------------------------------------
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
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")

def _format_schema(schema: dict[str, Any]) -> Schema:
    """Adapt JSON schemas to the Google generative AI schema."""
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
            # Only keep recognized formats...
            if schema.get("type") == "string" and val not in ("enum", "date-time"):
                continue
            if schema.get("type") == "number" and val not in ("float", "double"):
                continue
            if schema.get("type") == "integer" and val not in ("int32", "int64"):
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

def _escape_decode(value: Any) -> Any:
    """Recursively call codecs.escape_decode on all strings."""
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
    """Convert HA's tool results into a Google Content containing function responses."""
    return Content(
        parts=[
            Part.from_function_response(name=tool_result.tool_name, response=tool_result.tool_result)
            for tool_result in content
        ]
    )

def _convert_content(
    content: conversation.UserContent
    | conversation.AssistantContent
    | conversation.SystemContent,
) -> Content:
    """Convert a Home Assistant content to Google Generative AI content."""
    if content.role != "assistant" or not content.tool_calls:  # type: ignore
        role = "model" if content.role == "assistant" else content.role
        return Content(
            role=role,
            parts=[Part.from_text(text=content.content if content.content else "")],
        )

    # If we have Assistant content with tool calls, we separate them
    parts: list[Part] = []
    if content.content:
        parts.append(Part.from_text(text=content.content))

    if content.tool_calls:
        for tool_call in content.tool_calls:
            tool_args = _escape_decode(tool_call.tool_args)
            parts.append(Part.from_function_call(name=tool_call.tool_name, args=tool_args))

    return Content(role="model", parts=parts)


class GoogleGenerativeAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Google Generative AI conversation agent with function-calling integration."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the entity."""
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
        return MATCH_ALL

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
        """When entity is removed."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a user query."""
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
        """Send a request to the Google Generative AI model and handle tool/function calls."""

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

        # --- 1. Build Tools from the user’s function definitions (similar to openai) ---
        tools: list[Tool] = []
        use_functions = options.get(CONF_USE_FUNCTIONS, DEFAULT_USE_FUNCTIONS)
        if use_functions:
            user_functions = load_functions_from_config(self.hass, self.entry)
            for fdef in user_functions:
                spec = fdef.get("spec", {})
                # Convert the JSON schema to Google's format:
                parameters = None
                if "parameters" in spec:
                    # Use voluptuous_openapi.convert() to produce a dict
                    # Then use _format_schema to adapt it for Google
                    parameters = _format_schema(
                        convert(spec["parameters"])
                    )
                # Create a Google "Tool" with the function name and any schema
                tool = Tool(
                    function_declarations=[
                        FunctionDeclaration(
                            name=spec.get("name", "unnamed_tool"),
                            description=spec.get("description", ""),
                            parameters=parameters,
                        )
                    ]
                )
                tools.append(tool)
        else:
            # fallback: user might also have pipelines from llm_api
            if chat_log.llm_api:
                tools = [
                    Tool(
                        function_declarations=[
                            FunctionDeclaration(
                                name=t.name,
                                description=t.description,
                                parameters=(
                                    _format_schema(
                                        convert(t.parameters, custom_serializer=chat_log.llm_api.custom_serializer)
                                    )
                                    if t.parameters and t.parameters.schema
                                    else None
                                ),
                            )
                        ]
                    )
                    for t in chat_log.llm_api.tools
                ]

        # --- 2. Prepare the user’s conversation messages for Google ---
        model_name = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        supports_system_instruction = ("gemini-1.0" not in model_name and "gemini-pro" not in model_name)

        prompt_content = cast(conversation.SystemContent, chat_log.content[0])
        if not prompt_content.content:
            raise HomeAssistantError("Invalid or empty system prompt content.")

        # Flatten conversation into Google Content objects
        messages: list[Content] = []
        tool_results: list[conversation.ToolResultContent] = []
        # We skip the final item (user message) because we supply it below:
        for chat_content in chat_log.content[1:-1]:
            if chat_content.role == "tool_result":
                tool_results.append(cast(conversation.ToolResultContent, chat_content))
                continue
            # If we have pending tool results, flush them as a single "function response"
            if tool_results:
                messages.append(_create_google_tool_response_content(tool_results))
                tool_results.clear()

            messages.append(
                _convert_content(
                    cast(
                        conversation.UserContent
                        | conversation.SystemContent
                        | conversation.AssistantContent,
                        chat_content,
                    )
                )
            )
        # If leftover tool results:
        if tool_results:
            messages.append(_create_google_tool_response_content(tool_results))

        # The final user message in chat_log.content[-1] is presumably user_input.text
        # We do not convert it into a separate Content because the send_message method
        # below uses chat_request = user_input.text.

        # --- 3. Build the config to call the model ---
        generateContentConfig = GenerateContentConfig(
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
            system_instruction=(prompt_content.content if supports_system_instruction else None),
            # Google “automatic_function_calling” can be toggled if you want the LLM
            # to auto-call functions. We'll keep it disabled and parse manually:
            automatic_function_calling=AutomaticFunctionCallingConfig(
                disable=True,
                maximum_remote_calls=None,
            ),
        )

        if not supports_system_instruction:
            # Prepend the system prompt as user content
            messages = [
                Content(role="user", parts=[Part.from_text(text=prompt_content.content)]),
                Content(role="model", parts=[Part.from_text(text="Ok")]),
                *messages,
            ]

        # Create a chat session with the model.
        chat = self._genai_client.aio.chats.create(
            model=model_name,
            history=messages,
            config=generateContentConfig,
        )
        chat_request: str | Content = user_input.text  # The final user message

        # --- 4. Repeatedly send messages to handle function calls & tool calls ---
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                chat_response = await chat.send_message(message=chat_request)
                if chat_response.prompt_feedback:
                    raise HomeAssistantError(
                        f"Your message was blocked for: {chat_response.prompt_feedback.block_reason_message}"
                    )
            except (APIError, ValueError) as err:
                LOGGER.error("Error in chat.send_message: %s", err)
                raise HomeAssistantError(f"Problem talking to Generative AI: {err}")

            response_parts = chat_response.candidates[0].content.parts
            if not response_parts:
                raise HomeAssistantError("Empty response from Generative AI.")

            # Collect the text portion for user
            content = " ".join([part.text.strip() for part in response_parts if part.text])

            # Gather any function calls from the response
            tool_calls = []
            for part in response_parts:
                if part.function_call:
                    tool_calls.append(
                        llm.ToolInput(
                            tool_name=part.function_call.name,
                            tool_args=_escape_decode(part.function_call.args),
                        )
                    )

            # Register the assistant content & see if any tool calls result in function calls
            chat_request = _create_google_tool_response_content(
                [
                    tool_response
                    async for tool_response in chat_log.async_add_assistant_content(
                        conversation.AssistantContent(
                            agent_id=user_input.agent_id,
                            content=content,
                            tool_calls=tool_calls or None,
                        )
                    )
                ]
            )

            # 5. If no calls, we're done
            if not tool_calls:
                break

            # 6. If user has enabled function usage, attempt to run them
            if not use_functions:
                # If user didn't enable function usage, the LLM won't get real results
                # We'll just feed an empty "function result" back
                continue

            # Load user function definitions again:
            user_functions = load_functions_from_config(self.hass, self.entry)
            # or combine with LLM API’s tools if you want

            # For each function call, match it to our known function, run it, produce a “tool_result”
            tool_results_list: list[conversation.ToolResultContent] = []
            for call in tool_calls:
                tool_name = call.tool_name
                found_spec = next(
                    (f for f in user_functions if f["spec"]["name"] == tool_name),
                    None
                )
                if not found_spec:
                    # Unknown function
                    tool_results_list.append(
                        conversation.ToolResultContent(
                            tool_name=tool_name,
                            tool_result={"error": f"Function '{tool_name}' not found."},
                        )
                    )
                    continue

                # We have a known function. Now see what "type" it is & run it
                function_type = found_spec["function"].get("type")
                try:
                    executor = get_function_executor(function_type)
                except FunctionNotFound:
                    tool_results_list.append(
                        conversation.ToolResultContent(
                            tool_name=tool_name,
                            tool_result={"error": f"Function type '{function_type}' missing."},
                        )
                    )
                    continue

                # Parse the call arguments as JSON
                import json
                try:
                    args = json.loads(call.tool_args)
                except json.JSONDecodeError:
                    args = {}

                # Execute it
                exposed_entities = chat_log.exposed_entities  # or however you store them
                try:
                    result = await executor.execute(
                        self.hass, found_spec["function"], args, user_input, exposed_entities
                    )
                    tool_results_list.append(
                        conversation.ToolResultContent(
                            tool_name=tool_name,
                            tool_result=result,
                        )
                    )
                except HomeAssistantError as e:
                    tool_results_list.append(
                        conversation.ToolResultContent(
                            tool_name=tool_name,
                            tool_result={"error": str(e)},
                        )
                    )

            # After running all calls, we feed them back to the model as function responses
            chat_request = _create_google_tool_response_content(tool_results_list)
            # The conversation loop will continue

        # 7. Construct final conversation response to user
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(content)
        return conversation.ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update (reload if needed)."""
        await hass.config_entries.async_reload(entry.entry_id)
