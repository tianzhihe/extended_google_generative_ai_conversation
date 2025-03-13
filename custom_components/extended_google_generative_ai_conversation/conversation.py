from __future__ import annotations

"""
Google Generative AI Conversation integration
with support for function definitions that specify OpenAPI-like schemas.
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

# Max number of iterations for repeated tool calls
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
    """Adapt OpenAPI-like schema to something Gemini will accept."""
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
            # Filter out unsupported formats
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
        # Gemini doesn't handle open-ended objects well. Fallback to a JSON string.
        result["properties"] = {"json": {"type": "STRING"}}
        result["required"] = []
    return cast(Schema, result)

def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> Tool:
    """Format a Home Assistant llm.Tool into a Gemini Tool."""
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
    """Recursively handle escaped sequences in strings."""
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
    """Create a Content object from tool results."""
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
    """
    Convert Home Assistant conversation content into
    a format suitable for Google Generative AI.
    """
    if content.role != "assistant" or not content.tool_calls:
        role = "model" if content.role == "assistant" else content.role
        return Content(
            role=role,
            parts=[
                Part.from_text(text=content.content if content.content else ""),
            ],
        )

    # We have assistant content with tool calls
    assert type(content) is conversation.AssistantContent
    parts: list[Part] = []
    if content.content:
        parts.append(Part.from_text(text=content.content))

    for tool_call in content.tool_calls:
        parts.append(
            Part.from_function_call(
                name=tool_call.tool_name,
                args=_escape_decode(tool_call.tool_args),
            )
        )

    return Content(role="model", parts=parts)

class GoogleGenerativeAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Google Generative AI conversation agent."""

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
        """Return the list of supported languages."""
        return MATCH_ALL

    def get_functions(self):
        """Load function definitions from YAML or defaults, for example usage."""
        try:
            function_str = self.entry.options.get(CONF_FUNCTIONS)
            result = yaml.safe_load(function_str) if function_str else DEFAULT_CONF_FUNCTIONS
            if result:
                for setting in result:
                    func_exec = get_function_executor(setting["function"]["type"])
                    # Convert the function block so we can run it later
                    setting["function"] = func_exec.to_arguments(setting["function"])
            return result
        except (InvalidFunction, FunctionNotFound) as err:
            raise err
        except Exception:
            raise FunctionLoadFailed()

    async def async_added_to_hass(self) -> None:
        """Run when entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity is removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process user input."""
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
        """Handle the conversation with possible function calls."""
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

        # Convert HA llm.Tools to Google Tools
        tools: list[Tool | Callable[..., Any]] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        # Now read your YAML-based function definitions and convert them into Tools
        user_functions = self.get_functions() or []
        for fn_def in user_functions:
            spec = fn_def["spec"]
            # If 'parameters' is present, translate it using _format_schema
            schema_obj = None
            if "parameters" in spec:
                schema_obj = _format_schema(spec["parameters"])

            fn_tool = Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name=spec["name"],
                        description=spec.get("description", ""),
                        parameters=schema_obj,
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
        if not prompt_content.content:
            raise HomeAssistantError("Invalid prompt content")

        prompt = prompt_content.content

        messages: list[Content] = []
        tool_results: list[conversation.ToolResultContent] = []

        # Convert prior conversation content into Gemini Content
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

        # For older models that don't allow system_instruction, prepend prompt
        if not supports_system_instruction:
            messages = [
                Content(role="user", parts=[Part.from_text(text=prompt)]),
                Content(role="model", parts=[Part.from_text(text="Ok")]),
                *messages,
            ]

        chat_request: str | Content = user_input.text
        resp_content = ""
        for _ in range(MAX_TOOL_ITERATIONS):
            resp_content, found_tool_calls = await self._query_model(
                messages=messages,
                message_input=chat_request,
                model=model_name,
                config=generate_config,
            )

            if resp_content is None:
                raise HomeAssistantError("Prompt blocked or empty response from Generative AI.")

            # If no tool calls, we're done
            if not found_tool_calls:
                await chat_log.async_add_assistant_content(
                    conversation.AssistantContent(
                        agent_id=user_input.agent_id,
                        content=resp_content,
                    )
                )
                break

            # Otherwise, record the assistant's text + the tool calls
            combined_parts = await chat_log.async_add_assistant_content(
                conversation.AssistantContent(
                    agent_id=user_input.agent_id,
                    content=resp_content,
                    tool_calls=found_tool_calls,
                )
            )

            # Execute each tool call and gather results
            tool_result_contents: list[conversation.ToolResultContent] = []
            for call in found_tool_calls:
                result = await self._execute_function_call(call)
                tool_result_contents.append(
                    conversation.ToolResultContent(
                        tool_name=call.tool_name,
                        tool_result=result
                    )
                )

            # Pass function-call outputs back into the model
            chat_request = _create_google_tool_response_content(tool_result_contents)

        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(resp_content)
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
        """Send data to Google Generative AI and parse the result."""
        chat = self._genai_client.aio.chats.create(
            model=model,
            history=messages,
            config=config,
        )
        try:
            chat_response = await chat.send_message(message=message_input)
        except (APIError, ValueError) as err:
            LOGGER.error("Error sending message to Google Generative AI: %s", err)
            raise HomeAssistantError(
                f"Problem talking to Google Generative AI: {err}"
            ) from err

        # Safety block or empty response
        if chat_response.prompt_feedback:
            return None, None

        response_parts = chat_response.candidates[0].content.parts
        if not response_parts:
            return None, None

        assistant_text = " ".join(part.text.strip() for part in response_parts if part.text)

        # Find any function calls
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
        """
        Locate and execute a user-defined function by name.
        Handle composite or native calls.
        """
        all_funcs = self.get_functions() or []
        fn_name = tool_call.tool_name

        for fn_def in all_funcs:
            if fn_def["spec"]["name"] == fn_name:
                # For example, either native or composite
                func_exec = get_function_executor(fn_def["function"]["type"])
                try:
                    # Convert JSON string to dictionary
                    # If you have logic for partial JSON or other format,
                    # adapt as needed
                    from json import loads as json_load
                    args = json_load(tool_call.tool_args)
                except ValueError as err:
                    raise ParseArgumentsFailed(tool_call.tool_args) from err

                # Execute via function executor
                result = await func_exec.execute(
                    self.hass,
                    fn_def["function"],
                    args,
                    None,  # user_input can be passed if needed
                    None,  # exposed_entities can be used if needed
                )
                return str(result)

        # If no match found
        raise FunctionNotFound(fn_name)

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle config entry updates."""
        await hass.config_entries.async_reload(entry.entry_id)
