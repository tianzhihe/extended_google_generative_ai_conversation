"""Conversation support for the Extended Google Generative AI Conversation integration."""

from __future__ import annotations

import codecs
from collections.abc import Callable
from typing import Any, Literal, cast

# imports types and classes from google.genai for AI interactions.
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
)

# Max number of back and forth with the LLM to generate a response
# Sets a limit on how many times the conversation agent can repeatedly call tools to generate a single response.
MAX_TOOL_ITERATIONS = 10


# Creates an instance of GoogleGenerativeAIConversationEntity with the provided config entry, 
# and registers it (via async_add_entities) so Home Assistant recognizes it as a conversation provider.
async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = GoogleGenerativeAIConversationEntity(config_entry)
    async_add_entities([agent])

# Contains a set of keys recognized by the Gemini API for schema definitions. 
# Keys outside this list are filtered out when adapting schemas for the AI model.
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

# Goes through each character, inserting an underscore before uppercase letters, then strips leading underscores at the end.
def _camel_to_snake(name: str) -> str:
    """Convert camel case to snake case."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


def _format_schema(schema: dict[str, Any]) -> Schema:
    """Format the schema to be compatible with Gemini API."""
    if subschemas := schema.get("allOf"):
        for subschema in subschemas:  # Gemini API does not support allOf keys
            if "type" in subschema:  # Fallback to first subschema with 'type' field
                return _format_schema(subschema)
        return _format_schema(
            subschemas[0]
        )  # Or, if not found, to any of the subschemas

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
            # Gemini API does not support all formats, see: https://ai.google.dev/api/caching#Schema
            # formats that are not supported are ignored
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
        # enum is only allowed for STRING type. This is safe as long as the schema
        # contains vol.Coerce for the respective type, for example:
        # vol.All(vol.Coerce(int), vol.In([1, 2, 3]))
        result["type"] = "STRING"
        result["enum"] = [str(item) for item in result["enum"]]

    if result.get("type") == "OBJECT" and not result.get("properties"):
        # An object with undefined properties is not supported by Gemini API.
        # Fallback to JSON string. This will probably fail for most tools that want it
        # (but we don't have a better fallback strategy so far.)
        result["properties"] = {"json": {"type": "STRING"}}
        result["required"] = []
    return cast(Schema, result) # Returns a Schema object in a format that the Gemini API can handle for function/tool parameters.

# Prepares a Home Assistant llm.Tool for use with the Gemini API.
def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> Tool:
    """Format tool specification."""

    # Converts the tool’s parameter schema to a Gemini-compatible schema using _format_schema, if a schema is present.
    if tool.parameters.schema:
        parameters = _format_schema(
            convert(tool.parameters, custom_serializer=custom_serializer)
        )
    else:
        parameters = None

    # Builds a Tool object with a single FunctionDeclaration. 
    # The AI service can then call the tool using these declared parameters.
    return Tool(
        function_declarations=[
            FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=parameters,
            )
        ]
    )

# Safely handles certain escaped string characters in nested data structures (strings, lists, dictionaries) by decoding them.
def _escape_decode(value: Any) -> Any:
    """Recursively call codecs.escape_decode on all values."""
    if isinstance(value, str):
        return codecs.escape_decode(bytes(value, "utf-8"))[0].decode("utf-8")  # type: ignore[attr-defined]
    if isinstance(value, list):
        return [_escape_decode(item) for item in value]
    if isinstance(value, dict):
        return {k: _escape_decode(v) for k, v in value.items()}
    return value

# Constructs a Content object that represents the AI’s use of a “tool result” (i.e., a response or outcome from a tool/function call).
def _create_google_tool_response_content(
    content: list[conversation.ToolResultContent],
) -> Content:
    return Content(
        parts=[
            Part.from_function_response(
                name=tool_result.tool_name, response=tool_result.tool_result # map each tool result into a Part, storing the name and response in Gemini’s recognized format.
            )
            for tool_result in content # Wraps these parts in a single Content object, which the Generative AI model can interpret as a function response.
        ]
    )

# Translates Home Assistant conversation.*Content objects into Gemini-compatible Content.
def _convert_content(
    content: conversation.UserContent
    | conversation.AssistantContent
    | conversation.SystemContent,
) -> Content:
    """Convert HA content to Google content."""
    
    # If the content isn’t from the “assistant” role or has no tool calls, it simply puts the text into a Part and sets the Gemini role accordingly.
    if content.role != "assistant" or not content.tool_calls:  # type: ignore[union-attr]
        role = "model" if content.role == "assistant" else content.role
        return Content(
            role=role,
            parts=[
                Part.from_text(text=content.content if content.content else ""),
            ],
        )

    # If it’s truly assistant content with tool calls:
    # 
    # Handle the Assistant content with tool calls.
    assert type(content) is conversation.AssistantContent
    parts: list[Part] = []

    # Adds a Part for any text from the assistant (content.content).
    if content.content:
        parts.append(Part.from_text(text=content.content))

    # Adds a Part for each tool call using Part.from_function_call.
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

    # Marks the role as "model", which is how Gemini recognizes AI/assistant output.
    return Content(role="model", parts=parts)

#  This class acts as a Home Assistant conversation “entity,” managing user interactions with the Generative AI model.
class GoogleGenerativeAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Google Generative AI conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent."""

        # Stores the config_entry and retrieves the AI client from its runtime data.
        self.entry = entry
        self._genai_client = entry.runtime_data

        # Sets a unique ID for device tracking within Home Assistant.
        self._attr_unique_id = entry.entry_id

        # Creates device_info that identifies this agent as a service from Google.
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Google",
            model="Generative AI",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

        # If the user has set an LLM Control API (CONF_LLM_HASS_API), 
        # then this entity declares the CONTROL feature, which helps indicate it can run advanced conversation tasks.
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    # Declares that this entity supports all languages, so Home Assistant won’t limit language usage.
    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    #  Called when Home Assistant has added this entity to the system.
    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        # ensure conversation handling is updated if needed.
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        # Registers this entity as the conversation agent in the conversation component.
        conversation.async_set_agent(self.hass, self.entry, self)
        # Sets up a listener for entry updates, so if the user changes config options, the entity can reload them.
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    # Ensures this agent is de-registered from Home Assistant’s conversation system 
    # when the entity is removed (for example, if the integration is uninstalled or disabled).
    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    #  Main entry point when Home Assistant wants to process user input through this AI agent.
    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        with (
            # Acquires a chat session associated with the conversation_id, creating one if needed.
            chat_session.async_get_chat_session(
                self.hass, user_input.conversation_id
            ) as session,
            # Retrieves or creates a chat_log to record the conversation.
            conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            # Delegates to self._async_handle_message for the actual AI request.
            return await self._async_handle_message(user_input, chat_log)

    # Performs the heavy lifting of generating an AI response and possibly calling tools/functions.
    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        options = self.entry.options

        try:
            # Informs the conversation log about the chosen LLM settings (API, prompt, etc.)
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # logs the intent and ensures that we included the tools 
        # (if for some reason tools were optional, you might programmatically ensure they’re added when these keywords appear)
        user_text = user_input.text.lower()
        if "automation" in user_text and "add" in user_text:
            LOGGER.debug("User is requesting an automation; tool call likely needed.")
        if "energy" in user_text:
            LOGGER.debug("User is requesting energy stats; tool call likely needed.")

        # If an LLM API is set and it has available tools, 
        # transforms them via _format_tool into the Gemini-compatible Tool structure.
        tools: list[Tool | Callable[..., Any]] | None = None
        
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]
        
        # Append the new Home Assistant tools
        tools += [
            Tool(function_declarations=[
                FunctionDeclaration(
                    name="add_automation",
                    description="Add a new Home Assistant automation from a YAML configuration.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "automation_config": {
                                "type": "string",
                                "description": "YAML for the new automation"
                            }
                        },
                        "required": ["automation_config"]
                    }
                ),
                FunctionDeclaration(
                    name="get_energy",
                    description="Get current energy usage statistics from Home Assistant’s energy manager.",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                )
            ])
        ]


        # Chooses which model to use (user-chosen or recommended).
        model_name = self.entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        # Gemini 1.0 doesn't support system_instruction while 1.5 does.
        # Assume future versions will support it (if not, the request fails with a
        # clear message at which point we can fix).
        supports_system_instruction = (
            "gemini-1.0" not in model_name and "gemini-pro" not in model_name
        )

        # The first element in chat_log.content is assumed to be the “system” prompt that orients the conversation.
        prompt_content = cast(
            conversation.SystemContent,
            chat_log.content[0],
        )

        if prompt_content.content:
            prompt = prompt_content.content
        else:
            raise HomeAssistantError("Invalid prompt content")

        messages: list[Content] = []

        # Google groups tool results, we do not. Group them before sending.
        tool_results: list[conversation.ToolResultContent] = []

        # Non-tool results are converted via _convert_content to the Gemini Content format.
        for chat_content in chat_log.content[1:-1]:
            if chat_content.role == "tool_result":
                # mypy doesn't like picking a type based on checking shared property 'role'
                tool_results.append(cast(conversation.ToolResultContent, chat_content))
                continue

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

        if tool_results:
            messages.append(_create_google_tool_response_content(tool_results))
        generateContentConfig = GenerateContentConfig(
            temperature=self.entry.options.get(
                CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
            ),
            # Collects user or default options for AI generation, such as temperature, token limits, etc.
            top_k=self.entry.options.get(CONF_TOP_K, RECOMMENDED_TOP_K),
            top_p=self.entry.options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            max_output_tokens=self.entry.options.get(
                CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
            ),
            # Lists safety settings (Hate speech, harassment, etc.) based on user-configured thresholds.
            safety_settings=[
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=self.entry.options.get(
                        CONF_HATE_BLOCK_THRESHOLD, RECOMMENDED_HARM_BLOCK_THRESHOLD
                    ),
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=self.entry.options.get(
                        CONF_HARASSMENT_BLOCK_THRESHOLD,
                        RECOMMENDED_HARM_BLOCK_THRESHOLD,
                    ),
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=self.entry.options.get(
                        CONF_DANGEROUS_BLOCK_THRESHOLD, RECOMMENDED_HARM_BLOCK_THRESHOLD
                    ),
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=self.entry.options.get(
                        CONF_SEXUAL_BLOCK_THRESHOLD, RECOMMENDED_HARM_BLOCK_THRESHOLD
                    ),
                ),
            ],
            tools=tools,
            system_instruction=prompt if supports_system_instruction else None,
            # Disables automatic function calling so that the conversation flow is controlled 
            # by the integration (though tools can still be called within the conversation loop).
            automatic_function_calling=AutomaticFunctionCallingConfig(
                disable=True, maximum_remote_calls=None
            ),
        )

        # For older models that cannot take a system_instruction, 
        # inserts the prompt as if it were user text, followed by a model acknowledgement.
        if not supports_system_instruction:
            messages = [
                Content(role="user", parts=[Part.from_text(text=prompt)]),
                Content(role="model", parts=[Part.from_text(text="Ok")]),
                *messages,
            ]
        # Creates a new chat session with the configured model and a conversation history.
        chat = self._genai_client.aio.chats.create(
            model=model_name, history=messages, config=generateContentConfig
        )
        chat_request: str | Content = user_input.text
        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                chat_response = await chat.send_message(message=chat_request)

                # Checks if the response is blocked by safety filters
                if chat_response.prompt_feedback:
                    raise HomeAssistantError(
                        f"The message got blocked due to content violations, reason: {chat_response.prompt_feedback.block_reason_message}"
                    )

            except (
                APIError,
                ValueError,
            ) as err:
                LOGGER.error("Error sending message: %s %s", type(err), err)
                error = f"Sorry, I had a problem talking to Google Generative AI: {err}"
                raise HomeAssistantError(error) from err

            response_parts = chat_response.candidates[0].content.parts
            if not response_parts:
                raise HomeAssistantError(
                    "Sorry, I had a problem getting a response from Google Generative AI."
                )
            content = " ".join(
                [part.text.strip() for part in response_parts if part.text]
            )

            # Collects any function calls in the response (tool_calls).
            tool_calls = []
            for part in response_parts:
                if not part.function_call:
                    continue
                tool_call = part.function_call
                tool_name = tool_call.name
                tool_args = _escape_decode(tool_call.args)
                tool_calls.append(
                    llm.ToolInput(tool_name=tool_name, tool_args=tool_args)
                )
                LOGGER.info("Google Gemini: Detected request for %s function call", tool_name)
                LOGGER.info("Google Gemini: Executed %s -> Result: %s", tool_name, tool_result)

            # If function calls are present, the code re-invokes the model with the tool responses as a new message. 
            # This loop allows for “multi-turn” usage of the tools.
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

            # Exits once no more tool calls remain or the iteration limit is hit
            if not tool_calls:
                break

        # Creates a final Home Assistant intent response, populating it with the AI’s text output.
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(
            " ".join([part.text.strip() for part in response_parts if part.text])
        )
        # Returns this result to the user.
        return conversation.ConversationResult(
            response=response, conversation_id=chat_log.conversation_id
        )

    # React to any updates in the config entry options (e.g., user changes the model or safety thresholds).
    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)
