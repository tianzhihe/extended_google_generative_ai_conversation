from __future__ import annotations

"""The Extended Google Generative AI Conversation Integration."""
"""A Google Generative AI Conversation integration, rewritten to match the structure of extended OpenAI."""


import codecs
import json
import logging
from collections.abc import Callable
from typing import Any, Literal, cast

import yaml

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm, intent
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid

from google.genai.errors import APIError
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Content,
    FunctionCall,
    FunctionDeclaration,
    GenerateContentConfig,
    HarmCategory,
    Part,
    SafetySetting,
    Schema,
    Tool,
)

from .const import (
    # You might keep or remove these constants as needed.
    DOMAIN,
    LOGGER,
    CONF_PROMPT,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_K,
    RECOMMENDED_TOP_P,
    RECOMMENDED_MAX_TOKENS,
    CONF_HATE_BLOCK_THRESHOLD,
    CONF_HARASSMENT_BLOCK_THRESHOLD,
    CONF_DANGEROUS_BLOCK_THRESHOLD,
    CONF_SEXUAL_BLOCK_THRESHOLD,
    RECOMMENDED_HARM_BLOCK_THRESHOLD,
    CONF_CHAT_MODEL,
    RECOMMENDED_CHAT_MODEL,
    CONF_FUNCTIONS,
    DEFAULT_CONF_FUNCTIONS,
    CONF_USE_TOOLS,
    DEFAULT_USE_TOOLS,
    MAX_TOOL_ITERATIONS,
)

from .exceptions import (
    InvalidFunction,
    FunctionNotFound,
    FunctionLoadFailed,
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from .helpers import (
    get_function_executor,
)
# You can remove or adapt these if desired

_LOGGER = logging.getLogger(__name__)

DATA_AGENT = "agent"

# You can decide if you still want a config schema
def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Google Generative AI Conversation integration (dummy setup)."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Google Generative AI from a config entry with a new flow."""
    agent = GoogleGenerativeAIAgent(hass, entry)
    hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})[DATA_AGENT] = agent
    conversation.async_set_agent(hass, entry, agent)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Google Generative AI conversation."""
    hass.data[DOMAIN].pop(entry.entry_id, None)
    conversation.async_unset_agent(hass, entry)
    return True


# Keys we support for the schema of function parameters in Gemini
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
    """Format the schema for Google Generative AI."""
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
        # enum is only allowed for STRING type in Google Generative AI
        result["type"] = "STRING"
        result["enum"] = [str(item) for item in result["enum"]]

    if result.get("type") == "OBJECT" and not result.get("properties"):
        # Fallback to JSON string
        result["properties"] = {"json": {"type": "STRING"}}
        result["required"] = []
    return cast(Schema, result)


def _escape_decode(value: Any) -> Any:
    """Recursively call codecs.escape_decode on all values."""
    if isinstance(value, str):
        return codecs.escape_decode(bytes(value, "utf-8"))[0].decode("utf-8")  # type: ignore[attr-defined]
    if isinstance(value, list):
        return [_escape_decode(item) for item in value]
    if isinstance(value, dict):
        return {k: _escape_decode(v) for k, v in value.items()}
    return value


def _format_tool(tool: llm.Tool) -> Tool:
    """Format tool specification for Google Generative AI."""
    parameters = None
    if tool.parameters.schema:
        parameters = _format_schema(tool.parameters.schema)
    return Tool(
        function_declarations=[
            FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=parameters,
            )
        ]
    )


class GoogleGenerativeAIAgent(conversation.AbstractConversationAgent):
    """
    A Google Generative AI Agent, adopting the step-by-step function call structure
    used in Code 2 (OpenAI Conversation).

    This class manages conversation histories, function calls, and responses
    for Google Generative AI (Gemini).
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent with a Google GenAI client."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[dict[str, Any]]] = {}
        # The runtime_data should hold the Google Generative AI client.
        # For example, entry.runtime_data = google.genai.Genie(...)
        self._genai_client = entry.runtime_data
        self.logger = LOGGER

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """
        Process user input in a conversation, adopting a structure like Code 2.
        We keep a local history, determine if a function call is needed,
        and respond accordingly.
        """
        # Obtain or create a conversation ID
        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            user_input.conversation_id = conversation_id

            # Create a system message from the user config or default
            prompt = self.entry.options.get(CONF_PROMPT, "")
            if not prompt:
                prompt = "You are a helpful assistant."
            system_message = {"role": "system", "content": prompt}
            messages = [system_message]

        # Append user message
        messages.append({"role": "user", "content": user_input.text})

        # Start the query cycle
        try:
            response_data = await self.query(messages, user_input, 0)
        except (APIError, ValueError) as err:
            _LOGGER.error("Error talking to Google Generative AI: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Problem with Google Generative AI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        except HomeAssistantError as err:
            _LOGGER.error("Error: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN, str(err)
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        # Store the assistant message in the conversation
        messages.append(response_data["assistant_message"])
        self.history[conversation_id] = messages

        # Construct final Home Assistant conversation result
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_data["assistant_message"]["content"])
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def get_functions(self) -> list[dict[str, Any]]:
        """
        Load user-defined functions from YAML or defaults. We keep them in
        a format suitable for Gemini, but we do not rely on the old approach
        from the original code. We only parse them for usage in the new flow.
        """
        try:
            function = self.entry.options.get(CONF_FUNCTIONS)
            result = yaml.safe_load(function) if function else DEFAULT_CONF_FUNCTIONS
            if result:
                for setting in result:
                    function_executor = get_function_executor(
                        setting["function"]["type"]
                    )
                    setting["function"] = function_executor.to_arguments(
                        setting["function"]
                    )
            return result
        except (InvalidFunction, FunctionNotFound) as exc:
            raise exc
        except Exception as exc:
            raise FunctionLoadFailed from exc

    async def query(
        self,
        messages: list[dict[str, Any]],
        user_input: conversation.ConversationInput,
        iteration: int,
    ) -> dict[str, Any]:
        """
        Core method that sends messages to the Google Generative AI service
        and manages function call logic, similar to Code 2's approach.
        """
        # Prepare the request config
        model_name = self.entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        generateContentConfig = GenerateContentConfig(
            temperature=self.entry.options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            top_k=self.entry.options.get(CONF_TOP_K, RECOMMENDED_TOP_K),
            top_p=self.entry.options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            max_output_tokens=self.entry.options.get(
                CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
            ),
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
            automatic_function_calling=AutomaticFunctionCallingConfig(
                disable=True, maximum_remote_calls=None
            ),
        )

        # Convert local text-based messages into Gemini Contents
        gemini_messages = self._convert_messages_to_gemini(messages)
        chat = self._genai_client.aio.chats.create(
            model=model_name, history=gemini_messages, config=generateContentConfig
        )

        # Attempt to get a response
        for _ in range(MAX_TOOL_ITERATIONS):
            chat_response = await chat.send_message(message="")  # We'll send no new text
            candidate = chat_response.candidates[0]
            parts = candidate.content.parts

            # If the LLM returns blocked content
            if chat_response.prompt_feedback:
                block_msg = chat_response.prompt_feedback.block_reason_message
                raise HomeAssistantError(f"Content blocked: {block_msg}")

            # Combine text from parts
            assistant_text = " ".join([p.text for p in parts if p.text])

            # Check for function calls
            function_calls = [
                p.function_call for p in parts if p.function_call is not None
            ]
            if not function_calls:
                # No function call needed. Return final text to user.
                return {
                    "assistant_message": {
                        "role": "assistant",
                        "content": assistant_text,
                    }
                }

            # If there's a function call, handle it
            response_data = await self.execute_function_calls(
                function_calls, assistant_text, messages, user_input
            )
            if response_data.get("done"):
                return {
                    "assistant_message": {
                        "role": "assistant",
                        "content": response_data["assistant_content"],
                    }
                }
            # If not done, the loop continues to allow new calls
        # If we exceed MAX_TOOL_ITERATIONS, return what we have
        return {
            "assistant_message": {
                "role": "assistant",
                "content": "I'm stuck in a loop; can't finish.",
            }
        }

    async def execute_function_calls(
        self,
        function_calls: list[FunctionCall],
        assistant_text: str,
        messages: list[dict[str, Any]],
        user_input: conversation.ConversationInput,
    ) -> dict[str, Any]:
        """
        Execute each function call indicated by the LLM, append results,
        and signal if we are done or need to continue calling the LLM.
        """
        functions_config = self.get_functions()
        # If no functions are available, we do nothing
        if not functions_config:
            return {"done": True, "assistant_content": assistant_text}

        # We'll store any new function results as "function" messages
        for fn_call in function_calls:
            function_name = fn_call.name
            # Try to find a matching function
            function_data = next(
                (
                    f
                    for f in functions_config
                    if f["spec"]["name"] == function_name
                ),
                None,
            )
            if not function_data:
                raise FunctionNotFound(function_name)

            # Attempt to parse arguments
            try:
                arguments = json.loads(_escape_decode(fn_call.args))
            except json.JSONDecodeError as exc:
                raise ParseArgumentsFailed(fn_call.args) from exc

            # Execute
            function_executor = get_function_executor(function_data["function"]["type"])
            result = await function_executor.execute(
                self.hass, function_data["function"], arguments, user_input, None
            )

            # Append a "function" role message
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": str(result),
                }
            )

        # After executing all function calls, we return a signal that the conversation
        # should continue. The query loop will re-issue messages to see if more calls are needed.
        return {"done": False, "assistant_content": assistant_text}

    def _convert_messages_to_gemini(self, messages: list[dict[str, Any]]) -> list[Content]:
        """
        Convert local conversation messages into the Google Generative AI
        Content objects. For system/user/assistant roles, we adopt the same
        approach as Code 2 but for Gemini.
        """
        gemini_messages: list[Content] = []
        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")

            # 'assistant' or 'system' roles become 'model' or 'system' in Gemini
            if role == "assistant":
                gemini_role = "model"
            else:
                gemini_role = role

            gemini_messages.append(
                Content(
                    role=gemini_role,
                    parts=[Part.from_text(text=text)],
                )
            )
        return gemini_messages
