"""Google Generative AI Conversation integration with function-calling style support."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import voluptuous as vol
from google import genai  # type: ignore[attr-defined]
from google.genai.errors import APIError, ClientError
from requests.exceptions import Timeout

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_API_KEY,
    Platform,
    MATCH_ALL,
)
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    intent,
)
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryError,
    ConfigEntryNotReady,
    HomeAssistantError,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid

_LOGGER = logging.getLogger(__name__)

DOMAIN = "google_generative_ai_conversation"
DATA_AGENT = "agent"

CONF_CHAT_MODEL = "chat_model"
CONF_PROMPT = "prompt"
CONF_IMAGE_FILENAME = "image_filename"
CONF_FILENAMES = "filenames"

# Example: Provide your “functions” in a YAML-like or JSON-like structure
# so you can load them at runtime. This can then be documented or 
# stored in ConfigEntry options.
CONF_FUNCTIONS = "functions"

RECOMMENDED_CHAT_MODEL = "models/chat-bison-001"
TIMEOUT_MILLIS = 60000

# Services
SERVICE_GENERATE_CONTENT = "generate_content"

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
PLATFORMS = (Platform.CONVERSATION,)

# Example function placeholder
def execute_example_function(hass: HomeAssistant, arguments: dict) -> str:
    """Example function that might toggle a light, look up weather, etc."""
    # Implementation will depend on your needs
    return f"Executed example_function with arguments: {arguments}"

def load_functions(raw_functions: str) -> list[dict]:
    """Load function specifications from YAML or JSON in config options."""
    # This snippet uses JSON for illustration; adjust as needed for YAML.
    try:
        spec_list = json.loads(raw_functions)
    except (json.JSONDecodeError, TypeError):
        spec_list = []
    return spec_list

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the integration (register services, etc.)."""

    async def generate_content(call: ServiceCall) -> ServiceResponse:
        """Generate content from text and optionally images using Google Generative AI."""
        # This portion shows how you'd still handle your existing service
        # to generate text (and potentially attach files).
        if call.data.get(CONF_IMAGE_FILENAME):
            # Deprecated issue example
            async_create_issue(
                hass,
                DOMAIN,
                "deprecated_image_filename_parameter",
                breaks_in_ha_version="2025.9.0",
                is_fixable=False,
                severity=IssueSeverity.WARNING,
                translation_key="deprecated_image_filename_parameter",
            )

        prompt_parts = [call.data[CONF_PROMPT]]

        config_entry: ConfigEntry = hass.config_entries.async_entries(DOMAIN)[0]
        client = config_entry.runtime_data

        def append_files_to_prompt():
            image_filenames = call.data[CONF_IMAGE_FILENAME]
            filenames = call.data[CONF_FILENAMES]
            for filename in set(image_filenames + filenames):
                if not hass.config.is_allowed_path(filename):
                    raise HomeAssistantError(
                        f"Cannot read `{filename}`, no access to path; "
                        "`allowlist_external_dirs` may need to be adjusted in "
                        "`configuration.yaml`"
                    )
                if not Path(filename).exists():
                    raise HomeAssistantError(f"`{filename}` does not exist")
                prompt_parts.append(client.files.upload(file=filename))

        await hass.async_add_executor_job(append_files_to_prompt)

        try:
            response = await client.aio.models.generate_content(
                model=RECOMMENDED_CHAT_MODEL,
                contents=prompt_parts,
            )
        except (APIError, ValueError) as err:
            raise HomeAssistantError(f"Error generating content: {err}") from err

        if response.prompt_feedback:
            raise HomeAssistantError(
                "Error generating content due to content violations, "
                f"reason: {response.prompt_feedback.block_reason_message}"
            )

        if not response.candidates or not response.candidates[0].content.parts:
            raise HomeAssistantError("Unknown error generating content")

        return {"text": response.text}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        generate_content,
        schema=vol.Schema(
            {
                vol.Required(CONF_PROMPT): cv.string,
                vol.Optional(CONF_IMAGE_FILENAME, default=[]): vol.All(
                    cv.ensure_list, [cv.string]
                ),
                vol.Optional(CONF_FILENAMES, default=[]): vol.All(
                    cv.ensure_list, [cv.string]
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the integration from a config entry (creates the conversation agent, etc.)."""
    try:
        client = genai.Client(api_key=entry.data[CONF_API_KEY])
        # Validate we can connect to at least one model
        await client.aio.models.get(
            model=entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            config={"http_options": {"timeout": TIMEOUT_MILLIS}},
        )
    except (APIError, Timeout) as err:
        if isinstance(err, ClientError) and "API_KEY_INVALID" in str(err):
            raise ConfigEntryAuthFailed("Invalid API Key") from err
        if isinstance(err, Timeout):
            raise ConfigEntryNotReady("Request timed out") from err
        raise ConfigEntryError(f"Error setting up entry: {err}") from err

    entry.runtime_data = client

    # Create the conversation agent
    agent = GoogleGenerativeAIAgent(hass, entry)
    # Register the agent with the conversation component
    conversation.async_set_agent(hass, entry, agent)

    # Forward the Conversation platform (so HA sets up conversation for this domain)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload the integration."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False
    conversation.async_unset_agent(hass, entry)
    return True


class GoogleGenerativeAIAgent(conversation.AbstractConversationAgent):
    """
    A sample agent that uses Google Generative AI to manage conversation, 
    including function-call style prompts.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.client = entry.runtime_data
        # Keep track of messages for each conversation
        self.history: dict[str, list[dict]] = {}

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """
        Process a conversation request. This method is where you'll incorporate 
        the system prompt, user message, function definitions, and detect 
        whether the AI is requesting a function call.
        """

        # Find or create a conversation ID
        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            user_input.conversation_id = conversation_id
            system_message = self._generate_system_message()
            messages = [system_message]

        # Append the current user message
        messages.append({"role": "user", "content": user_input.text})
        self.history[conversation_id] = messages

        # Send request to Google Generative AI
        try:
            response_text = await self._call_generative_ai(messages)
        except HomeAssistantError as err:
            _LOGGER.error("Error during generative AI call: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN, str(err)
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        # Examine the response to see if it is requesting a function call
        function_call_result = await self._maybe_execute_function(response_text)

        # If a function call was made, we might want to ask the model how to handle it
        if function_call_result:
            # The model’s response indicated a function call
            # so we can append the function call result and re-query for a final message
            messages.append(
                {
                    "role": "function",
                    "content": function_call_result,
                }
            )
            # Then call the model again with updated messages
            try:
                response_text = await self._call_generative_ai(messages)
            except HomeAssistantError as err:
                _LOGGER.error("Error after function call: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN, str(err)
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )

        # Record the final AI message
        messages.append({"role": "assistant", "content": response_text})

        # Prepare the final conversation response
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _generate_system_message(self) -> dict:
        """
        Generate a system message that includes instructions and 
        the list of functions the model may call.
        """
        # In the OpenAI approach, you might load the function schema. 
        # Here, we embed it in the prompt for demonstration.
        raw_functions = self.entry.options.get(CONF_FUNCTIONS, "[]")
        functions_list = load_functions(raw_functions)

        # Provide usage instructions plus some kind of JSON format for function calls
        # This is a sample approach; you should refine your instructions.
        system_prompt = (
            "You are Home Assistant's Google Generative AI Agent. "
            "You can respond normally, or if you need to use one of the following functions, "
            "please return ONLY valid JSON in this exact format:\n\n"
            '{\n  "function_call": {\n    "name": "<FUNCTION_NAME>",\n    '
            '"arguments": { "arg1": "value", ... }\n  }\n}\n\n'
            "Here are the functions you have available:\n"
        )

        for func in functions_list:
            name = func.get("name")
            desc = func.get("description", "")
            args = func.get("parameters", {})
            system_prompt += f"\n- Name: {name}\n  Description: {desc}\n  Parameters: {json.dumps(args)}\n"

        return {"role": "system", "content": system_prompt}

    async def _call_generative_ai(self, messages: list[dict]) -> str:
        """
        Convert the messages into a single text prompt and call the 
        Google Generative AI API. 
        """
        # Flatten the conversation into text. You can refine for better context.
        conversation_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            conversation_text += f"{role.upper()}:\n{content}\n\n"

        # Prepare the contents for Google’s client
        contents = [conversation_text]

        try:
            response = await self.client.aio.models.generate_content(
                model=self.entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                contents=contents,
            )
        except (APIError, ValueError) as err:
            raise HomeAssistantError(f"Generative AI error: {err}") from err

        if response.prompt_feedback:
            raise HomeAssistantError(
                f"Content violation: {response.prompt_feedback.block_reason_message}"
            )

        if not response.candidates or not response.candidates[0].content.parts:
            raise HomeAssistantError("No valid response from Generative AI.")

        return response.text

    async def _maybe_execute_function(self, response_text: str) -> str | None:
        """
        Look for a JSON block that indicates a function call.
        If found, parse it, execute the function, and return the result.
        If no function call is indicated, return None.
        """
        # Attempt to parse as JSON
        try:
            parsed = json.loads(response_text)
            if "function_call" not in parsed:
                return None
        except (json.JSONDecodeError, TypeError):
            return None

        call_data = parsed["function_call"]
        name = call_data.get("name")
        arguments = call_data.get("arguments", {})

        # Look up the function
        raw_functions = self.entry.options.get(CONF_FUNCTIONS, "[]")
        functions_list = load_functions(raw_functions)
        func_spec = next((f for f in functions_list if f["name"] == name), None)
        if not func_spec:
            return f"Function '{name}' not found."

        # Here you decide how to route to the correct function
        # For demonstration, we only handle a single example function
        if name == "example_function":
            try:
                result = execute_example_function(self.hass, arguments)
                return result
            except Exception as err:  # broad except for demonstration
                return f"Error in function '{name}': {err}"

        # If you had multiple functions, you'd add additional handling.
        return f"Function '{name}' is recognized but not implemented."
