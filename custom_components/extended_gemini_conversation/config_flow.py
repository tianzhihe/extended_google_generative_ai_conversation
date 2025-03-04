"""Config flow for Google Generative AI Conversation integration (with function calling)."""
from __future__ import annotations

from collections.abc import Mapping
import logging
from types import MappingProxyType
from typing import Any

from google import genai  # type: ignore[attr-defined]
from google.genai.errors import APIError, ClientError
from requests.exceptions import Timeout
import voluptuous as vol
import yaml

from homeassistant.config_entries import (
    SOURCE_REAUTH,
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    BooleanSelector,
)

from .const import (
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    TIMEOUT_MILLIS,
    CONF_CHAT_MODEL,
    CONF_PROMPT,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_RECOMMENDED,
    CONF_HARASSMENT_BLOCK_THRESHOLD,
    CONF_HATE_BLOCK_THRESHOLD,
    CONF_SEXUAL_BLOCK_THRESHOLD,
    CONF_DANGEROUS_BLOCK_THRESHOLD,
    RECOMMENDED_HARM_BLOCK_THRESHOLD,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_K,
    RECOMMENDED_TOP_P,
)
_LOGGER = logging.getLogger(__name__)

# Additional keys for function calling support
CONF_FUNCTIONS = "functions"
CONF_USE_FUNCTIONS = "use_functions"

# Provide some defaults for function calling
DEFAULT_USE_FUNCTIONS = False
DEFAULT_FUNCTIONS = "[]"

STEP_API_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}


async def validate_input(data: dict[str, Any]) -> None:
    """Validate the user's input to ensure we can connect."""
    client = genai.Client(api_key=data[CONF_API_KEY])
    # Attempt a simple list call, which will fail if the API Key is invalid or unreachable
    await client.aio.models.list(
        config={
            "http_options": {"timeout": TIMEOUT_MILLIS},
            "query_base": True,
        }
    )


class GoogleGenerativeAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Config flow for Google Generative AI Conversation."""

    VERSION = 1

    async def async_step_api(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step to set up API credentials."""
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                await validate_input(user_input)
            except (APIError, Timeout) as err:
                if isinstance(err, ClientError) and "API_KEY_INVALID" in str(err):
                    errors["base"] = "invalid_auth"
                else:
                    errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                # If reauth, update
                if self.source == SOURCE_REAUTH:
                    return self.async_update_reload_and_abort(
                        self._get_reauth_entry(),
                        data=user_input,
                    )

                # Create a fresh entry with recommended defaults
                return self.async_create_entry(
                    title="Google Generative AI",
                    data=user_input,
                    options=RECOMMENDED_OPTIONS,
                )

        return self.async_show_form(
            step_id="api",
            data_schema=STEP_API_DATA_SCHEMA,
            description_placeholders={
                "api_key_url": "https://aistudio.google.com/app/apikey"
            },
            errors=errors,
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial user step."""
        return await self.async_step_api()

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Begin reauthorization if needed."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Dialog that informs the user that reauth is required."""
        if user_input is not None:
            return await self.async_step_api()

        reauth_entry = self._get_reauth_entry()
        return self.async_show_form(
            step_id="reauth_confirm",
            description_placeholders={
                CONF_NAME: reauth_entry.title,
                CONF_API_KEY: reauth_entry.data.get(CONF_API_KEY, ""),
            },
        )

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Return the OptionsFlow for advanced configuration."""
        return GoogleGenerativeAIOptionsFlow(config_entry)


class GoogleGenerativeAIOptionsFlow(OptionsFlow):
    """Google Generative AI options flow handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize the options flow."""
        self.config_entry = config_entry
        self.last_rendered_recommended = config_entry.options.get(CONF_RECOMMENDED, False)
        self._genai_client = config_entry.runtime_data

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the configuration options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options

        if user_input is not None:
            # If the recommended toggle didn't change, we can commit the changes
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if user_input[CONF_LLM_HASS_API] == "none":
                    user_input.pop(CONF_LLM_HASS_API)
                return self.async_create_entry(title="", data=user_input)

            # If the user changed the recommended toggle, we update and re-show the form
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]
            options = {
                CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                CONF_PROMPT: user_input[CONF_PROMPT],
                CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
                # We keep existing advanced fields if we want to re-show them
                CONF_FUNCTIONS: user_input.get(CONF_FUNCTIONS, DEFAULT_FUNCTIONS),
                CONF_USE_FUNCTIONS: user_input.get(CONF_USE_FUNCTIONS, DEFAULT_USE_FUNCTIONS),
            }

        schema = await google_generative_ai_config_option_schema(
            self.hass, options, self._genai_client
        )
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


async def google_generative_ai_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
    genai_client: genai.Client,
) -> dict:
    """Build the dynamic options schema for Google Generative AI, including function calling."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(label="No control", value="none")
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    # Basic items always shown (Prompt, LLM API control, recommended toggle).
    schema = {
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT)},
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Required(
            CONF_RECOMMENDED,
            default=options.get(CONF_RECOMMENDED, False),
        ): bool,
    }

    # If user has chosen "recommended" we skip advanced settings.
    if options.get(CONF_RECOMMENDED):
        return schema

    # Otherwise, build advanced settings (temperature, top_p, etc.).
    # We also retrieve available models from genai_client.
    api_models_pager = await genai_client.aio.models.list(config={"query_base": True})
    api_models = [api_model async for api_model in api_models_pager]
    models = []
    for api_model in sorted(api_models, key=lambda x: x.display_name or ""):
        # Filter out any model that lacks "generateContent" or is a vision-based model
        if (
            api_model.name
            and api_model.display_name
            and api_model.supported_actions
            and "vision" not in api_model.name
            and "generateContent" in api_model.supported_actions
        ):
            # Skip potential duplicates if needed
            if api_model.name == "models/gemini-1.0-pro":
                continue
            models.append(
                SelectOptionDict(
                    label=api_model.display_name,
                    value=api_model.name,
                )
            )

    harm_block_thresholds: list[SelectOptionDict] = [
        SelectOptionDict(label="Block none", value="BLOCK_NONE"),
        SelectOptionDict(label="Block few", value="BLOCK_ONLY_HIGH"),
        SelectOptionDict(label="Block some", value="BLOCK_MEDIUM_AND_ABOVE"),
        SelectOptionDict(label="Block most", value="BLOCK_LOW_AND_ABOVE"),
    ]

    harm_block_thresholds_selector = SelectSelector(
        SelectSelectorConfig(
            mode=SelectSelectorMode.DROPDOWN,
            options=harm_block_thresholds,
        )
    )

    # Add advanced completion parameters plus function-calling parameters
    schema.update(
        {
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)},
                default=RECOMMENDED_CHAT_MODEL,
            ): SelectSelector(SelectSelectorConfig(mode=SelectSelectorMode.DROPDOWN, options=models)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P, RECOMMENDED_TOP_P)},
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TOP_K,
                description={"suggested_value": options.get(CONF_TOP_K, RECOMMENDED_TOP_K)},
                default=RECOMMENDED_TOP_K,
            ): int,
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_HARASSMENT_BLOCK_THRESHOLD,
                description={"suggested_value": options.get(CONF_HARASSMENT_BLOCK_THRESHOLD)},
                default=RECOMMENDED_HARM_BLOCK_THRESHOLD,
            ): harm_block_thresholds_selector,
            vol.Optional(
                CONF_HATE_BLOCK_THRESHOLD,
                description={"suggested_value": options.get(CONF_HATE_BLOCK_THRESHOLD)},
                default=RECOMMENDED_HARM_BLOCK_THRESHOLD,
            ): harm_block_thresholds_selector,
            vol.Optional(
                CONF_SEXUAL_BLOCK_THRESHOLD,
                description={"suggested_value": options.get(CONF_SEXUAL_BLOCK_THRESHOLD)},
                default=RECOMMENDED_HARM_BLOCK_THRESHOLD,
            ): harm_block_thresholds_selector,
            vol.Optional(
                CONF_DANGEROUS_BLOCK_THRESHOLD,
                description={"suggested_value": options.get(CONF_DANGEROUS_BLOCK_THRESHOLD)},
                default=RECOMMENDED_HARM_BLOCK_THRESHOLD,
            ): harm_block_thresholds_selector,
            # New fields for function usage
            vol.Optional(
                CONF_USE_FUNCTIONS,
                description={"suggested_value": options.get(CONF_USE_FUNCTIONS, DEFAULT_USE_FUNCTIONS)},
                default=DEFAULT_USE_FUNCTIONS,
            ): BooleanSelector(),
            vol.Optional(
                CONF_FUNCTIONS,
                description={"suggested_value": options.get(CONF_FUNCTIONS, DEFAULT_FUNCTIONS)},
                default=DEFAULT_FUNCTIONS,
            ): TemplateSelector(),
        }
    )

    return schema
