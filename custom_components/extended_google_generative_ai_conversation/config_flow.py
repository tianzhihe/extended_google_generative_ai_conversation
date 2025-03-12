"""Config flow for Google Generative AI Conversation integration."""

from __future__ import annotations

# Brings in logging, plus helpers from Python’s standard library (Mapping, MappingProxyType, Any, etc.) for type annotations and read-only dictionaries.
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
    BooleanSelector
)

#  Several constants imported from .const that will be used when constructing the flow and building default values.
from .const import (
    CONF_CHAT_MODEL,
    CONF_DANGEROUS_BLOCK_THRESHOLD,
    CONF_HARASSMENT_BLOCK_THRESHOLD,
    CONF_HATE_BLOCK_THRESHOLD,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_SEXUAL_BLOCK_THRESHOLD,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_HARM_BLOCK_THRESHOLD,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_K,
    RECOMMENDED_TOP_P,
    TIMEOUT_MILLIS,
    CONF_FUNCTIONS,
    DEFAULT_CONF_FUNCTIONS,
    CONF_USE_TOOLS,
    DEFAULT_USE_TOOLS,
)

# Dumps a default functions structure (from Python to YAML) so it can be used in the form as the default value.
DEFAULT_CONF_FUNCTIONS_STR = yaml.dump(DEFAULT_CONF_FUNCTIONS, sort_keys=False)

_LOGGER = logging.getLogger(__name__)

# Describes the required data for the initial configuration step. Specifically, it requires an api_key from the user.
STEP_API_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)

# A dictionary containing default or recommended options for the integration when the user completes setup successfully. 
# It includes recommended flags and a default prompt.
# Need to change it to specialized LLM
RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}


async def validate_input(data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    client = genai.Client(api_key=data[CONF_API_KEY])
    #  Calls the models.list method with a set timeout to confirm that the API key is valid and the service is reachable.
    await client.aio.models.list( 
        config={
            "http_options": {
                "timeout": TIMEOUT_MILLIS,
            },
            "query_base": True,
        }
    )

# defines the steps and logic needed to set up a new configuration entry (or update an existing one) for this integration.
class GoogleGenerativeAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Google Generative AI Conversation."""

    # Used to track config flow versions, allowing migrations if the flow changes in the future.
    VERSION = 1

    async def async_step_api(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                await validate_input(user_input) # validate user input
            except (APIError, Timeout) as err:
                if isinstance(err, ClientError) and "API_KEY_INVALID" in str(err):
                    errors["base"] = "invalid_auth"
                else:
                    errors["base"] = "cannot_connect" # invalid_auth is set if the API key is invalid, otherwise cannot_connect is used.
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                # If this is a reauthentication flow (SOURCE_REAUTH), the code updates the existing entry’s credentials and aborts the flow.
                if self.source == SOURCE_REAUTH:
                    return self.async_update_reload_and_abort(
                        self._get_reauth_entry(),
                        data=user_input,
                    )
                # Otherwise, a new entry is created with a default title (“Google Extended Generative AI”) and prefilled recommended options.
                return self.async_create_entry(
                    title="Google Generative AI",
                    data=user_input,
                    options=RECOMMENDED_OPTIONS,
                )
        # If no user input is provided, or if an error occurred, the code displays a form to collect or re-collect the API key.
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
        """Handle the initial step."""
        return await self.async_step_api()

    # Triggered when the system detects that the existing credentials need to be reauthorized (for example, if an API key becomes invalid).
    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Handle configuration by re-auth."""
        return await self.async_step_reauth_confirm()

    # If the user submits new credentials, it again calls async_step_api for validation.
    # If no credentials are provided yet, it displays a form explaining that reauthentication is necessary.
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

    # Ties an Options Flow to this config flow, meaning that once the integration is set up, 
    # the user can open “Options” in Home Assistant’s UI and adjust settings without removing and recreating the entire config entry.
    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return GoogleGenerativeAIOptionsFlow(config_entry)

# Manages user-adjustable settings after the integration is installed.
class GoogleGenerativeAIOptionsFlow(OptionsFlow):
    """Google Generative AI config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        # Tracks whether the user has toggled recommended settings or not, so the form can dynamically change which fields are shown.
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )
        # A reference to the existing AI client, allowing it to pull available models or other dynamic data.
        self._genai_client = config_entry.runtime_data

    # If user_input is already provided, the flow checks whether the user changed the “recommended” toggle.
    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options

        if user_input is not None:
            # If the user did not change that toggle and chooses “No control” ("none") for the CONF_LLM_HASS_API, 
            # it removes the CONF_LLM_HASS_API key entirely from final options, then stores the updated data.
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if user_input[CONF_LLM_HASS_API] == "none":
                    user_input.pop(CONF_LLM_HASS_API)
                return self.async_create_entry(title="", data=user_input)

            # Re-render the options again, now with the recommended options shown/hidden
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]

            options = {
                CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                CONF_PROMPT: user_input[CONF_PROMPT],
                CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
            }

        schema = await google_generative_ai_config_option_schema(
            self.hass, options, self._genai_client
        )
        # Builds the form fields by calling google_generative_ai_config_option_schema, 
        # ensuring the user sees either a simplified or expanded set of configuration options based on their choices.
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


async def google_generative_ai_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
    genai_client: genai.Client,
) -> dict:
    """Return a schema for Google Generative AI completion options."""
    # Collects a list of possible Home Assistant “LLM APIs” or simply “no control” (none).
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    # Basic fields shown whenever the user edits the integration: a prompt (can be a template), 
    # a choice of controlling or not controlling LLM behaviors within Home Assistant, 
    # and a toggle that indicates if recommended settings should be displayed.
    schema = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }

    # If the user has chosen recommended settings, the function returns this simple set of fields.
    if options.get(CONF_RECOMMENDED):
        return schema

    # If the user does not select recommended defaults, the function fetches all available models from the Google AI client.
    api_models_pager = await genai_client.aio.models.list(config={"query_base": True})
    api_models = [api_model async for api_model in api_models_pager]
    models = [
        SelectOptionDict(
            label=api_model.display_name,
            value=api_model.name,
        )
        for api_model in sorted(api_models, key=lambda x: x.display_name or "")
        if (
            api_model.name != "models/gemini-1.0-pro"  # duplicate of gemini-pro
            and api_model.display_name
            and api_model.name
            and api_model.supported_actions
            and "vision" not in api_model.name
            and "generateContent" in api_model.supported_actions
        )
    ]

    # The user can then pick a model, specify text-generation parameters like temperature, top_p, top_k, and set thresholds to block harmful content.
    harm_block_thresholds: list[SelectOptionDict] = [
        SelectOptionDict(
            label="Block none",
            value="BLOCK_NONE",
        ),
        SelectOptionDict(
            label="Block few",
            value="BLOCK_ONLY_HIGH",
        ),
        SelectOptionDict(
            label="Block some",
            value="BLOCK_MEDIUM_AND_ABOVE",
        ),
        SelectOptionDict(
            label="Block most",
            value="BLOCK_LOW_AND_ABOVE",
        ),
    ]
    harm_block_thresholds_selector = SelectSelector(
        SelectSelectorConfig(
            mode=SelectSelectorMode.DROPDOWN, options=harm_block_thresholds
        )
    )

    # Those fields are appended to the original schema.
    schema.update(
        {
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                default=RECOMMENDED_CHAT_MODEL,
            ): SelectSelector(
                SelectSelectorConfig(mode=SelectSelectorMode.DROPDOWN, options=models)
            ),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TOP_K,
                description={"suggested_value": options.get(CONF_TOP_K)},
                default=RECOMMENDED_TOP_K,
            ): int,
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_HARASSMENT_BLOCK_THRESHOLD,
                description={
                    "suggested_value": options.get(CONF_HARASSMENT_BLOCK_THRESHOLD)
                },
                default=RECOMMENDED_HARM_BLOCK_THRESHOLD,
            ): harm_block_thresholds_selector,
            vol.Optional(
                CONF_HATE_BLOCK_THRESHOLD,
                description={"suggested_value": options.get(CONF_HATE_BLOCK_THRESHOLD)},
                default=RECOMMENDED_HARM_BLOCK_THRESHOLD,
            ): harm_block_thresholds_selector,
            vol.Optional(
                CONF_SEXUAL_BLOCK_THRESHOLD,
                description={
                    "suggested_value": options.get(CONF_SEXUAL_BLOCK_THRESHOLD)
                },
                default=RECOMMENDED_HARM_BLOCK_THRESHOLD,
            ): harm_block_thresholds_selector,
            vol.Optional(
                CONF_DANGEROUS_BLOCK_THRESHOLD,
                description={
                    "suggested_value": options.get(CONF_DANGEROUS_BLOCK_THRESHOLD)
                },
                default=RECOMMENDED_HARM_BLOCK_THRESHOLD,
            ): harm_block_thresholds_selector,
            vol.Optional(
                CONF_FUNCTIONS,
                description={"suggested_value": options.get(CONF_FUNCTIONS)},
                default=DEFAULT_CONF_FUNCTIONS_STR,
            ): TemplateSelector(),
            vol.Optional(
                CONF_USE_TOOLS,
                description={"suggested_value": options.get(CONF_USE_TOOLS)},
                default=DEFAULT_USE_TOOLS,
            ): BooleanSelector(),
        }
    )
    return schema
