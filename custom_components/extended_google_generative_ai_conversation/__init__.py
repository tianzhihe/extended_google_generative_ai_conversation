"""The Extended Google Generative AI Conversation Integration."""

# enable postponed evaluation of annotations
from __future__ import annotations

# for filesystem path handling.
from pathlib import Path

from google import genai  # type: ignore[attr-defined]
from google.genai.errors import APIError, ClientError 
from requests.exceptions import Timeout # handle request timeouts.
import voluptuous as vol # a data validation library often used by Home Assistant.

from homeassistant.config_entries import ConfigEntry # Home Assistant’s internal classes and types for its component/integration infrastructure.
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryError,
    ConfigEntryNotReady,
    HomeAssistantError, #  a general exception base class for Home Assistant.
)
from homeassistant.helpers import config_validation as cv # assists with validating user or configuration data.
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue  # create warnings or issues for the user in the Home Assistant interface.
from homeassistant.helpers.typing import ConfigType

# Local Constants for configuration and defaults.
from .const import (
    CONF_CHAT_MODEL,
    CONF_PROMPT,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    TIMEOUT_MILLIS,
)

SERVICE_GENERATE_CONTENT = "generate_content"
CONF_IMAGE_FILENAME = "image_filename"
CONF_FILENAMES = "filenames"

# A schema definition indicating that configuration can only be set up through Home Assistant’s config entries for this domain 
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
# Lists the Home Assistant platforms used by this integration. Here, the only platform is Platform.CONVERSATION, which enables conversation features.
PLATFORMS = (Platform.CONVERSATION,)

type GoogleGenerativeAIConfigEntry = ConfigEntry[genai.Client]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Google Generative AI Conversation."""

    async def generate_content(call: ServiceCall) -> ServiceResponse:
        """Generate content from text and optionally images."""

        if call.data[CONF_IMAGE_FILENAME]:
            # Deprecated in 2025.3, to remove in 2025.9
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

        config_entry: GoogleGenerativeAIConfigEntry = hass.config_entries.async_entries(
            DOMAIN
        )[0]

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
                model=RECOMMENDED_CHAT_MODEL, contents=prompt_parts
            )
        except (
            APIError,
            ValueError,
        ) as err:
            raise HomeAssistantError(f"Error generating content: {err}") from err

        if response.prompt_feedback:
            raise HomeAssistantError(
                f"Error generating content due to content violations, reason: {response.prompt_feedback.block_reason_message}"
            )

        if not response.candidates[0].content.parts:
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


async def async_setup_entry(
    hass: HomeAssistant, entry: GoogleGenerativeAIConfigEntry
) -> bool:
    """Set up Google Generative AI Conversation from a config entry."""

    try:
        client = genai.Client(api_key=entry.data[CONF_API_KEY])
        await client.aio.models.get(
            model=entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            config={"http_options": {"timeout": TIMEOUT_MILLIS}},
        )
    except (APIError, Timeout) as err:
        if isinstance(err, ClientError) and "API_KEY_INVALID" in str(err):
            raise ConfigEntryAuthFailed(err.message) from err
        if isinstance(err, Timeout):
            raise ConfigEntryNotReady(err) from err
        raise ConfigEntryError(err) from err
    else:
        entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: GoogleGenerativeAIConfigEntry
) -> bool:
    """Unload GoogleGenerativeAI."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    return True
