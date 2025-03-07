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

from .helpers import (
    get_function_executor,
    is_azure,
    validate_authentication,
)

SERVICE_GENERATE_CONTENT = "generate_content"
CONF_IMAGE_FILENAME = "image_filename"
CONF_FILENAMES = "filenames"
SERVICE_ADD_AUTOMATION = "add_automation"
SERVICE_GET_ENERGY = "get_energy"

# A schema definition indicating that configuration can only be set up through Home Assistant’s config entries for this domain 
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
# Lists the Home Assistant platforms used by this integration. Here, the only platform is Platform.CONVERSATION, which enables conversation features.
PLATFORMS = (Platform.CONVERSATION,)

# This creates a type alias showing that the ConfigEntry will store and manage an instance of genai.Client at runtime. It helps with code clarity and type checking.
type GoogleGenerativeAIConfigEntry = ConfigEntry[genai.Client]

# This is the entry point for setting up the integration within Home Assistant. 
# It returns a boolean indicating whether setup was successful.
async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Google Generative AI Conversation."""

    # Defines a service (named generate_content) that anyone using Home Assistant can call to request text generation (and optional image uploads) through Google’s Generative AI.
    async def generate_content(call: ServiceCall) -> ServiceResponse:
        """Generate content from text and optionally images."""

        # This checks if image_filename was included in the service data. If present, it creates a warning message because that parameter is planned to be removed in a future version.
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

        # A list that will collect all pieces needed for the request to the AI model. It starts with the main prompt string.
        prompt_parts = [call.data[CONF_PROMPT]]

        # fetches the active config entry for this integration (DOMAIN) and retrieves its stored client (client), which is the genai.Client used to interact with Google’s AI API.
        config_entry: GoogleGenerativeAIConfigEntry = hass.config_entries.async_entries(
            DOMAIN
        )[0]

        client = config_entry.runtime_data

        # Gathers all filenames (both image filenames and general file names).
        # Verifies each file is permitted by Home Assistant’s security settings (is_allowed_path).
        # Checks if the file physically exists.
        # Uploads the file to the AI service via client.files.upload and appends the result to prompt_parts.
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

        #  runs that function in a thread pool to avoid blocking the event loop, which is best practice in async contexts.
        await hass.async_add_executor_job(append_files_to_prompt)

        # Once prompt_parts is ready, the code calls generate_content on the client’s asynchronous model interface.
        try:
            response = await client.aio.models.generate_content(
                model=RECOMMENDED_CHAT_MODEL, contents=prompt_parts
            )
        except (
            APIError,
            ValueError,
        ) as err:
            raise HomeAssistantError(f"Error generating content: {err}") from err # 
        # If an API error (like invalid credentials, bad request) or ValueError occurs, it wraps and re-raises it as a HomeAssistantError. This is so Home Assistant can handle it gracefully.
        if response.prompt_feedback:
            raise HomeAssistantError(
                f"Error generating content due to content violations, reason: {response.prompt_feedback.block_reason_message}"
            )

        if not response.candidates[0].content.parts:
            raise HomeAssistantError("Unknown error generating content")

        return {"text": response.text}

    # makes the generate_content function available as a Home Assistant service under the integration’s domain (DOMAIN) and service name (SERVICE_GENERATE_CONTENT).
    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        generate_content,
        schema=vol.Schema(
            {
                vol.Required(CONF_PROMPT): cv.string, # the prompt field must be provided and must be a string.
                vol.Optional(CONF_IMAGE_FILENAME, default=[]): vol.All(
                    cv.ensure_list, [cv.string]
                ),
                vol.Optional(CONF_FILENAMES, default=[]): vol.All(
                    cv.ensure_list, [cv.string]
                ), # each can be provided as either a single string or a list of strings, but ultimately validated as lists.
            }
        ),
        supports_response=SupportsResponse.ONLY, # the service can return data back to the caller, rather than just performing an action.
    )
    return True
    # The function ends with return True, signaling successful setup at this stage.



async def async_setup_entry(
    hass: HomeAssistant, entry: GoogleGenerativeAIConfigEntry
) -> bool:
    """Set up Google Generative AI Conversation from a config entry."""

    try:
        client = genai.Client(api_key=entry.data[CONF_API_KEY]) #  object is created with the API key stored in the config entry.
        await client.aio.models.get(
            model=entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            config={"http_options": {"timeout": TIMEOUT_MILLIS}},
        ) # Confirm the given model is reachable and functional before proceeding. This also applies a timeout (TIMEOUT_MILLIS) so the integration does not hang indefinitely.
    except (APIError, Timeout) as err:
        # If the API key is invalid, ConfigEntryAuthFailed will be raised, prompting Home Assistant to handle re-authentication or report the problem to the user.
        if isinstance(err, ClientError) and "API_KEY_INVALID" in str(err):
            raise ConfigEntryAuthFailed(err.message) from err
        #  If the request times out, ConfigEntryNotReady is raised, telling Home Assistant to try again later.
        if isinstance(err, Timeout):
            raise ConfigEntryNotReady(err) from err
        raise ConfigEntryError(err) from err
    # If no errors occur, the code attaches the client to entry.runtime_data. This makes the client accessible to other parts of the integration, like the service handler.
    else:
        entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True

async def add_automation_service(hass: HomeAssistant, call: ServiceCall) -> ServiceResponse:
    """Handle the add_automation service call."""
    yaml_str = call.data["automation_config"]
    try:
        # Parse YAML string to dict
        automation_config = yaml.safe_load(yaml_str)
    except Exception as err:
        raise HomeAssistantError(f"Invalid YAML: {err}") 
    # Validate and add the automation (similar to conversation tool logic)
    config = {"id": str(round(time.time() * 1000))}  # generate a unique ID
    if isinstance(automation_config, list):
        config.update(automation_config[0])
    elif isinstance(automation_config, dict):
        config.update(automation_config)
    # Validate the automation configuration structure
    await _async_validate_config_item(hass, config, full_config=True, raise_on_errors=True)
    # Append to automations YAML and reload
    automations_path = os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH)
    current = []
    if os.path.exists(automations_path):
        current = yaml.safe_load(open(automations_path, "r")) or []
    current.append(config)
    with open(automations_path, "w", encoding="utf-8") as f:
        yaml.dump(current, f, allow_unicode=True, sort_keys=False)
    # Reload automations to apply the new one
    await hass.services.async_call("automation", "reload", {})
    # Return a success result
    return {"result": "success", "id": config.get("id")}  # include the new automation ID
    # register these in async_setup (or async_setup_entry after obtaining an API client) with schemas
    hass.services.async_register(
        DOMAIN, SERVICE_ADD_AUTOMATION, add_automation_service,
        schema=vol.Schema({ vol.Required("automation_config"): cv.string }),
        supports_response=SupportsResponse.ONLY
    )

async def get_energy_service(hass: HomeAssistant, call: ServiceCall) -> ServiceResponse:
    """Handle the get_energy service call."""
    energy_manager = await hass.helpers.energy.async_get_manager(hass)
    data = energy_manager.data  # get energy stats data structure
    return {"result": data}
    
    hass.services.async_register(
        DOMAIN, SERVICE_GET_ENERGY, get_energy_service,
        schema=vol.Schema({}),  # no inputs
        supports_response=SupportsResponse.ONLY
    )

# Manages unloading and cleanup when the user or system removes or disables the integration’s config entry.
async def async_unload_entry(
    hass: HomeAssistant, entry: GoogleGenerativeAIConfigEntry
) -> bool:
    """Unload GoogleGenerativeAI."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    return True
