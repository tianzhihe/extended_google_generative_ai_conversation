# Standard Python libraries for base64 encoding, logging, MIME type handling, and file paths
import base64
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

# AsyncOpenAI and OpenAIError are used to handle OpenAI model calls and potential exceptions.
from openai import AsyncOpenAI
from openai._exceptions import OpenAIError

# ChatCompletionContentPartImageParam defines how image parameters can be sent to OpenAI in the new multi-part messages.
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)

# voluptuous is a library for data validation. 
import voluptuous as vol

# Home Assistant imports for core functionality, service calls/responses, exceptions, and config validation.
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.typing import ConfigType

# Constants from this integration's domain and service names
from .const import DOMAIN, SERVICE_QUERY_IMAGE

# Defines the schema for validating the service call to query images with OpenAI.
QUERY_IMAGE_SCHEMA = vol.Schema(
    {
        # config_entry selector ensures we pick the correct integration config entry (e.g., an OpenAI config).
        vol.Required("config_entry"): selector.ConfigEntrySelector(
            {
                "integration": DOMAIN,
            }
        ),
        # 'model' defaults to "gpt-4-vision-preview"
        vol.Required("model", default="gpt-4-vision-preview"): cv.string,
        # 'prompt' is required and must be a string
        vol.Required("prompt"): cv.string,
        # 'images' must be a list of dictionaries, each containing a "url" entry
        vol.Required("images"): vol.All(cv.ensure_list, [{"url": cv.string}]),
        # 'max_tokens' is optional with a default of 300
        vol.Optional("max_tokens", default=300): cv.positive_int,
    }
)

# Set up a logger named after the current package/module
_LOGGER = logging.getLogger(__package__)


async def async_setup_services(hass: HomeAssistant, config: ConfigType) -> None:
    """Set up services for the extended openai conversation component."""

    # Define the service function that will handle image-related prompts to the OpenAI model.
    async def query_image(call: ServiceCall) -> ServiceResponse:
        """Query an image."""
        try:
            # Extract the model type from the service call data.
            model = call.data["model"]

            # Convert each image URL to a suitable parameter for the OpenAI message.
            # This will check if it's a local file or an external URL and handle accordingly.
            images = [
                {"type": "image_url", "image_url": to_image_param(hass, image)}
                for image in call.data["images"]
            ]

            # Construct a message that combines a text prompt and the image references.
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": call.data["prompt"]}] + images,
                }
            ]

            # Log the outgoing prompt (for debugging and clarity).
            _LOGGER.info("Prompt for %s: %s", model, messages)

            # Create a new AsyncOpenAI client with the API key associated with the config entry.
            response = await AsyncOpenAI(
                api_key=hass.data[DOMAIN][call.data["config_entry"]]["api_key"]
            ).chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=call.data["max_tokens"],
            )

            # Convert the response to a standard dictionary for easy logging and returning.
            response_dict = response.model_dump()
            _LOGGER.info("Response %s", response_dict)

        except OpenAIError as err:
            # Convert any OpenAI-specific errors to Home Assistant errors.
            raise HomeAssistantError(f"Error generating image: {err}") from err

        # Return the entire response dictionary.
        return response_dict

    # Register the new service (SERVICE_QUERY_IMAGE) with Home Assistant, 
    # providing the schema, function handler, and indicating it supports a response object.
    hass.services.async_register(
        DOMAIN,
        SERVICE_QUERY_IMAGE,
        query_image,
        schema=QUERY_IMAGE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


def to_image_param(hass: HomeAssistant, image) -> ChatCompletionContentPartImageParam:
    """Convert url to base64 encoded image if local."""
    # Extract the URL from the 'image' dictionary.
    url = image["url"]

    # Check whether the URL scheme is one of the recognized external protocols (http, https, etc.).
    if urlparse(url).scheme in cv.EXTERNAL_URL_PROTOCOL_SCHEMA_LIST:
        # If the URL is external, we simply return it as-is (no need to encode).
        return image

    # If it's not an external URL, we consider it local and see if Home Assistant can access it (allowlist).
    if not hass.config.is_allowed_path(url):
        raise HomeAssistantError(
            f"Cannot read `{url}`, no access to path; "
            "`allowlist_external_dirs` may need to be adjusted in "
            "`configuration.yaml`"
        )

    # Confirm the file actually exists.
    if not Path(url).exists():
        raise HomeAssistantError(f"`{url}` does not exist")

    # Try to guess the file’s MIME type to ensure it's an image.
    mime_type, _ = mimetypes.guess_type(url)
    if mime_type is None or not mime_type.startswith("image"):
        raise HomeAssistantError(f"`{url}` is not an image")

    # Convert the local image file into a data URL with base64 encoding.
    image["url"] = f"data:{mime_type};base64,{encode_image(url)}"
    return image


def encode_image(image_path):
    """Convert to base64 encoded image."""
    # Open the image file in binary read mode.
    with open(image_path, "rb") as image_file:
        # Read the file content, encode it as base64, and decode it into a UTF-8 string.
        return base64.b64encode(image_file.read()).decode("utf-8")
