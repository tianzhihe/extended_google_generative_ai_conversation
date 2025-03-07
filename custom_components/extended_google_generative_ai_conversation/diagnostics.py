"""Diagnostics support for Google Generative AI Conversation."""

from __future__ import annotations

from typing import Any

# a utility function for removing or masking sensitive information in diagnostic data.
from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant

TO_REDACT = {CONF_API_KEY}

# Called by Home Assistant whenever a user or system process requests diagnostic information for this config entry.
# provides a single function that Home Assistant uses to safely expose diagnostic information about the integration. 
# It ensures that sensitive data (like API keys) is masked, preserving user privacy and maintaining secure handling of secrets.
async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    return async_redact_data(
        {
            "title": entry.title,
            "data": entry.data,
            "options": entry.options,
        },
        TO_REDACT,
    )
