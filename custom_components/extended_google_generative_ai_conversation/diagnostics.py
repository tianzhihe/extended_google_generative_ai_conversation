"""Diagnostics support for Google Generative AI Conversation."""

from __future__ import annotations

from typing import Any

# a utility function for removing or masking sensitive information in diagnostic data.
from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant

TO_REDACT = {CONF_API_KEY}

# maintain a simple in-memory record each time a function is called: in the service handlers or conversation tool execution, increment a counter or store the timestamp. 
hass.data.setdefault(DOMAIN, {})
hass.data[DOMAIN]["last_add_automation"] = {"time": time.time(), "yaml_length": len(yaml_str)}


# Called by Home Assistant whenever a user or system process requests diagnostic information for this config entry.
# provides a single function that Home Assistant uses to safely expose diagnostic information about the integration. 
# It ensures that sensitive data (like API keys) is masked, preserving user privacy and maintaining secure handling of secrets.
async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    diag = {
        "entry": {
            "title": entry.title,
            "data": async_redact_data(entry.data, TO_REDACT),
            "options": entry.options,
        },
        "function_calls": hass.data.get(DOMAIN, {})
    }
    return diag
