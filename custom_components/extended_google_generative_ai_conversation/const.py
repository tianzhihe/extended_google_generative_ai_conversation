"""Constants for the Google Generative AI Conversation integration."""

import logging

# This serves as the unique identifier for the entire integration within Home Assistant. 
# Any services, configuration entries, or logs associated with this integration are tracked under this string.
DOMAIN = "extended_google_generative_ai_conversation"

# A standard Python logger associated with the current package. 
# It makes it possible to log messages that relate specifically to this integration.
LOGGER = logging.getLogger(__package__)

# A configuration key representing the text prompt that the user wants to send to the Generative AI model.
CONF_PROMPT = "prompt"

CONF_RECOMMENDED = "recommended"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "models/gemini-2.0-flash"
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 0.95
CONF_TOP_K = "top_k"
RECOMMENDED_TOP_K = 64
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 200
CONF_HARASSMENT_BLOCK_THRESHOLD = "harassment_block_threshold"
CONF_HATE_BLOCK_THRESHOLD = "hate_block_threshold"
CONF_SEXUAL_BLOCK_THRESHOLD = "sexual_block_threshold"
CONF_DANGEROUS_BLOCK_THRESHOLD = "dangerous_block_threshold"
RECOMMENDED_HARM_BLOCK_THRESHOLD = "BLOCK_MEDIUM_AND_ABOVE"

CONF_FUNCTIONS = "functions"
DEFAULT_CONF_FUNCTIONS = [
    {
        "spec": {
            "name": "execute_services",
            "description": "Use this function to execute service of devices in Home Assistant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "domain": {
                                    "type": "string",
                                    "description": "The domain of the service",
                                },
                                "service": {
                                    "type": "string",
                                    "description": "The service to be called",
                                },
                                "service_data": {
                                    "type": "object",
                                    "description": "The service data object to indicate what to control.",
                                    "properties": {
                                        "entity_id": {
                                            "type": "string",
                                            "description": "The entity_id retrieved from available devices. It must start with domain, followed by dot character.",
                                        }
                                    },
                                    "required": ["entity_id"],
                                },
                            },
                            "required": ["domain", "service", "service_data"],
                        },
                    }
                },
            },
        },
        "function": {"type": "native", "name": "execute_service"},
    },
        {
        "spec": {
            "name": "get_energy_statistic_ids",
            "description": "Get statistics",
            "parameters": {
                "type": "object",
                "properties": {
                    "dummy": {
                        "type": "string",
                        "description": "Nothing",
                    }
                },
            },
        },
        "function": {
            "type": "composite",
            "sequence": [
                {
                    "type": "native",
                    "name": "get_energy",
                    "response_variable": "result"
                },
                {
                    "type": "template",
                    "value_template": "{{result.device_consumption | map(attribute='stat_consumption') | list}}"
                }
            ]
        }
    }
]

CONF_USE_TOOLS = "use_tools"
DEFAULT_USE_TOOLS = True

TIMEOUT_MILLIS = 10000

CONF_PAYLOAD_TEMPLATE = "payload_template"

EVENT_AUTOMATION_REGISTERED = "automation_registered_via_extended_google_generative_ai_conversation"
EVENT_CONVERSATION_FINISHED = "extended_google_generative_ai_conversation.conversation.finished"

MAX_TOOL_ITERATIONS = 10
