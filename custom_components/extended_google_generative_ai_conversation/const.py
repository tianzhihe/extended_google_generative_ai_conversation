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



TIMEOUT_MILLIS = 10000

DEFAULT_NAME = "Extended Google AI Conversation"
CONF_ORGANIZATION = "organization"
CONF_BASE_URL = "base_url"
DEFAULT_CONF_BASE_URL = "https://ai.google.dev/gemini-api/docs"
# CONF_API_VERSION = "api_version"
# CONF_SKIP_AUTHENTICATION = "skip_authentication"
# DEFAULT_SKIP_AUTHENTICATION = False

EVENT_AUTOMATION_REGISTERED = "automation_registered_via_extended_googleai_conversation"
EVENT_CONVERSATION_FINISHED = "extended_googleai_conversation.conversation.finished"

CONF_PROMPT = "prompt"
DEFAULT_PROMPT = """I want you to act as smart home manager of Home Assistant.
I will provide information of smart home along with a question, you will truthfully make correction or answer using information provided in one sentence in everyday language.

Current Time: {{now()}}

Available Devices:
```csv
entity_id,name,state,aliases
{% for entity in exposed_entities -%}
{{ entity.entity_id }},{{ entity.name }},{{ entity.state }},{{entity.aliases | join('/')}}
{% endfor -%}
```

The current state of devices is provided in available devices.
Use execute_services function only for requested action, not for current states.
Do not execute service without user's confirmation.
Do not restate or appreciate what user says, rather make a quick inquiry.
"""
# CONF_CHAT_MODEL = "chat_model"
# DEFAULT_CHAT_MODEL = "gpt-4o-mini"
# CONF_MAX_TOKENS = "max_tokens"
# DEFAULT_MAX_TOKENS = 150
# CONF_TOP_P = "top_p"
# DEFAULT_TOP_P = 1
# CONF_TEMPERATURE = "temperature"
# DEFAULT_TEMPERATURE = 0.5
CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION = "max_function_calls_per_conversation"
DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION = 1
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
    }
]
CONF_ATTACH_USERNAME = "attach_username"
DEFAULT_ATTACH_USERNAME = False
CONF_USE_TOOLS = "use_tools"
DEFAULT_USE_TOOLS = False
CONF_CONTEXT_THRESHOLD = "context_threshold"
# DEFAULT_CONTEXT_THRESHOLD = 13000
CONTEXT_TRUNCATE_STRATEGIES = [{"key": "clear", "label": "Clear All Messages"}]
CONF_CONTEXT_TRUNCATE_STRATEGY = "context_truncate_strategy"
DEFAULT_CONTEXT_TRUNCATE_STRATEGY = CONTEXT_TRUNCATE_STRATEGIES[0]["key"]

SERVICE_QUERY_IMAGE = "query_image"

CONF_PAYLOAD_TEMPLATE = "payload_template"
