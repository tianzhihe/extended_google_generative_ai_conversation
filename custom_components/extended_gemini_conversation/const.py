"""Constants for the Google Generative AI Conversation integration with function calling."""

import logging

DOMAIN = "google_generative_ai_conversation"
DEFAULT_NAME = "Google Generative AI Conversation"

LOGGER = logging.getLogger(__package__)

# Event that can be fired when a conversation finishes
EVENT_CONVERSATION_FINISHED = f"{DOMAIN}.conversation.finished"

# Basic Prompt Handling
CONF_PROMPT = "prompt"
DEFAULT_PROMPT = """You are a Google Generative AI agent for Home Assistant.
You can respond to user questions or requests about their smart home.
Include only necessary details in your responses.
"""

# Model Settings
CONF_CHAT_MODEL = "chat_model"
DEFAULT_CHAT_MODEL = "models/gemini-2.0-flash"

# Token & Sampling Settings
CONF_MAX_TOKENS = "max_tokens"
DEFAULT_MAX_TOKENS = 150

CONF_TOP_P = "top_p"
DEFAULT_TOP_P = 0.95

CONF_TOP_K = "top_k"
DEFAULT_TOP_K = 64

CONF_TEMPERATURE = "temperature"
DEFAULT_TEMPERATURE = 1.0

# Google-Specific Harm Block Thresholds
CONF_HARASSMENT_BLOCK_THRESHOLD = "harassment_block_threshold"
CONF_HATE_BLOCK_THRESHOLD = "hate_block_threshold"
CONF_SEXUAL_BLOCK_THRESHOLD = "sexual_block_threshold"
CONF_DANGEROUS_BLOCK_THRESHOLD = "dangerous_block_threshold"
RECOMMENDED_HARM_BLOCK_THRESHOLD = "BLOCK_MEDIUM_AND_ABOVE"

# “Recommended” mode (skips advanced fields) as in the original integration
CONF_RECOMMENDED = "recommended"

# Timeouts
TIMEOUT_MILLIS = 10000

# Function Calling
CONF_FUNCTIONS = "functions"
DEFAULT_FUNCTIONS = """[]"""

# Toggle whether the AI can request function calls
CONF_USE_FUNCTIONS = "use_functions"
DEFAULT_USE_FUNCTIONS = False

# Example of how you might define a default set of functions (similar to extended OpenAI)
DEFAULT_CONF_FUNCTIONS = [
    {
        "spec": {
            "name": "execute_services",
            "description": "Use this function to call Home Assistant services.",
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
                                    "description": "The domain (e.g., light, switch).",
                                },
                                "service": {
                                    "type": "string",
                                    "description": "The specific service to call in that domain (e.g., turn_on).",
                                },
                                "service_data": {
                                    "type": "object",
                                    "description": "Parameters for the service.",
                                    "properties": {
                                        "entity_id": {
                                            "type": "string",
                                            "description": "An entity to target (must match the domain).",
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

# Username Attachment (similar to OpenAI)
CONF_ATTACH_USERNAME = "attach_username"
DEFAULT_ATTACH_USERNAME = False

# Whether to use “tools,” if needed
CONF_USE_TOOLS = "use_tools"
DEFAULT_USE_TOOLS = False

# Limit how many times the model can request function calls in a single conversation
CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION = "max_function_calls_per_conversation"
DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION = 1

# Conversation Context/Truncation
CONF_CONTEXT_THRESHOLD = "context_threshold"
DEFAULT_CONTEXT_THRESHOLD = 13000

CONTEXT_TRUNCATE_STRATEGIES = [{"key": "clear", "label": "Clear All Messages"}]
CONF_CONTEXT_TRUNCATE_STRATEGY = "context_truncate_strategy"
DEFAULT_CONTEXT_TRUNCATE_STRATEGY = CONTEXT_TRUNCATE_STRATEGIES[0]["key"]

# Example “query_image” service
SERVICE_QUERY_IMAGE = "query_image"

# Payload Template
CONF_PAYLOAD_TEMPLATE = "payload_template"
