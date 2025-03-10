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
