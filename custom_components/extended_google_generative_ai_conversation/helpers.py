
# The built-in 'abc' module provides infrastructure for
# defining abstract base classes (ABC) and methods (abstractmethod).
from abc import ABC, abstractmethod

# 'timedelta' is used for date/time manipulations.
from datetime import timedelta

# 'partial' allows partial application of functions, capturing some arguments beforehand.
from functools import partial

# 'logging' is for creating log messages in different severity levels.
import logging

# 'os' allows interaction with the operating system, such as file paths.
import os

# 're' provides regular expression matching.
import re

# 'sqlite3' is the built-in database interface for SQLite in Python.
import sqlite3

# 'time' offers time-related functions, such as getting the current time in seconds.
import time

# 'typing' offers type hints like 'Any' to annotate variable/argument types.
from typing import Any

# 'parse' from urllib is used to handle URL parsing operations.
from urllib import parse

# 'BeautifulSoup' from bs4 is used to parse HTML and XML documents.
from bs4 import BeautifulSoup

# The 'openai' package provides functionalities to interact with OpenAI services.
# Here we import asynchronous classes for Azure OpenAI and OpenAI usage.
from openai import AsyncAzureOpenAI, AsyncOpenAI

# 'voluptuous' (vol) is a Python data validation library used to define schemas and validate data.
import voluptuous as vol

# 'yaml' helps in reading and writing YAML files.
import yaml

# These imports are from the Home Assistant framework:
#   - automation: for handling automations
#   - conversation: for managing conversation-based features
#   - energy: for energy management
#   - recorder: for recording state changes
#   - rest: for REST-based functionalities
#   - scrape: for scraping web pages
from homeassistant.components import (
    automation,
    conversation,
    energy,
    recorder,
    rest,
    scrape,
)

# '_async_validate_config_item' is used internally by Home Assistant to validate automation configuration entries.
from homeassistant.components.automation.config import _async_validate_config_item

# 'SCRIPT_ENTITY_SCHEMA' outlines the allowed structure of script entities.
from homeassistant.components.script.config import SCRIPT_ENTITY_SCHEMA

# 'AUTOMATION_CONFIG_PATH' is the file path where Home Assistant stores automations.
from homeassistant.config import AUTOMATION_CONFIG_PATH

# Various constants from Home Assistant.
from homeassistant.const import (
    CONF_ATTRIBUTE,
    CONF_METHOD,
    CONF_NAME,
    CONF_PAYLOAD,
    CONF_RESOURCE,
    CONF_RESOURCE_TEMPLATE,
    CONF_TIMEOUT,
    CONF_VALUE_TEMPLATE,
    CONF_VERIFY_SSL,
    SERVICE_RELOAD,
)

# Core functionalities and classes from Home Assistant:
#   - HomeAssistant: the main class representing the instance
#   - State: used to handle entity state
from homeassistant.core import HomeAssistant, State

# Custom exception classes that Home Assistant defines for specific error handling.
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound

# Helpers for data validation, HTTP client usage, and scripting within Home Assistant.
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.script import Script
from homeassistant.helpers.template import Template

# 'dt_util' provides date/time utilities that integrate with Home Assistant's time zones.
import homeassistant.util.dt as dt_util

# These constants and exceptions are specific to the custom integration or module:
from .const import CONF_PAYLOAD_TEMPLATE, DOMAIN, EVENT_AUTOMATION_REGISTERED
from .exceptions import (
    CallServiceError,
    EntityNotExposed,
    EntityNotFound,
    FunctionNotFound,
    InvalidFunction,
    NativeNotFound,
)

# Prepare a logger instance for this module.
_LOGGER = logging.getLogger(__name__)

# Regular expression pattern used to detect Azure domain in a URL.
AZURE_DOMAIN_PATTERN = r"\.(openai\.azure\.com|azure-api\.net)"

# The following function retrieves a function executor from a predefined dictionary.
# If the requested function type does not exist, it raises a 'FunctionNotFound' error.
def get_function_executor(value: str):
    function_executor = FUNCTION_EXECUTORS.get(value)
    if function_executor is None:
        raise FunctionNotFound(value)
    return function_executor

# Checks if the provided base_url matches the Azure domain pattern above.
def is_azure(base_url: str):
    if base_url and re.search(AZURE_DOMAIN_PATTERN, base_url):
        return True
    return False

# Converts certain keys in a dictionary or list structure into Home Assistant Templates,
# if they match certain template key names. Useful for dynamically rendering fields.
def convert_to_template(
    settings,
    template_keys=["data", "event_data", "target", "service"],
    hass: HomeAssistant | None = None,
):
    _convert_to_template(settings, template_keys, hass, [])

# Helper function to recursively convert items to Template objects if they match the criteria.
def _convert_to_template(settings, template_keys, hass, parents: list[str]):
    if isinstance(settings, dict):
        for key, value in settings.items():
            if isinstance(value, str) and (
                key in template_keys or set(parents).intersection(template_keys)
            ):
                settings[key] = Template(value, hass)
            if isinstance(value, dict):
                parents.append(key)
                _convert_to_template(value, template_keys, hass, parents)
                parents.pop()
            if isinstance(value, list):
                parents.append(key)
                for item in value:
                    _convert_to_template(item, template_keys, hass, parents)
                parents.pop()
    if isinstance(settings, list):
        for setting in settings:
            _convert_to_template(setting, template_keys, hass, parents)

# Prepares and returns REST data by applying templates and merging them back into the configuration.
def _get_rest_data(hass, rest_config, arguments):
    # Default values for REST configuration if not specified.
    rest_config.setdefault(CONF_METHOD, rest.const.DEFAULT_METHOD)
    rest_config.setdefault(CONF_VERIFY_SSL, rest.const.DEFAULT_VERIFY_SSL)
    rest_config.setdefault(CONF_TIMEOUT, rest.data.DEFAULT_TIMEOUT)
    rest_config.setdefault(rest.const.CONF_ENCODING, rest.const.DEFAULT_ENCODING)

    # If a resource_template is present, render it and store it as 'CONF_RESOURCE'.
    resource_template: Template | None = rest_config.get(CONF_RESOURCE_TEMPLATE)
    if resource_template is not None:
        rest_config.pop(CONF_RESOURCE_TEMPLATE)
        rest_config[CONF_RESOURCE] = resource_template.async_render(
            arguments, parse_result=False
        )

    # If a payload_template is present, render it and store it as 'CONF_PAYLOAD'.
    payload_template: Template | None = rest_config.get(CONF_PAYLOAD_TEMPLATE)
    if payload_template is not None:
        rest_config.pop(CONF_PAYLOAD_TEMPLATE)
        rest_config[CONF_PAYLOAD] = payload_template.async_render(
            arguments, parse_result=False
        )

    # Create and return a RestData instance based on the configuration.
    return rest.create_rest_data_from_config(hass, rest_config)

# Validates the provided OpenAI or Azure OpenAI credentials by attempting to list available models.
# If skip_authentication is True, the check is bypassed.
async def validate_authentication(
    hass: HomeAssistant,
    api_key: str,
    base_url: str,
    api_version: str,
    organization: str = None,
    skip_authentication=False,
) -> None:
    if skip_authentication:
        return

    if is_azure(base_url):
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
            organization=organization,
            http_client=get_async_client(hass),
        )
    else:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            http_client=get_async_client(hass),
        )

    await hass.async_add_executor_job(partial(client.models.list, timeout=10))

# Abstract base class for function executors. 
# Each executor must define the 'execute' method.
class FunctionExecutor(ABC):
    def __init__(self, data_schema=vol.Schema({})) -> None:
        """initialize function executor"""
        # Each executor requires a base schema. A 'type' key is required for identification.
        self.data_schema = data_schema.extend({vol.Required("type"): str})
   
    # Converts incoming arguments to a validated schema form, or raises an error if invalid.
    def to_arguments(self, arguments):
        """to_arguments function"""
        try:
            return self.data_schema(arguments)
        except vol.error.Error as e:
            # Attempt to figure out which function type it wa
            function_type = next(
                (key for key, value in FUNCTION_EXECUTORS.items() if value == self),
                None,
            )
            raise InvalidFunction(function_type) from e

    # Verifies that provided entity IDs exist in Home Assistant and are among the exposed ones.
    def validate_entity_ids(self, hass: HomeAssistant, entity_ids, exposed_entities):
        if any(hass.states.get(entity_id) is None for entity_id in entity_ids):
            raise EntityNotFound(entity_ids)
        exposed_entity_ids = map(lambda e: e["entity_id"], exposed_entities)
        if not set(entity_ids).issubset(exposed_entity_ids):
            raise EntityNotExposed(entity_ids)

    @abstractmethod
    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        # Each executor must implement its own logic in this method.
        """execute function"""

# Handles "native" function calls that operate directly on Home Assistant entities
# or system-level features (e.g., calling services, adding automations, retrieving history).
class NativeFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize native function"""
        # Native executor requires at least a 'name' in the arguments.
        super().__init__(vol.Schema({vol.Required("name"): str}))

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        # Routes to different internal methods based on the 'name' field.
        name = function["name"]
        if name == "execute_service":
            return await self.execute_service(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "execute_service_single":
            return await self.execute_service_single(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "add_automation":
            return await self.add_automation(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "get_history":
            return await self.get_history(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "get_energy":
            return await self.get_energy(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "get_statistics":
            return await self.get_statistics(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "get_user_from_user_id":
            return await self.get_user_from_user_id(
                hass, function, arguments, user_input, exposed_entities
            )

        # If the requested native method is not recognized, raise an error.
        raise NativeNotFound(name)

    # Executes a single Home Assistant service call, including entity or area IDs if present.
    # The key to the service of controlling single device.
    async def execute_service_single(
        self,
        hass: HomeAssistant,
        function,
        service_argument,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        domain = service_argument["domain"]
        service = service_argument["service"]
        service_data = service_argument.get(
            "service_data", service_argument.get("data", {})
        )
        entity_id = service_data.get("entity_id", service_argument.get("entity_id"))
        area_id = service_data.get("area_id")
        device_id = service_data.get("device_id")

        # Convert entity_id from string to a list, if needed.
        if isinstance(entity_id, str):
            entity_id = [e.strip() for e in entity_id.split(",")]
        service_data["entity_id"] = entity_id

        # Require at least one targeting method (entity_id, area_id, device_id).
        if entity_id is None and area_id is None and device_id is None:
            raise CallServiceError(domain, service, service_data)
        # Ensure the service itself exists.
        if not hass.services.has_service(domain, service):
            raise ServiceNotFound(domain, service)
        # Validate that entity IDs are exposed.
        self.validate_entity_ids(hass, entity_id or [], exposed_entities)

        # Attempt the service call; handle any Home Assistant errors.
        try:
            await hass.services.async_call(
                domain=domain,
                service=service,
                service_data=service_data,
            )
            return {"success": True}
        except HomeAssistantError as e:
            _LOGGER.error(e)
            return {"error": str(e)}

    # Executes multiple service calls by iterating over a list of service definitions.
    # The key to the service of controlling devices.
    async def execute_service(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        result = []
        for service_argument in arguments.get("list", []):
            result.append(
                await self.execute_service_single(
                    hass, function, service_argument, user_input, exposed_entities
                )
            )
        return result

    # Adds a new automation to the Home Assistant configuration file and triggers a reload.
    async def add_automation(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        # Load the automation YAML provided in the argument.
        automation_config = yaml.safe_load(arguments["automation_config"])
        config = {"id": str(round(time.time() * 1000))}

        # Merge the loaded config with a generated ID.
        if isinstance(automation_config, list):
            config.update(automation_config[0])
        if isinstance(automation_config, dict):
            config.update(automation_config)

        # Validate the final automation configuration.
        await _async_validate_config_item(hass, config, True, False)

        automations = [config]
        # Read the existing automations file.
        with open(
            os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH),
            "r",
            encoding="utf-8",
        ) as f:
            current_automations = yaml.safe_load(f.read())

        # Append (or create) the new automation entry.
        with open(
            os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH),
            "a" if current_automations else "w",
            encoding="utf-8",
        ) as f:
            raw_config = yaml.dump(automations, allow_unicode=True, sort_keys=False)
            f.write("\n" + raw_config)

        # Reload the automation integration to apply changes immediately.
        await hass.services.async_call(automation.config.DOMAIN, SERVICE_RELOAD)

        # Fire an event indicating a new automation was registered.
        hass.bus.async_fire(
            EVENT_AUTOMATION_REGISTERED,
            {"automation_config": config, "raw_config": raw_config},
        )
        return "Success"

    # Retrieves historical state data for specified entity IDs in a given time range.
    async def get_history(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        start_time = arguments.get("start_time")
        end_time = arguments.get("end_time")
        entity_ids = arguments.get("entity_ids", [])
        include_start_time_state = arguments.get("include_start_time_state", True)
        significant_changes_only = arguments.get("significant_changes_only", True)
        minimal_response = arguments.get("minimal_response", True)
        no_attributes = arguments.get("no_attributes", True)

        now = dt_util.utcnow()
        one_day = timedelta(days=1)
        # Determine a valid UTC range for the history request.
        start_time = self.as_utc(start_time, now - one_day, "start_time not valid")
        end_time = self.as_utc(end_time, start_time + one_day, "end_time not valid")

        # Ensure the requested entity IDs are exposed.
        self.validate_entity_ids(hass, entity_ids, exposed_entities)

        # Use the recorder's session to retrieve states.
        with recorder.util.session_scope(hass=hass, read_only=True) as session:
            result = await recorder.get_instance(hass).async_add_executor_job(
                recorder.history.get_significant_states_with_session,
                hass,
                session,
                start_time,
                end_time,
                entity_ids,
                None,
                include_start_time_state,
                significant_changes_only,
                minimal_response,
                no_attributes,
            )
            
        # The result is a dict keyed by entity_id, each containing a list of states.
        # Convert each list of State objects or dicts to a list of dicts.
        return [[self.as_dict(item) for item in sublist] for sublist in result.values()]

    # Returns the energy management data from the system.
    async def get_energy(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        energy_manager: energy.data.EnergyManager = await energy.async_get_manager(hass)
        return energy_manager.data

    # Retrieves user details by the user_id present in the conversation input.
    async def get_user_from_user_id(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        user = await hass.auth.async_get_user(user_input.context.user_id)
        return {'name': user.name if user and hasattr(user, 'name') else 'Unknown'}

    # Returns statistical data for specific entity IDs in a given time range.
    async def get_statistics(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        statistic_ids = arguments.get("statistic_ids", [])
        start_time = dt_util.as_utc(dt_util.parse_datetime(arguments["start_time"]))
        end_time = dt_util.as_utc(dt_util.parse_datetime(arguments["end_time"]))

        return await recorder.get_instance(hass).async_add_executor_job(
            recorder.statistics.statistics_during_period,
            hass,
            start_time,
            end_time,
            statistic_ids,
            arguments.get("period", "day"),
            arguments.get("units"),
            arguments.get("types", {"change"}),
        )

    # Helper method to convert a string-based datetime to UTC or return a default if not valid.
    def as_utc(self, value: str, default_value, parse_error_message: str):
        if value is None:
            return default_value

        parsed_datetime = dt_util.parse_datetime(value)
        if parsed_datetime is None:
            raise HomeAssistantError(parse_error_message)

        return dt_util.as_utc(parsed_datetime)

    # Converts a State object or dict into a dictionary for easier consumption.
    def as_dict(self, state: State | dict[str, Any]):
        if isinstance(state, State):
            return state.as_dict()
        return state


# Handles script-based function calls. Script sequences in Home Assistant are similar to automations,
# but can be called like standalone routines.
class ScriptFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize script function"""
        # This executor requires arguments that conform to SCRIPT_ENTITY_SCHEMA.
        super().__init__(SCRIPT_ENTITY_SCHEMA)

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        # Creates a Script object using the provided sequence.
        script = Script(
            hass,
            function["sequence"],
            "extended_openai_conversation",
            DOMAIN,
            running_description="[extended_openai_conversation] function",
            logger=_LOGGER,
        )

        # Runs the script and returns any result saved in _function_result.
        result = await script.async_run(
            run_variables=arguments, context=user_input.context
        )
        return result.variables.get("_function_result", "Success")

# Handles rendering values through Home Assistant templates.
class TemplateFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize template function"""
        super().__init__(
            vol.Schema(
                {
                    vol.Required("value_template"): cv.template,
                    vol.Optional("parse_result"): bool,
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        # Renders the template with the possibility of interpreting the result as JSON.
        return function["value_template"].async_render(
            arguments,
            parse_result=function.get("parse_result", False),
        )

# Handles REST calls, similar to the standard Home Assistant REST sensor.
class RestFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize Rest function"""
        super().__init__(
            vol.Schema(rest.RESOURCE_SCHEMA).extend(
                {
                    vol.Optional("value_template"): cv.template,
                    vol.Optional("payload_template"): cv.template,
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        # Prepare the REST data and send the request.
        config = function
        rest_data = _get_rest_data(hass, config, arguments)

        # Fetch the content from the remote resource.
        await rest_data.async_update()
        value = rest_data.data_without_xml()
        value_template = config.get(CONF_VALUE_TEMPLATE)

        # If there's a value_template, render the fetched data through it.
        if value is not None and value_template is not None:
            value = value_template.async_render_with_possible_json_value(
                value, None, arguments
            )

        return value

# Handles web scraping by using the 'scrape' component logic, including HTML parsing via BeautifulSoup.
class ScrapeFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize Scrape function"""
        # Merges the scrape sensor schema with optional template fields.
        super().__init__(
            scrape.COMBINED_SCHEMA.extend(
                {
                    vol.Optional("value_template"): cv.template,
                    vol.Optional("payload_template"): cv.template,
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        config = function
        
        # Prepare the REST data for scraping (this reuses the REST logic).
        rest_data = _get_rest_data(hass, config, arguments)

        # Create a coordinator object to refresh the data from the remote page.
        coordinator = scrape.coordinator.ScrapeCoordinator(
            hass,
            rest_data,
            scrape.const.DEFAULT_SCAN_INTERVAL,
        )
        await coordinator.async_config_entry_first_refresh()

        new_arguments = dict(arguments)

        # For each sensor definition, parse out the content from the HTML and store it.
        for sensor_config in config["sensor"]:
            name: Template = sensor_config.get(CONF_NAME)
            value = self._async_update_from_rest_data(
                coordinator.data, sensor_config, arguments
            )
            new_arguments["value"] = value
            if name:
                new_arguments[name.async_render()] = value

        # Optionally apply a top-level value_template if present.
        result = new_arguments["value"]
        value_template = config.get(CONF_VALUE_TEMPLATE)

        if value_template is not None:
            result = value_template.async_render_with_possible_json_value(
                result, None, new_arguments
            )

        return result

    # Extracts content from the HTML data based on the sensor's configuration.
    def _async_update_from_rest_data(
        self,
        data: BeautifulSoup,
        sensor_config: dict[str, Any],
        arguments: dict[str, Any],
    ) -> None:
        """Update state from the rest data."""
        value = self._extract_value(data, sensor_config)
        value_template = sensor_config.get(CONF_VALUE_TEMPLATE)

        if value_template is not None:
            value = value_template.async_render_with_possible_json_value(
                value, None, arguments
            )

        return value

     # Locates the text or attribute from the BeautifulSoup object based on CSS selectors.
    def _extract_value(self, data: BeautifulSoup, sensor_config: dict[str, Any]) -> Any:
        """Parse the html extraction in the executor."""
        value: str | list[str] | None
        select = sensor_config[scrape.const.CONF_SELECT]
        index = sensor_config.get(scrape.const.CONF_INDEX, 0)
        attr = sensor_config.get(CONF_ATTRIBUTE)
        try:
            if attr is not None:
                value = data.select(select)[index][attr]
            else:
                tag = data.select(select)[index]
                if tag.name in ("style", "script", "template"):
                    value = tag.string
                else:
                    value = tag.text
        except IndexError:
            _LOGGER.warning("Index '%s' not found", index)
            value = None
        except KeyError:
            _LOGGER.warning("Attribute '%s' not found", attr)
            value = None
        _LOGGER.debug("Parsed value: %s", value)
        return value


# Composes multiple sub-executors into a single sequence. Each step can store its result 
# in a shared variable for subsequent steps to use.
class CompositeFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize composite function"""
        # Expects a list of function calls under 'sequence'. Each must match a known function schema.
        super().__init__(
            vol.Schema(
                {
                    vol.Required("sequence"): vol.All(
                        cv.ensure_list, [self.function_schema]
                    )
                }
            )
        )

     # Validates that each function in the sequence is known and has valid parameters.
    def function_schema(self, value: Any) -> dict:
        """Validate a composite function schema."""
        if not isinstance(value, dict):
            raise vol.Invalid("expected dictionary")

        composite_schema = {vol.Optional("response_variable"): str}
        function_executor = get_function_executor(value["type"])

        return function_executor.data_schema.extend(composite_schema)(value)

    # Executes each function in the sequence in order. 
    # Results can be stored in 'arguments' under a response_variable if specified.
    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        config = function
        sequence = config["sequence"]

        for executor_config in sequence:
            function_executor = get_function_executor(executor_config["type"])
            result = await function_executor.execute(
                hass, executor_config, arguments, user_input, exposed_entities
            )

            # If there's a response_variable, store the function result in 'arguments'.
            response_variable = executor_config.get("response_variable")
            if response_variable:
                arguments[response_variable] = result

        # Return the last function's result.
        return result


# Allows read-only queries on a local or remote SQLite database. 
# It supports rendering the query via Home Assistant templates, and ensures the database is accessed read-only.
class SqliteFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize sqlite function"""
        # Expects a 'query', 'db_url', and optionally 'single' (to fetch only the first row).
        super().__init__(
            vol.Schema(
                {
                    vol.Optional("query"): str,
                    vol.Optional("db_url"): str,
                    vol.Optional("single"): bool,
                }
            )
        )

    # Checks if an entity is in the list of exposed entities.
    def is_exposed(self, entity_id, exposed_entities) -> bool:
        return any(
            exposed_entity["entity_id"] == entity_id
            for exposed_entity in exposed_entities
        )

    # Looks for any exposed entity IDs inside the query string as a simple check for security or usage.
    def is_exposed_entity_in_query(self, query: str, exposed_entities) -> bool:
        exposed_entity_ids = list(
            map(lambda e: f"'{e['entity_id']}'", exposed_entities)
        )
        return any(
            exposed_entity_id in query for exposed_entity_id in exposed_entity_ids
        )

    # Helper to raise a HomeAssistantError if something goes wrong in the template logic.
    def raise_error(self, msg="Unexpected error occurred."):
        raise HomeAssistantError(msg)

    # Gets the default read-only path to Home Assistant's internal database file.
    def get_default_db_url(self, hass: HomeAssistant) -> str:
        db_file_path = os.path.join(hass.config.config_dir, recorder.DEFAULT_DB_FILE)
        return f"file:{db_file_path}?mode=ro"

    # Ensures the URL is set to read-only mode by adding or overriding 'mode=ro'.
    def set_url_read_only(self, url: str) -> str:
        scheme, netloc, path, query_string, fragment = parse.urlsplit(url)
        query_params = parse.parse_qs(query_string)

        # Force read-only mode.
        query_params["mode"] = ["ro"]
        new_query_string = parse.urlencode(query_params, doseq=True)

        return parse.urlunsplit((scheme, netloc, path, new_query_string, fragment))

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        # Retrieve the database URL, defaulting to Home Assistant's recorder DB if missing.
        db_url = self.set_url_read_only(
            function.get("db_url", self.get_default_db_url(hass))
        )
        query = function.get("query", "{{query}}")

        template_arguments = {
            "is_exposed": lambda e: self.is_exposed(e, exposed_entities),
            "is_exposed_entity_in_query": lambda q: self.is_exposed_entity_in_query(
                q, exposed_entities
            ),
            "exposed_entities": exposed_entities,
            "raise": self.raise_error,
        }
        template_arguments.update(arguments)

        # Render the SQL query with the provided or default template.
        q = Template(query, hass).async_render(template_arguments)
        _LOGGER.info("Rendered query: %s", q)

        # Connect to the SQLite database in read-only mode and execute the query.
        with sqlite3.connect(db_url, uri=True) as conn:
            cursor = conn.cursor().execute(q)
            names = [description[0] for description in cursor.description]

            # If 'single' is True, fetch the first row and return it as a dict.
            if function.get("single") is True:
                row = cursor.fetchone()
                return {name: val for name, val in zip(names, row)}

            # Otherwise, fetch all rows and return a list of dicts.
            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append({name: val for name, val in zip(names, row)})
            return result

# A dictionary linking function types to their respective executor classes.
FUNCTION_EXECUTORS: dict[str, FunctionExecutor] = {
    "native": NativeFunctionExecutor(),
    "script": ScriptFunctionExecutor(),
    "template": TemplateFunctionExecutor(),
    "rest": RestFunctionExecutor(),
    "scrape": ScrapeFunctionExecutor(),
    "composite": CompositeFunctionExecutor(),
    "sqlite": SqliteFunctionExecutor(),
}
