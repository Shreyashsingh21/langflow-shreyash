from langchain_core.tools import StructuredTool

from langflow.base.agents.agent import LCToolsAgentComponent
from langflow.base.agents.events import ExceptionWithMessageError
from langflow.base.models.model_input_constants import (
    ALL_PROVIDER_FIELDS,
)
from langflow.base.models.model_utils import get_model_name
from langflow.components.helpers.current_date import CurrentDateComponent
from langflow.components.helpers.memory import MemoryComponent
from langflow.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from langflow.custom.custom_component.component import _get_component_toolkit
from langflow.custom.utils import update_component_build_config
from langflow.field_typing import Tool
from langflow.io import BoolInput, DropdownInput, IntInput, MultilineInput, Output, FloatInput
from langflow.logging import logger
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message


def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


MODEL_PROVIDERS_LIST = [
    # Custom local models only
    "GPT-OSS (Crimson)",
    "Llama Local (Crimson)",
]


class AgentComponent(ToolCallingAgentComponent):
    display_name: str = "Self Hosted Agent"
    description: str = "Define the agent's instructions, then enter a task to complete using tools."
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "SelfHostedAgent"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    inputs = [
        DropdownInput(
            name="agent_llm",
            display_name="Model Provider",
            info="Select which local model the agent will use.",
            options=MODEL_PROVIDERS_LIST,
            value="GPT-OSS (Crimson)",
            real_time_refresh=True,
            input_types=[],
            options_metadata=[{"icon": "OpenAI"}, {"icon": "Ollama"}],
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            value=0.7,
            info="Controls randomness in responses",
            advanced=False,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            value=20000,
            info="Maximum number of tokens to generate in the response",
            advanced=False,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            value=20000,
            info="Maximum number of tokens to generate in the response",
            advanced=False,
        ),
        # Timeout is fixed to 60 seconds in code; no UI control
        # JSON mode is fixed to False in code; no UI control
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: Initial instructions and context provided to guide the agent's behavior.",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
            advanced=False,
        ),
        IntInput(
            name="n_messages",
            display_name="Number of Chat History Messages",
            value=100,
            info="Number of chat history messages to retrieve.",
            advanced=True,
            show=True,
        ),
        *LCToolsAgentComponent._base_inputs,
        # removed memory inputs from agent component
        # *memory_inputs,
        BoolInput(
            name="enable_parallel_tool_calls",
            display_name="Enable Parallel Tool Calls",
            advanced=True,
            info="If true, the agent will attempt to execute multiple tool calls concurrently when possible.",
            value=False,
        ),
        IntInput(
            name="parallel_tool_concurrency",
            display_name="Max Parallel Tools",
            value=3,
            info="Maximum number of tools to execute concurrently when parallel tool calls are enabled.",
            advanced=True,
            show=True,
        ),
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
    ]
    outputs = [Output(name="response", display_name="Response", method="message_response")]

    async def message_response(self) -> Message:
        try:
            # Get LLM model and validate
            llm_model, display_name = self.get_llm()
            if llm_model is None:
                msg = "No language model selected. Please choose a model to proceed."
                raise ValueError(msg)
            self.model_name = get_model_name(llm_model, display_name=display_name)

            # Get memory data
            self.chat_history = await self.get_memory_data()
            if isinstance(self.chat_history, Message):
                self.chat_history = [self.chat_history]

            # Add current date tool if enabled
            if self.add_current_date_tool:
                if not isinstance(self.tools, list):  # type: ignore[has-type]
                    self.tools = []
                current_date_tool = (await CurrentDateComponent(**self.get_base_args()).to_toolkit()).pop(0)
                if not isinstance(current_date_tool, StructuredTool):
                    msg = "CurrentDateComponent must be converted to a StructuredTool"
                    raise TypeError(msg)
                self.tools.append(current_date_tool)
            # note the tools are not required to run the agent, hence the validation removed.

            # Set up and run agent
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=self.system_prompt,
            )
            # Expose parallel tool execution preferences to the underlying agent/runtime
            try:
                self.parallel_tool_calls = bool(getattr(self, "enable_parallel_tool_calls", False))
                self.parallel_tool_concurrency = int(getattr(self, "parallel_tool_concurrency", 3))
            except Exception:
                self.parallel_tool_calls = False
                self.parallel_tool_concurrency = 3
            agent = self.create_agent_runnable()
            return await self.run_agent(agent)

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"{type(e).__name__}: {e!s}")
            raise
        except ExceptionWithMessageError as e:
            logger.error(f"ExceptionWithMessageError occurred: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e!s}")
            raise

    async def get_memory_data(self):
        # TODO: This is a temporary fix to avoid message duplication. We should develop a function for this.
        messages = (
            await MemoryComponent(**self.get_base_args())
            .set(session_id=self.graph.session_id, order="Ascending", n_messages=self.n_messages)
            .retrieve_messages()
        )
        return [
            message for message in messages if getattr(message, "id", None) != getattr(self.input_value, "id", None)
        ]

    def get_llm(self):
        if not isinstance(self.agent_llm, str):
            return self.agent_llm, None

        try:
            # Handle custom local models without relying on MODEL_PROVIDERS_DICT
            if self.agent_llm in ("GPT-OSS (Crimson)", "Llama Local (Crimson)"):
                # Lazy import to avoid circular deps during build
                from langflow.components.OwnDeployedModels.combine_models import (
                    LocalLanguageModelComponent as CrimsonLocalComponent,
                )
                # Map provider display name to component provider value
                provider_map = {
                    "GPT-OSS (Crimson)": "GPT-OSS",
                    "Llama Local (Crimson)": "Llama Local",
                }
                provider_value = provider_map.get(self.agent_llm, "GPT-OSS")

                # Build with safe defaults; these align with the component defaults
                # Read values from UI (fallback to defaults) and validate
                raw_temperature = getattr(self, "temperature", 0.7)
                try:
                    temperature_val = float(raw_temperature)
                except Exception:
                    temperature_val = 0.7
                # Clamp temperature to [0,1]
                temperature_val = max(0.0, min(1.0, temperature_val))

                raw_max_tokens = getattr(self, "max_tokens", 20000)
                try:
                    max_tokens_val = int(raw_max_tokens)
                except Exception:
                    max_tokens_val = 20000
                # Minimum sensible floor
                max_tokens_val = max(16, max_tokens_val)

                # Force timeout to 60 seconds regardless of UI
                timeout_val = 60

                # Force json_mode to False regardless of UI
                json_mode_val = False

                crimson_component = CrimsonLocalComponent().set(
                    provider=provider_value,
                    temperature=temperature_val,
                    max_tokens=max_tokens_val,
                    timeout=timeout_val,
                    json_mode=json_mode_val,
                )
                # Indicate whether values came from the UI (user) or defaults
                def src(name: str, default_marker: str = "default") -> str:
                    try:
                        return "user" if (name in getattr(self, "__dict__", {})) else default_marker
                    except Exception:
                        return default_marker
                try:
                    print(
                        f"[SelfHostedAgent] Initializing {self.agent_llm} (provider={provider_value}) "
                        f"with temperature={temperature_val} ({src('temperature')}), "
                        f"max_tokens={max_tokens_val} ({src('max_tokens')}), "
                        f"timeout={timeout_val} (fixed), "
                        f"json_mode={json_mode_val} (fixed)"
                    )
                except Exception:
                    pass
                return crimson_component.build_model(), self.agent_llm

            msg = f"Invalid model provider: {self.agent_llm}"
            raise ValueError(msg)

        except Exception as e:
            logger.error(f"Error building {self.agent_llm} language model: {e!s}")
            msg = f"Failed to initialize language model: {e!s}"
            raise ValueError(msg) from e

    def _build_llm_model(self, component, inputs, prefix=""):
        # Deprecated path for built-in providers. Not used for custom-only setup.
        return component.set().build_model()

    def delete_fields(self, build_config: dotdict, fields: dict | list[str]) -> None:
        """Delete specified fields from build_config."""
        for field in fields:
            build_config.pop(field, None)

    def update_input_types(self, build_config: dotdict) -> dotdict:
        """Update input types for all fields in build_config."""
        for key, value in build_config.items():
            if isinstance(value, dict):
                if value.get("input_types") is None:
                    build_config[key]["input_types"] = []
            elif hasattr(value, "input_types") and value.input_types is None:
                value.input_types = []
        return build_config

    async def update_build_config(
        self, build_config: dotdict, field_value: str, field_name: str | None = None
    ) -> dotdict:
        # Custom-only: just propagate the selected provider value and keep inputs stable
        if field_name in ("agent_llm",):
            if "agent_llm" in build_config:
                build_config["agent_llm"]["value"] = field_value
                build_config["agent_llm"]["input_types"] = []
        # Ensure required keys exist minimally
        for key in [
            "agent_llm",
            "tools",
            "input_value",
            "add_current_date_tool",
            "system_prompt",
            "enable_parallel_tool_calls",
            "parallel_tool_concurrency",
        ]:
            if key not in build_config:
                continue
        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})

    async def _get_tools(self) -> list[Tool]:
        component_toolkit = _get_component_toolkit()
        tools_names = self._build_tools_names()
        agent_description = self.get_tool_description()
        # TODO: Agent Description Depreciated Feature to be removed
        description = f"{agent_description}{tools_names}"
        tools = component_toolkit(component=self).get_tools(
            tool_name="Call_Agent", tool_description=description, callbacks=self.get_langchain_callbacks()
        )
        if hasattr(self, "tools_metadata"):
            tools = component_toolkit(component=self, metadata=self.tools_metadata).update_tools_metadata(tools=tools)
        return tools
