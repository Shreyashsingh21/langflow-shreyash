from typing import Any

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs.inputs import BoolInput
from langflow.io import DropdownInput, MessageInput, MultilineInput, SliderInput
from langflow.schema.dotdict import dotdict



class LocalLanguageModelComponent(LCModelComponent):
    display_name = "Local Language Model"
    description = "Runs one of the local chat models (GPT-OSS or Llama Local)."
    documentation: str = "https://docs.langflow.org/components-models"
    icon = "brain-circuit"
    category = "models"
    priority = 0

    inputs = [
        DropdownInput(
            name="provider",
            display_name="Model Provider",
            options=["GPT-OSS", "Llama Local"],
            value="GPT-OSS",
            info="Select the local model provider",
            real_time_refresh=True,
            options_metadata=[{"icon": "OpenAI"}, {"icon": "Ollama"}],
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            # Expose the hardcoded defaults from each implementation for clarity
            options=[
                "gpt-oss:20b",  # from gptoss.py
                "meta-llama/Llama-3.1-8B-Instruct",  # from llama.py
            ],
            value="gpt-oss:20b",
            info="Select the model preset (provider-specific)",
            real_time_refresh=True,
        ),
        MessageInput(
            name="input_value",
            display_name="Input",
            info="The input text to send to the model",
        ),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            info="A system message that helps set the behavior of the assistant",
            advanced=False,
        ),
        BoolInput(
            name="stream",
            display_name="Stream",
            info="Whether to stream the response",
            value=False,
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.7,
            info="Controls randomness in responses",
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        SliderInput(
            name="max_tokens",
            display_name="Max Tokens",
            value=512,
            info="Maximum number of tokens to generate in the response",
            range_spec=RangeSpec(min=16, max=8192, step=16),
            advanced=True,
        ),
        SliderInput(
            name="timeout",
            display_name="Request Timeout (s)",
            value=60,
            info="Timeout for API requests in seconds",
            range_spec=RangeSpec(min=5, max=300, step=5),
            advanced=True,
        ),
        BoolInput(
            name="json_mode",
            display_name="JSON Mode",
            info="If True, output JSON format response",
            value=False,
            advanced=True,
        ),
    ]

    def build_model(self) -> LanguageModel:
        provider = self.provider
        temperature = self.temperature
        stream = self.stream
        json_mode = self.json_mode
        max_tokens = self.max_tokens
        timeout = self.timeout

        if provider == "GPT-OSS":
            # Instantiate the LocalChatModel from installed OwnDeployedModels to avoid relative import resolution issues
            from langflow.components.OwnDeployedModels.gptoss import LocalChatModel as GPTOSSLocalChatModel
            return GPTOSSLocalChatModel(
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        if provider == "Llama Local":
            # Instantiate the LocalChatModel from installed OwnDeployedModels to avoid relative import resolution issues
            from langflow.components.OwnDeployedModels.llama import LocalChatModel as LlamaLocalChatModel
            return LlamaLocalChatModel(
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        msg = f"Unknown provider: {provider}"
        raise ValueError(msg)

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None) -> dotdict:
        if field_name == "provider":
            if field_value == "GPT-OSS":
                build_config["model_name"]["options"] = ["gpt-oss:20b"]
                build_config["model_name"]["value"] = "gpt-oss:20b"
            elif field_value == "Llama Local":
                build_config["model_name"]["options"] = ["meta-llama/Llama-3.1-8B-Instruct"]
                build_config["model_name"]["value"] = "meta-llama/Llama-3.1-8B-Instruct"
        return build_config


