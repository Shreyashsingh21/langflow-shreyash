from __future__ import annotations

from math import ceil

from langflow.custom.custom_component.component import Component
from langflow.io import DropdownInput, MultilineInput, Output


def _estimate_tokens_bytes(text: str) -> int:
    try:
        byte_len = len(text.encode("utf-8"))
    except Exception:
        byte_len = len(text)
    # Heuristic: ~4 bytes per token for most English text
    return max(0, ceil(byte_len / 4))


def _count_tokens(text: str, model: str | None = None) -> int:
    # Prefer tiktoken if available; otherwise fallback to byte heuristic
    try:
        import tiktoken  # type: ignore

        # Map common model aliases to encodings
        enc = None
        try:
            if model:
                enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = None
        if enc is None:
            # Default to cl100k_base which covers GPT-4/3.5-ish
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return _estimate_tokens_bytes(text or "")


class TokenCountToolComponent(Component):
    display_name: str = "Token Count Tool"
    description: str = "Calculate approximate token count for a given text. Uses tiktoken if available, otherwise a byte-length heuristic."
    icon = "hash"
    category = "tools"
    priority = 0
    category = "tools"
    priority = 0

    inputs = [
        MultilineInput(
            name="text",
            display_name="Text",
            info="Text to analyze",
            value="",
        ),
        DropdownInput(
            name="model",
            display_name="Tokenizer Model (optional)",
            options=[
                "gpt-4o",
                "gpt-4",
                "gpt-3.5-turbo",
                "gpt-oss:20b",
                "meta-llama/Llama-3.1-8B-Instruct",
            ],
            value=None,
            info="If set and tiktoken supports it, will use that tokenizer.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(name="count", display_name="Token Count", method="count"),
    ]

    async def count(self) -> str:  # noqa: D401
        """Return the token count for provided inputs as a string."""
        try:
            text_value = getattr(self, "text", "")
            model_value = getattr(self, "model", None)
        except Exception:
            text_value = ""
            model_value = None
        count_value = _count_tokens(text=str(text_value or ""), model=str(model_value) if model_value else None)
        return str(count_value)


