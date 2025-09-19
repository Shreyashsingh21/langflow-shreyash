import requests
import os
import json
from typing import Any, List, Dict, Optional, Union, Iterator, AsyncIterator
from typing_extensions import override
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.tools import BaseTool

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.inputs import BoolInput, DictInput, StrInput, MultilineInput, IntInput, FloatInput

# Safety import to ensure openai_constants is available during flow building
try:
    import langflow.base.models.openai_constants
except ImportError:
    pass  # Ignore if not available

API_BASE_URL = os.environ.get("GPTOSS_API_BASE", "https://gptoss-api.trinka.ai/gpt-oss-20B_chat")
MODEL_NAME_HARDCODED = os.environ.get("GPTOSS_MODEL", "gpt-oss:20b")


# Custom LangChain Chat Model Wrapper for Local LLM
class LocalChatModel(BaseChatModel):
    """Custom chat model that wraps the local Llama API for agent and tool use."""

    api_base: str = API_BASE_URL
    model_name: str = MODEL_NAME_HARDCODED
    json_mode: bool = False
    temperature: float = 0.3  # Lower temperature for faster, more deterministic responses
    max_tokens: int = 256  # Reduced for faster responses
    timeout: int = 60  # Allow slower local servers
    _session: Optional[requests.Session] = None
    def _sanitize_final_content(self, content: str) -> str:
        """Remove leaked tool-call JSON and collapse duplicate consecutive lines, preserving code fences."""
        try:
            import re
            text = content or ""
            # Preserve fenced code blocks; only strip inline tool-call JSON blobs
            text = re.sub(r"\{\s*\"actions\"\s*:\s*\[[\s\S]*?\]\s*\}", "", text)
            text = re.sub(r"\{\s*\"action\"\s*:\s*\"[^\"]+\"[\s\S]*?\}", "", text)
            lines = text.splitlines()
            deduped = []
            prev = None
            for ln in lines:
                if ln != prev:
                    deduped.append(ln)
                prev = ln
            cleaned = "\n".join(deduped).strip()
            return cleaned or (content.strip() if content else "")
        except Exception:
            return content
    
    def _debug(self, message: str, data: Optional[Any] = None) -> None:
        try:
            if os.environ.get("GOTOSS_DEBUG", "0") == "1":
                prefix = "[GOTOSS DEBUG]"
                if data is None:
                    print(f"{prefix} {message}")
                else:
                    # Render small JSON safely
                    try:
                        rendered = data
                        if not isinstance(rendered, str):
                            rendered = json.dumps(rendered, ensure_ascii=False)[:2000]
                        print(f"{prefix} {message}: {rendered}")
                    except Exception:
                        print(f"{prefix} {message}: <unserializable>")
        except Exception:
            pass
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response from the local LLM."""
        
        # Convert LangChain messages to API format
        api_messages = self._convert_messages_to_api_format(messages)
        # Build minimal variants of messages for stricter providers
        minimal_messages: List[Dict[str, str]] = []
        system_texts: List[str] = []
        try:
            for m in api_messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    system_texts.append(content)
                    continue
                if role not in ("user", "assistant"):
                    role = "user"
                minimal_messages.append({"role": role, "content": content})
        except Exception:
            minimal_messages = api_messages

        # Optional: strict mode to avoid injecting extra system/tool instructions
        strict_messages = os.environ.get("GPTOSS_STRICT_MESSAGES", "0") == "1"
        
        # Add a concise system instruction that still allows comprehensive, multi-tool answers
        general_instruction = {
            "role": "system",
            "content": """You are a helpful AI assistant. Use tools when needed to answer accurately and comprehensively. You may call multiple tools and iterate when useful. Synthesize information from all tools into a clear final answer. Do not mention tool names in final answers."""
        }
        
        # Insert general instruction at the beginning unless strict
        if not strict_messages:
            api_messages.insert(0, general_instruction)
        
        # Add tool information if tools are bound
        if (not strict_messages) and hasattr(self, '_tools') and self._tools:
            tool_descriptions = self._format_tools_for_prompt()
            if tool_descriptions:
                # Add tool information to the system message or as a separate message
                tool_message = {
                    "role": "system", 
                    "content": f"""Tools available:
{tool_descriptions}

TOOL USE INSTRUCTIONS:
1. Single call: {{"action": "tool_name", "action_input": {{"param": "value"}}}}
2. Batch multiple calls: {{"actions": [{{"action": "toolA", "action_input": {{...}}}}, {{"action": "toolB", "action_input": {{...}}}}]}}
3. Use the EXACT parameter names listed in each tool's Parameters.
4. Required parameters MUST be provided; optional can be omitted.
5. Analyze results and call additional tools if needed for complete information.
6. Final answer must be clean prose (no JSON), synthesizing all tool results.

PARAMETER MATCHING RULES:
- If you see "input_value" in parameters → use "input_value"
- If you see "search_query" in parameters → use "search_query"
- If you see "url" in parameters → use "url"
- Always match the exact parameter name from the tool description

EXAMPLE WORKFLOW:
1. User asks question → If multiple independent lookups: {{"actions": [ ... ]}}
2. Tools return results → Analyze results and, if needed, call more tools
3. When sufficient information is gathered → Provide a comprehensive final answer."""
                }
                # Insert tool message at the beginning (after any existing system message)
                if api_messages and api_messages[0]["role"] == "system":
                    api_messages.insert(1, tool_message)
                else:
                    api_messages.insert(0, tool_message)
        
        # Build headers with API key support
        api_key = os.environ.get("GPTOSS_API_KEY") or os.environ.get("TRINKA_API_KEY")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            # Try both common auth header styles
            headers["Authorization"] = f"Bearer {api_key}"
            headers["x-api-key"] = api_key

        # Prepare multiple payload variants to handle provider schema quirks
        payload_variants: List[Dict[str, Any]] = []

        # Variant A: OpenAI-like (default)
        payload_variants.append({
            "model": self.model_name,
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop or [],
            "stream": False,
        })

        # Variant B: same as A but omit model (some providers fix model server-side)
        payload_variants.append({
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop or [],
            "stream": False,
        })

        # Variant C: alternative keys used by some backends
        payload_variants.append({
            "messages": api_messages,
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "stream": False,
        })

        # Variant D: generic "input" wrapper
        payload_variants.append({
            "input": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        })

        # Variant E: minimal messages only (drop system)
        payload_variants.append({
            "messages": minimal_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        })

        # Variant F: minimal messages with model
        payload_variants.append({
            "model": self.model_name,
            "messages": minimal_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        })

        # Variant G: prompt string (concatenate)
        try:
            prompt_lines: List[str] = []
            if system_texts:
                prompt_lines.append("\n\n".join(system_texts))
            for m in minimal_messages:
                prompt_lines.append(f"{m.get('role','user')}: {m.get('content','')}")
            prompt_text = "\n".join(prompt_lines).strip()
        except Exception:
            prompt_text = ""
        if prompt_text:
            payload_variants.append({
                "prompt": prompt_text,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            })

        # Variant H: inputs schema used by some inference servers
        if prompt_text:
            payload_variants.append({
                "inputs": {
                    "past_user_inputs": [],
                    "generated_responses": [],
                    "text": prompt_text,
                },
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                }
            })
        
        try:
            # Use session with connection pooling for better performance
            if self._session is None:
                self._session = requests.Session()
                # Configure retry strategy for better reliability
                retry_strategy = Retry(
                    total=2,  # Only 2 retries for speed
                    backoff_factor=0.1,  # Very short backoff
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=1, pool_maxsize=1)
                self._session.mount("http://", adapter)
                self._session.mount("https://", adapter)
            
            # Try each payload variant until one succeeds (non-4xx or specifically not 422)
            request_timeout = (10, self.timeout)
            last_response = None
            last_error_body: Optional[str] = None
            for idx, payload in enumerate(payload_variants):
                # Log outgoing request (without auth header)
                self._debug("Requesting completion", {"url": self.api_base, "variant": idx, "payload": payload})
                try:
                    response = self._session.post(self.api_base, headers=headers, json=payload, timeout=request_timeout)
                except requests.exceptions.ReadTimeout:
                    # Retry once with reduced tokens and extended timeout
                    reduced_payload = dict(payload)
                    if "max_tokens" in reduced_payload:
                        reduced_payload["max_tokens"] = max(64, int(self.max_tokens / 2))
                    if "max_new_tokens" in reduced_payload:
                        reduced_payload["max_new_tokens"] = max(64, int(self.max_tokens / 2))
                    self._debug("Read timeout. Retrying with reduced tokens and longer timeout", reduced_payload.get("max_tokens", reduced_payload.get("max_new_tokens", None)))
                    response = self._session.post(self.api_base, headers=headers, json=reduced_payload, timeout=(10, self.timeout * 2))

                last_response = response
                self._debug("Response status", response.status_code)
                try:
                    self._debug("Raw response text", response.text[:4000])
                    if response.status_code >= 400:
                        last_error_body = response.text
                except Exception:
                    pass

                # If not 422, or success, break and handle
                if response.status_code != 422:
                    break

            # If we tried multiple and still have a response, raise for status here
            response = last_response if last_response is not None else response
            self._debug("Response status", response.status_code)
            # Try to log JSON or raw text
            try:
                self._debug("Raw response text", response.text[:4000])
            except Exception:
                pass
            try:
                response.raise_for_status()
            except Exception as e:
                # Surface the server error body to the user for faster debugging
                details = (last_error_body or getattr(response, 'text', '') or "").strip()
                raise type(e)(f"{str(e)} | body: {details[:800]}")
            data = {}
            try:
                data = response.json()
            except Exception:
                self._debug("Failed to parse JSON; using empty data")
                data = {}

            # Extract content and potential tool calls defensively
            content = ""
            message_dict = {}
            first_choice = {}
            try:
                choices = data.get("choices") or []
                if isinstance(choices, list) and choices:
                    first_choice = choices[0] if isinstance(choices[0], dict) else {}
                # Primary: OpenAI-like schema
                message_dict = (first_choice.get("message") or {})
                # Some providers put the last message in an array
                if not message_dict and isinstance(first_choice.get("messages"), list) and first_choice["messages"]:
                    message_dict = first_choice["messages"][-1] or {}
                # Content extraction attempts in order
                content = (
                    (message_dict.get("content") if isinstance(message_dict, dict) else "")
                    or first_choice.get("text", "")
                    or data.get("content", "")
                    or data.get("text", "")
                    or data.get("output_text", "")
                    or data.get("response", "")
                )
                # Some streaming-like responses may have delta blocks
                if not content and isinstance(first_choice.get("delta"), dict):
                    content = first_choice["delta"].get("content", "")
            except Exception:
                content = ""
            if not content:
                # Fallbacks for non-standard providers
                content = data.get("content", "") or data.get("text", "") or ""
            self._debug("Parsed content", content if content else "<empty>")
            
            if self.json_mode:
                # For JSON mode, try to parse and return structured data
                try:
                    parsed_content = json.loads(content)
                    content = json.dumps(parsed_content, indent=2)
                except json.JSONDecodeError:
                    # If parsing fails, wrap in JSON structure
                    content = json.dumps({"response": content})
            
            # Import json for tool call parsing
            
            # Create chat generation with tool calling support
            message = AIMessage(content=content)

            # Try to parse tool calls from structured JSON first
            if hasattr(self, '_tools') and self._tools:
                structured_tool_calls = []
                try:
                    # Tool calls may appear in multiple locations depending on provider
                    api_tool_calls = (
                        (message_dict.get("tool_calls") if isinstance(message_dict, dict) else None)
                        or first_choice.get("tool_calls")
                        or data.get("tool_calls")
                        or []
                    )
                    if isinstance(api_tool_calls, list) and api_tool_calls:
                        for call in api_tool_calls:
                            if isinstance(call, dict) and call.get("type") == "function" and "function" in call:
                                func = call.get("function", {})
                                arguments = func.get("arguments", "{}")
                                # Ensure arguments is JSON object
                                args_obj = {}
                                try:
                                    args_obj = json.loads(arguments) if isinstance(arguments, str) else arguments
                                except Exception:
                                    args_obj = {}
                                structured_tool_calls.append({
                                    "name": func.get("name", ""),
                                    "args": args_obj,
                                    "id": call.get("id", f"call_{len(structured_tool_calls)}")
                                })
                except Exception:
                    pass

                if structured_tool_calls:
                    self._debug("Tool calls (structured) detected", structured_tool_calls)
                    message = AIMessage(content="", tool_calls=structured_tool_calls)
                else:
                    # Fallback: parse tool calls from content text
                    tool_calls = self._parse_tool_calls(content)
                    if tool_calls:
                        self._debug("Tool calls (text) detected", tool_calls)
                        langchain_tool_calls = []
                        for tool_call in tool_calls:
                            langchain_tool_calls.append({
                                "name": tool_call["function"]["name"],
                                "args": json.loads(tool_call["function"]["arguments"]),
                                "id": tool_call["id"]
                            })
                        message = AIMessage(content="", tool_calls=langchain_tool_calls)
                    else:
                        self._debug("No tool calls detected")
                        # Sanitize final prose to avoid leaking JSON tool calls and duplicates
                        if message.content:
                            message = AIMessage(content=self._sanitize_final_content(message.content))
            
            # If still empty message and no tool calls, provide a meaningful placeholder
            if (not getattr(message, 'tool_calls', None)) and (not message.content):
                # Last-ditch: attempt to stringify any 'reasoning' or provider-specific fields
                fallback_text = (
                    data.get("reasoning", "")
                    or (first_choice.get("reasoning", "") if isinstance(first_choice, dict) else "")
                )
                message = AIMessage(content=fallback_text or "(no content returned by model)")
                self._debug("Empty content after parsing; using placeholder")
            
            # Additional debug info for troubleshooting
            self._debug("Final message content length", len(message.content) if message.content else 0)
            self._debug("Final message has tool calls", bool(getattr(message, 'tool_calls', None)))
            
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            error_message = f"Error calling local LLM: {str(e)}"
            self._debug("Exception in _generate", error_message)
            message = AIMessage(content=error_message)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate a chat response from the local LLM."""
        import asyncio
        
        # For now, we'll use the sync version wrapped in async
        # In a production implementation, you'd use aiohttp
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self._generate(messages, stop, None, **kwargs)
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat response from the local LLM."""
        
        try:
            # Get the full response first
            result = self._generate(messages, stop, run_manager, **kwargs)
            
            if not result.generations:
                # Yield an empty chunk to satisfy the iterator contract
                self._debug("Stream: no generations; yielding empty chunk")
                yield ChatGenerationChunk(message=AIMessageChunk(content="(no generations)"))
                return
                
            message = result.generations[0].message
            
            # Handle tool calls - yield as single chunk
            if hasattr(message, 'tool_calls') and message.tool_calls:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=message.content,
                        tool_calls=message.tool_calls
                    )
                )
                if run_manager:
                    run_manager.on_llm_new_token(message.content)
                yield chunk
                return
            
            # Handle regular content - split into chunks
            content = message.content
            if not content:
                self._debug("Stream: empty content; yielding placeholder chunk")
                yield ChatGenerationChunk(message=AIMessageChunk(content="(empty)"))
                return
                
            # Split content into chunks for streaming effect
            import re
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            for sentence in sentences:
                if sentence.strip():
                    chunk_content = sentence + " "
                    message_chunk = AIMessageChunk(content=chunk_content)
                    chunk = ChatGenerationChunk(message=message_chunk)
                    if run_manager:
                        run_manager.on_llm_new_token(chunk_content)
                    yield chunk
            
        except Exception as e:
            error_message = f"Error streaming from local LLM: {str(e)}"
            message = AIMessageChunk(content=error_message)
            chunk = ChatGenerationChunk(message=message)
            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream chat response from the local LLM."""
        import asyncio
        
        try:
            # Use the sync streaming method wrapped in async
            loop = asyncio.get_event_loop()
            
            def get_sync_chunks():
                return list(self._stream(messages, stop, None, **kwargs))
            
            chunks = await loop.run_in_executor(None, get_sync_chunks)
            
            # Ensure we always yield at least one chunk
            if not chunks:
                yield ChatGenerationChunk(message=AIMessageChunk(content=""))
                return
            
            for chunk in chunks:
                if run_manager and hasattr(chunk.message, 'content') and chunk.message.content:
                    await run_manager.on_llm_new_token(chunk.message.content)
                yield chunk
                # Small delay to simulate streaming
                await asyncio.sleep(0.01)
                
        except Exception as e:
            error_message = f"Error in async streaming: {str(e)}"
            message = AIMessageChunk(content=error_message)
            chunk = ChatGenerationChunk(message=message)
            yield chunk

    def _convert_messages_to_api_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to API format."""
        api_messages = []
        has_tool_response = False
        
        def normalize_content(raw: Any) -> str:
            try:
                if raw is None:
                    return ""
                if isinstance(raw, list):
                    parts: List[str] = []
                    for part in raw:
                        if isinstance(part, dict):
                            if isinstance(part.get("text"), str):
                                parts.append(part["text"])
                            elif isinstance(part.get("content"), str):
                                parts.append(part["content"])
                            else:
                                parts.append(str(part))
                        else:
                            parts.append(str(part))
                    return "\n".join([p for p in parts if p]).strip()
                if isinstance(raw, dict):
                    if isinstance(raw.get("text"), str):
                        return raw["text"]
                    if isinstance(raw.get("content"), str):
                        return raw["content"]
                    try:
                        return json.dumps(raw, ensure_ascii=False)
                    except Exception:
                        return str(raw)
                return str(raw)
            except Exception:
                try:
                    return str(raw)
                except Exception:
                    return ""

        for message in messages:
            if isinstance(message, SystemMessage):
                api_messages.append({"role": "system", "content": normalize_content(message.content)})
            elif isinstance(message, HumanMessage):
                api_messages.append({"role": "user", "content": normalize_content(message.content)})
            elif isinstance(message, AIMessage):
                api_messages.append({"role": "assistant", "content": normalize_content(message.content)})
            else:
                # Check if this is a tool result message
                content = normalize_content(message.content)
                if any(indicator in content.lower() for indicator in ['tool result:', 'search result:', 'function result:', 'output:']):
                    has_tool_response = True
                    # Encourage comprehensive synthesis and allow additional tool calls if needed
                    enhanced_content = f"TOOL RESPONSE RECEIVED: {content}\n\nYou have received a tool response. You may:\n1. Use this information to answer comprehensively\n2. Call additional tools if you need more information\n3. Provide a final answer once you have sufficient information"
                    api_messages.append({"role": "user", "content": enhanced_content})
                else:
                    # Default to user message for unknown types
                    api_messages.append({"role": "user", "content": content})
        
        # If we detected a tool response, add a controller note emphasizing comprehensive synthesis
        if has_tool_response and api_messages:
            api_messages.append({
                "role": "system", 
                "content": "CONTROLLER: Tool responses are provided above. Analyze all results and, if needed, call additional tools (output ONLY JSON for actions). Provide a comprehensive final answer in clean prose without any tool-call JSON."
            })
        
        return api_messages

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "api_base": self.api_base,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
        }

    @property
    def _llm_type(self) -> str:
        return "local-chat-llm"
    
    def __del__(self):
        """Clean up session when model is destroyed."""
        if hasattr(self, '_session') and self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
    
    def bind_tools(
        self,
        tools: List[Union[Dict[str, Any], type, BaseTool]],
        **kwargs: Any,
    ) -> "LocalChatModel":
        """Bind tools to the model for tool calling.
        Mutates current instance to ensure agent uses the same model with tools.
        """
        try:
            self._tools = tools
        except Exception:
            # Fallback attribute set
            setattr(self, "_tools", tools)
        return self
    
    @property
    def _bound_tools(self) -> List[Union[Dict[str, Any], type, BaseTool]]:
        """Get bound tools."""
        return getattr(self, '_tools', [])
    
    @_bound_tools.setter
    def _bound_tools(self, tools: List[Union[Dict[str, Any], type, BaseTool]]) -> None:
        """Set bound tools."""
        self._tools = tools
    
    def _format_tools_for_prompt(self) -> str:
        """Format tools for inclusion in the prompt with automatic schema detection."""
        if not hasattr(self, '_tools') or not self._tools:
            return ""
        
        tool_descriptions = []
        for tool in self._tools:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                # This is a BaseTool - extract schema automatically
                tool_name = tool.name
                tool_desc = tool.description
                param_info = self._extract_tool_parameters(tool)
                tool_descriptions.append(f"- {tool_name}: {tool_desc}{param_info}")
                
            elif isinstance(tool, dict):
                # This is a tool dictionary
                name = tool.get('name', 'unknown')
                description = tool.get('description', 'No description')
                # Try to extract parameters from dict schema if available
                param_info = ""
                if 'parameters' in tool or 'args_schema' in tool:
                    param_info = self._extract_dict_tool_parameters(tool)
                tool_descriptions.append(f"- {name}: {description}{param_info}")
                
            elif hasattr(tool, '__name__'):
                # This is a function - extract from signature/annotations
                func_name = tool.__name__
                func_doc = tool.__doc__ or "No description"
                param_info = self._extract_function_parameters(tool)
                tool_descriptions.append(f"- {func_name}: {func_doc}{param_info}")
        
        return "\n".join(tool_descriptions)
    
    def _extract_tool_parameters(self, tool) -> str:
        """Extract parameters from a BaseTool's schema."""
        try:
            if not hasattr(tool, 'args_schema') or not tool.args_schema:
                return ""
                
            schema = tool.args_schema.schema()
            if 'properties' not in schema:
                return ""
                
            params = []
            required = schema.get('required', [])
            
            for param_name, param_details in schema['properties'].items():
                param_type = param_details.get('type', 'string')
                param_desc = param_details.get('description', f'{param_type} parameter')
                req_marker = ' (required)' if param_name in required else ' (optional)'
                
                # Create a concise but informative parameter description
                if param_desc:
                    params.append(f'"{param_name}"{req_marker}: {param_desc}')
                else:
                    params.append(f'"{param_name}"{req_marker}: {param_type} value')
            
            if params:
                return f" | Parameters: {', '.join(params)}"
                
        except Exception as e:
            # If schema extraction fails, try to infer from tool name patterns
            return self._infer_parameters_from_name(getattr(tool, 'name', ''))
        
        return ""
    
    def _extract_dict_tool_parameters(self, tool_dict) -> str:
        """Extract parameters from a dictionary-based tool definition."""
        try:
            params = []
            
            # Check various possible parameter definition formats
            param_sources = [
                tool_dict.get('parameters', {}),
                tool_dict.get('args_schema', {}),
                tool_dict.get('schema', {}),
                tool_dict.get('inputs', {})
            ]
            
            for param_source in param_sources:
                if isinstance(param_source, dict) and param_source:
                    if 'properties' in param_source:
                        # JSON Schema format
                        required = param_source.get('required', [])
                        for param_name, param_details in param_source['properties'].items():
                            param_desc = param_details.get('description', 'parameter')
                            req_marker = ' (required)' if param_name in required else ' (optional)'
                            params.append(f'"{param_name}"{req_marker}: {param_desc}')
                    else:
                        # Simple key-value format
                        for param_name, param_desc in param_source.items():
                            if isinstance(param_desc, str):
                                params.append(f'"{param_name}": {param_desc}')
                    
                    if params:
                        break
            
            if params:
                return f" | Parameters: {', '.join(params)}"
                
        except Exception:
            pass
        
        return ""
    
    def _extract_function_parameters(self, func) -> str:
        """Extract parameters from a function's signature."""
        try:
            import inspect
            sig = inspect.signature(func)
            params = []
            
            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'cls']:
                    continue
                    
                param_type = 'string'
                if param.annotation != inspect.Parameter.empty:
                    param_type = getattr(param.annotation, '__name__', str(param.annotation))
                
                req_marker = '' if param.default != inspect.Parameter.empty else ' (required)'
                params.append(f'"{param_name}"{req_marker}: {param_type}')
            
            if params:
                return f" | Parameters: {', '.join(params)}"
                
        except Exception:
            pass
        
        return ""
    
    def _infer_parameters_from_name(self, tool_name: str) -> str:
        """Fallback: infer likely parameters from tool name patterns."""
        tool_name_lower = tool_name.lower()
        
        # Common patterns for different tool types
        if any(keyword in tool_name_lower for keyword in ['search', 'query', 'find']):
            return ' | Parameters: "input_value" or "query" (required): search query text'
        elif any(keyword in tool_name_lower for keyword in ['arxiv', 'paper', 'research']):
            return ' | Parameters: "search_query" (required): research query'
        elif any(keyword in tool_name_lower for keyword in ['api', 'request', 'fetch']):
            return ' | Parameters: "url" (required): API endpoint, "params": additional parameters'
        elif any(keyword in tool_name_lower for keyword in ['file', 'read', 'write']):
            return ' | Parameters: "file_path" (required): path to file'
        elif any(keyword in tool_name_lower for keyword in ['calculate', 'compute', 'math']):
            return ' | Parameters: "expression" (required): mathematical expression'
        else:
            return ' | Parameters: check tool documentation for required parameters'
    
    def _parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse one or multiple tool calls from model response.
        Supports single action {action, action_input} and batched {actions: [{action,...}, ...]}.
        """
        tool_calls: List[Dict[str, Any]] = []
        try:
            import json
            import re

            def append_action(action_obj):
                name = action_obj.get('action') or action_obj.get('name')
                args = action_obj.get('action_input') or action_obj.get('arguments') or {}
                if name:
                    # Remove any "tool." prefix that might be incorrectly added
                    if name.startswith('tool.'):
                        name = name[5:]  # Remove "tool." prefix
                    
                    tool_calls.append({
                        'id': f'call_{len(tool_calls)}',
                        'type': 'function',
                        'function': {
                            'name': name,
                            'arguments': json.dumps(args if isinstance(args, dict) else {})
                        }
                    })

            # Try whole content as JSON
            try:
                parsed = json.loads(content.strip())
                if isinstance(parsed, dict):
                    if 'actions' in parsed and isinstance(parsed['actions'], list):
                        for act in parsed['actions']:
                            if isinstance(act, dict):
                                append_action(act)
                        if tool_calls:
                            return tool_calls
                    if 'action' in parsed:
                        append_action(parsed)
                        if tool_calls:
                            return tool_calls
            except Exception:
                pass

            # Search JSON objects within text (greedy and fenced)
            json_patterns = [
                r'```json\s*(.*?)\s*```',
                r'```\s*(.*?)\s*```',
                r'\{[\s\S]*?\}'
            ]
            for pattern in json_patterns:
                matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
                for match in matches:
                    json_str = match.strip() if isinstance(match, str) else str(match)
                    if not (json_str.startswith('{') and json_str.endswith('}')):
                        continue
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            if 'actions' in parsed and isinstance(parsed['actions'], list):
                                for act in parsed['actions']:
                                    if isinstance(act, dict):
                                        append_action(act)
                            elif 'action' in parsed:
                                append_action(parsed)
                    except Exception:
                        continue
            return tool_calls
        except Exception:
            return tool_calls


# Langflow Component
class GPTOSSModelComponent(LCModelComponent):
    display_name = "GPT OSS Model (Crimson)"
    description = "Generate text using a locally hosted Llama-3.1-8B-Instruct model. Compatible with agents and tools."
    icon = "OpenAI"

    inputs = [
        *LCModelComponent._base_inputs,
        FloatInput(
            name="temperature",
            display_name="Temperature",
            info="Controls randomness in the model's output. Lower values = faster, more deterministic responses.",
            value=0.3,
            advanced=False,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            info="Maximum number of tokens to generate in the response. Lower values = faster responses.",
            value=256,
            advanced=False,
        ),
        IntInput(
            name="timeout",
            display_name="Request Timeout",
            info="Timeout for API requests in seconds. Lower values = faster failures.",
            value=15,
            advanced=True,
        ),
        BoolInput(
            name="json_mode",
            display_name="JSON Mode",
            advanced=True,
            info="If True, output JSON format response.",
        ),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            advanced=True,
            info="Additional keyword arguments to pass to the model (currently unused).",
        ),
    ]

    def build_model(self) -> LanguageModel:
        """Build the chat model for agent and tool compatibility."""
        model = LocalChatModel(
            json_mode=self.json_mode,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        
        # Verify tool calling support for debugging
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Built LocalChatModel with tool calling support: {hasattr(model, 'bind_tools')}")
        
        return model

    def supports_tool_calling(self, model: LanguageModel) -> bool:
        """Override to confirm this model supports tool calling."""
        # Our model has bind_tools implemented
        # Check if model supports tool calling
        return hasattr(model, 'bind_tools') and callable(getattr(model, 'bind_tools'))

    def _get_exception_message(self, e: Exception):
        return str(e)