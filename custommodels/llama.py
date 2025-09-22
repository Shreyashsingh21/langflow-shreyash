import requests
from typing import Any, List, Dict, Optional, Union, Iterator, AsyncIterator
from typing_extensions import override

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

API_BASE_URL = "http://216.48.181.172:8090/v1/chat/completions"
MODEL_NAME_HARDCODED = "meta-llama/Llama-3.1-8B-Instruct"


# Custom LangChain Chat Model Wrapper for Local LLM
class LocalChatModel(BaseChatModel):
    """Custom chat model that wraps the local Llama API for agent and tool use."""

    api_base: str = API_BASE_URL
    model_name: str = MODEL_NAME_HARDCODED
    json_mode: bool = False
    temperature: float = 0.7  
    max_tokens: int = 20000  
    timeout: int = 60
    def _sanitize_final_content(self, content: str) -> str:
        """Remove leaked tool-call JSON and collapse duplicate text, preserving code fences.

        Heuristics:
        - Preserve fenced JSON/code blocks
        - Remove inline {action/...} and {actions:[...]} snippets
        - Deduplicate consecutive identical lines
        - If the response body appears duplicated end-to-end, keep the first half
        """
        try:
            import re
            text = content or ""
            # Preserve fenced code blocks; only strip inline tool-call JSON blobs
            # Remove inline action JSON objects (single or batched)
            text = re.sub(r"\{\s*\"actions\"\s*:\s*\[[\s\S]*?\]\s*\}", "", text)
            text = re.sub(r"\{\s*\"action\"\s*:\s*\"[^\"]+\"[\s\S]*?\}", "", text)

            # Collapse duplicate consecutive lines
            lines = text.splitlines()
            deduped: List[str] = []
            prev = None
            for ln in lines:
                if ln != prev:
                    deduped.append(ln)
                prev = ln
            cleaned = "\n".join(deduped).strip()

            # If the entire content got duplicated (A + A), keep first half
            if cleaned:
                mid = len(cleaned) // 2
                first_half = cleaned[:mid].strip()
                second_half = cleaned[mid:].strip()
                if first_half and first_half == second_half:
                    cleaned = first_half

            # Paragraph-level dedupe (handle repeated paragraphs with minor whitespace differences)
            if cleaned:
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
                unique_paragraphs: List[str] = []
                seen = set()
                for p in paragraphs:
                    key = " ".join(p.split())  # normalize whitespace
                    if key not in seen:
                        unique_paragraphs.append(p)
                        seen.add(key)
                cleaned = "\n\n".join(unique_paragraphs).strip()

            return cleaned or (content.strip() if content else "")
        except Exception:
            return content
    
    # Conservative context window for this endpoint (inputs + max_new_tokens)
    _context_limit: int = 4096
    
    def _estimate_tokens(self, messages: List[BaseMessage]) -> int:
        """Rough token estimate: chars/4 heuristic across all message contents."""
        try:
            total_chars = 0
            for m in messages:
                try:
                    total_chars += len(str(getattr(m, 'content', '') or ''))
                except Exception:
                    continue
            # Approximate 4 chars per token
            return max(1, total_chars // 4)
        except Exception:
            return 512  # safe fallback

    def _estimate_tokens_api_messages(self, api_messages: List[Dict[str, str]]) -> int:
        try:
            total_chars = 0
            for msg in api_messages:
                total_chars += len(str(msg.get('content', '')))
            return max(1, total_chars // 4)
        except Exception:
            return 1024

    def _shrink_messages_to_fit(self, api_messages: List[Dict[str, str]], max_total_tokens: int) -> List[Dict[str, str]]:
        """Destructively trims oversized tool/user contents to fit context window."""
        def estimate() -> int:
            return self._estimate_tokens_api_messages(api_messages)

        # First pass: aggressively trim known tool response wrappers
        for msg in api_messages:
            content = str(msg.get('content', ''))
            if content.startswith('TOOL RESPONSE RECEIVED:') and len(content) > 2000:
                msg['content'] = content[:2000] + "\n...[truncated]"

        # Recheck; if still too large, trim largest contents iteratively
        safety_counter = 0
        while estimate() > max_total_tokens and safety_counter < 10:
            # Find the longest content among user/tool/assistant (prefer trimming user/tool first)
            idx_longest = -1
            longest_len = 0
            for i, msg in enumerate(api_messages):
                role = msg.get('role', '')
                content = str(msg.get('content', ''))
                # Avoid trimming system first
                if role == 'system':
                    continue
                if len(content) > longest_len:
                    longest_len = len(content)
                    idx_longest = i
            if idx_longest == -1 or longest_len < 400:
                break
            # Trim by half down to a floor of 400 chars
            current = api_messages[idx_longest]['content']
            new_len = max(400, len(current) // 2)
            api_messages[idx_longest]['content'] = current[:new_len] + "\n...[truncated]"
            safety_counter += 1

        # Final safety: if still too large, keep only last few messages (preserve first system and last 2 turns)
        if estimate() > max_total_tokens and len(api_messages) > 4:
            preserved = []
            # keep first system if present
            if api_messages and api_messages[0].get('role') == 'system':
                preserved.append(api_messages[0])
            # keep last three messages
            preserved.extend(api_messages[-3:])
            api_messages = preserved
        return api_messages
    
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
        
        # Add a general system instruction to handle tool results properly
        general_instruction = {
            "role": "system",
            "content": """You are a helpful AI assistant. CRITICAL: You MUST use tools when available. Do NOT answer questions directly if tools can help.

MANDATORY TOOL USAGE:
- For time/date questions → ALWAYS call get_current_date tool
- For math/arithmetic → ALWAYS call evaluate_expression tool  
- For search/research/paper queries → ALWAYS call fetch_content_dataframe tool
- For ticket numbers, ticket IDs, or support ticket queries → ALWAYS call Trinka_ticket_context_tool
- NEVER compute math in your head, guess dates, or make up ticket information

TOOL USAGE POLICY:
1. WHEN TO USE TOOLS: Use tools when they help you answer accurately.
2. MULTIPLE TOOLS: You may call multiple tools in one round if they are independent. Prefer batching them as an array.
3. ITERATIVE STEPS: It's OK to run tools, read results, then call more tools if needed.
4. FINAL ANSWER: When you have enough information, provide a clear final answer.
5. DO NOT MENTION TOOLS: Do not mention internal tool names or that you are calling a tool. Just answer with the result.

TOOL RESPONSE HANDLING:
- If tool returns useful data → Use it directly in your reasoning and answer.
- If tool returns empty/no results → Explain briefly and suggest alternatives.
- Keep tool calls concise and only as needed."""
        }
        
        # Insert general instruction at the beginning
        api_messages.insert(0, general_instruction)
        
        # Add tool information if tools are bound
        if hasattr(self, '_tools') and self._tools:
            tool_descriptions = self._format_tools_for_prompt()
            if tool_descriptions:
                # Add tool information to the system message or as a separate message
                tool_message = {
                    "role": "system", 
                    "content": f"""You have access to the following tools:
{tool_descriptions}

CRITICAL: You MUST use tools for time, math, and search questions. Do NOT answer directly.

MANDATORY TOOL USAGE:
- Time/date questions → get_current_date
- Math/arithmetic → evaluate_expression with {{"expression": "..."}}
- Search/research → fetch_content_dataframe with {{"search_query": "..."}}

TOOL USE INSTRUCTIONS:
- Call multiple tools in one step using "actions" array when possible
- Match parameter names EXACTLY as listed in Parameters
- Provide all required parameters; optional may be omitted
- Do not include tool-call JSON in final prose answer

JSON FORMATS:
- Single: {{"action": "ToolName", "action_input": {{"param": "value"}}}}
- Batch: {{"actions": [{{"action": "ToolA", "action_input": {{...}}}}, {{"action": "ToolB", "action_input": {{...}}}}]}}

PARAMETER MAPPING:
- "query", "search_query", "srsearch" → user's query text
- "expression" → mathematical expression from user
- "url" → URL from user
- "input_value" → user's message when unsure

EXAMPLE for "tell me current time and 2+3 and about attention paper":
{{"actions": [
  {{"action": "get_current_date", "action_input": {{}}}},
  {{"action": "evaluate_expression", "action_input": {{"expression": "2+3"}}}},
  {{"action": "fetch_content_dataframe", "action_input": {{"search_query": "attention paper"}}}}
]}}

After tools return, provide clean final answer without JSON."""
                }
                # Insert tool message at the beginning (after any existing system message)
                if api_messages and api_messages[0]["role"] == "system":
                    api_messages.insert(1, tool_message)
                else:
                    api_messages.insert(0, tool_message)
        
        headers = {"Content-Type": "application/json", "Authorization": "Bearer token-abc123"}
        # Clamp max_tokens AND shrink inputs to avoid 422 (inputs + max_new_tokens <= context_limit)
        # Start by estimating on api_messages (more accurate, includes tool responses)
        allowed_total_tokens = self._context_limit
        # Temporarily assemble payload to compute and shrink
        temp_api_messages = list(api_messages)
        # First, shrink if needed with a conservative allowance for output
        prelim_allowed_new = max(16, min(self.max_tokens, 1024))
        temp_api_messages = self._shrink_messages_to_fit(temp_api_messages, allowed_total_tokens - prelim_allowed_new)
        prompt_tokens_est = self._estimate_tokens_api_messages(temp_api_messages)
        allowed_new_tokens = max(16, allowed_total_tokens - prompt_tokens_est)
        safe_max_tokens = max(16, min(self.max_tokens, allowed_new_tokens))
        payload = {
            "model": self.model_name,
            "messages": temp_api_messages,
            "temperature": self.temperature,
            "max_tokens": safe_max_tokens,
            "stop": stop or [],
        }
        
        try:
            # Print payload for every API call (mask secrets)
            try:
                safe_headers = dict(headers)
                if "Authorization" in safe_headers and isinstance(safe_headers["Authorization"], str):
                    safe_headers["Authorization"] = "***"
                import json as _json
                print(
                    f"[LocalChatModel Llama] POST {self.api_base} \nheaders={safe_headers} \npayload="
                    + _json.dumps(payload, ensure_ascii=False)[:4000]
                )
            except Exception:
                pass
            response = requests.post(self.api_base, headers=headers, json=payload, timeout=self.timeout)
            
            # Handle 422 errors specifically
            if response.status_code == 422:
                error_detail = "Unprocessable Entity"
                try:
                    error_data = response.json()
                    error_detail = str(error_data)
                except Exception:
                    error_detail = response.text[:500]
                raise Exception(f"422 Client Error: {error_detail}")
            
            response.raise_for_status()
            data = response.json()
            # Defensive extraction of content and potential tool_calls
            content = ""
            choice = {}
            message_obj = {}
            try:
                choices = data.get("choices") or []
                if isinstance(choices, list) and choices:
                    choice = choices[0] if isinstance(choices[0], dict) else {}
                message_obj = (choice.get("message") or {}) if isinstance(choice, dict) else {}
                content = (
                    (message_obj.get("content") if isinstance(message_obj, dict) else "")
                    or choice.get("text", "")
                    or data.get("content", "")
                    or data.get("text", "")
                )
            except Exception:
                content = data.get("content", "") or ""
            
            if self.json_mode:
                # For JSON mode, try to parse and return structured data
                try:
                    import json
                    parsed_content = json.loads(content)
                    content = json.dumps(parsed_content, indent=2)
                except json.JSONDecodeError:
                    # If parsing fails, wrap in JSON structure
                    content = json.dumps({"response": content})
            
            # Create chat generation with tool calling support
            message = AIMessage(content=content)

            # Tool-calling: prefer structured tool_calls if present, else parse from text
            if hasattr(self, '_tools') and self._tools:
                structured_calls = []
                try:
                    api_tool_calls = (
                        (message_obj.get("tool_calls") if isinstance(message_obj, dict) else None)
                        or choice.get("tool_calls")
                        or data.get("tool_calls")
                        or []
                    )
                    if isinstance(api_tool_calls, list) and api_tool_calls:
                        for call in api_tool_calls:
                            if isinstance(call, dict) and call.get("type") == "function" and "function" in call:
                                func = call.get("function", {})
                                arguments = func.get("arguments", "{}")
                                try:
                                    args_obj = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
                                except Exception:
                                    args_obj = {}
                                # Normalize name by stripping incorrect prefixes like "tool."
                                name = (func.get("name", "") or "")
                                if name.startswith("tool."):
                                    name = name[5:]
                                structured_calls.append({
                                    "name": name,
                                    "args": args_obj,
                                    "id": call.get("id", f"call_{len(structured_calls)}")
                                })
                except Exception:
                    structured_calls = []

                if structured_calls:
                    # Backfill missing args when possible
                    backfilled = self._backfill_missing_tool_args(structured_calls, messages)
                    message = AIMessage(content="", tool_calls=backfilled)
                else:
                    # Fallback: parse JSON-ish tool calls from text
                    tool_calls = self._parse_tool_calls(content)
                    if tool_calls:
                        langchain_tool_calls = []
                        for tool_call in tool_calls:
                            # Strip any tool. prefix
                            name = tool_call["function"].get("name", "")
                            if name.startswith("tool."):
                                name = name[5:]
                            try:
                                args_obj = json.loads(tool_call["function"].get("arguments", "{}"))
                            except Exception:
                                args_obj = {}
                            langchain_tool_calls.append({
                                "name": name,
                                "args": args_obj,
                                "id": tool_call.get("id", "")
                            })
                        backfilled = self._backfill_missing_tool_args(langchain_tool_calls, messages)
                        message = AIMessage(content="", tool_calls=backfilled)
                    else:
                        if message.content:
                            message = AIMessage(content=self._sanitize_final_content(message.content))
            
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            error_message = f"Error calling local LLM: {str(e)}"
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
                yield ChatGenerationChunk(message=AIMessageChunk(content=""))
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
                yield ChatGenerationChunk(message=AIMessageChunk(content=""))
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
        
        for message in messages:
            if isinstance(message, SystemMessage):
                api_messages.append({"role": "system", "content": str(message.content)})
            elif isinstance(message, HumanMessage):
                api_messages.append({"role": "user", "content": str(message.content)})
            elif isinstance(message, AIMessage):
                # Handle AIMessage with tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # For tool calls, send the tool call information
                    api_messages.append({"role": "assistant", "content": str(message.content)})
                else:
                    api_messages.append({"role": "assistant", "content": str(message.content)})
            else:
                # Check if this is a tool result message
                content = str(message.content)
                if any(indicator in content.lower() for indicator in ['tool result:', 'search result:', 'function result:', 'output:', 'invoking:', 'error calling']):
                    has_tool_response = True
                    # Allow continued tool use if needed; instruct JSON-only when calling tools
                    enhanced_content = (
                        f"TOOL RESPONSE RECEIVED: {content}\n\n"
                        "You have received a tool response. You may: (1) use this to answer now,"
                        " (2) call additional tools if still needed (output ONLY JSON for actions),"
                        " then provide a clear final answer."
                    )
                    api_messages.append({"role": "user", "content": enhanced_content})
                else:
                    # Default to user message for unknown types
                    api_messages.append({"role": "user", "content": content})
        
        # If we detected a tool response, add a gentle controller note allowing follow-ups
        if has_tool_response and api_messages:
            api_messages.append({
                "role": "system", 
                "content": "CONTROLLER: A tool response was provided above. You may either (a) call additional tools if still needed (output ONLY JSON for actions), or (b) provide a clear final answer. Do NOT include tool-call JSON in the final prose answer."
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
                    # Normalize name by stripping accidental "tool." prefix
                    if isinstance(name, str) and name.startswith('tool.'):
                        name = name[5:]
                    
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

    def _backfill_missing_tool_args(self, tool_calls: List[Dict[str, Any]], messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Heuristically fill missing args from the latest user message using generic rules.

        - Infer likely parameters by name: query/search_query/srsearch/url/expression/input_value
        - Use the latest user text as the default source when appropriate
        - Do not hardcode tool names; rely on parameter names and schemas when available
        """
        try:
            import re
            last_user = ""
            for msg in reversed(messages):
                from langchain_core.messages import HumanMessage
                if isinstance(msg, HumanMessage):
                    last_user = str(getattr(msg, 'content', '') or '')
                    break

            # Build a quick map of tool name -> required parameter hints if available
            schema_hints: Dict[str, List[str]] = {}
            try:
                if hasattr(self, '_tools') and self._tools:
                    for t in self._tools:
                        name = getattr(t, 'name', None)
                        if not name and isinstance(t, dict):
                            name = t.get('name')
                        if not name:
                            continue
                        params: List[str] = []
                        # args_schema for BaseTool
                        if hasattr(t, 'args_schema') and t.args_schema:
                            try:
                                schema = t.args_schema.schema()
                                props = (schema or {}).get('properties', {})
                                params.extend(list(props.keys()))
                            except Exception:
                                pass
                        # dict-based tool schema
                        if isinstance(t, dict):
                            for key in ['parameters', 'args_schema', 'schema', 'inputs']:
                                if key in t and isinstance(t[key], dict):
                                    props = t[key].get('properties') or t[key]
                                    if isinstance(props, dict):
                                        params.extend(list(props.keys()))
                        schema_hints[name.lower()] = list(dict.fromkeys(params))
            except Exception:
                schema_hints = {}

            for call in tool_calls:
                name = (call.get('name') or '')
                args = call.get('args') or {}
                lowered = name.lower()
                hinted_params = schema_hints.get(lowered, [])

                # If expression-like param exists and is empty, try to extract math expression
                if any(p in hinted_params or p in args for p in ['expression']):
                    if not args.get('expression') and last_user:
                        candidates = re.findall(r"([\d\s\(\)\+\-\*\/\^\.]{3,})", last_user)
                        if candidates:
                            expr = candidates[-1].strip()
                            expr = re.sub(r"\s+", "", expr)
                            if expr:
                                args['expression'] = expr

                # If query-like parameters exist, set from last_user
                for qparam in ['search_query', 'srsearch', 'query']:
                    if (qparam in hinted_params or qparam in args) and not args.get(qparam) and last_user:
                        args[qparam] = last_user

                # Fallback: input_value
                if (('input_value' in hinted_params) or ('input_value' in args)) and not args.get('input_value') and last_user:
                    args['input_value'] = last_user

                call['args'] = args
            return tool_calls
        except Exception:
            return tool_calls


# Langflow Component
class LlamaLocalLLMComponent(LCModelComponent):
    display_name = "Llama Chat LLM (Crimson)"
    description = "Generate text using a locally hosted Llama-3.1-8B-Instruct model. Compatible with agents and tools."
    icon = "Ollama"

    inputs = [
        *LCModelComponent._base_inputs,
        FloatInput(
            name="temperature",
            display_name="Temperature",
            info="Controls randomness in the model's output. Higher values make output more random.",
            value=0.7,
            advanced=False,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            info="Maximum number of tokens to generate in the response.",
            value=512,
            advanced=False,
        ),
        IntInput(
            name="timeout",
            display_name="Request Timeout",
            info="Timeout for API requests in seconds.",
            value=60,
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
