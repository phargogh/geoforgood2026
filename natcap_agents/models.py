"""smolagents Models for the two kinds of backend, neither going through LiteLLM:

* `VertexAIServerModel` talks to Gemini through Google's own `google-genai`
  SDK. LiteLLM's `vertex_ai/` provider authenticates only via a service
  account (google-auth / ADC) and ignores a plain API key. Google's
  `google-genai` SDK, by contrast, supports Vertex with an API key
  (`Client(vertexai=True, api_key=...)`, "Express mode") as well as the
  service-account path, so this gives smolagents a Vertex backend that works
  with either credential.
* `OllamaModel` talks to a local Ollama server over its native HTTP API.

Both reuse smolagents' own message/tool normalization
(`_prepare_completion_kwargs` -> OpenAI-format), then convert that to their
backend's calls, so behavior matches the other smolagents models.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

import requests
from google import genai
from google.genai import types
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
    TokenUsage,
)


def _openai_to_gemini(messages: list[dict], tools: list[dict] | None):
    """Convert smolagents/OpenAI-format messages + tools to google-genai inputs.

    Returns (system_instruction, contents, gemini_tools).
    Assumes message `content` is a plain string (flatten_messages_as_text=True).
    """
    system_bits: list[str] = []
    contents: list[types.Content] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):  # safety if not flattened
            content = "".join(part.get("text", "") for part in content)
        content = content or ""

        if role == "system":
            system_bits.append(content)
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
        else:  # user, tool -> present as user turns (tool results as user text)
            text = content if role != "tool" else f"Tool result:\n{content}"
            contents.append(types.Content(role="user", parts=[types.Part(text=text)]))

    gemini_tools = None
    if tools:
        decls = []
        for tool in tools:
            fn = tool.get("function", tool)
            decls.append(
                types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters") or None,
                )
            )
        gemini_tools = [types.Tool(function_declarations=decls)]

    system_instruction = "\n\n".join(system_bits) if system_bits else None
    return system_instruction, contents, gemini_tools


def _normalize_messages(messages) -> list:
    """Accept both hand-written string content and the agents' list-of-parts.

    smolagents' message cleaner expects `content` to be a list of typed parts;
    a plain string breaks it. We wrap any string content so direct calls like
    `model([ChatMessage(role=USER, content='hi')])` work as well as agent calls.
    """
    norm = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                msg = {**msg, "content": [{"type": "text", "text": content}]}
            norm.append(msg)
        else:  # ChatMessage-like object
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                role = getattr(msg, "role", "user")
                role = getattr(role, "value", role)
                norm.append({"role": role, "content": [{"type": "text", "text": content}]})
            else:
                norm.append(msg)
    return norm


class VertexAIServerModel(Model):
    """smolagents Model for Gemini via google-genai (Vertex or Gemini API).

    Args:
        model_id: bare Gemini id, e.g. 'gemini-2.5-pro'.
        api_key: if given, authenticates with this API key (Vertex Express when
            use_vertex=True, Gemini API when use_vertex=False). No service account needed.
        project, location: used for the service-account/ADC path when api_key is None.
        use_vertex: route via Vertex AI (True) or the Gemini Developer API (False).
        temperature: sampling temperature.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        api_key: str | None = None,
        project: str | None = None,
        location: str = "us-central1",
        use_vertex: bool = True,
        temperature: float = 0.2,
        **kwargs,
    ):
        super().__init__(flatten_messages_as_text=True, model_id=model_id, **kwargs)
        self.temperature = temperature

        if api_key:
            self._client = genai.Client(vertexai=use_vertex, api_key=api_key)
        elif use_vertex:
            # Service-account / ADC path (GOOGLE_APPLICATION_CREDENTIALS).
            self._client = genai.Client(vertexai=True, project=project, location=location)
        else:
            self._client = genai.Client()  # Gemini API via GOOGLE_API_KEY env

    def generate(
        self,
        messages,
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list | None = None,
        **kwargs,
    ) -> ChatMessage:
        ck = self._prepare_completion_kwargs(
            messages=_normalize_messages(messages),
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
        )
        system_instruction, contents, gemini_tools = _openai_to_gemini(
            ck["messages"], ck.get("tools")
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=kwargs.get("temperature", self.temperature),
            stop_sequences=ck.get("stop"),
            tools=gemini_tools,
        )

        response = self._client.models.generate_content(
            model=self.model_id, contents=contents, config=config
        )

        # Extract text + any function calls from the first candidate.
        text_parts: list[str] = []
        tool_calls: list[ChatMessageToolCall] = []
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", None) or []
            for i, part in enumerate(parts):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_calls.append(
                        ChatMessageToolCall(
                            id=getattr(fc, "id", None) or f"call_{i}",
                            type="function",
                            function=ChatMessageToolCallFunction(
                                name=fc.name, arguments=dict(fc.args or {})
                            ),
                        )
                    )

        usage = getattr(response, "usage_metadata", None)
        token_usage = (
            TokenUsage(
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            )
            if usage
            else None
        )

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="".join(text_parts) or None,
            tool_calls=tool_calls or None,
            raw=response,
            token_usage=token_usage,
        )


def _ollama_base_url(host: str) -> str:
    """OLLAMA_HOST -> a base URL, read the way the ollama CLI reads it.

    Accepts '127.0.0.1:11434', 'localhost', 'http://host:port', etc.; a missing
    scheme means http, and a missing port on plain http means 11434.
    """
    parts = urlsplit(host if "://" in host else f"http://{host}")
    netloc = parts.netloc
    if parts.port is None and parts.scheme == "http":
        netloc += ":11434"
    return f"{parts.scheme}://{netloc}{parts.path.rstrip('/')}"


def _first_stop(text: str, stops: list[str]) -> int | None:
    """Index of the earliest stop sequence in `text`, or None if there is none."""
    hits = [i for i in (text.find(s) for s in stops) if i != -1]
    return min(hits) if hits else None


class OllamaModel(Model):
    """smolagents Model for a local Ollama server, via its native /api/chat.

    The native API rather than Ollama's OpenAI-compatible /v1 endpoint, because
    only the native one takes `num_ctx` per request: the server's default
    context (often 4096 tokens) is smaller than the orchestrator's prompt, and
    Ollama truncates an over-long prompt without saying so.

    Stop sequences: Ollama applies `stop` to a model's thinking as well as its
    answer, so a '</code>' or 'Observation:' drafted while thinking would end
    the reply before the answer starts. So when the model may think, the stops
    are matched here, against the streamed answer only, and leaving the stream
    early closes the connection, which makes Ollama stop generating.

    Args:
        model_id: an Ollama model tag, e.g. 'qwen2.5-coder:14b'.
        host: the server, as in OLLAMA_HOST.
        num_ctx: context window, in tokens.
        think: True/False to turn thinking on/off, 'low'/'medium'/'high' for
            gpt-oss, or None to leave it at the model's default.
        max_tokens: cap on the tokens one reply may generate, thinking
            included. Ollama's own default is no cap, and no stop sequence
            can end a thought that never reaches an answer.
        temperature: sampling temperature, or None for the model's own (its
            Modelfile's), the value its publisher tuned it with.
        timeout: seconds to wait for each streamed chunk. The first one waits
            for the model to load and read the whole prompt.
    """

    def __init__(
        self,
        model_id: str,
        host: str = "http://localhost:11434",
        num_ctx: int = 32768,
        think: bool | str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
        timeout: float = 600,
        **kwargs,
    ):
        super().__init__(flatten_messages_as_text=True, model_id=model_id, **kwargs)
        self.base_url = _ollama_base_url(host)
        self.num_ctx = num_ctx
        self.think = think
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # Fail now, at setup, if the server is down or the model isn't pulled,
        # rather than on the first step of the first run.
        shown = self._post("/api/show", {"model": model_id}).json()
        capabilities = shown.get("capabilities")
        can_think = capabilities is None or "thinking" in capabilities  # None: older Ollama, unknown
        if think and not can_think:
            raise RuntimeError(
                f"Ollama model '{model_id}' does not support thinking. Clear "
                "OLLAMA_THINK in .env, or pick a model that thinks."
            )
        self._may_think = can_think and think is not False
        # Without Ollama's `tools` capability a request carrying tools is
        # refused. Leave them out: the agent's prompt already lists them, and
        # smolagents parses a tool call written as JSON in the text.
        self._native_tools = capabilities is None or "tools" in capabilities
        # A request's `stop` replaces the Modelfile's instead of adding to it,
        # so send the model's own along too (deepseek-r1 ends turns on them).
        parameters = (line.partition(" ") for line in (shown.get("parameters") or "").splitlines())
        self._model_stops = [v.strip().strip('"') for key, _, v in parameters if key == "stop"]

    def _post(self, path: str, payload: dict, stream: bool = False) -> requests.Response:
        try:
            response = requests.post(
                self.base_url + path, json=payload, stream=stream, timeout=(10, self.timeout)
            )
        except requests.ConnectionError as error:
            raise RuntimeError(
                f"Can't reach Ollama at {self.base_url}. Start it (the Ollama app, "
                "or `ollama serve`), or point OLLAMA_HOST in .env at your server."
            ) from error
        if response.ok:
            return response

        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        response.close()
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama has no model '{self.model_id}' ({message}). Pull it with "
                f"`ollama pull {self.model_id}`, or set ORCHESTRATOR_MODEL / "
                f"WORKER_MODEL in .env to one you have: {self._local_models()}."
            )
        raise RuntimeError(f"Ollama model '{self.model_id}': {message}")

    def _local_models(self) -> str:
        try:
            tags = requests.get(self.base_url + "/api/tags", timeout=10).json()
            return ", ".join(m["name"] for m in tags.get("models", [])) or "none pulled yet"
        except (requests.RequestException, ValueError):
            return "see `ollama list`"

    def generate(
        self,
        messages,
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list | None = None,
        **kwargs,
    ) -> ChatMessage:
        ck = self._prepare_completion_kwargs(
            messages=_normalize_messages(messages),
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
        )
        stops = ck.get("stop") or []

        chat_messages = []
        for msg in ck["messages"]:
            content = msg.get("content")
            if isinstance(content, list):  # safety if not flattened
                content = "".join(part.get("text", "") for part in content)
            chat_messages.append({"role": msg["role"], "content": content or ""})

        options: dict[str, Any] = {"num_ctx": self.num_ctx, "num_predict": self.max_tokens}
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            options["temperature"] = temperature
        if stops and not self._may_think:
            options["stop"] = self._model_stops + stops  # safe server-side: no thinking to cut short
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": chat_messages,
            "options": options,
            "stream": True,
        }
        if ck.get("tools") and self._native_tools:
            payload["tools"] = ck["tools"]
        if self.think is not None:
            payload["think"] = self.think

        content, thinking, raw_tool_calls, done = "", "", [], None
        with self._post("/api/chat", payload, stream=True) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(f"Ollama model '{self.model_id}': {chunk['error']}")
                message = chunk.get("message") or {}
                thinking += message.get("thinking") or ""
                content += message.get("content") or ""
                raw_tool_calls += message.get("tool_calls") or []
                cut = _first_stop(content, stops)
                if cut is not None:
                    content = content[:cut]
                    break  # closing the stream stops the generation
                if chunk.get("done"):
                    done = chunk
                    break

        tool_calls = [
            ChatMessageToolCall(
                id=call.get("id") or f"call_{i}",
                type="function",
                function=ChatMessageToolCallFunction(
                    name=call["function"]["name"],
                    arguments=call["function"].get("arguments") or {},
                ),
            )
            for i, call in enumerate(raw_tool_calls)
        ]

        # Only the final chunk carries the token counts, so a reply cut short
        # by a stop matched here has none to report.
        token_usage = (
            TokenUsage(
                input_tokens=done.get("prompt_eval_count", 0) or 0,
                output_tokens=done.get("eval_count", 0) or 0,
            )
            if done
            else None
        )

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content or None,
            tool_calls=tool_calls or None,
            raw={
                **(done or {}),
                "message": {"role": "assistant", "content": content,
                            "thinking": thinking, "tool_calls": raw_tool_calls},
            },
            token_usage=token_usage,
        )
