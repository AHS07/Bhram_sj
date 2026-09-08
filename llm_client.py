"""
Single point of contact with the LLM provider.

Two supported profiles (set via .env — no code changes needed to switch):

  TESTING  — Hugging Face free Inference Providers tier
             Base URL : https://router.huggingface.co/v1
             Text model: deepseek-ai/DeepSeek-V4-Flash-0731:baseten
             Vision    : not available on HF — ingestion falls back to native PyMuPDF
             Token     : HF read-scope access token (hf_...)

  PRODUCTION — DeepSeek official API
             Base URL : https://api.deepseek.com
             Text model: deepseek-v4-flash  (routes to 0731 GA automatically)
             Vision    : deepseek-v4-flash-vision-exp
             Token     : DeepSeek API key (sk-...)

JSON mode: DeepSeek does not support response_format={"type":"json_object"}.
JSON output is enforced through the prompt and the parse-then-retry loop.

Thinking mode: DeepSeek V4-Flash supports disabling chain-of-thought via
extra_body={"thinking":{"type":"disabled"}}. This is passed on every call.
The HF router may not support this parameter — if it causes a 400, the call
is retried without it (thinking tokens are then billed but output still works).
"""
import os
import json
import logging
import threading
from openai import OpenAI, BadRequestError

logger = logging.getLogger("bhram.llm")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "TODO_SET_ME")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

TEXT_MODEL   = os.environ.get("DEEPSEEK_TEXT_MODEL",   "deepseek-v4-flash")
VISION_MODEL = os.environ.get("DEEPSEEK_VISION_MODEL", "deepseek-v4-flash-vision-exp")

# Request timeout for all LLM calls (seconds).
# Prevents pipeline workers from blocking indefinitely on a hung TCP connection.
_REQUEST_TIMEOUT = 120

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# Minimum token budget: even with thinking disabled, leave enough room for a
# full JSON extraction response (up to 10 facts × ~150 tokens each).
_MIN_MAX_TOKENS = 512

# Thread-safe flag: whether this provider supports the thinking-disable parameter.
# Starts True; flipped to False permanently on first BadRequestError so subsequent
# calls skip the parameter without another round trip.
# Protected by _thinking_lock to prevent a race condition when multiple asyncio
# worker threads simultaneously inspect and mutate this flag.
_thinking_supported: bool = True
_thinking_lock = threading.Lock()


def _make_params(model: str, messages: list, max_tokens: int) -> dict:
    params = {
        "model":      model,
        "messages":   messages,
        "max_tokens": max_tokens,
        "timeout":    _REQUEST_TIMEOUT,
    }
    with _thinking_lock:
        if _thinking_supported:
            params["extra_body"] = {"thinking": {"type": "disabled"}}
    return params


def call_json(
    system_prompt: str,
    user_content,
    model: str = TEXT_MODEL,
    max_tokens: int = 2000,
) -> dict:
    """
    Calls chat completions and parses the response as JSON. Retries once on
    empty or malformed output. Raises ValueError if both attempts fail.

    Thinking is disabled where supported (DeepSeek direct) to prevent the
    model from spending its entire token budget on chain-of-thought. On
    providers that do not support the parameter (e.g. HF router), the call
    is transparently retried without it.
    """
    global _thinking_supported
    effective_max = max(max_tokens, _MIN_MAX_TOKENS)
    messages = [
        {"role": "system", "content": system_prompt + "\nRespond with valid JSON only."},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(2):
        params = _make_params(model, messages, effective_max)
        try:
            response = client.chat.completions.create(**params)
        except BadRequestError as e:
            # Provider doesn't support the thinking parameter — disable it
            # permanently for this process and retry the same call without it.
            with _thinking_lock:
                if _thinking_supported and "thinking" in str(e).lower():
                    logger.info("Provider does not support thinking parameter — disabling for all future calls.")
                    _thinking_supported = False
                else:
                    raise
            params = _make_params(model, messages, effective_max)
            response = client.chat.completions.create(**params)

        choice = response.choices[0]
        raw = choice.message.content

        if not raw:
            reasoning_tokens = getattr(
                getattr(response.usage, "completion_tokens_details", None),
                "reasoning_tokens", "?"
            )
            logger.warning(
                "call_json attempt %d: empty content (finish_reason=%s, reasoning_tokens=%s)",
                attempt + 1, choice.finish_reason, reasoning_tokens,
            )
            continue

        # Strip markdown code fences if the model wraps its JSON in them.
        stripped = raw.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            stripped = "\n".join(inner).strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            # Model sometimes appends trailing text after the closing brace.
            # Try to extract just the first complete JSON object.
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(stripped)
                return obj
            except json.JSONDecodeError as e:
                logger.warning(
                    "call_json attempt %d: JSON parse failed (%s). raw=%r",
                    attempt + 1, e, raw[:300],
                )
                continue

    raise ValueError("LLM returned no parseable JSON after retry")
