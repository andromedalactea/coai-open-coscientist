"""
LLM calling utilities using litellm.

Provides a clean interface for calling LLMs with proper error handling
and JSON parsing.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import jsonschema
from jsonschema.exceptions import ValidationError
import litellm

from .cache import get_cache
from .constants import (
    DEEPSEEK_MAX_OUTPUT_TOKENS,
    DEEPSEEK_REASONING_EFFORT,
    LARGE_CONTEXT_MODEL,
    LLM_HEARTBEAT_INTERVAL,
    LLM_REQUEST_TIMEOUT,
    LLM_TRANSIENT_MAX_ATTEMPTS,
)

logger = logging.getLogger(__name__)

_deepseek_max_tokens_warned = False  # log the clamp warning only once


class _EmptyLLMResponse(Exception):
    """Raised internally when a provider returns no usable content.

    Treated as a transient error so the request is retried (often a different
    OpenRouter provider) instead of crashing the whole pipeline.
    """


def _is_transient_error(exc: Exception) -> bool:
    """Heuristic: should this failure be retried against the provider?

    Covers provider drops, gateway/5xx errors, timeouts, rate limits and the
    empty-content case (common with flaky OpenRouter provider routing).
    """
    if isinstance(exc, _EmptyLLMResponse):
        return True
    s = str(exc).lower()
    markers = (
        "network connection lost",
        "provider_unavailable",
        "provider returned error",
        "overloaded",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection error",
        "temporarily unavailable",
        "rate limit",
        "too many requests",
        "'code': 429",
        "'code': 500",
        "'code': 502",
        "'code': 503",
        "'code': 504",
    )
    return any(m in s for m in markers)


async def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff (capped) between transient retries."""
    delay = min(2 ** (attempt - 1), 15)
    await asyncio.sleep(delay)


async def _acompletion_with_heartbeat(
    completion_args: Dict[str, Any],
    *,
    model_name: str,
    attempt: int,
    max_attempts: int,
    phase: str,
) -> Any:
    """Run litellm.acompletion with an explicit timeout and periodic heartbeat logs.

    Without this, OpenRouter hangs can sit silent for ~600s (provider default)
    before surfacing a Timeout — looking like a freeze. Heartbeats make the wait
    visible; the timeout fails the attempt so transient retries can kick in.
    """
    args = {**completion_args, "timeout": LLM_REQUEST_TIMEOUT}
    prompt_chars = 0
    messages = args.get("messages") or []
    if messages:
        content = messages[0].get("content") if isinstance(messages[0], dict) else None
        if isinstance(content, str):
            prompt_chars = len(content)
    tools = args.get("tools")
    tool_count = len(tools) if tools else 0

    logger.info(
        f"LLM request start [{phase}] model={model_name} "
        f"attempt={attempt}/{max_attempts} timeout={LLM_REQUEST_TIMEOUT:.0f}s "
        f"prompt_chars={prompt_chars} tools={tool_count} "
        f"max_tokens={args.get('max_tokens')}"
    )

    start = time.monotonic()
    task = asyncio.create_task(litellm.acompletion(**args))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=LLM_HEARTBEAT_INTERVAL)
            if done:
                break
            elapsed = time.monotonic() - start
            logger.info(
                f"LLM still waiting [{phase}] model={model_name} "
                f"attempt={attempt}/{max_attempts} elapsed={elapsed:.0f}s "
                f"(timeout={LLM_REQUEST_TIMEOUT:.0f}s)"
            )
        return task.result()
    except Exception:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        elapsed = time.monotonic() - start
        logger.warning(
            f"LLM request failed [{phase}] model={model_name} "
            f"attempt={attempt}/{max_attempts} elapsed={elapsed:.0f}s"
        )
        raise


def _looks_like_llm_refusal(text: str) -> bool:
    """Heuristic: detect short refusal / policy-block replies that are not JSON.

    Seen in the wild from OpenRouter/DeepSeek: Chinese refusals like
    '你好，我无法给到相关内容。' which then poison validation synthesis parsing
    and trigger expensive individual retries.
    """
    if not text or not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # Real JSON hypotheses responses are long; refusals are typically short.
    if len(stripped) > 800:
        return False
    if stripped.startswith("{") or stripped.startswith("["):
        return False
    lower = stripped.lower()
    markers = (
        "i cannot",
        "i can't",
        "i'm unable",
        "i am unable",
        "unable to",
        "cannot provide",
        "can't provide",
        "won't provide",
        "will not provide",
        "not able to",
        "as an ai",
        "against my",
        "content policy",
        "我无法",
        "无法给到",
        "无法提供",
        "不能提供",
        "抱歉",
        "对不起",
    )
    return any(m in lower or m in stripped for m in markers)


def _maybe_add_deepseek_reasoning(completion_args: Dict[str, Any], model_name: str) -> None:
    """Inject the configured reasoning effort for DeepSeek reasoning models.

    Forwarded to OpenRouter via the provider-native ``reasoning.effort`` body
    field (supports DeepSeek-specific values like ``xhigh``). No-op when the
    model is not DeepSeek or no effort is configured.
    """
    if not DEEPSEEK_REASONING_EFFORT or not _is_deepseek_model(model_name):
        return
    extra_body = completion_args.setdefault("extra_body", {})
    extra_body["reasoning"] = {"effort": DEEPSEEK_REASONING_EFFORT}


# suppress Pydantic serialization warnings from LiteLLM globally
# these occur when LiteLLM response objects (Pydantic models) are serialized
# and have mismatched field counts between streaming/non-streaming responses
warnings.filterwarnings("ignore", message=r".*Pydantic serializer warnings.*", category=UserWarning)


def _is_deepseek_model(model_name: str) -> bool:
    """Check if a model name refers to a DeepSeek model."""
    return "deepseek" in model_name.lower()


def _is_context_window_error(exc: Exception) -> bool:
    """Return True if the exception signals a context-window / token-limit overflow.

    Covers litellm.ContextWindowExceededError and the generic
    litellm.BadRequestError whose message contains context-window-related keywords.
    """
    exc_type = type(exc).__name__
    if "ContextWindowExceeded" in exc_type:
        return True
    exc_str = str(exc).lower()
    return (
        "context" in exc_str
        and ("exceed" in exc_str or "length" in exc_str or "maximum" in exc_str)
    )


def _build_deepseek_json_prompt_suffix(json_schema: Dict[str, Any]) -> str:
    """
    Build a prompt suffix that instructs DeepSeek to output JSON matching a schema.

    DeepSeek does not support the `json_schema` response_format. Instead, we
    inject the schema description into the prompt and use `json_object` mode.
    Per DeepSeek docs, the prompt MUST contain the word "json".

    Args:
        json_schema: The JSON schema dict (may have a nested "schema" key).

    Returns:
        A prompt suffix string describing the expected JSON structure.
    """
    actual_schema = json_schema.get("schema", json_schema)
    schema_name = json_schema.get("name", "response")

    def _describe_properties(props: Dict[str, Any], required: list = None) -> str:
        """Recursively describe schema properties for prompt injection."""
        if required is None:
            required = []
        lines = []
        for key, prop in props.items():
            prop_type = prop.get("type", "string")
            desc = prop.get("description", "")
            is_required = key in required
            req_marker = " (required)" if is_required else " (optional)"

            if prop_type == "object" and "properties" in prop:
                nested = _describe_properties(
                    prop["properties"], prop.get("required", [])
                )
                lines.append(f'  "{key}": {{  // object{req_marker}{" - " + desc if desc else ""}\n{nested}\n  }}')
            elif prop_type == "array":
                item_type = prop.get("items", {}).get("type", "string")
                if item_type == "object" and "properties" in prop.get("items", {}):
                    nested = _describe_properties(
                        prop["items"]["properties"],
                        prop["items"].get("required", []),
                    )
                    lines.append(f'  "{key}": [  // array of objects{req_marker}{" - " + desc if desc else ""}\n    {{\n{nested}\n    }}\n  ]')
                else:
                    lines.append(f'  "{key}": []  // array of {item_type}{req_marker}{" - " + desc if desc else ""}')
            elif "enum" in prop:
                enum_vals = ", ".join(f'"{v}"' for v in prop["enum"])
                lines.append(f'  "{key}": "..."  // {prop_type}, one of [{enum_vals}]{req_marker}{" - " + desc if desc else ""}')
            else:
                lines.append(f'  "{key}": "..."  // {prop_type}{req_marker}{" - " + desc if desc else ""}')
        return "\n".join(lines)

    properties = actual_schema.get("properties", {})
    required_fields = actual_schema.get("required", [])
    structure = _describe_properties(properties, required_fields)

    return (
        f"\n\n--- REQUIRED JSON OUTPUT FORMAT ---\n"
        f"You MUST respond with valid JSON matching the \"{schema_name}\" schema below.\n"
        f"Output ONLY the JSON object, no markdown fences, no extra text.\n"
        f"IMPORTANT: Keep all string values concise (1-3 sentences each) to fit within token limits.\n"
        f"IMPORTANT: In string values, escape every backslash (write \\\\mu not \\mu, "
        f"\\\\Delta not \\Delta). Prefer plain Unicode (μm, ΔG) over LaTeX escapes.\n\n"
        f"Expected JSON structure:\n{{\n{structure}\n}}\n"
        f"--- END FORMAT ---"
    )


def _fix_invalid_json_escapes(s: str) -> str:
    """
    Escape backslashes that are not valid JSON escape sequences.

    LLMs often emit LaTeX-like or scientific notation (\\mu, \\Delta, \\approx)
    inside JSON strings; those are invalid escapes and break json.loads mid-document.
    """

    def _replace(match: re.Match) -> str:
        seq = match.group(0)
        # Keep valid escapes: \" \\ \/ \b \f \n \r \t \uXXXX
        if seq in ('\\"', "\\\\", "\\/", "\\b", "\\f", "\\n", "\\r", "\\t"):
            return seq
        if re.fullmatch(r"\\u[0-9a-fA-F]{4}", seq):
            return seq
        # Incomplete \\uXXXX or any other \\X → escape the backslash
        return "\\" + seq

    return re.sub(r"\\u[0-9a-fA-F]{0,4}|\\.", _replace, s)


def _strip_json_comments_and_control_chars(s: str) -> str:
    """Remove // comments and non-whitespace control characters that break JSON."""
    # Strip // line comments (common LLM artifact); keep URLs like https://
    s = re.sub(r"(?<!:)//[^\n]*", "", s)
    # Replace raw control chars (except \t \n \r) with spaces
    return "".join(ch if (ord(ch) >= 32 or ch in "\t\n\r") else " " for ch in s)


def attempt_json_repair(
    json_str: str, allow_major_repairs: bool = False
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Attempt to repair common JSON syntax errors from LLM outputs.

    With json_schema response formats, most responses should be valid JSON.
    This function first tries to parse as-is, and only attempts repairs if needed.

    Args:
        json_str: Potentially malformed JSON string
        allow_major_repairs: If True, attempt major repairs (indicate truncation).
                           If False, only attempt minor repairs (safe syntax fixes).

    Returns:
        Tuple of (parsed JSON dict if successful, was_major_repair: bool)
        Returns (None, False) if all repair attempts failed
    """
    # First, try parsing as-is (should work for json_schema responses)
    try:
        result = json.loads(json_str)
        if isinstance(result, dict):
            return result, False
    except json.JSONDecodeError:
        # JSON is malformed, proceed with repair strategies
        pass

    def close_truncated_json(s: str) -> str:
        """Try to close truncated JSON by adding missing braces/brackets."""
        # Count open vs closed braces and brackets
        open_braces = s.count("{") - s.count("}")
        open_brackets = s.count("[") - s.count("]")

        # Enhanced unterminated string detection
        # Check if the string ends mid-value (unterminated string)
        stripped = s.rstrip()

        # Pattern 1: Ends with opening quote after colon/comma (e.g., ':"text)
        if re.search(r'[:,]\s*"[^"]*$', stripped):
            s = s + '"'
            logger.debug("repaired: unterminated string after colon/comma")

        # Pattern 2: Ends with partial field name (e.g., '"field_na)
        elif re.search(r'"\w+$', stripped):
            # Find if we're in a string literal or field name
            # Count quotes before this position to determine context
            before_partial = stripped[:-20] if len(stripped) > 20 else ""
            quote_count = before_partial.count('"')
            if quote_count % 2 == 1:  # Odd number = we're inside a string
                s = s + '"'
                logger.debug("repaired: unterminated field name/string")

        # Pattern 3: Ends mid-array without closing (e.g., '"item1", "item2)
        elif stripped.endswith(",") or (stripped[-1].isalnum() and "[" in stripped):
            # Likely truncated mid-array or mid-value
            # Try to close intelligently based on context
            last_open_bracket = stripped.rfind("[")
            last_close_bracket = stripped.rfind("]")
            if last_open_bracket > last_close_bracket:
                # We're inside an unclosed array
                # Check if we need to close a string first
                after_bracket = stripped[last_open_bracket:]
                quote_count = after_bracket.count('"')
                if quote_count % 2 == 1:
                    s = s + '"'
                    logger.debug("repaired: unterminated string in array")

        # Remove trailing comma if present
        s = re.sub(r",\s*$", "", s)

        # Add missing closing characters
        # Close arrays first, then objects (proper nesting)
        result = s + ("]" * open_brackets) + ("}" * open_braces)

        if open_braces > 0 or open_brackets > 0:
            logger.debug(f"repaired: added {open_brackets} ']' and {open_braces} '}}'")

        return result

    def _loads_cleaned(s: str) -> Dict[str, Any]:
        cleaned = _strip_json_comments_and_control_chars(s)
        cleaned = _fix_invalid_json_escapes(cleaned)
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            raise TypeError("Parsed JSON is not a dictionary")
        return result

    # Minor repairs (safe, don't indicate truncation)
    minor_repairs = [
        # Fix invalid escapes / control chars / trailing commas (common with scientific text)
        _loads_cleaned,
        # Remove trailing commas before closing braces/brackets
        lambda s: json.loads(re.sub(r",(\s*[}\]])", r"\1", s)),
    ]

    # Major repairs (indicate truncation/incomplete, only on final retry)
    major_repairs = [
        # Close unterminated strings and truncated JSON (most common Gemini issue)
        lambda s: json.loads(close_truncated_json(s)),
        # Clean escapes then close truncation
        lambda s: _loads_cleaned(close_truncated_json(s)),
        # Remove trailing commas AND close truncated JSON
        lambda s: json.loads(close_truncated_json(re.sub(r",(\s*[}\]])", r"\1", s))),
        # Aggressively remove incomplete trailing content and close JSON
        lambda s: json.loads(close_truncated_json(re.sub(r',?\s*"[^"]*$', "", s))),
        # Remove incomplete field (key OR value) and close
        lambda s: json.loads(close_truncated_json(re.sub(r'[:,]\s*"[^"]*$', "", s))),
        # Find last complete comma, truncate there, then close
        lambda s: json.loads(close_truncated_json(s[: s.rfind(",") + 1] if "," in s else s)),
        # Extract first complete JSON object using regex
        lambda s: (
            json.loads(re.search(r"\{.*\}", s, re.DOTALL).group(0))
            if re.search(r"\{.*\}", s, re.DOTALL)
            else None
        ),
    ]

    # Try minor repairs first
    for i, repair_fn in enumerate(minor_repairs):
        try:
            result = repair_fn(json_str)
            if result:
                logger.debug(f"JSON repaired using minor repair strategy {i}")
                return result, False
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.debug(f"minor repair strategy {i} failed: {e}")
            continue

    # If major repairs are allowed, try them
    if allow_major_repairs:
        for i, repair_fn in enumerate(major_repairs):
            try:
                result = repair_fn(json_str)
                if result:
                    logger.warning(
                        f"JSON repaired using major repair strategy {i} (indicates truncation/incomplete response)"
                    )
                    return result, True
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                if i < 2:  # Only log for first few strategies
                    logger.debug(f"major repair strategy {i} failed: {e}")
                continue

    return None, False


def validate_json_schema(result: Dict[str, Any], json_schema: Optional[Dict[str, Any]]) -> None:
    """
    Validate parsed JSON against the provided schema.

    Args:
        result: Parsed JSON dictionary to validate
        json_schema: Optional JSON schema dict (may have nested "schema" key)

    Raises:
        ValidationError: If the result doesn't match the schema
    """
    if json_schema is None:
        # No schema provided, skip validation
        return

    # Extract actual schema from nested structure if present
    actual_schema = json_schema.get("schema", json_schema)

    try:
        jsonschema.validate(instance=result, schema=actual_schema)
        logger.debug("JSON schema validation passed")
    except ValidationError as e:
        logger.warning(f"JSON schema validation failed: {e.message}")
        logger.debug(f"validation error path: {'.'.join(str(p) for p in e.path)}")
        logger.debug(f"first 500 chars of result: {str(result)[:500]}")
        raise


def get_fallback_response(json_schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Get fallback placeholder data for non-critical nodes that failed.

    Args:
        json_schema: Optional JSON schema dict (may have "name" field to identify node)

    Returns:
        Placeholder data matching schema structure, or None if node is critical
    """
    if json_schema is None:
        return None

    schema_name = json_schema.get("name")

    # Non-critical nodes that can fail gracefully
    if schema_name == "proximity_analysis":
        logger.warning(
            "Proximity analysis failed after all retries. "
            "Returning fallback data to continue workflow."
        )
        return {
            "similarity_clusters": [],
            "diversity_assessment": "Analysis failed - skipping deduplication",
            "redundancy_assessment": "Analysis failed - skipping deduplication",
        }

    # Critical nodes - no fallback
    return None


async def call_llm(
    prompt: str,
    model_name: str,
    max_tokens: int = 25000,
    temperature: float = 0.7,
    force_json: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call an LLM via litellm and return the response.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Model name in litellm format (e.g., "gpt-4o-mini", "gemini/gemini-3-flash")
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        force_json: If True, try to force JSON mode (model support varies)
        json_schema: Optional JSON schema to constrain the response format

    Returns:
        String response from the LLM

    Raises:
        Exception: If the LLM call fails
    """
    # clamp temperature for gemini 3 models (requires temp >= 1.0)
    if "gemini-3" in model_name.lower() and temperature < 1.0:
        original_temp = temperature
        temperature = 1.0
        logger.debug(
            f"clamping temperature {original_temp} -> 1.0 for gemini 3 model "
            f"(gemini 3 requires temp >= 1.0 to avoid degraded performance)"
        )

    # clamp max_tokens for deepseek models (API limit is 8192 for deepseek-chat)
    global _deepseek_max_tokens_warned
    if _is_deepseek_model(model_name) and max_tokens > DEEPSEEK_MAX_OUTPUT_TOKENS:
        original_max_tokens = max_tokens
        max_tokens = DEEPSEEK_MAX_OUTPUT_TOKENS
        if not _deepseek_max_tokens_warned:
            logger.warning(
                f"clamping max_tokens {original_max_tokens} -> {DEEPSEEK_MAX_OUTPUT_TOKENS} for DeepSeek model "
                f"(DeepSeek API limit; set DEEPSEEK_MAX_OUTPUT_TOKENS env var to adjust)"
            )
            _deepseek_max_tokens_warned = True

    # Check cache first
    cache = get_cache()
    cached_response = cache.get(
        prompt, model_name, temperature, max_tokens, json_schema=json_schema, force_json=force_json
    )
    if cached_response is not None:
        logger.debug("using cached llm response")
        return cached_response["text"]

    logger.debug(f"cache miss for prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

    # Build completion args once (deterministic across retries)
    completion_args = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "drop_params": True,
    }

    # Try to add response_format based on schema or force_json
    is_deepseek = _is_deepseek_model(model_name)

    if json_schema:
        if is_deepseek:
            # DeepSeek only supports json_object mode, not json_schema.
            # Inject schema description into the prompt instead.
            completion_args["response_format"] = {"type": "json_object"}
            schema_suffix = _build_deepseek_json_prompt_suffix(json_schema)
            completion_args["messages"][0]["content"] += schema_suffix
            logger.debug(
                "DeepSeek model detected: using json_object mode "
                "with schema injected into prompt"
            )
        else:
            completion_args["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
    elif force_json:
        completion_args["response_format"] = {"type": "json_object"}
        if is_deepseek:
            # DeepSeek requires the word "json" in the prompt for json_object mode
            msg_content = completion_args["messages"][0]["content"]
            if "json" not in msg_content.lower():
                completion_args["messages"][0]["content"] += (
                    "\n\nRespond with valid JSON output only."
                )
                logger.debug(
                    "DeepSeek model detected: added 'json' keyword to prompt"
                )

    _maybe_add_deepseek_reasoning(completion_args, model_name)

    last_exc: Optional[Exception] = None
    for attempt in range(1, LLM_TRANSIENT_MAX_ATTEMPTS + 1):
        try:
            logger.info(
                f"Executing LLM call to model: {model_name} "
                f"(attempt {attempt}/{LLM_TRANSIENT_MAX_ATTEMPTS})"
            )
            response = await _acompletion_with_heartbeat(
                completion_args,
                model_name=model_name,
                attempt=attempt,
                max_attempts=LLM_TRANSIENT_MAX_ATTEMPTS,
                phase="call_llm",
            )

            content = response.choices[0].message.content

            if content is None or not content.strip():
                # Often a transient provider drop (e.g. reasoning consumed the
                # response or the upstream connection was lost) - retry.
                raise _EmptyLLMResponse(
                    f"LLM returned None or empty content. Model: {model_name}"
                )

            if _looks_like_llm_refusal(content):
                logger.error(
                    f"LLM refusal detected for '{model_name}' "
                    f"(attempt {attempt}/{LLM_TRANSIENT_MAX_ATTEMPTS}): "
                    f"{content[:200]!r}"
                )
                raise _EmptyLLMResponse(
                    f"LLM refused to answer (non-JSON refusal). Model: {model_name}. "
                    f"Preview: {content[:120]!r}"
                )

            cache.set(
                prompt,
                model_name,
                temperature,
                max_tokens,
                {"text": content},
                json_schema=json_schema,
                force_json=force_json,
            )

            return content

        except Exception as e:
            last_exc = e

            # --- large-context fallback (not retryable on same model) ---
            if _is_context_window_error(e) and model_name != LARGE_CONTEXT_MODEL:
                logger.warning(
                    f"Context window exceeded for model '{model_name}'. "
                    f"Retrying with large-context fallback model '{LARGE_CONTEXT_MODEL}'."
                )
                return await call_llm(
                    prompt=prompt,
                    model_name=LARGE_CONTEXT_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    force_json=force_json,
                    json_schema=json_schema,
                )

            # --- transient provider errors: retry with backoff ---
            if _is_transient_error(e) and attempt < LLM_TRANSIENT_MAX_ATTEMPTS:
                logger.warning(
                    f"Transient LLM error on attempt {attempt}/{LLM_TRANSIENT_MAX_ATTEMPTS} "
                    f"for '{model_name}' ({type(e).__name__}: {e}). Retrying..."
                )
                await _backoff_sleep(attempt)
                continue

            logger.error(f"LLM call failed: {e}")
            logger.error(f"Model: {model_name}, max_tokens: {max_tokens}")
            raise

    # Should not reach here, but guard just in case
    raise last_exc if last_exc else RuntimeError("LLM call failed without exception")


async def call_llm_json(
    prompt: str,
    model_name: str,
    max_tokens: int = 25000,
    temperature: float = 0.7,
    json_schema: Optional[Dict[str, Any]] = None,
    max_attempts: int = 5,
) -> Dict[str, Any]:
    """
    Call an LLM and parse the response as JSON with validation and retry logic.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Model name in litellm format
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        json_schema: Optional JSON schema to constrain the response format
        max_attempts: Maximum number of retry attempts (default 5)

    Returns:
        Parsed JSON response as a dictionary

    Raises:
        json.JSONDecodeError: If response is not valid JSON after all repair attempts (for critical nodes)
        ValidationError: If response doesn't match schema after all retries (for critical nodes)
        Exception: If the LLM call fails or returns empty response
    """
    # Check cache first
    cache = get_cache()
    cached_response = cache.get(
        prompt, model_name, temperature, max_tokens, json_schema=json_schema
    )
    if cached_response is not None:
        logger.debug("using cached llm json response")
        return cached_response

    logger.debug(f"cache miss for prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    last_error = None
    last_response_text = None
    original_prompt = prompt  # save original for retries with feedback

    for attempt in range(1, max_attempts + 1):
        is_final_attempt = attempt == max_attempts

        if attempt > 1:
            logger.debug(f"retrying llm call (attempt {attempt}/{max_attempts})")

        try:
            # Call LLM
            response_text = await call_llm(
                prompt,
                model_name,
                max_tokens,
                temperature,
                force_json=True if not json_schema else False,
                json_schema=json_schema,
            )

            # Check for None or empty response
            if not response_text:
                logger.error("LLM returned None or empty response")
                raise ValueError(
                    "LLM returned None or empty response. Check API keys, rate limits, and model availability."
                )

            # Try to extract JSON from markdown code blocks if present
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            last_response_text = response_text

            # Step 1: Try simple parse first
            result = None
            parse_error = None
            try:
                result = json.loads(response_text)
                if not isinstance(result, dict):
                    parse_error = ValueError("Parsed JSON is not a dictionary")
                    result = None
            except json.JSONDecodeError as e:
                parse_error = e
                result = None

            # Step 2: If parsing succeeded, validate schema
            if result is not None:
                if json_schema is not None:
                    try:
                        validate_json_schema(result, json_schema)
                        # Success! Cache and return
                        cache.set(
                            prompt,
                            model_name,
                            temperature,
                            max_tokens,
                            result,
                            json_schema=json_schema,
                        )
                        return result
                    except ValidationError as e:
                        last_error = e
                        logger.warning(
                            f"Schema validation failed on attempt {attempt}: {e.message}"
                        )

                        # add validation feedback to prompt for next retry
                        if not is_final_attempt:
                            error_path = ".".join(str(p) for p in e.path) if e.path else "root"
                            validation_feedback = f"\n\n--- VALIDATION ERROR FROM PREVIOUS ATTEMPT ---\nError: {e.message}\nLocation: {error_path}\nPlease ensure your JSON output strictly matches the required schema structure.\n---"
                            prompt = original_prompt + validation_feedback
                            logger.debug("added validation feedback to retry prompt")

                        # Retry on validation failure
                        continue
                else:
                    # No schema, parsing succeeded - we're done
                    cache.set(
                        prompt, model_name, temperature, max_tokens, result, json_schema=json_schema
                    )
                    return result

            # Step 3: Parsing failed, attempt repairs
            was_major_repair = False
            if parse_error is not None:
                # Attempt repairs (minor only unless final attempt)
                result, was_major_repair = attempt_json_repair(
                    response_text, allow_major_repairs=is_final_attempt
                )

                if result is not None:
                    # Repair succeeded, validate schema if provided
                    if json_schema is not None:
                        try:
                            validate_json_schema(result, json_schema)
                            # Success! Cache and return
                            cache.set(
                                prompt,
                                model_name,
                                temperature,
                                max_tokens,
                                result,
                                json_schema=json_schema,
                            )
                            return result
                        except ValidationError as e:
                            last_error = e
                            logger.warning(
                                f"Schema validation failed after repair on attempt {attempt}: {e.message}"
                            )

                            # add validation feedback to prompt for next retry
                            if not is_final_attempt:
                                error_path = ".".join(str(p) for p in e.path) if e.path else "root"
                                validation_feedback = f"\n\n--- VALIDATION ERROR FROM PREVIOUS ATTEMPT ---\nError: {e.message}\nLocation: {error_path}\nPlease ensure your JSON output strictly matches the required schema structure.\n---"
                                prompt = original_prompt + validation_feedback
                                logger.debug(
                                    "added validation feedback to retry prompt after repair"
                                )

                            # Retry on validation failure
                            continue
                    else:
                        # No schema, repair succeeded - we're done
                        cache.set(
                            prompt,
                            model_name,
                            temperature,
                            max_tokens,
                            result,
                            json_schema=json_schema,
                        )
                        return result

                # If major repair was needed but we're not on final attempt, retry immediately
                if was_major_repair and not is_final_attempt:
                    logger.info("Major repair needed (truncation detected), retrying immediately")
                    continue

            # All repairs exhausted for this attempt — feed parse error back for next try
            last_error = parse_error or ValueError("All repair strategies failed")
            if not is_final_attempt and parse_error is not None:
                err_ctx = ""
                if isinstance(parse_error, json.JSONDecodeError) and last_response_text:
                    pos = parse_error.pos or 0
                    err_ctx = last_response_text[max(0, pos - 80) : pos + 80]
                parse_feedback = (
                    "\n\n--- JSON PARSE ERROR FROM PREVIOUS ATTEMPT ---\n"
                    f"Error: {parse_error}\n"
                    f"Context around error: ...{err_ctx}...\n"
                    "Return ONLY valid JSON. Escape all backslashes in strings "
                    "(write \\\\mu not \\mu). Do not include comments or trailing commas.\n"
                    "---"
                )
                prompt = original_prompt + parse_feedback
                logger.warning(
                    f"JSON parse failed on attempt {attempt}: {parse_error}. "
                    "Added parse feedback to retry prompt."
                )

        except Exception as e:
            last_error = e
            logger.error(f"LLM call failed on attempt {attempt}: {e}")
            if is_final_attempt:
                raise

    # All retries exhausted
    # Check for fallback for non-critical nodes
    fallback = get_fallback_response(json_schema)
    if fallback is not None:
        logger.warning("Returning fallback data for non-critical node after all retries exhausted")
        return fallback

    # No fallback available - raise appropriate error
    if last_response_text:
        # Log the full response for debugging
        logger.error("Failed to parse JSON response after all repair attempts.")
        logger.error(f"Response length: {len(last_response_text)} chars")
        logger.error(f"First 500 chars: {last_response_text[:500]}")
        logger.error(f"Last 500 chars: {last_response_text[-500:]}")

        # Log middle section too (where errors often are)
        if len(last_response_text) > 1000:
            mid_point = len(last_response_text) // 2
            logger.error(
                f"Middle 500 chars (around char {mid_point}): {last_response_text[mid_point-250:mid_point+250]}"
            )

        # Try to find where JSON is broken
        try:
            # Count braces
            open_braces = last_response_text.count("{")
            close_braces = last_response_text.count("}")
            logger.error(f"Brace count: {{ = {open_braces}, }} = {close_braces}")

            # Try to find first JSON error position
            for i in range(0, len(last_response_text), 100):
                chunk = last_response_text[: i + 100]
                try:
                    json.loads(chunk)
                except json.JSONDecodeError as e:
                    if i > len(last_response_text) - 200:  # Near the end
                        logger.error(f"JSON error near position {e.pos}: {e.msg}")
                        logger.error(
                            f"Context around error: ...{last_response_text[max(0,e.pos-100):e.pos+100]}..."
                        )
                        break
        except Exception as debug_err:
            logger.error(f"Error during debugging: {debug_err}")

    # Raise appropriate error
    if isinstance(last_error, ValidationError):
        raise ValidationError(
            f"Schema validation failed after {max_attempts} attempts: {last_error.message}",
            instance=last_error.instance,
            schema=last_error.schema,
            schema_path=last_error.schema_path,
            path=last_error.path,
        )
    elif isinstance(last_error, json.JSONDecodeError):
        raise json.JSONDecodeError(
            f"Could not parse LLM response as JSON after {max_attempts} attempts",
            last_response_text or "",
            last_error.pos if hasattr(last_error, "pos") else 0,
        )
    else:
        raise json.JSONDecodeError(
            f"Could not parse LLM response as JSON after {max_attempts} attempts",
            last_response_text or "",
            0,
        )


async def call_llm_with_tools(
    prompt: str,
    model_name: str,
    tools: List[Dict[str, Any]],
    tool_executor: Callable,
    max_tokens: int = 25000,
    temperature: float = 0.7,
    max_iterations: int = 10,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Call an LLM with tool access and handle tool execution loop.

    This function implements an agent loop where the LLM can call tools,
    see the results, and continue iterating until it produces a final response.

    Args:
        prompt: The initial user prompt
        model_name: Model name in litellm format
        tools: List of tools in OpenAI format
        tool_executor: Async callable that executes tool calls and returns tool response messages
        max_tokens: Maximum tokens per LLM call
        temperature: Sampling temperature
        max_iterations: Maximum number of LLM calls (prevents infinite loops)

    Returns:
        Tuple of (final_response_text, complete_message_history)

    Raises:
        Exception: If the LLM call fails or max iterations reached
    """
    # clamp temperature for gemini 3 models (requires temp >= 1.0)
    if "gemini-3" in model_name.lower() and temperature < 1.0:
        original_temp = temperature
        temperature = 1.0
        logger.debug(
            f"clamping temperature {original_temp} -> 1.0 for gemini 3 model "
            f"(gemini 3 requires temp >= 1.0 to avoid degraded performance)"
        )

    # clamp max_tokens for deepseek models (API limit is 8192 for deepseek-chat)
    global _deepseek_max_tokens_warned
    if _is_deepseek_model(model_name) and max_tokens > DEEPSEEK_MAX_OUTPUT_TOKENS:
        original_max_tokens = max_tokens
        max_tokens = DEEPSEEK_MAX_OUTPUT_TOKENS
        if not _deepseek_max_tokens_warned:
            logger.warning(
                f"clamping max_tokens {original_max_tokens} -> {DEEPSEEK_MAX_OUTPUT_TOKENS} for DeepSeek model "
                f"(DeepSeek API limit; set DEEPSEEK_MAX_OUTPUT_TOKENS env var to adjust)"
            )
            _deepseek_max_tokens_warned = True

    # Check cache first
    cache = get_cache()
    cached_response = cache.get(prompt, model_name, temperature, max_tokens, tools=tools)
    if cached_response is not None:
        logger.debug("using cached llm tool call response")
        return cached_response["final_response"], cached_response["message_history"]

    logger.debug(f"cache miss for prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

    messages = [{"role": "user", "content": prompt}]

    async def _finalize_without_tools(reason: str) -> tuple[str, List[Dict[str, Any]]]:
        """Force one final completion with tools disabled after budget exhaustion.

        Models (esp. tool-happy ones) often keep calling search tools until the
        hard iteration cap, then the whole pipeline crashes. Soft-stop instead:
        keep tool results already gathered and demand the final JSON answer.
        """
        tool_summary = _summarize_tool_loop_history(messages)
        logger.warning(
            f"Forcing tool-loop finalization ({reason}). "
            f"History: {tool_summary}"
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "TOOL BUDGET EXHAUSTED. Do NOT call any more tools. "
                    "Using the literature/tool results already in this conversation, "
                    "respond NOW with ONLY the required final JSON object "
                    "(no markdown fences, no preamble)."
                ),
            }
        )
        final_args = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "drop_params": True,
            # Omit tools entirely — more reliable than tool_choice=none across providers
        }
        _maybe_add_deepseek_reasoning(final_args, model_name)

        response = None
        for net_attempt in range(1, LLM_TRANSIENT_MAX_ATTEMPTS + 1):
            try:
                response = await _acompletion_with_heartbeat(
                    final_args,
                    model_name=model_name,
                    attempt=net_attempt,
                    max_attempts=LLM_TRANSIENT_MAX_ATTEMPTS,
                    phase="call_llm_with_tools:forced_final",
                )
                break
            except Exception as net_e:
                if _is_context_window_error(net_e):
                    raise
                if _is_transient_error(net_e) and net_attempt < LLM_TRANSIENT_MAX_ATTEMPTS:
                    logger.warning(
                        f"Transient error on forced finalization attempt "
                        f"{net_attempt}/{LLM_TRANSIENT_MAX_ATTEMPTS} for '{model_name}' "
                        f"({type(net_e).__name__}: {net_e}). Retrying..."
                    )
                    await _backoff_sleep(net_attempt)
                    continue
                raise

        final_content = response.choices[0].message.content or ""
        if not final_content.strip():
            raise ValueError(
                f"Forced tool-loop finalization returned empty content. Model: {model_name}"
            )
        if _looks_like_llm_refusal(final_content):
            raise ValueError(
                f"Forced tool-loop finalization got a refusal. Model: {model_name}. "
                f"Preview: {final_content[:120]!r}"
            )

        messages.append({"role": "assistant", "content": final_content})
        logger.info(
            f"Forced tool-loop finalization succeeded "
            f"(final_chars={len(final_content)}; {tool_summary})"
        )
        cache.set(
            prompt,
            model_name,
            temperature,
            max_tokens,
            {"final_response": final_content, "message_history": messages},
            tools=tools,
        )
        return final_content, messages

    for iteration in range(max_iterations):
        remaining_after = max_iterations - (iteration + 1)
        logger.info(
            f"LLM tool-call iteration {iteration + 1}/{max_iterations} "
            f"(remaining_after={remaining_after})"
        )

        try:
            # Call LLM with tools
            logger.info(f"Executing LLM tool call to model: {model_name}")
            tool_completion_args = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "drop_params": True,
            }
            # On the last budgeted turn, omit tools so the model must answer.
            if remaining_after > 0:
                tool_completion_args["tools"] = tools
            else:
                logger.info(
                    f"Last tool-loop turn ({iteration + 1}/{max_iterations}): "
                    "omitting tools to force final answer"
                )
            _maybe_add_deepseek_reasoning(tool_completion_args, model_name)

            # Inner retry for transient provider failures (network drops, 5xx, refusals)
            response = None
            message = None
            for net_attempt in range(1, LLM_TRANSIENT_MAX_ATTEMPTS + 1):
                try:
                    response = await _acompletion_with_heartbeat(
                        tool_completion_args,
                        model_name=model_name,
                        attempt=net_attempt,
                        max_attempts=LLM_TRANSIENT_MAX_ATTEMPTS,
                        phase=f"call_llm_with_tools:iter{iteration + 1}",
                    )
                    message = response.choices[0].message
                    has_tools = bool(getattr(message, "tool_calls", None))
                    content = message.content if message.content else ""
                    if not has_tools and _looks_like_llm_refusal(content):
                        raise _EmptyLLMResponse(
                            f"LLM refused to answer during tool-call loop. "
                            f"Model: {model_name}. Preview: {content[:120]!r}"
                        )
                    break
                except Exception as net_e:
                    if _is_context_window_error(net_e):
                        raise
                    if _is_transient_error(net_e) and net_attempt < LLM_TRANSIENT_MAX_ATTEMPTS:
                        logger.warning(
                            f"Transient LLM tool-call error on attempt "
                            f"{net_attempt}/{LLM_TRANSIENT_MAX_ATTEMPTS} for '{model_name}' "
                            f"({type(net_e).__name__}: {net_e}). Retrying..."
                        )
                        await _backoff_sleep(net_attempt)
                        continue
                    raise

            # Convert message to dict format for history
            message_dict = {
                "role": message.role,
                "content": message.content,
            }

            # Add tool calls if present
            if hasattr(message, "tool_calls") and message.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]

            messages.append(message_dict)

            # Check if LLM wants to call tools
            if hasattr(message, "tool_calls") and message.tool_calls:
                logger.info(
                    f"LLM tool-call iter {iteration + 1}/{max_iterations}: "
                    f"requested {len(message.tool_calls)} tool(s): "
                    + ", ".join(tc.function.name for tc in message.tool_calls)
                )

                # Last budgeted turn still requested tools despite tool_choice=none
                # (some providers ignore it) — execute once, then force finalize.
                is_last_turn = remaining_after == 0

                # Execute all tool calls in parallel
                tool_start = time.monotonic()
                tool_results = await asyncio.gather(
                    *[tool_executor(tc) for tc in message.tool_calls]
                )
                tool_elapsed = time.monotonic() - tool_start
                logger.info(
                    f"LLM tool-call iter {iteration + 1}: executed "
                    f"{len(message.tool_calls)} tool(s) in {tool_elapsed:.1f}s"
                )

                # Add tool results to message history
                messages.extend(tool_results)

                if is_last_turn:
                    return await _finalize_without_tools(
                        f"last iteration {iteration + 1}/{max_iterations} still requested tools"
                    )

                # Nudge the model when budget is nearly gone
                if remaining_after <= 2:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"You have only {remaining_after} tool-loop turn(s) left. "
                                "Stop searching and produce the final JSON answer now "
                                "unless one more tool call is strictly necessary."
                            ),
                        }
                    )
                    logger.info(
                        f"Injected tool-budget nudge ({remaining_after} turn(s) remaining)"
                    )

                # Continue loop - LLM will see tool results and respond
                continue
            else:
                # No tool calls - this is the final response
                final_content = message.content if message.content else ""

                # Validate response before caching
                if not final_content.strip():
                    logger.error("LLM returned empty final response in tool call loop")
                    raise ValueError(f"LLM returned empty final response. Model: {model_name}")

                logger.info(
                    f"LLM tool-call finished after {iteration + 1} iterations "
                    f"(final_chars={len(final_content)})"
                )

                # Cache the successful result (only reached if content is valid)
                cache.set(
                    prompt,
                    model_name,
                    temperature,
                    max_tokens,
                    {"final_response": final_content, "message_history": messages},
                    tools=tools,
                )

                return final_content, messages

        except Exception as e:
            logger.error(f"Error in LLM tool call loop (iteration {iteration + 1}): {e}")
            # --- large-context fallback ---
            if _is_context_window_error(e) and model_name != LARGE_CONTEXT_MODEL:
                logger.warning(
                    f"Context window exceeded for model '{model_name}' on iteration "
                    f"{iteration + 1}. Restarting entire tool call loop with "
                    f"large-context fallback model '{LARGE_CONTEXT_MODEL}'."
                )
                return await call_llm_with_tools(
                    prompt=prompt,
                    model_name=LARGE_CONTEXT_MODEL,
                    tools=tools,
                    tool_executor=tool_executor,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_iterations=max_iterations,
                )
            raise

    # Max iterations reached without a clean final answer — soft-stop instead of crash
    return await _finalize_without_tools(
        f"exceeded max iterations ({max_iterations})"
    )


def _summarize_tool_loop_history(messages: List[Dict[str, Any]]) -> str:
    """Compact summary of tool-loop history for diagnostics."""
    assistant_turns = 0
    tool_calls = 0
    tool_names: Dict[str, int] = {}
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            assistant_turns += 1
            for tc in msg.get("tool_calls") or []:
                tool_calls += 1
                name = (
                    tc.get("function", {}).get("name")
                    if isinstance(tc, dict)
                    else getattr(getattr(tc, "function", None), "name", "unknown")
                )
                tool_names[name or "unknown"] = tool_names.get(name or "unknown", 0) + 1
    names = ", ".join(f"{n}={c}" for n, c in tool_names.items()) or "none"
    return (
        f"assistant_turns={assistant_turns}, tool_calls={tool_calls}, tools=[{names}], "
        f"messages={len(messages)}"
    )