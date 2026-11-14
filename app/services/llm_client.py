"""Provider-agnostic JSON completion. Callers ask for JSON matching a schema; the provider is
chosen by LLM_PROVIDER in .env ("anthropic" or "groq")."""
import json
from functools import lru_cache

from app.config import get_settings


class LLMUnavailable(RuntimeError):
    """No provider configured, or the provider declined / returned unusable output."""


def complete_json(system: str, user: str, schema: dict, max_tokens: int = 4000) -> dict:
    s = get_settings()
    if not s.llm_enabled:
        raise LLMUnavailable(f"No API key configured for LLM_PROVIDER={s.llm_provider}")
    if s.llm_provider == "groq":
        return _groq(system, user, schema, max_tokens)
    return _anthropic(system, user, schema, max_tokens)


# ── Anthropic ────────────────────────────────────────────────────────────────

@lru_cache
def _anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key)


def _anthropic(system: str, user: str, schema: dict, max_tokens: int) -> dict:
    response = _anthropic_client().beta.messages.create(
        model=get_settings().anthropic_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise LLMUnavailable("Model declined the request")
    return json.loads(next(b.text for b in response.content if b.type == "text"))


# ── Groq (OpenAI-compatible chat completions) ────────────────────────────────

@lru_cache
def _groq_client():
    from groq import Groq
    return Groq(api_key=get_settings().groq_api_key)


def _groq(system: str, user: str, schema: dict, max_tokens: int) -> dict:
    # JSON mode guarantees valid JSON but not the schema, so the schema goes in the prompt
    # and the result is checked for required keys.
    system = (f"{system}\n\nRespond with only a JSON object matching this JSON Schema:\n"
              f"{json.dumps(schema)}")
    response = _groq_client().chat.completions.create(
        model=get_settings().groq_model,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    try:
        out = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError) as e:
        raise LLMUnavailable("Groq returned invalid JSON") from e
    missing = [k for k in schema.get("required", []) if k not in out]
    if missing:
        raise LLMUnavailable(f"Groq response missing fields: {missing}")
    return out
