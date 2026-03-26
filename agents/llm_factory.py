"""
LLM Factory — Centralized model selection with Anthropic primary + OpenRouter fallback.

Priority:
  1. Anthropic API (ANTHROPIC_API_KEY) — primary
     - Parallel agents  (Profiler, Analyst)        → claude-haiku-4-5  (fast & cheap)
     - Sequential agents (Gap Analyzer, Strategist) → claude-sonnet-4-6 (best reasoning)

  2. OpenRouter fallback (OPENAI_API_KEY + OPENAI_BASE_URL)
     - All agents → AI_MODEL (default: x-ai/grok-4-fast)
     - Uses AI_TEMPERATURE / AI_RECOMMENDATION_TEMPERATURE
     - Uses AI_MAX_TOKENS

Shared settings from .env:
  AI_MAX_TOKENS                   — max output tokens (default: 1200)
  AI_TEMPERATURE                  — default temperature for extraction agents (default: 0.35)
  AI_RECOMMENDATION_TEMPERATURE   — temperature for Strategist roadmap (default: 0.3)
"""
import os
import logging

logger = logging.getLogger(__name__)

# Anthropic primary
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# OpenRouter fallback (uses standard OPENAI_* variable names)
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
AI_MODEL        = os.getenv("AI_MODEL", "x-ai/grok-4-fast")

# Shared LLM settings
AI_MAX_TOKENS                  = int(os.getenv("AI_MAX_TOKENS", "1200"))
AI_TEMPERATURE                 = float(os.getenv("AI_TEMPERATURE", "0.35"))
AI_RECOMMENDATION_TEMPERATURE  = float(os.getenv("AI_RECOMMENDATION_TEMPERATURE", "0.3"))

# Anthropic model IDs
HAIKU_MODEL  = os.getenv("PROFILER_MODEL", "claude-haiku-4-5-20251001")
SONNET_MODEL = os.getenv("GAP_MODEL",      "claude-sonnet-4-6")


def _use_anthropic() -> bool:
    return bool(ANTHROPIC_API_KEY)


def _use_openrouter() -> bool:
    return bool(OPENAI_API_KEY)


def get_fast_llm(temperature: float = None, max_tokens: int = None):
    """
    Fast/cheap model for parallel agents (Profiler, Analyst).
    - Anthropic: claude-haiku-4-5  ($1/$5 per MTok)
    - OpenRouter fallback: AI_MODEL (x-ai/grok-4-fast)

    Uses AI_TEMPERATURE and AI_MAX_TOKENS from .env if not explicitly passed.
    """
    temp       = temperature if temperature is not None else AI_TEMPERATURE
    max_tok    = max_tokens  if max_tokens  is not None else AI_MAX_TOKENS

    if _use_anthropic():
        from langchain_anthropic import ChatAnthropic
        logger.debug(f"LLM [fast]: Anthropic {HAIKU_MODEL}")
        return ChatAnthropic(
            model=HAIKU_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=temp,
            max_tokens=max_tok,
        )

    if _use_openrouter():
        from langchain_openai import ChatOpenAI
        logger.debug(f"LLM [fast]: OpenRouter {AI_MODEL}")
        return ChatOpenAI(
            model=AI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=temp,
            max_tokens=max_tok,
        )

    raise EnvironmentError(
        "Tidak ada API key LLM. Set ANTHROPIC_API_KEY atau OPENAI_API_KEY di file .env"
    )


def get_quality_llm(temperature: float = None, max_tokens: int = None):
    """
    High-quality model for critical sequential agents (Gap Analyzer, Strategist).
    - Anthropic: claude-sonnet-4-6  ($3/$15 per MTok)
    - OpenRouter fallback: AI_MODEL (x-ai/grok-4-fast)

    Strategist uses AI_RECOMMENDATION_TEMPERATURE from .env.
    """
    temp    = temperature if temperature is not None else AI_TEMPERATURE
    max_tok = max_tokens  if max_tokens  is not None else AI_MAX_TOKENS

    if _use_anthropic():
        from langchain_anthropic import ChatAnthropic
        logger.debug(f"LLM [quality]: Anthropic {SONNET_MODEL}")
        return ChatAnthropic(
            model=SONNET_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=temp,
            max_tokens=max_tok,
        )

    if _use_openrouter():
        from langchain_openai import ChatOpenAI
        logger.debug(f"LLM [quality]: OpenRouter {AI_MODEL}")
        return ChatOpenAI(
            model=AI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=temp,
            max_tokens=max_tok,
        )

    raise EnvironmentError(
        "Tidak ada API key LLM. Set ANTHROPIC_API_KEY atau OPENAI_API_KEY di file .env"
    )
