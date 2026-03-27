"""
LLM Factory — Per-agent model assignment via OpenRouter.

Semua agent menggunakan OpenRouter (https://openrouter.ai/api/v1) sebagai provider.
Setiap agent bisa dikonfigurasi model yang berbeda via .env — tidak perlu ubah kode.

Model defaults:
  Profiler   (paralel, ekstraksi CV)      → x-ai/grok-4-fast
  Analyst    (paralel, riset pasar)        → x-ai/grok-4-fast
  Gap Analyzer (sequential, reasoning)    → anthropic/claude-sonnet-4-6
  Strategist  (sequential, roadmap panjang)→ anthropic/claude-sonnet-4-6

Semua model diakses via satu endpoint OpenRouter — bisa mix Opus, Grok, GPT-4o, dll.
Paralel tetap berjalan normal — LangGraph Fan-Out/Fan-In tidak terpengaruh model.

Shared settings (dari .env):
  AI_MAX_TOKENS                   — max output tokens (default: 1200)
  AI_TEMPERATURE                  — temperature untuk agent ekstraksi (default: 0.35)
  AI_RECOMMENDATION_TEMPERATURE   — temperature untuk Strategist (default: 0.3)
"""
import os
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# OpenRouter — satu provider untuk semua agent
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

# Model per agent — dikonfigurasi via .env
PROFILER_MODEL   = os.getenv("PROFILER_MODEL",   "x-ai/grok-4-fast")
ANALYST_MODEL    = os.getenv("ANALYST_MODEL",    "x-ai/grok-4-fast")
GAP_MODEL        = os.getenv("GAP_MODEL",        "anthropic/claude-sonnet-4-6")
STRATEGIST_MODEL = os.getenv("STRATEGIST_MODEL", "anthropic/claude-sonnet-4-6")

# Shared LLM settings
AI_MAX_TOKENS                 = int(os.getenv("AI_MAX_TOKENS", "1200"))
AI_TEMPERATURE                = float(os.getenv("AI_TEMPERATURE", "0.35"))
AI_RECOMMENDATION_TEMPERATURE = float(os.getenv("AI_RECOMMENDATION_TEMPERATURE", "0.3"))


def _build_llm(model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY tidak ditemukan. Set OPENAI_API_KEY di file .env "
            "(gunakan API key dari https://openrouter.ai)"
        )
    logger.debug(f"LLM: OpenRouter model={model} temp={temperature} max_tokens={max_tokens}")
    return ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_profiler_llm(temperature: float = None, max_tokens: int = None) -> ChatOpenAI:
    """
    Profiler Agent — berjalan PARALEL dengan Analyst.
    Default: x-ai/grok-4-fast (cepat, murah, cukup untuk ekstraksi CV).
    Override: set PROFILER_MODEL di .env (contoh: anthropic/claude-opus-4)
    """
    return _build_llm(
        model=PROFILER_MODEL,
        temperature=temperature if temperature is not None else AI_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else AI_MAX_TOKENS,
    )


def get_analyst_llm(temperature: float = None, max_tokens: int = None) -> ChatOpenAI:
    """
    Market Analyst Agent — berjalan PARALEL dengan Profiler.
    Default: x-ai/grok-4-fast (cepat, cukup untuk agregasi data pasar).
    Override: set ANALYST_MODEL di .env
    """
    return _build_llm(
        model=ANALYST_MODEL,
        temperature=temperature if temperature is not None else AI_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else AI_MAX_TOKENS,
    )


def get_gap_llm(temperature: float = None, max_tokens: int = None) -> ChatOpenAI:
    """
    Gap Analyzer Agent — sequential (Fan-In), butuh reasoning lebih baik.
    Default: anthropic/claude-sonnet-4-6 (via OpenRouter).
    Override: set GAP_MODEL di .env (contoh: openai/gpt-4o)
    """
    return _build_llm(
        model=GAP_MODEL,
        temperature=temperature if temperature is not None else AI_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else AI_MAX_TOKENS,
    )


def get_strategist_llm(temperature: float = None, max_tokens: int = None) -> ChatOpenAI:
    """
    Strategist Agent — sequential, output terpanjang (roadmap 6 bulan Bahasa Indonesia).
    Default: anthropic/claude-sonnet-4-6 (via OpenRouter).
    Override: set STRATEGIST_MODEL di .env (contoh: anthropic/claude-opus-4)
    Uses AI_RECOMMENDATION_TEMPERATURE by default.
    """
    return _build_llm(
        model=STRATEGIST_MODEL,
        temperature=temperature if temperature is not None else AI_RECOMMENDATION_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else AI_MAX_TOKENS,
    )


# ---------------------------------------------------------------------------
# Legacy aliases — dipertahankan agar tidak ada import error di file lain
# yang masih memanggil get_fast_llm() / get_quality_llm()
# ---------------------------------------------------------------------------
def get_fast_llm(temperature: float = None, max_tokens: int = None) -> ChatOpenAI:
    """Alias untuk get_profiler_llm() — backward compat."""
    return get_profiler_llm(temperature=temperature, max_tokens=max_tokens)


def get_quality_llm(temperature: float = None, max_tokens: int = None) -> ChatOpenAI:
    """Alias untuk get_gap_llm() — backward compat."""
    return get_gap_llm(temperature=temperature, max_tokens=max_tokens)
