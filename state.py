from typing import TypedDict, Annotated
import operator


def _last_value(existing, new):
    """Reducer: keep the latest non-empty value (for status/error fields)."""
    return new if new else existing


class CareerState(TypedDict):
    # --- Inputs (set once at graph entry) ---
    cv_text: str
    cv_filename: str
    target_role: str
    user_name: str
    github_url: str

    # --- Agent outputs ---
    user_profile: dict      # From Profiler
    market_data: dict       # From Market Analyst
    skill_gaps: dict        # From Gap Analyzer
    roadmap: str            # From Strategist

    # --- Metadata ---
    # append-only: both parallel agents can safely write without conflict
    messages: Annotated[list, operator.add]
    # last-write-wins with empty-value guard: parallel agents can both write
    error: Annotated[str, _last_value]
    status: Annotated[str, _last_value]
