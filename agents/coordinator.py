"""
Coordinator Agent
Entry-point node that validates input and triggers Fan-Out to Profiler and Market Analyst.
"""
import logging
from langgraph.types import Send
from state import CareerState

logger = logging.getLogger(__name__)


def coordinator_node(state: CareerState) -> dict:
    """
    Validates that cv_text and target_role are present.
    If validation fails, sets error and returns immediately (routes to END via error check).
    Does NOT call LLM — pure routing/validation logic.
    """
    cv_text = state.get("cv_text", "").strip()
    target_role = state.get("target_role", "").strip()

    if not cv_text:
        return {
            "error": "CV tidak ditemukan. Silakan upload file PDF CV Anda.",
            "status": "Error: CV kosong",
            "messages": ["Coordinator: CV text kosong, analisis dihentikan."],
        }

    if not target_role:
        return {
            "error": "Posisi yang dituju belum diisi. Silakan masukkan target role Anda.",
            "status": "Error: Target role kosong",
            "messages": ["Coordinator: Target role kosong, analisis dihentikan."],
        }

    logger.info(f"Coordinator: memulai analisis untuk '{target_role}'")
    return {
        "status": "Memulai analisis paralel...",
        "messages": [
            f"Coordinator: Memulai analisis karier untuk posisi '{target_role}'.",
            "Coordinator: Profiler dan Market Analyst berjalan secara paralel.",
        ],
        "error": "",
    }


def fan_out_router(state: CareerState):
    """
    LangGraph Fan-Out router using the Send API.
    Returns a list of Send objects to dispatch Profiler and Market Analyst IN PARALLEL.

    This is the correct LangGraph pattern for true parallelism:
    - Both nodes receive the full current state
    - They run on separate threads
    - Gap Analyzer waits for BOTH to complete (Fan-In)
    """
    # If there's an error from coordinator, don't fan out
    if state.get("error"):
        return []

    return [
        Send("profiler", state),
        Send("analyst", state),
    ]
