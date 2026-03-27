"""
FastAPI Entry Point — Indo-Career AI Backend
Serves the LangGraph multi-agent pipeline via HTTP API.
"""
import os
# Force PyTorch backend for sentence-transformers before any imports
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tools.cv_parser import extract_text_from_pdf
from tools.vector_store import get_or_build_job_collection, search_similar_jobs, get_sbert_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: mode real-time aktif — tidak ada dataset lokal yang di-load.
    Lowongan diambil langsung dari LinkedIn, JobStreet, Glints saat query.
    """
    logger.info("Indo-Career AI starting up — mode real-time (LinkedIn + job boards).")
    yield
    logger.info("Indo-Career AI shutting down.")


app = FastAPI(
    title="Indo-Career AI API",
    description="Multi-Agent Career Recommendation System untuk pasar kerja Indonesia",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Returns service status. Mode real-time: lowongan diambil langsung dari LinkedIn."""
    return {
        "status": "ok",
        "mode": "realtime",
        "job_sources": ["LinkedIn", "JobStreet", "Glints"],
        "llm_provider": "Anthropic" if os.getenv("ANTHROPIC_API_KEY") else "OpenRouter",
        "llm_model": os.getenv("AI_MODEL", os.getenv("GAP_MODEL", "not configured")),
    }


# ---------------------------------------------------------------------------
# Main analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
async def analyze_career(
    cv_file: UploadFile = File(..., description="CV dalam format PDF"),
    target_role: str = Form(..., description="Posisi yang dituju"),
    user_name: str = Form(default="", description="Nama pengguna"),
    github_url: str = Form(default="", description="URL GitHub (opsional)"),
):
    """
    Main endpoint: accepts a PDF CV, runs the LangGraph multi-agent pipeline,
    and returns the full CareerState as JSON.

    Graph flow:
      coordinator → [profiler ∥ analyst] → gap_analyzer → strategist
    """
    # Validate file type
    if not cv_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diterima.")

    # Read and parse PDF
    pdf_bytes = await cv_file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="File PDF kosong.")

    try:
        cv_text = extract_text_from_pdf(pdf_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not cv_text.strip():
        raise HTTPException(status_code=422, detail="Tidak dapat mengekstrak teks dari PDF. Pastikan PDF bukan hasil scan.")

    # Build initial state
    initial_state = {
        "cv_text": cv_text,
        "cv_filename": cv_file.filename,
        "target_role": target_role.strip(),
        "user_name": user_name.strip(),
        "github_url": github_url.strip(),
        "user_profile": {},
        "market_data": {},
        "skill_gaps": {},
        "roadmap": "",
        "messages": [],
        "error": "",
        "status": "Memulai...",
    }

    # Import here to avoid circular imports at module load time
    from graph import career_graph

    logger.info(f"Starting analysis for '{target_role}' — CV: {cv_file.filename}")

    # Run LangGraph pipeline asynchronously
    loop = asyncio.get_event_loop()
    try:
        final_state = await loop.run_in_executor(
            None,
            lambda: career_graph.invoke(initial_state),
        )
    except Exception as e:
        logger.error(f"Graph execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan sistem: {str(e)}")

    # Check for agent errors
    if final_state.get("error"):
        logger.warning(f"Graph completed with error: {final_state['error']}")

    # Remove raw cv_text from response (large, not needed by UI)
    response = dict(final_state)
    response.pop("cv_text", None)

    logger.info(f"Analysis complete — match: {final_state.get('skill_gaps', {}).get('persentase_kecocokan', 'N/A')}%")
    return response


# ---------------------------------------------------------------------------
# Job search endpoint (no LLM — just ChromaDB)
# ---------------------------------------------------------------------------

class JobSearchRequest(BaseModel):
    query: str
    limit: int = 10
    filter_city: Optional[str] = None
    filter_industry: Optional[str] = None


@app.post("/api/search-jobs")
async def search_jobs(request: JobSearchRequest):
    """
    Fast semantic job search using ChromaDB + SBERT.
    Does NOT invoke any LLM — returns results in ~100ms after warmup.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")

    try:
        collection = get_or_build_job_collection()
        model = get_sbert_model()

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: search_similar_jobs(
                collection,
                request.query,
                model=model,
                n_results=min(request.limit, 50),
                filter_city=request.filter_city,
                filter_industry=request.filter_industry,
            ),
        )

        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Individual job detail endpoint
# ---------------------------------------------------------------------------

@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    """Returns full metadata for a specific job by ID."""
    try:
        collection = get_or_build_job_collection()
        result = collection.get(ids=[f"job_{job_id}"], include=["metadatas", "documents"])
        if not result["metadatas"]:
            raise HTTPException(status_code=404, detail=f"Lowongan dengan ID '{job_id}' tidak ditemukan.")
        return {
            "metadata": result["metadatas"][0],
            "description": result["documents"][0],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
