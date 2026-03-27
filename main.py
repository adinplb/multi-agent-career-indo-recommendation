"""
FastAPI Entry Point — Indo-Career AI Backend
Serves the LangGraph multi-agent pipeline via HTTP API.
Job data: Tavily real-time search + FAISS semantic embeddings.
"""
import os
import io
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: mode real-time aktif.
    Tavily + FAISS — tidak ada dataset lokal yang di-load.
    """
    logger.info("Indo-Career AI starting up — Tavily + FAISS embeddings mode.")
    yield
    logger.info("Indo-Career AI shutting down.")


app = FastAPI(
    title="Indo-Career AI API",
    description="Multi-Agent Career Recommendation System untuk pasar kerja Indonesia",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Attachment text extraction helper
# ---------------------------------------------------------------------------

async def extract_attachment_text(files: list[UploadFile]) -> str:
    """
    Ekstrak teks dari file lampiran (PDF, DOCX, TXT).
    Truncate: 5.000 char per file, 15.000 char total.
    """
    if not files:
        return ""

    all_texts = []
    for f in files:
        if not f or not f.filename:
            continue
        try:
            content = await f.read()
            fname = f.filename.lower()

            if fname.endswith(".pdf"):
                text = extract_text_from_pdf(content)
            elif fname.endswith(".docx"):
                from docx import Document
                doc = Document(io.BytesIO(content))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            elif fname.endswith(".txt"):
                text = content.decode("utf-8", errors="ignore")
            else:
                continue

            text = text.strip()[:5000]
            if text:
                all_texts.append(f"[{f.filename}]\n{text}")

        except Exception as e:
            logger.warning("Gagal ekstrak lampiran '%s': %s", f.filename, e)

    combined = "\n\n---\n\n".join(all_texts)
    return combined[:15000]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Returns service status."""
    return {
        "status": "ok",
        "mode": "realtime",
        "job_sources": ["Tavily"],
        "search_backend": "Tavily + FAISS embeddings",
        "llm_provider": "Anthropic" if os.getenv("ANTHROPIC_API_KEY") else "OpenRouter",
        "llm_model": os.getenv("AI_MODEL", os.getenv("GAP_MODEL", "not configured")),
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
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
    additional_context: str = Form(default="", description="Konteks tambahan: preferensi, minat, tujuan karier"),
    extra_files: list[UploadFile] = File(default=[], description="File lampiran tambahan: PDF, DOCX, TXT"),
):
    """
    Main endpoint: menerima CV PDF + konteks, menjalankan LangGraph multi-agent pipeline.

    Graph flow:
      coordinator → [profiler ∥ analyst] → gap_analyzer → strategist
    """
    if not cv_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diterima.")

    pdf_bytes = await cv_file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="File PDF kosong.")

    try:
        cv_text = extract_text_from_pdf(pdf_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not cv_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Tidak dapat mengekstrak teks dari PDF. Pastikan PDF bukan hasil scan.",
        )

    # Ekstrak teks dari file lampiran
    attachments_text = await extract_attachment_text(extra_files)

    initial_state = {
        "cv_text": cv_text,
        "cv_filename": cv_file.filename,
        "target_role": target_role.strip(),
        "user_name": user_name.strip(),
        "github_url": github_url.strip(),
        "additional_context": additional_context.strip(),
        "attachments_text": attachments_text,
        "user_profile": {},
        "market_data": {},
        "skill_gaps": {},
        "roadmap": "",
        "messages": [],
        "error": "",
        "status": "Memulai...",
    }

    from graph import career_graph

    logger.info(
        "Mulai analisis '%s' — CV: %s | konteks: %d chars | lampiran: %d chars",
        target_role, cv_file.filename, len(additional_context), len(attachments_text),
    )

    loop = asyncio.get_event_loop()
    try:
        final_state = await loop.run_in_executor(
            None,
            lambda: career_graph.invoke(initial_state),
        )
    except Exception as e:
        logger.error("Graph execution error: %s", e)
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan sistem: {str(e)}")

    if final_state.get("error"):
        logger.warning("Graph selesai dengan error: %s", final_state["error"])

    response = dict(final_state)
    response.pop("cv_text", None)
    response.pop("attachments_text", None)  # jangan kirim ke UI (bisa besar)

    logger.info(
        "Analisis selesai — match: %s%%",
        final_state.get("skill_gaps", {}).get("persentase_kecocokan", "N/A"),
    )
    return response


# ---------------------------------------------------------------------------
# Job search endpoint (Tavily — no LLM)
# ---------------------------------------------------------------------------

class JobSearchRequest(BaseModel):
    query: str
    limit: int = 10
    filter_city: Optional[str] = None
    filter_industry: Optional[str] = None


@app.post("/api/search-jobs")
async def search_jobs(request: JobSearchRequest):
    """
    Pencarian lowongan real-time via Tavily.
    Returns hasil dari LinkedIn/JobStreet/Glints dalam ~2-5 detik.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")

    try:
        from tools.tavily_search import search_jobs_tavily

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: search_jobs_tavily(
                request.query,
                location="Indonesia",
                num_results=min(request.limit, 20),
            ),
        )
        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
