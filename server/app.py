import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from raganything.services.local_rag import LocalRagService, LocalRagSettings

APP_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "templates"))

API_KEY_ENV = "RAGANYTHING_API_KEY"
UPLOAD_DIR = Path(os.getenv("RAGANYTHING_UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAGAnything Local Service")
_service: Optional[LocalRagService] = None


def get_service() -> LocalRagService:
    global _service
    if _service is None:
        settings = LocalRagSettings.from_env()
        _service = LocalRagService(settings)
    return _service


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.getenv(API_KEY_ENV, "").strip()
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


class QueryRequest(BaseModel):
    doc_id: str
    query: str
    mode: str = "hybrid"
    top_k: int = 15
    chunk_top_k: int = 30
    enable_rerank: bool = True
    vlm_enhanced: bool = True


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(default=None),
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    file_path = UPLOAD_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    doc_id = await service.ingest(str(file_path), doc_id=doc_id)
    return {"doc_id": doc_id}


@app.post("/query")
async def query(
    payload: QueryRequest,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    result = await service.query(
        payload.doc_id,
        payload.query,
        mode=payload.mode,
        top_k=payload.top_k,
        chunk_top_k=payload.chunk_top_k,
        enable_rerank=payload.enable_rerank,
        vlm_enhanced=payload.vlm_enhanced,
    )
    return {"answer": result}


@app.get("/health")
def health():
    return {"status": "ok"}
