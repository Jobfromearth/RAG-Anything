import os
import json
from pathlib import Path
from typing import Optional, Set
import glob

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from raganything.services.local_rag import LocalRagService, LocalRagSettings

# --- 配置与初始化 ---
APP_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "templates"))

API_KEY_ENV = "RAGANYTHING_API_KEY"
UPLOAD_DIR = Path(os.getenv("RAGANYTHING_UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAGAnything Local Service")
_service: Optional[LocalRagService] = None

# --- [核心逻辑 1] 强制在 hybrid_auto 中查找文件 ---
def _find_md_in_hybrid_auto(filename: str) -> Path:
    """
    不管 doc_id 是什么，直接在 output 下的所有子目录里的 hybrid_auto 文件夹找文件。
    路径模式: output / * / hybrid_auto / filename
    """
    settings = LocalRagSettings.from_env()
    output_root = Path(settings.output_dir).resolve()
    
    if not output_root.exists():
        raise HTTPException(status_code=500, detail="Output root directory does not exist")

    # 1. 构造 Glob 模式：找任意子文件夹下的 hybrid_auto 目录下的该文件
    # 模式: */hybrid_auto/filename
    pattern = f"*/hybrid_auto/{filename}"
    
    candidates = list(output_root.glob(pattern))
    
    if candidates:
        return candidates[0]
    
    # 2. 如果没找到，尝试模糊匹配 (比如大小写问题)
    fuzzy_pattern = f"*/hybrid_auto/*{filename}*"
    candidates = list(output_root.glob(fuzzy_pattern))
    
    # 过滤 .md
    md_candidates = [p for p in candidates if p.suffix.lower() == '.md']
    if md_candidates:
        return md_candidates[0]

    raise HTTPException(status_code=404, detail=f"File '{filename}' not found in any 'hybrid_auto' folder.")

# --- 依赖注入 ---
def get_service() -> LocalRagService:
    global _service
    if _service is None:
        settings = LocalRagSettings.from_env()
        _service = LocalRagService(settings)
    return _service

def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.getenv(API_KEY_ENV, "").strip()
    if not expected: return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

# --- 数据模型 ---
class QueryRequest(BaseModel):
    doc_id: str
    query: str
    mode: str = "hybrid"
    top_k: int = 15
    chunk_top_k: int = 30
    enable_rerank: bool = True
    vlm_enhanced: bool = True

# --- 路由 ---

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})

# --- [核心逻辑 2] 列出所有 hybrid_auto 下的 md 文件 ---
@app.get("/files/{doc_id}")
def list_workspace_files(doc_id: str, _auth: None = Depends(verify_api_key)):
    """
    忽略 doc_id 参数！
    直接扫描 /output/ 下所有符合 */hybrid_auto/*.md 结构的文件。
    """
    settings = LocalRagSettings.from_env()
    output_root = Path(settings.output_dir).resolve()
    
    file_list = set()
    
    if output_root.exists():
        # 核心：使用 glob 扫描两层目录下的 hybrid_auto
        md_files = list(output_root.glob("*/hybrid_auto/*.md"))
        
        for p in md_files:
            file_list.add(p.name)

    # 排序返回
    return {"files": sorted(list(file_list))}

# --- [核心逻辑 3] 获取内容 ---
@app.get("/content/{doc_id}")
async def get_document_content(
    doc_id: str, 
    filename: Optional[str] = None,
    _auth: None = Depends(verify_api_key)
):
    # 如果前端没传 filename，自动取列表里的第一个
    if not filename or filename == "undefined":
        files_resp = list_workspace_files(doc_id, _auth)
        files = files_resp["files"]
        if files:
            filename = files[0]
        else:
            raise HTTPException(status_code=404, detail="No processed .md files found in output directory.")

    try:
        # 使用强制 hybrid_auto 查找逻辑
        md_path = _find_md_in_hybrid_auto(filename)
        content = md_path.read_text(encoding="utf-8")
        return {"content": content, "filename": md_path.name}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")

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
    final_id = await service.ingest(str(file_path), doc_id=doc_id)
    return {"doc_id": final_id}

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

@app.get("/workspaces")
def list_workspaces(_auth: None = Depends(verify_api_key)):
    settings = LocalRagSettings.from_env()
    root = Path(settings.working_dir_root).resolve()
    items = []
    if root.exists():
        for entry in root.iterdir():
            if entry.is_dir():
                items.append({"doc_id": entry.name})
    return {"workdir_root": str(root), "workspaces": items}