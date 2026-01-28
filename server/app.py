import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import random

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

# --- 依赖注入函数 ---
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

# --- 数据模型 ---
class QueryRequest(BaseModel):
    doc_id: str
    query: str
    mode: str = "hybrid"
    top_k: int = 15
    chunk_top_k: int = 30
    enable_rerank: bool = True
    vlm_enhanced: bool = True

# --- 路由接口 ---

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """渲染首页"""
    return TEMPLATES.TemplateResponse("index.html", {"request": request})

@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(default=None),
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    """上传并处理文档"""
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
    """执行 RAG 查询"""
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
    """健康检查"""
    return {"status": "ok"}

def _resolve_workspace_root(root: Optional[str]) -> Path:
    settings = LocalRagSettings.from_env()
    if root:
        candidate = Path(root).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
    else:
        candidate = Path(settings.working_dir_root).resolve()

    if not candidate.exists() or not candidate.is_dir():
        raise HTTPException(status_code=400, detail="Workspace root not found.")
    return candidate


def _list_workspaces(root: Path) -> list[dict]:
    """获取本地所有工作空间（文档库）"""
    items = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        items.append(
            {
                "doc_id": entry.name,
                "path": str(entry.resolve()),
                "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items

def _resolve_graph_path(doc_id: str, graph_path: Optional[str]) -> Path:
    settings = LocalRagSettings.from_env()
    base = Path(settings.working_dir_root).resolve()
    if graph_path:
        raw_path = Path(graph_path)
        if raw_path.is_absolute():
            candidate = raw_path.resolve()
        else:
            candidate = (base / doc_id / raw_path).resolve()
    else:
        candidate = (base / doc_id / "graph_chunk_entity_relation.graphml").resolve()

    if base != candidate and base not in candidate.parents:
        raise HTTPException(status_code=400, detail="Graph path is not allowed.")
    return candidate

@app.get("/workspaces")
def list_workspaces(
    root: Optional[str] = None,
    _auth: None = Depends(verify_api_key),
):
    """工作空间列表 API"""
    resolved_root = _resolve_workspace_root(root)
    return {
        "workdir_root": str(resolved_root),
        "workspaces": _list_workspaces(resolved_root),
    }

@app.get("/graph", response_class=HTMLResponse)
def show_graph(
    doc_id: str,
    graph_path: Optional[str] = None,
    _auth: None = Depends(verify_api_key),
):
    """知识图谱可视化 HTML"""
    graph_file = _resolve_graph_path(doc_id, graph_path)
    if not graph_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"GraphML not found: {graph_file}",
        )

    try:
        import networkx as nx
        from pyvis.network import Network
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Missing dependency: {exc}",
        )

    graph = nx.read_graphml(str(graph_file))
    net = Network(height="100vh", width="100%", notebook=False)
    net.from_nx(graph)

    for node in net.nodes:
        node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        description = node.get("description")
        if description:
            node["title"] = description

    for edge in net.edges:
        description = edge.get("description")
        if description:
            edge["title"] = description

    return HTMLResponse(net.generate_html())