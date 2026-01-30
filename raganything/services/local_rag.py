#!/usr/bin/env python
"""
新开图谱 (处理整个文件夹)：
python raganything/services/local_rag.py -p ./data/my_paper_folder -i My_New_Graph

补充文件 (向已有图谱添加)：
python raganything/services/local_rag.py -p ./data/extra.pdf -i My_New_Graph
"""

import asyncio
import hashlib
import logging
import logging.config
import os
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

from raganything import RAGAnything, RAGAnythingConfig

_MODEL_CACHE: Dict[str, Any] = {}


@dataclass
class LocalRagSettings:
    tiktoken_cache_dir: str =  "/data/h50056787/workspaces/lightrag/tiktoken_cache"
    embedding_model_path: str =  "/data/h50056787/models/bge-m3"
    rerank_model_path: str = "/data/h50056787/models/bge-reranker-v2-m3"

    working_dir_root: str = "./rag_workspace"
    output_dir: str = "./output"
    log_dir: str = "./logs"
    
    vllm_api_base: str = "http://localhost:8001/v1"
    vllm_api_key: str = "EMPTY"
    llm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    device: str = "cuda:0"

    embedding_dim: int = 1024
    max_token_size: int = 8192
    temperature: float = 0.1
    max_tokens: int = 1024
    vision_max_tokens: int = 2048
    max_prompt_chars: int = 12000
    max_vlm_prompt_chars: int = 12000
    max_vlm_images: int = 5
    default_top_k: int = 15
    default_chunk_top_k: int = 30
    base_system_prompt: str = (
        "You are Qwen2-VL, an expert multimodal AI assistant. "
        "1. First, analyze ALL images provided at the start. "
        "2. Then, read the retrieved text context carefully. "
        "3. Answer the user's question based ONLY on the provided images and text. "
        "4. CRITICAL: Cite your sources using the [Source ID] format found in the text. "
        "5. If the requested item (e.g., Equation N, Table N, Figure N) is not present in the provided context/images, "
        "say 'I don't know' and do NOT substitute a different item. "
        "6. If the answer is not in the context/images, say 'I don't know'. "
        "7. SPECIFICALLY FOR TABLES OR NUMBERS: If the table/value is not explicitly provided, DO NOT guess any numbers. "
        "8. Never invent identifiers, equation numbers, or table numbers. "
        "9. If you answer with a number, it must appear verbatim in the provided context/images and be cited."
    )

    @classmethod
    def from_env(cls) -> "LocalRagSettings":
        return cls(
            tiktoken_cache_dir=os.getenv("TIKTOKEN_CACHE_DIR", "/data/h50056787/workspaces/lightrag/tiktoken_cache"),
            embedding_model_path=os.getenv(
                "RAGANYTHING_EMBEDDING_MODEL_PATH", "/data/h50056787/models/bge-m3"
            ),
            rerank_model_path=os.getenv(
                "RAGANYTHING_RERANK_MODEL_PATH", "/data/h50056787/models/bge-reranker-v2-m3"
            ),
            log_dir=os.getenv("RAGANYTHING_LOG_DIR", "./logs"),
            vllm_api_base=os.getenv("VLLM_API_BASE", "http://localhost:8001/v1"),
            vllm_api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            llm_model_name=os.getenv(
                "LLM_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct"
            ),
            device=os.getenv("RAGANYTHING_DEVICE", "cuda:0"),
            working_dir_root=os.getenv("RAGANYTHING_WORKDIR_ROOT", "./rag_workspace"),
            output_dir=os.getenv("RAGANYTHING_OUTPUT_DIR", "./output"),
            embedding_dim=int(os.getenv("RAGANYTHING_EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("RAGANYTHING_MAX_TOKEN_SIZE", "8192")),
            temperature=float(os.getenv("RAGANYTHING_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("RAGANYTHING_MAX_TOKENS", "4096")),
            vision_max_tokens=int(os.getenv("RAGANYTHING_VISION_MAX_TOKENS", "2048")),
            max_prompt_chars=int(os.getenv("RAGANYTHING_MAX_PROMPT_CHARS", "12000")),
            max_vlm_prompt_chars=int(os.getenv("RAGANYTHING_MAX_VLM_PROMPT_CHARS", "12000")),
            max_vlm_images=int(os.getenv("RAGANYTHING_MAX_VLM_IMAGES", "5")),
            default_top_k=int(os.getenv("RAGANYTHING_DEFAULT_TOP_K", "15")),
            default_chunk_top_k=int(os.getenv("RAGANYTHING_DEFAULT_CHUNK_TOP_K", "30")),
            base_system_prompt=os.getenv(
                "RAGANYTHING_BASE_SYSTEM_PROMPT",
                cls().base_system_prompt,
            ),
        )


def configure_logging(settings: LocalRagSettings) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = Path(settings.log_dir) / f"run_{timestamp}.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "detailed",
                    "filename": str(log_file_path),
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "": {"handlers": ["console", "file"], "level": "INFO"},
            },
        }
    )
    return logging.getLogger(__name__)


def _model_cache_key(settings: LocalRagSettings) -> str:
    return f"{settings.embedding_model_path}|{settings.rerank_model_path}|{settings.device}"


def load_models(settings: LocalRagSettings) -> tuple[SentenceTransformer, CrossEncoder]:
    key = _model_cache_key(settings)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    st_model = SentenceTransformer(
        settings.embedding_model_path,
        trust_remote_code=True,
        device=settings.device,
    )
    reranker_model = CrossEncoder(
        settings.rerank_model_path,
        device=settings.device,
        trust_remote_code=True,
    )
    _MODEL_CACHE[key] = (st_model, reranker_model)
    return st_model, reranker_model


def build_embedding_func(
    settings: LocalRagSettings, st_model: SentenceTransformer
) -> EmbeddingFunc:
    async def _compute_embedding(texts: list[str]) -> np.ndarray:
        return st_model.encode(texts, normalize_embeddings=True)

    return EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=settings.max_token_size,
        func=_compute_embedding,
    )


def build_rerank_func(reranker_model: CrossEncoder, logger: logging.Logger):
    async def rerank_func(query: str, documents: list[str], top_n: int) -> list[dict]:
        if not documents:
            return []
        try:
            pairs = [[query, doc] for doc in documents]
            scores = reranker_model.predict(pairs)
            results = [
                {"index": i, "relevance_score": float(score)}
                for i, score in enumerate(scores)
            ]
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:top_n]
        except Exception as exc:
            logger.error(f"Rerank Error: {exc}")
            return []

    return rerank_func


def build_llm_model_func(
    settings: LocalRagSettings, client: AsyncOpenAI, logger: logging.Logger
):
    async def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        history_messages = history_messages or []
        cleaned_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["hashing_kv", "keyword_extraction", "enable_cot"]
        }
        if isinstance(prompt, str) and settings.max_prompt_chars > 0:
            if len(prompt) > settings.max_prompt_chars:
                logger.warning(
                    f"Prompt too long ({len(prompt)} chars), truncating to {settings.max_prompt_chars}"
                )
                prompt = prompt[: settings.max_prompt_chars]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        try:
            response = await client.chat.completions.create(
                model=settings.llm_model_name,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                **cleaned_kwargs,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.error(f"LLM Error: {exc}")
            return ""

    return llm_model_func


def build_vision_model_func(
    settings: LocalRagSettings, client: AsyncOpenAI, logger: logging.Logger
):
    async def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        history_messages = history_messages or []
        cleaned_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["hashing_kv", "keyword_extraction", "enable_cot"]
        }

        if messages:
            original_sys = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            if original_sys:
                if settings.base_system_prompt and settings.base_system_prompt in original_sys:
                    full_system = original_sys
                else:
                    full_system = (
                        f"{settings.base_system_prompt}\n\nAdditional Instructions:\n"
                        f"{original_sys}"
                    )
            else:
                full_system = settings.base_system_prompt

            user_content_list = []
            for msg in messages:
                if msg["role"] == "user":
                    if isinstance(msg["content"], list):
                        user_content_list.extend(msg["content"])
                    else:
                        user_content_list.append(
                            {"type": "text", "text": str(msg["content"])}
                        )

            images_part = [
                item
                for item in user_content_list
                if item.get("type") == "image_url"
            ]

            max_images = settings.max_vlm_images
            if max_images > 0 and len(images_part) > max_images:
                logger.warning(
                    f"Too many images ({len(images_part)}), truncating to {max_images}"
                )
                images_part = images_part[:max_images]

            texts_part = [
                item.get("text", "").strip()
                for item in user_content_list
                if item.get("type") == "text"
            ]
            full_text_context = "\n\n".join([t for t in texts_part if t])
            if settings.max_vlm_prompt_chars > 0 and len(full_text_context) > settings.max_vlm_prompt_chars:
                logger.warning(
                    f"VLM prompt too long ({len(full_text_context)} chars), truncating to {settings.max_vlm_prompt_chars}"
                )
                full_text_context = full_text_context[: settings.max_vlm_prompt_chars]

            citation_reminder = (
                "\n\n----------------\n"
                "FINAL INSTRUCTION:\n"
                "You MUST cite your sources using the format [doc_id] or [Source ID] "
                "at the end of every sentence where you use information from the context. "
            )
            final_text_payload = (
                f"--- RETRIEVED CONTEXT & QUESTION ---\n"
                f"{full_text_context}{citation_reminder}"
            )

            final_user_content = []
            final_user_content.extend(images_part)
            final_user_content.append({"type": "text", "text": final_text_payload})

            final_messages = [
                {"role": "system", "content": full_system},
                {"role": "user", "content": final_user_content},
            ]

            try:
                response = await client.chat.completions.create(
                    model=settings.llm_model_name,
                    messages=final_messages,
                    temperature=settings.temperature,
                    max_tokens=settings.vision_max_tokens,
                    **cleaned_kwargs,
                )
                return response.choices[0].message.content
            except Exception as exc:
                logger.error(f"Vision LLM Error: {exc}")
                raise

        if image_data:
            import base64
            if isinstance(image_data, bytes):
                base64_image = base64.b64encode(image_data).decode("utf-8")
            else:
                base64_image = str(image_data)
            
            user_content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
            
        return await build_llm_model_func(settings, client, logger)(
            prompt, system_prompt, history_messages, **kwargs
        )

    return vision_model_func


def _safe_doc_id(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_"))
    if cleaned:
        return cleaned
    return hashlib.md5(name.encode("utf-8")).hexdigest()


class LocalRagService:
    def __init__(self, settings: Optional[LocalRagSettings] = None):
        self.settings = settings or LocalRagSettings.from_env()
        os.environ["TIKTOKEN_CACHE_DIR"] = self.settings.tiktoken_cache_dir
        self.logger = configure_logging(self.settings)
        self.client = AsyncOpenAI(
            api_key=self.settings.vllm_api_key, base_url=self.settings.vllm_api_base
        )
        self._rag_instances: Dict[str, RAGAnything] = {}
        self._init_lock = asyncio.Lock()

        st_model, reranker_model = load_models(self.settings)
        self.embedding_func = build_embedding_func(self.settings, st_model)
        self.rerank_func = build_rerank_func(reranker_model, self.logger)
        self.llm_model_func = build_llm_model_func(
            self.settings, self.client, self.logger
        )
        self.vision_model_func = build_vision_model_func(
            self.settings, self.client, self.logger
        )

    def _build_rag(self, working_dir: str) -> RAGAnything:
        config = RAGAnythingConfig(
            working_dir=working_dir,
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        return RAGAnything(
            config=config,
            llm_model_func=self.llm_model_func,
            vision_model_func=self.vision_model_func,
            embedding_func=self.embedding_func,
            lightrag_kwargs={
                "top_k": self.settings.default_top_k,
                "chunk_top_k": self.settings.default_chunk_top_k,
                "rerank_model_func": self.rerank_func,
            },
        )

    async def get_rag(self, doc_id: str) -> RAGAnything:
        async with self._init_lock:
            if doc_id in self._rag_instances:
                return self._rag_instances[doc_id]
            working_dir = str(Path(self.settings.working_dir_root) / doc_id)
            rag = self._build_rag(working_dir)
            self._rag_instances[doc_id] = rag
            return rag

    async def ingest(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Input not found: {file_path}")

        doc_id = doc_id or _safe_doc_id(file_path_obj.stem)
        rag = await self.get_rag(doc_id)
        output_dir = output_dir or self.settings.output_dir

        if file_path_obj.is_file():
            await rag.process_document_complete(
                file_path=str(file_path_obj),
                output_dir=output_dir,
                parse_method="auto",
            )
        else:
            await rag.process_folder_complete(str(file_path_obj), recursive=False)

        return doc_id

    async def query(self, doc_id: str, query: str, **kwargs) -> str:
        rag = await self.get_rag(doc_id)
        return await rag.aquery(query, **kwargs)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="RAG 后台管理工具")
    parser.add_argument("--path", "-p", required=True, help="要入库的文件或文件夹路径")
    parser.add_argument("--id", "-i", required=True, help="工作空间名称 (doc_id)")
    args = parser.parse_args()
    
    async def main():
        print(f"正在初始化 RAG 服务...")
        settings = LocalRagSettings.from_env()
        service = LocalRagService(settings)

        target_path = args.path
        workspace_name = args.id
        
        print(f"开始处理: {target_path}")
        print(f"目标工作区: {settings.working_dir_root}/{workspace_name}")

        try:
            await service.ingest(file_path=target_path, doc_id=workspace_name)

            print(f"\n 入库成功！")
            print(f"知识图谱已更新: {settings.working_dir_root}/{workspace_name}/graph_chunk_entity_relation.graphml")
            print(f"Markdown 已生成: {settings.output_dir}/{workspace_name}/")

        except Exception as e:
            print(f"\n 发生错误: {e}")
    
    asyncio.run(main())