import asyncio
import hashlib
import logging
import logging.config
import os
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
    tiktoken_cache_dir: str = "./tiktoken_cache"
    embedding_model_path: str = "./models/bge-m3"
    rerank_model_path: str = "./models/bge-reranker-v2-m3"
    log_dir: str = "./logs"
    vllm_api_base: str = "http://localhost:8001/v1"
    vllm_api_key: str = "EMPTY"
    llm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    device: str = "cuda:0"
    working_dir_root: str = "./rag_workspace"
    output_dir: str = "./output"
    embedding_dim: int = 1024
    max_token_size: int = 8192
    temperature: float = 0.1
    max_tokens: int = 4096
    vision_max_tokens: int = 2048
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
            tiktoken_cache_dir=os.getenv("TIKTOKEN_CACHE_DIR", "./tiktoken_cache"),
            embedding_model_path=os.getenv(
                "RAGANYTHING_EMBEDDING_MODEL_PATH", "./models/bge-m3"
            ),
            rerank_model_path=os.getenv(
                "RAGANYTHING_RERANK_MODEL_PATH", "./models/bge-reranker-v2-m3"
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
            full_system = (
                f"{settings.base_system_prompt}\n\nAdditional Instructions:\n"
                f"{original_sys}"
            )

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
            texts_part = [
                item.get("text", "").strip()
                for item in user_content_list
                if item.get("type") == "text"
            ]
            full_text_context = "\n\n".join([t for t in texts_part if t])

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
            raise NotImplementedError("image_data path is not implemented.")

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

