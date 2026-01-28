#!/usr/bin/env python
"""
RAG-Anything Local Multimodal Pipeline (Fixed Version)
------------------------------------------------------
This script demonstrates a local deployment of a Multimodal RAG system using:
1. RAG-Anything & LightRAG for document processing and retrieval.
2. vLLM (hosting Qwen2-VL-7B) for high-performance inference.
3. Local BGE-M3 for embedding generation.

Key Fixes:
- Preserves RAG-Anything's original prompt structure to maintain reference generation
- Enhances rather than replaces system prompts for visual capabilities
- Removes redundant question injection
- Simplifies vision model function logic
- Adds better debugging and error handling

Usage:
    python raganything_local_fixed.py --input ./data/Attention.pdf
"""

import os
import sys
import logging
import logging.config
import asyncio
import argparse
import numpy as np
from datetime import datetime

# ==========================================
# 1. Environment & Path Configuration
# ==========================================

# Path to Tiktoken cache to prevent re-downloading
os.environ["TIKTOKEN_CACHE_DIR"] = "/data/h50056787/workspaces/lightrag/tiktoken_cache"

# Model paths and API configuration
EMBEDDING_MODEL_PATH = "/data/h50056787/models/bge-m3"
VLLM_API_BASE = "http://localhost:8001/v1"
VLLM_API_KEY = "EMPTY"
LLM_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# ==========================================
# 2. Core Library Imports
# ==========================================
from raganything import RAGAnything, RAGAnythingConfig
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI

# Global initialization
print(f"Loading Embedding Model: {EMBEDDING_MODEL_PATH} ...")
try:
    st_model = SentenceTransformer(EMBEDDING_MODEL_PATH, trust_remote_code=True, device="cuda:0")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    sys.exit(1)

# Initialize Async OpenAI Client for vLLM
client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE)

# ==========================================
# 3. Logging Configuration
# ==========================================
def configure_logging():
    """Configure logging to both console and rotating file."""
    log_dir = "/data/h50056787/workspaces/lightrag/logs"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"run_{timestamp}.log")

    print(f"\nRAGAnything log file: {log_file_path}\n")
    os.makedirs(log_dir, exist_ok=True)

    log_max_bytes = 10485760  # 10MB
    log_backup_count = 5

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(levelname)s: %(message)s",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout", 
            },
            "file": {
                "formatter": "detailed",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file_path,
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": "INFO",
            },
        },
    })
    
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

# ==========================================
# 4. RAG Processing Logic (Fixed Implementation)
# ==========================================
async def process_with_rag(
    file_path: str,
    output_dir: str,
    working_dir: str = None,
):
    """
    Main asynchronous function to process documents and execute queries using RAG.
    """
    try:
        # Initialize RAG configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_workspace",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # ---------------------------------------------------------
        # A. Define LLM Function (Text-only / Entity Extraction)
        # ---------------------------------------------------------
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            """
            Standard LLM function for text-only tasks.
            Used by RAG for entity extraction, keyword extraction, etc.
            """
            # Clean kwargs to prevent API conflicts
            cleaned_kwargs = {
                k: v for k, v in kwargs.items() 
                if k not in ['hashing_kv', 'keyword_extraction', 'enable_cot']
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            try:
                response = await client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                    frequency_penalty=1.0,
                    **cleaned_kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"LLM Error: {e}")
                return ""

        # ---------------------------------------------------------
        # B. Define Vision Function (Fixed Version)
        # ---------------------------------------------------------
        async def vision_model_func(
            prompt, 
            system_prompt=None, 
            history_messages=[], 
            image_data=None, 
            messages=None,
            **kwargs
        ):
            """
            Enhanced vision model function that preserves RAG's prompt structure.
            
            Key improvements:
            1. Preserves RAG's system prompt (includes reference instructions)
            2. Enhances rather than replaces the prompt
            3. No redundant question injection (RAG handles it)
            4. Cleaner logic flow
            """
            # Clean kwargs (messages is a parameter, not in kwargs)
            cleaned_kwargs = {
                k: v for k, v in kwargs.items() 
                if k not in ['hashing_kv', 'keyword_extraction', 'enable_cot']
            }
            
            # --- Strategy A: RAG provided pre-built messages (Standard Path) ---
            if messages:
                logger.info("Processing RAG-built multimodal messages")
                
                # Enhance the messages while preserving structure
                enhanced_messages = []
                
                for msg in messages:
                    if msg['role'] == 'system':
                        # CRITICAL FIX: Enhance rather than replace
                        original_content = msg['content']
                        
                        # Add visual capability instructions to the existing prompt
                        visual_enhancement = (
                            "IMPORTANT VISUAL CAPABILITIES:\n"
                            "- You are Qwen2-VL with strong multimodal understanding\n"
                            "- Carefully analyze ALL provided images, tables, and charts\n"
                            "- Extract precise numerical values from visual content\n"
                            "- Pay special attention to table cells, axis labels, and figure captions\n"
                            "- Cross-reference visual content with textual descriptions\n\n"
                        )
                        
                        enhanced_content = visual_enhancement + original_content
                        enhanced_messages.append({"role": "system", "content": enhanced_content})
                        
                        # Debug: Show what we're preserving
                        logger.debug(f"Original system prompt length: {len(original_content)}")
                        logger.debug(f"Enhanced system prompt length: {len(enhanced_content)}")
                    
                    elif msg['role'] == 'user':
                        # Keep user messages as-is (RAG already formatted them correctly)
                        content = msg['content']
                        
                        # Count images if content is multimodal
                        if isinstance(content, list):
                            image_count = sum(1 for item in content if item.get('type') == 'image_url')
                            logger.info(f"User message contains {image_count} images")
                        
                        enhanced_messages.append(msg)
                    
                    else:
                        # Keep other message types unchanged
                        enhanced_messages.append(msg)
                
                # Send to vLLM
                try:
                    response = await client.chat.completions.create(
                        model=LLM_MODEL_NAME,
                        messages=enhanced_messages,
                        temperature=0.1,
                        max_tokens=2048,
                        **cleaned_kwargs
                    )
                    return response.choices[0].message.content
                
                except Exception as e:
                    logger.error(f"Vision LLM Error with messages: {e}")
                    # If it's a token limit error, try with fewer images
                    if "token" in str(e).lower() or "limit" in str(e).lower():
                        logger.warning("Possible token limit exceeded, consider reducing top_k or image count")
                    raise
            
            # --- Strategy B: Raw image_data provided (Fallback Path) ---
            elif image_data:
                logger.info("Processing raw image_data (fallback mode)")
                
                user_content = []
                
                # Handle image quota (check your vLLM startup args for actual limit)
                MAX_IMAGES = 20  # Conservative limit, adjust based on vLLM config
                imgs = image_data if isinstance(image_data, list) else [image_data]
                
                if len(imgs) > MAX_IMAGES:
                    logger.warning(f"Image count {len(imgs)} exceeds limit {MAX_IMAGES}, truncating")
                
                # Add images
                for img in imgs[:MAX_IMAGES]:
                    user_content.append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                    })
                
                # Combine text content
                text_parts = []
                if system_prompt:
                    text_parts.append(f"Context:\n{system_prompt}")
                if prompt:
                    text_parts.append(f"\nRetrieved Information:\n{prompt}")
                
                user_content.append({
                    "type": "text", 
                    "text": "\n".join(text_parts)
                })
                
                # Build messages
                messages_to_send = [
                    {
                        "role": "system",
                        "content": (
                            "You are Qwen2-VL, a helpful multimodal AI assistant. "
                            "Analyze the provided images and text carefully. "
                            "Always cite your sources in your answers."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ]
                
                try:
                    response = await client.chat.completions.create(
                        model=LLM_MODEL_NAME,
                        messages=messages_to_send,
                        temperature=0.1,
                        max_tokens=2048,
                        **cleaned_kwargs
                    )
                    return response.choices[0].message.content
                
                except Exception as e:
                    logger.error(f"Vision LLM Error with image_data: {e}")
                    raise
            
            # --- Strategy C: Pure Text Fallback ---
            else:
                logger.info("No images provided, using text-only LLM")
                return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # ---------------------------------------------------------
        # C. Define Embedding Function
        # ---------------------------------------------------------
        async def _compute_embedding(texts: list[str]) -> np.ndarray:
            """Compute embeddings using local BGE-M3 model."""
            return st_model.encode(texts, normalize_embeddings=True)

        from lightrag.utils import EmbeddingFunc
        embedding_func = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=_compute_embedding
        )

        # ---------------------------------------------------------
        # D. Initialize and Run RAG
        # ---------------------------------------------------------
        logger.info("Initializing RAG-Anything...")
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Process the input
        if os.path.isfile(file_path):
            logger.info(f"Processing single file: {file_path}")
            await rag.process_document_complete(
                file_path=file_path, 
                output_dir=output_dir, 
                parse_method="auto"
            )
        elif os.path.isdir(file_path):
            logger.info(f"Processing folder: {file_path}")
            await rag.process_folder_complete(file_path, recursive=False)
        else:
            logger.error(f"Invalid path: {file_path}")
            return

        logger.info("Index built successfully.")

        # ---------------------------------------------------------
        # E. Execute Queries
        # ---------------------------------------------------------
        queries = [
            "Explain the architecture shown in Figure 1.",
            "In table 3, what is the number of the params of the base and big model?",
            "Explain the equation (1) in the attention paper.",
            "What is the name of the chapter that mentions the table 1 in the rag-anything paper?",
            "Explain the equation 1 in the rag-anything paper page 3?",
            "Give me the latex format expression of the equation (1) in the rag-anything paper."
        ]

        for i, query in enumerate(queries, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Query {i}/{len(queries)}: {query}")
            logger.info(f"{'='*80}")
            
            # Configure query parameters
            query_param = {
                "mode": "hybrid",  # Combines local + global search
                "top_k": 10,       # Reduced from 15 to minimize noise
            }
            
            try:
                result = await rag.aquery(query, **query_param)
                logger.info(f"\n✅ Answer:\n{result}\n")
                
                # Check if reference is present
                if '[' in result and ']' in result:
                    logger.info("✓ Reference detected in answer")
                else:
                    logger.warning("⚠ No reference found in answer (may indicate issue)")
            
            except Exception as e:
                logger.error(f"❌ Query failed: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

        logger.info("\n" + "="*80)
        logger.info("All queries completed!")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# ==========================================
# 5. Main Entry Point
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="RAGAnything Local Multimodal Example (Fixed Version)"
    )
    parser.add_argument(
        "--input", "-i", 
        default="./data/RAG_anything.pdf", 
        help="Path to input file or directory"
    )
    parser.add_argument(
        "--working_dir", "-w", 
        default="./rag_workspace", 
        help="Path to RAG storage directory"
    )
    parser.add_argument(
        "--output", "-o", 
        default="./output", 
        help="Path to intermediate output directory"
    )
    
    args = parser.parse_args()
    
    # Input validation
    if not os.path.exists(args.input):
        if not os.path.exists("./data"): 
            print(f"❌ Input not found: {args.input}")
            print("Please provide a valid input file or directory.")
            return
        else:
            print(f"⚠ Warning: {args.input} not found, but ./data directory exists")

    # Run the async pipeline
    asyncio.run(
        process_with_rag(
            args.input, 
            args.output, 
            args.working_dir
        )
    )

if __name__ == "__main__":
    configure_logging()
    main()
