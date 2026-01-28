#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG-Anything Local Multimodal Pipeline (Final Optimized Version)
----------------------------------------------------------------
Key Features:
1. Native Rerank Integration via lightrag_kwargs (No monkey patching).
2. Prompt Reordering (Images -> Text -> Citation Reminder) for Qwen2-VL.
3. optimized Retrieval Parameters (chunk_top_k=30, top_k=15).
"""

import asyncio
import argparse
import os

from raganything.services.local_rag import LocalRagService, LocalRagSettings

# ==========================================
# 1. Configuration & Global Init
# ==========================================

async def process_with_rag(service: LocalRagService, file_path: str):
    doc_id = await service.ingest(file_path)

    queries = [
        "Explain the architecture shown in Figure 1.",
        "In table 3, what is the number of the params of the base and big model?",
        "What is the name of the chapter that mentions the table 1 in the rag-anything paper?",
        "Give me the latex format expression of the equation (1) in the rag-anything paper.",
        "Explain the equation 1",
    ]

    for i, query in enumerate(queries, 1):
        service.logger.info(f"\n{'='*80}")
        service.logger.info(f"Query {i}/{len(queries)}: {query}")
        service.logger.info(f"{'='*80}")

        query_param = {
            "mode": "hybrid",
            "top_k": 15,
            "chunk_top_k": 30,
            "enable_rerank": True,
            "vlm_enhanced": True,
        }

        result = await service.query(doc_id, query, **query_param)
        service.logger.info(f"\n✅ Answer:\n{result}\n")

        if "[" in result and "]" in result:
            service.logger.info("✓ Reference detected")
        else:
            service.logger.warning("⚠ No reference found")


# ==========================================
# 4. Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="RAGAnything Local Pipeline")
    parser.add_argument("--input", "-i", default="./data/RAG_anything.pdf", help="Input file/dir")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input not found: {args.input}")
        return

    settings = LocalRagSettings.from_env()
    service = LocalRagService(settings)
    asyncio.run(process_with_rag(service, args.input))

if __name__ == "__main__":
    main()