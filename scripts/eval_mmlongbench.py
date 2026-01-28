import argparse
import asyncio
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from raganything.services.local_rag import LocalRagService, LocalRagSettings

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None


def _load_dataset_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_jsonl:
        data_path = Path(args.dataset_jsonl)
        if not data_path.exists():
            raise FileNotFoundError(f"dataset_jsonl not found: {data_path}")
        return [
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if load_dataset is None:
        raise RuntimeError(
            "datasets is not installed. Install with: pip install datasets"
        )

    dataset = load_dataset(
        args.hf_dataset,
        name=args.hf_config,
        split=args.hf_split,
    )
    return [dict(row) for row in dataset]


def _find_pdf_for_doc(docs_dir: Path, doc_id: str) -> Path:
    direct = docs_dir / f"{doc_id}.pdf"
    if direct.exists():
        return direct
    nested = list((docs_dir / doc_id).glob("*.pdf"))
    if nested:
        return nested[0]
    raise FileNotFoundError(f"PDF not found for doc_id={doc_id} in {docs_dir}")


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _simple_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {}

    exact = 0
    num_total = 0
    num_correct = 0
    unanswerable_total = 0
    unanswerable_correct = 0

    for row in rows:
        answer = str(row.get("answer", "")).strip()
        sys_ans = str(row.get("sys_ans", "")).strip()
        answer_format = str(row.get("answer_format", "")).strip().lower()

        if answer_format in {"none", "unanswerable"}:
            unanswerable_total += 1
            if "i don't know" in sys_ans.lower() or "unknown" in sys_ans.lower():
                unanswerable_correct += 1
            continue

        if answer_format in {"int", "float"}:
            num_total += 1
            try:
                ans_num = float(answer)
                sys_num = float(sys_ans.split()[0])
                if math.isfinite(sys_num) and abs(sys_num - ans_num) <= 1e-3:
                    num_correct += 1
            except Exception:
                pass

        if _normalize_text(sys_ans) == _normalize_text(answer):
            exact += 1

    return {
        "total": total,
        "exact_match": exact / total if total else 0.0,
        "numeric_accuracy": num_correct / num_total if num_total else 0.0,
        "unanswerable_accuracy": unanswerable_correct / unanswerable_total
        if unanswerable_total
        else 0.0,
    }


async def run_eval(args: argparse.Namespace) -> None:
    settings = LocalRagSettings.from_env()
    settings.working_dir_root = str(Path(args.workdir_root))
    settings.output_dir = str(Path(args.output_dir) / "parsed")
    service = LocalRagService(settings)

    docs_dir = Path(args.docs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{args.system_name}_results.jsonl"

    rows = _load_dataset_rows(args)
    rows_by_doc = defaultdict(list)
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            continue
        rows_by_doc[doc_id].append(row)

    processed_rows: list[dict[str, Any]] = []

    for doc_id, doc_rows in rows_by_doc.items():
        pdf_path = _find_pdf_for_doc(docs_dir, doc_id)
        await service.ingest(str(pdf_path), doc_id=doc_id)

        for row in doc_rows[: args.max_questions_per_doc]:
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            sys_ans = await service.query(
                doc_id,
                question,
                mode=args.mode,
                top_k=args.top_k,
                chunk_top_k=args.chunk_top_k,
                enable_rerank=args.enable_rerank,
                vlm_enhanced=args.vlm_enhanced,
            )
            output_row = dict(row)
            output_row["sys_ans"] = sys_ans
            processed_rows.append(output_row)

            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    metrics = _simple_metrics(processed_rows)
    metrics_path = output_dir / f"{args.system_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RAGAnything on MMLongBench-Doc")
    parser.add_argument("--docs_dir", required=True, help="Directory containing PDFs by doc_id")
    parser.add_argument("--output_dir", default="./eval_outputs/mmlongbench", help="Output dir")
    parser.add_argument("--workdir_root", default="./rag_workspace/mmlongbench", help="Working dir root")
    parser.add_argument("--system_name", default="raganything", help="System name in output files")
    parser.add_argument("--mode", default="hybrid", help="LightRAG query mode")
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--chunk_top_k", type=int, default=30)
    parser.add_argument("--disable_rerank", action="store_false", dest="enable_rerank")
    parser.set_defaults(enable_rerank=True)
    parser.add_argument("--disable_vlm", action="store_false", dest="vlm_enhanced")
    parser.set_defaults(vlm_enhanced=True)
    parser.add_argument("--max_questions_per_doc", type=int, default=9999)

    parser.add_argument("--dataset_jsonl", default="", help="Optional local jsonl dataset")
    parser.add_argument("--hf_dataset", default="yubo2333/MMLongBench-Doc", help="HF dataset name")
    parser.add_argument("--hf_config", default="default", help="HF dataset config")
    parser.add_argument("--hf_split", default="train", help="HF dataset split")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
