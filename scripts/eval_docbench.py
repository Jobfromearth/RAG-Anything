import argparse
import asyncio
import json
from pathlib import Path

from raganything.services.local_rag import LocalRagService, LocalRagSettings


def _find_pdf(folder_path: Path) -> Path:
    pdfs = list(folder_path.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {folder_path}")
    return pdfs[0]


def _load_questions(folder_path: Path) -> list[dict]:
    qa_path = folder_path / f"{folder_path.name}_qa.jsonl"
    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")
    return [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run_eval(args: argparse.Namespace) -> None:
    settings = LocalRagSettings.from_env()
    settings.working_dir_root = str(Path(args.workdir_root))
    settings.output_dir = str(Path(args.output_dir) / "parsed")
    service = LocalRagService(settings)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_input_path = output_dir / f"{args.system_name}_eval_input.jsonl"
    results_path = output_dir / f"{args.system_name}_results.jsonl"

    folders = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if args.start_folder is not None:
        folders = [p for p in folders if p.name.isdigit() and int(p.name) >= args.start_folder]
    if args.max_folders is not None:
        folders = folders[: args.max_folders]

    for folder in folders:
        pdf_path = _find_pdf(folder)
        questions = _load_questions(folder)
        doc_id = folder.name
        await service.ingest(str(pdf_path), doc_id=doc_id)

        for item in questions:
            question = item.get("question", "").strip()
            if not question:
                continue

            result = await service.query(
                doc_id,
                question,
                mode=args.mode,
                top_k=args.top_k,
                chunk_top_k=args.chunk_top_k,
                enable_rerank=args.enable_rerank,
                vlm_enhanced=args.vlm_enhanced,
            )

            output_item = dict(item)
            output_item["sys_ans"] = result
            output_item["file"] = doc_id

            with eval_input_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(output_item, ensure_ascii=False) + "\n")

            with results_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "file": doc_id,
                            "question": question,
                            "sys_ans": result,
                            "answer": item.get("answer"),
                            "evidence": item.get("evidence"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RAGAnything on DocBench")
    parser.add_argument("--data_root", default="./data/docbench", help="DocBench root dir")
    parser.add_argument("--output_dir", default="./eval_outputs/docbench", help="Output dir")
    parser.add_argument("--workdir_root", default="./rag_workspace/docbench", help="Working dir root")
    parser.add_argument("--system_name", default="raganything", help="System name in output files")
    parser.add_argument("--mode", default="hybrid", help="LightRAG query mode")
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--chunk_top_k", type=int, default=30)
    parser.add_argument("--disable_rerank", action="store_false", dest="enable_rerank")
    parser.set_defaults(enable_rerank=True)
    parser.add_argument("--disable_vlm", action="store_false", dest="vlm_enhanced")
    parser.set_defaults(vlm_enhanced=True)
    parser.add_argument("--start_folder", type=int, default=None, help="Start folder id")
    parser.add_argument("--max_folders", type=int, default=None, help="Limit number of folders")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not Path(args.data_root).exists():
        raise FileNotFoundError(f"DocBench data_root not found: {args.data_root}")

    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
