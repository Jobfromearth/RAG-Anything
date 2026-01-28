import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


def _write_jsonl(rows: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _download_docs_from_manifest(manifest_path: Path, docs_dir: Path) -> None:
    import requests

    docs_dir.mkdir(parents=True, exist_ok=True)
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid manifest line: {line}")
        doc_id, url = parts[0].strip(), parts[1].strip()
        target = docs_dir / f"{doc_id}.pdf"
        if target.exists():
            continue
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        target.write_bytes(resp.content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download MMLongBench-Doc dataset")
    parser.add_argument(
        "--output_jsonl",
        default="./data/mmlongbench/mmlongbench.jsonl",
        help="Output JSONL path for questions",
    )
    parser.add_argument(
        "--hf_dataset",
        default="yubo2333/MMLongBench-Doc",
        help="Hugging Face dataset name",
    )
    parser.add_argument("--hf_config", default="default", help="HF dataset config")
    parser.add_argument("--hf_split", default="train", help="HF dataset split")
    parser.add_argument(
        "--docs_manifest",
        default="",
        help="Optional CSV manifest: doc_id,url (one per line) to download PDFs",
    )
    parser.add_argument(
        "--docs_dir",
        default="./data/mmlongbench/docs",
        help="Directory to store downloaded PDFs",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset = load_dataset(
        args.hf_dataset, name=args.hf_config, split=args.hf_split
    )
    _write_jsonl(dataset, Path(args.output_jsonl))

    if args.docs_manifest:
        _download_docs_from_manifest(Path(args.docs_manifest), Path(args.docs_dir))


if __name__ == "__main__":
    main()
