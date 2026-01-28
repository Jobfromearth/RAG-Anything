import argparse
import re
from pathlib import Path

import gdown


def _extract_id(url_or_id: str) -> str:
    match = re.search(r"(?:/folders/|id=)([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id


def download_docbench(folder_url_or_id: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    folder_id = _extract_id(folder_url_or_id)
    gdown.download_folder(id=folder_id, output=str(output_dir), quiet=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download DocBench dataset")
    parser.add_argument(
        "--folder",
        required=True,
        help="Google Drive folder URL or folder id for DocBench data",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/docbench",
        help="Output directory for DocBench data",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    download_docbench(args.folder, Path(args.output_dir))


if __name__ == "__main__":
    main()
