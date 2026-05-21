import argparse
import json
from pathlib import Path


DEFAULT_FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
DEFAULT_FINEWEB_CONFIG = "sample-10BT"
DEFAULT_WIKIPEDIA_REPO = "wikimedia/wikipedia"
DEFAULT_WIKIPEDIA_CONFIG = "20231101.en"
DEFAULT_ARXIV_REPO = "gfissore/arxiv-abstracts-2021"
DEFAULT_ARXIV_CONFIG = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a small multi-domain corpus from Hugging Face and write JSONL for train.py."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/minimal_corpus.jsonl"),
        help="Where to write the merged JSONL corpus.",
    )
    parser.add_argument(
        "--fineweb-bytes",
        type=int,
        default=1_500_000,
        help="Approximate UTF-8 byte budget for FineWeb-Edu text.",
    )
    parser.add_argument(
        "--wikipedia-bytes",
        type=int,
        default=1_000_000,
        help="Approximate UTF-8 byte budget for Wikipedia text.",
    )
    parser.add_argument(
        "--arxiv-bytes",
        type=int,
        default=1_000_000,
        help="Approximate UTF-8 byte budget for arXiv text.",
    )
    parser.add_argument(
        "--fineweb-repo",
        type=str,
        default=DEFAULT_FINEWEB_REPO,
        help="Hugging Face dataset repo for web text.",
    )
    parser.add_argument(
        "--fineweb-config",
        type=str,
        default=DEFAULT_FINEWEB_CONFIG,
        help="Dataset config for FineWeb-Edu. sample-10BT is used by default so the stream comes from a sampled subset.",
    )
    parser.add_argument(
        "--wikipedia-repo",
        type=str,
        default=DEFAULT_WIKIPEDIA_REPO,
        help="Hugging Face dataset repo for Wikipedia.",
    )
    parser.add_argument(
        "--wikipedia-config",
        type=str,
        default=DEFAULT_WIKIPEDIA_CONFIG,
        help="Wikipedia language/date config.",
    )
    parser.add_argument(
        "--arxiv-repo",
        type=str,
        default=DEFAULT_ARXIV_REPO,
        help="Hugging Face dataset repo for scientific papers.",
    )
    parser.add_argument(
        "--arxiv-config",
        type=str,
        default=DEFAULT_ARXIV_CONFIG,
        help="Dataset config for scientific papers.",
    )
    parser.add_argument(
        "--arxiv-field",
        type=str,
        default="abstract",
        choices=["abstract", "article"],
        help="Use abstracts by default to keep the local corpus small.",
    )
    return parser.parse_args()


def load_dataset_stream(repo: str, config: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required for prepare_corpus.py.\n"
            "Install it with: pip install datasets"
        ) from exc

    return load_dataset(repo, config, split="train", streaming=True)


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def iter_fineweb(args: argparse.Namespace):
    stream = load_dataset_stream(args.fineweb_repo, args.fineweb_config)
    for row in stream:
        text = row.get("text")
        if text:
            yield text


def iter_wikipedia(args: argparse.Namespace):
    stream = load_dataset_stream(args.wikipedia_repo, args.wikipedia_config)
    for row in stream:
        text = row.get("text")
        if text:
            yield text


def iter_arxiv(args: argparse.Namespace):
    stream = load_dataset_stream(args.arxiv_repo, args.arxiv_config)
    for row in stream:
        text = row.get(args.arxiv_field)
        if text:
            yield text


def write_rows(handle, source: str, texts, byte_budget: int) -> tuple[int, int]:
    total_bytes = 0
    total_rows = 0
    for text in texts:
        normalized = normalize_text(text)
        if not normalized:
            continue
        encoded = normalized.encode("utf-8", errors="replace")
        if total_bytes + len(encoded) > byte_budget:
            break
        handle.write(json.dumps({"text": normalized, "source": source}, ensure_ascii=False) + "\n")
        total_bytes += len(encoded)
        total_rows += 1
    return total_rows, total_bytes


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with args.output_path.open("w", encoding="utf-8") as handle:
        web_rows, web_bytes = write_rows(handle, "web", iter_fineweb(args), args.fineweb_bytes)
        wiki_rows, wiki_bytes = write_rows(handle, "wiki", iter_wikipedia(args), args.wikipedia_bytes)
        arxiv_rows, arxiv_bytes = write_rows(handle, "arxiv", iter_arxiv(args), args.arxiv_bytes)

    print(f"Wrote corpus to {args.output_path}")
    print(f"  web:   rows={web_rows} bytes={web_bytes}")
    print(f"  wiki:  rows={wiki_rows} bytes={wiki_bytes}")
    print(f"  arxiv: rows={arxiv_rows} bytes={arxiv_bytes}")


if __name__ == "__main__":
    main()
