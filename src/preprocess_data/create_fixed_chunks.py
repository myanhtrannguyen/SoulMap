import argparse
import json
import re
from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "data_process"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "data_for_retrieve" / "rag_documents_fixed_chunks.jsonl"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    return text.strip()


def iter_source_txt_files(input_dir: Path) -> list[Path]:
    txt_files = []

    for folder in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        folder_txt_files = sorted(folder.glob("*.txt"))
        txt_files.extend(folder_txt_files)

    return txt_files


def chunk_tokens(tokens: list[str], chunk_size: int, overlap: int) -> list[tuple[int, int, list[str]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append((start, end, tokens[start:end]))

        if end == len(tokens):
            break

        start += step

    return chunks


def build_fixed_chunks(input_dir: Path, chunk_size: int, overlap: int) -> list[dict]:
    docs = []

    for txt_path in iter_source_txt_files(input_dir):
        text = txt_path.read_text(encoding="utf-8")
        tokens = tokenize(text)
        source_name = txt_path.parent.name

        for chunk_index, (start_token, end_token, chunk) in enumerate(
            chunk_tokens(tokens, chunk_size=chunk_size, overlap=overlap)
        ):
            chunk_text = detokenize(chunk)
            if not chunk_text:
                continue

            docs.append(
                {
                    "id": str(uuid4()),
                    "doc_type": "fixed_chunk",
                    "source": source_name,
                    "source_file": str(txt_path.relative_to(BASE_DIR)).replace("\\", "/"),
                    "chunk_index": chunk_index,
                    "start_token": start_token,
                    "end_token": end_token,
                    "token_count": len(chunk),
                    "chunk_text": chunk_text,
                }
            )

    return docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed-size RAG chunks from txt files in data/data_process."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = build_fixed_chunks(
        input_dir=args.input_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Created {len(docs)} fixed chunks")
    output_display = args.output
    try:
        output_display = args.output.relative_to(BASE_DIR)
    except ValueError:
        pass
    print(f"Saved to {output_display}")


if __name__ == "__main__":
    main()
