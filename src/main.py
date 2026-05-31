import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from rag.chart_utils import build_chart_index
from rag.prompt_builder import build_followup_prompt, build_initial_prompt
from rag.retriever_client import RemoteRetriever
from rag.retriever import (
    MODE_FIXED_STRUCTURED_KEYWORD,
    MODE_STRUCTURED_KEYWORD,
    MODE_STRUCTURED_SIMILARITY,
    HybridRetriever,
    load_rag_documents,
    retrieve_initial_highlights,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_USER_DIR = BASE_DIR / "data" / "data_user" / "nguyễn_thu_huyền"
DEFAULT_STRUCTURED_RAG_PATH = (
    BASE_DIR / "data" / "data_for_retrieve" / "rag_documents_tu_vi_boi_toan.jsonl"
)
DEFAULT_FIXED_RAG_PATH = BASE_DIR / "data" / "data_for_retrieve" / "rag_documents_fixed_chunks.jsonl"
DEFAULT_MODEL = "gemini-2.5-flash"


def load_local_env(env_path: Path = BASE_DIR / ".env") -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def extract_gemini_text(response: dict) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini không trả về candidates: {response}")

    parts = candidates[0].get("content", {}).get("parts") or []
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    if not text_parts:
        raise RuntimeError(f"Gemini không trả về text: {response}")

    return "\n".join(text_parts).strip()


def call_gemini(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.4,
    api_key: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:

    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError("Missing API key")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": temperature
        },
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            request = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                method="POST",
            )

            with urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))

            return extract_gemini_text(data)

        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")

            if exc.code in (429, 500, 503):
                last_error = exc

            else:
                raise RuntimeError(f"Gemini HTTP {exc.code}: {body}")

        except URLError as exc:
            last_error = exc

        # retry delay (exponential backoff)
        time.sleep(retry_delay * (2 ** attempt))

    raise RuntimeError(f"Gemini failed after retries: {last_error}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hỏi đáp Tử Vi bằng RAG và Gemini.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Đường tình duyên của tôi như thế nào?",
        help="Câu hỏi người dùng muốn hỏi về lá số.",
    )
    parser.add_argument(
        "--user-dir",
        type=Path,
        default=DEFAULT_USER_DIR,
        help="Thư mục chứa user_chart.json, tuvi_chart.json và houses_chart.json của người dùng.",
    )
    parser.add_argument(
        "--rag-path",
        type=Path,
        default=DEFAULT_STRUCTURED_RAG_PATH,
        help="Đường dẫn file JSONL dữ liệu RAG cấu trúc.",
    )
    parser.add_argument(
        "--fixed-rag-path",
        type=Path,
        default=DEFAULT_FIXED_RAG_PATH,
        help="Đường dẫn file JSONL fixed-size chunks.",
    )
    parser.add_argument(
        "--retrieval-mode",
        default=MODE_FIXED_STRUCTURED_KEYWORD,
        choices=[
            "fixed_similarity",
            "fixed_structured_similarity",
            "fixed_structured_keyword",
            "structured_similarity",
            "structured_keyword",
        ],
        help="Pipeline retrieve: fixed only, fixed + structured, hoặc fixed + structured + BM25.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        help="Tên model Gemini dùng để sinh câu trả lời.",
    )
    parser.add_argument(
        "--retriever-url",
        default=os.getenv("RETRIEVER_URL"),
        help="URL retriever server, vi du http://127.0.0.1:8765. Neu bo trong se load retriever local.",
    )
    return parser.parse_args()


def build_retriever(args: argparse.Namespace):
    if args.retriever_url:
        return RemoteRetriever(args.retriever_url)

    structured_docs = load_rag_documents(str(args.rag_path))
    fixed_docs = []
    if args.retrieval_mode not in {MODE_STRUCTURED_SIMILARITY, MODE_STRUCTURED_KEYWORD}:
        fixed_docs = load_rag_documents(str(args.fixed_rag_path))

    return HybridRetriever(
        docs=structured_docs,
        fixed_docs=fixed_docs,
        mode=args.retrieval_mode,
    )


def main() -> None:
    load_local_env()
    args = parse_args()

    user_chart_path = args.user_dir / "user_chart.json"
    tuvi_chart_path = args.user_dir / "tuvi_chart.json"
    houses_chart_path = args.user_dir / "houses_chart.json"

    with user_chart_path.open("r", encoding="utf-8") as f:
        user_chart = json.load(f)
    with tuvi_chart_path.open("r", encoding="utf-8") as f:
        tuvi_chart = json.load(f)
    with houses_chart_path.open("r", encoding="utf-8") as f:
        houses_chart = json.load(f)

    chart_index = build_chart_index(houses_chart)

    retriever = build_retriever(args)

    initial_docs = retrieve_initial_highlights(
        retriever,
        chart_index,
        mode=args.retrieval_mode,
    )
    initial_prompt = build_initial_prompt(
        houses_chart=houses_chart,
        retrieved_docs=initial_docs,
        user_chart=user_chart,
        tuvi_chart=tuvi_chart,
    )
    initial_summary = call_gemini(initial_prompt, model=args.model)

    followup_docs = retriever.search(
        query=args.question,
        chart_index=chart_index,
        top_k=8,
        alpha=0.75,
        mode=args.retrieval_mode,
    )

    followup_prompt = build_followup_prompt(
        user_query=args.question,
        houses_chart=houses_chart,
        initial_summary=initial_summary,
        retrieved_docs=[x["doc"] for x in followup_docs],
        user_chart=user_chart,
        tuvi_chart=tuvi_chart,
    )

    answer = call_gemini(followup_prompt, model=args.model)

    print("\n=== Tóm tắt ban đầu ===\n")
    print(initial_summary)
    print("\n=== Câu trả lời ===\n")
    print(answer)


if __name__ == "__main__":
    main()
