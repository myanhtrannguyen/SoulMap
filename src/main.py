import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag.chart_utils import build_chart_index
from rag.prompt_builder import build_followup_prompt, build_initial_prompt
from rag.retriever import HybridRetriever, load_rag_documents, retrieve_initial_highlights


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_USER_DIR = BASE_DIR / "data" / "data_user" / "nguyễn_thu_huyền"
DEFAULT_RAG_PATH = BASE_DIR / "data" / "data_for_retrieve" / "rag_documents_tu_vi_boi_toan.jsonl"
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
    max_output_tokens: int = 2048,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu API key. Hãy đặt biến môi trường GEMINI_API_KEY "
            "hoặc GOOGLE_API_KEY trước khi chạy."
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Lỗi Gemini API HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Không kết nối được Gemini API: {exc.reason}") from exc

    return extract_gemini_text(data)


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
        help="Thư mục chứa houses_chart.json của người dùng.",
    )
    parser.add_argument(
        "--rag-path",
        type=Path,
        default=DEFAULT_RAG_PATH,
        help="Đường dẫn file JSONL dữ liệu RAG.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        help="Tên model Gemini dùng để sinh câu trả lời.",
    )
    return parser.parse_args()


def main() -> None:
    load_local_env()
    args = parse_args()

    houses_chart_path = args.user_dir / "houses_chart.json"
    with houses_chart_path.open("r", encoding="utf-8") as f:
        houses_chart = json.load(f)

    chart_index = build_chart_index(houses_chart)

    docs = load_rag_documents(str(args.rag_path))
    retriever = HybridRetriever(docs)

    initial_docs = retrieve_initial_highlights(retriever, chart_index)
    initial_prompt = build_initial_prompt(houses_chart, initial_docs)
    initial_summary = call_gemini(initial_prompt, model=args.model, max_output_tokens=1536)

    followup_docs = retriever.search(
        query=args.question,
        chart_index=chart_index,
        top_k=8,
        alpha=0.75,
    )

    followup_prompt = build_followup_prompt(
        user_query=args.question,
        houses_chart=houses_chart,
        initial_summary=initial_summary,
        retrieved_docs=[x["doc"] for x in followup_docs],
    )

    answer = call_gemini(followup_prompt, model=args.model)

    print("\n=== Tóm tắt ban đầu ===\n")
    print(initial_summary)
    print("\n=== Câu trả lời ===\n")
    print(answer)


if __name__ == "__main__":
    main()
