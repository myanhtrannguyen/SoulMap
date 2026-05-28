import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.retriever import (
    MODE_FIXED_STRUCTURED_KEYWORD,
    MODE_STRUCTURED_KEYWORD,
    MODE_STRUCTURED_SIMILARITY,
    HybridRetriever,
    load_rag_documents,
)


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_STRUCTURED_RAG_PATH = (
    BASE_DIR / "data" / "data_for_retrieve" / "rag_documents_tu_vi_boi_toan.jsonl"
)
DEFAULT_FIXED_RAG_PATH = BASE_DIR / "data" / "data_for_retrieve" / "rag_documents_fixed_chunks.jsonl"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host retriever model de tai model mot lan.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--rag-path",
        type=Path,
        default=DEFAULT_STRUCTURED_RAG_PATH,
        help="Duong dan file JSONL du lieu RAG cau truc.",
    )
    parser.add_argument(
        "--fixed-rag-path",
        type=Path,
        default=DEFAULT_FIXED_RAG_PATH,
        help="Duong dan file JSONL fixed-size chunks.",
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
    )
    parser.add_argument(
        "--embedding-model",
        default="bkai-foundation-models/vietnamese-bi-encoder",
        help="SentenceTransformer model dung cho retriever.",
    )
    return parser.parse_args()


class RetrieverRequestHandler(BaseHTTPRequestHandler):
    retriever: HybridRetriever | None = None

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        try:
            if self.retriever is None:
                raise RuntimeError("Retriever chua duoc khoi tao.")

            payload = self._read_json()

            if self.path == "/search":
                results = self.retriever.search(
                    query=payload["query"],
                    chart_index=payload.get("chart_index"),
                    top_k=int(payload.get("top_k", 8)),
                    alpha=float(payload.get("alpha", 0.75)),
                    mode=payload.get("mode"),
                )
                self._send_json(200, {"results": results})
                return

            self._send_json(404, {"error": "Not found"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


def main() -> None:
    args = parse_args()

    structured_docs = load_rag_documents(str(args.rag_path))
    fixed_docs = []
    if args.retrieval_mode not in {MODE_STRUCTURED_SIMILARITY, MODE_STRUCTURED_KEYWORD}:
        fixed_docs = load_rag_documents(str(args.fixed_rag_path))

    RetrieverRequestHandler.retriever = HybridRetriever(
        docs=structured_docs,
        fixed_docs=fixed_docs,
        mode=args.retrieval_mode,
        model_name=args.embedding_model,
    )

    server = ThreadingHTTPServer((args.host, args.port), RetrieverRequestHandler)
    print(f"Retriever server listening at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
