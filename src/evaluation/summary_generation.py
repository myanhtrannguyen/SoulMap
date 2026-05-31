import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    
import json
from pathlib import Path

from rag.chart_utils import build_chart_index
from rag.retriever_client import RemoteRetriever
from rag.prompt_builder import build_initial_prompt
from rag.retriever import (
    HybridRetriever,
    load_rag_documents,
    retrieve_initial_highlights,
)
from user_chart.generate_user_chart import generate_user_chart

from main import call_gemini


MODES = [
    "fixed_similarity",
    "fixed_structured_similarity",
    "structured_similarity",
]

TOP_K = 8
ALPHA = 0.75

USERS_PATH = "src/evaluation/eval_dataset/eval_users.jsonl"

OUTPUT_PATH = (
    "src/evaluation/eval_dataset/eval_initial_summaries.jsonl"
)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def build_retriever(mode):
    structured_docs = load_rag_documents(
        "data/data_for_retrieve/rag_documents_tu_vi_boi_toan.jsonl"
    )

    fixed_docs = []

    if mode != "structured_similarity" and mode != "structured_keyword":
        fixed_docs = load_rag_documents(
            "data/data_for_retrieve/rag_documents_fixed_chunks.jsonl"
        )

    return HybridRetriever(
        docs=structured_docs,
        fixed_docs=fixed_docs,
        mode=mode,
    )


retriever = RemoteRetriever(
    os.getenv("RETRIEVER_URL", "http://127.0.0.1:8765")
)


users = load_jsonl(USERS_PATH)[:10]

# resume support

completed = set()

if Path(OUTPUT_PATH).exists():
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            completed.add(
                (
                    row["mode"],
                    row["user_id"],
                )
            )


with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:

    for mode in MODES:

        print(f"\n=== MODE: {mode} ===")

        for user in users:

            key = (
                mode,
                user["id"],
            )

            if key in completed:
                continue

            print(
                f"Generating summary | "
                f"mode={mode} "
                f"user={user['id']}"
            )

            user_chart, tuvi_chart, houses_chart = (
                generate_user_chart(
                    user["full_name"],
                    user["gender"],
                    user["dob_solar_str"],
                )
            )

            chart_index = build_chart_index(
                houses_chart
            )

            initial_docs = retrieve_initial_highlights(
                retriever,
                chart_index,
                mode=mode,
            )

            initial_prompt = build_initial_prompt(
                houses_chart=houses_chart,
                retrieved_docs=initial_docs,
                user_chart=user_chart,
                tuvi_chart=tuvi_chart,
            )

            summary = call_gemini(
                initial_prompt
            )

            row = {
                "mode": mode,
                "user_id": user["id"],
                "initial_summary": summary,
            }

            fout.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            fout.flush()