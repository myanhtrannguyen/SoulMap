import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag.chart_utils import build_chart_index
from rag.retriever_client import RemoteRetriever
from rag.retriever import (
    HybridRetriever,
    load_rag_documents,
    MODE_FIXED_SIMILARITY,
    MODE_FIXED_STRUCTURED_SIMILARITY,
    MODE_FIXED_STRUCTURED_KEYWORD,
    MODE_STRUCTURED_SIMILARITY,
    MODE_STRUCTURED_KEYWORD,
)

from user_chart.generate_user_chart import generate_user_chart
from evaluation.retrieval_mode.retriever_metrics import *
from evaluation.answer_generation.coverage_grounding import build_retrieved_feature_set


# CONFIG

MODES = [
    "fixed_similarity",
    "fixed_structured_similarity",
    "fixed_structured_keyword",
    "structured_similarity",
    "structured_keyword",
]

TOP_K = 8
ALPHA = 0.75

USERS_PATH = "src/evaluation/eval_dataset/eval_users.jsonl"
QUESTIONS_PATH = "src/evaluation/eval_dataset/eval_questions.jsonl"
OUTPUT_PATH = "evaluation_results/retrieval_eval.jsonl"


# LOAD DATA

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


users = load_jsonl(USERS_PATH)
questions = load_jsonl(QUESTIONS_PATH)

all_user_charts = {}

for user in users:
    user_chart, tuvi_chart, houses_chart = generate_user_chart(
        user["full_name"],
        user["gender"],
        user["dob_solar_str"])
    
    all_user_charts[user["id"]] = {
        "houses_chart": houses_chart,
        "chart_index": build_chart_index(houses_chart),
    }


# BUILD RETRIEVERS

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

completed = set()

if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            completed.add(
                (
                    row["mode"],
                    row["user_id"],
                    row["question_id"],
                )
            )

# EVALUATION LOOP

with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
    for mode in MODES:
        print(f"\n=== Running mode: {mode} ===\n")

        for user in users:
            chart_index = all_user_charts[user["id"]]["chart_index"]

            for q in questions:
                current = (
                    mode,
                    user["id"],
                    q["id"],
                )

                if current in completed:
                    continue

                retrieved = retriever.search(
                    query=q["question"],
                    chart_index=chart_index,
                    top_k=TOP_K,
                    alpha=ALPHA,
                    mode=mode,
                )

                docs = [x["doc"] for x in retrieved]

                expected_features = q.get("expected_features", [])
                retrieved_features = build_retrieved_feature_set(docs)

                metrics = {
                    "feature_recall": feature_recall(expected_features, retrieved_features),
                    "query_success": query_success(expected_features, retrieved_features),
                    "missing_features": list(missing_features(expected_features, retrieved_features)),

                    "rr": reciprocal_rank(expected_features, docs),
                    "hit@1": hit_at_k(expected_features, docs, k=1),
                    "hit@3": hit_at_k(expected_features, docs, k=3),
                    "allhit@3": all_hit_at_k(expected_features, docs, k=3)
                }

                row = {
                    "mode": mode,
                    "user_id": user["id"],
                    "question_id": q["id"],
                    "difficulty": q.get("difficulty"),
                    "question": q["question"],

                    "expected_features": q.get("expected_features"),

                    "retrieved_doc_ids": [
                        d.get("id")
                        for d in docs
                    ],
                    "retrieved_features": sorted(list(retrieved_features)),
                    "retrieved_feature_count": len(retrieved_features),

                    "metrics": metrics,
                }

                fout.write(
                    json.dumps(row, ensure_ascii=False)
                    + "\n"
                )