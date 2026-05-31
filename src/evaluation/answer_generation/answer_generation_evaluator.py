import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
from rag.prompt_builder import build_followup_prompt
from evaluation.answer_generation.key_manager import KeyManager
from evaluation.answer_generation.coverage_grounding import compute_coverage_score, compute_prompt_grounding_score, compute_strict_grounding_score
from evaluation.answer_generation.perplexity import VietnamesePerplexityEvaluator

MODES = [
    "fixed_similarity",
    "fixed_structured_similarity",
    "structured_similarity",
]

TOP_K = 8
ALPHA = 0.75

USERS_PATH = "src/evaluation/eval_dataset/eval_users.jsonl"

QUESTIONS_PATH = "src/evaluation/eval_dataset/eval_questions.jsonl"
STRIDE = 9

SUMMARY_PATH = "src/evaluation/eval_dataset/eval_initial_summaries.jsonl"

OUTPUT_PATH = (
    "evaluation_results/generation_eval.jsonl"
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

# key manager

key_manager = KeyManager()

questions = load_jsonl(QUESTIONS_PATH)
users = load_jsonl(USERS_PATH)[:10]

def safe_call_gemini(prompt, key_manager):
    for _ in range(len(key_manager.keys)):
        try:
            return call_gemini(
                prompt,
                api_key=key_manager.next(),
                max_retries=3
            )
        except RuntimeError as e:
            if "RATE_LIMIT" in str(e) or "429" in str(e):
                continue
            raise

    raise RuntimeError("All API keys exhausted")

# load summary

initial_summary_cache = {}

with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        initial_summary_cache[(row["mode"], row["user_id"])] = row["initial_summary"]

# resume support

completed = set()

if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            completed.add((row["mode"], row["user_id"], row["question_id"]))


with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:

    for mode in MODES:

        print("MODE:", mode)

        for user in users:

            user_id = user["id"]

            initial_summary = initial_summary_cache[(mode, user_id)]

            user_chart, tuvi_chart, houses_chart = generate_user_chart(
                user["full_name"],
                user["gender"],
                user["dob_solar_str"],
            )

            chart_index = build_chart_index(houses_chart)

            # STRIDE QUESTION ASSIGNMENT
            for q in questions:

                if (q["id"] + user_id) % STRIDE != 0:
                    continue

                current = (mode, user_id, q["id"])

                if current in completed:
                    continue

                followup_docs = retriever.search(
                    query=q["question"],
                    chart_index=chart_index,
                    top_k=8,
                    alpha=0.75,
                    mode=mode,
                )

                prompt = build_followup_prompt(
                    user_query=q["question"],
                    houses_chart=houses_chart,
                    initial_summary=initial_summary,
                    retrieved_docs=[x["doc"] for x in followup_docs],
                    user_chart=user_chart,
                    tuvi_chart=tuvi_chart,
                )

                answer = safe_call_gemini(prompt, key_manager)

                evaluator = VietnamesePerplexityEvaluator()
                ppl_result = evaluator.evaluate_summary(answer)

                coverage_result = compute_coverage_score(
                    answer=answer,
                    retrieved_docs=[x["doc"] for x in followup_docs]
                )

                strict_grounding_result = compute_strict_grounding_score(
                    answer=answer,
                    retrieved_docs=[x["doc"] for x in followup_docs]
                )

                prompt_grounding_result = compute_prompt_grounding_score(
                    answer=answer,
                    houses_chart=houses_chart,
                    tuvi_chart=tuvi_chart,
                    user_chart=user_chart,
                    initial_summary=initial_summary,
                    retrieved_docs=[x["doc"] for x in followup_docs]
                )

                row = {
                    "mode": mode,
                    "user_id": user_id,
                    "question_id": q["id"],
                    "answer": answer,
                    "ppl_result": ppl_result,
                    "coverage_result": coverage_result,
                    "strict_grounding_result": strict_grounding_result,
                    "prompt_grounding_result": prompt_grounding_result
                }

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()