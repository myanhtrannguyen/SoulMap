import json
from rag.normalize import normalize_text

def load_rag_documents(path: str) -> list[dict]:
    docs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                docs.append(doc)

    return docs

def doc_matches_chart(doc: dict, chart_index: dict) -> bool:
    palace_id = normalize_text(doc.get("palace_id", ""))

    if palace_id not in chart_index:
        return False

    user_stars = set(chart_index[palace_id]["all_stars"])

    doc_star = normalize_text(doc.get("star_id", ""))

    if doc_star and doc_star not in user_stars:
        return False

    required_stars = doc.get("required_stars", {})

    for req_palace, req_stars in required_stars.items():
        req_palace = normalize_text(req_palace)

        if req_palace not in chart_index:
            return False

        if not req_stars:
            continue

        user_req_stars = set(chart_index[req_palace]["all_stars"])
        normalized_req_stars = {normalize_text(s) for s in req_stars}

        if not user_req_stars.intersection(normalized_req_stars):
            return False

    return True

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rag.normalize import normalize_text

class HybridRetriever:
    def __init__(self, docs: list[dict], model_name="bkai-foundation-models/vietnamese-bi-encoder"):
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        self.texts = [doc["chunk_text"] for doc in docs]
        self.embeddings = self.model.encode(self.texts, normalize_embeddings=True)

    def keyword_score(self, query: str, doc: dict) -> float:
        q = normalize_text(query)
        text = normalize_text(doc.get("chunk_text", ""))

        score = 0
        for token in q.split("_"):
            if token and token in text:
                score += 1

        return score

    def search(
        self,
        query: str,
        chart_index: dict,
        top_k: int = 8,
        alpha: float = 0.75
    ) -> list[dict]:

        candidate_docs = [
            doc for doc in self.docs
            if doc_matches_chart(doc, chart_index)
        ]

        if not candidate_docs:
            candidate_docs = self.docs

        candidate_texts = [doc["chunk_text"] for doc in candidate_docs]
        candidate_embeddings = self.model.encode(candidate_texts, normalize_embeddings=True)

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        semantic_scores = cosine_similarity(query_embedding, candidate_embeddings)[0]

        results = []

        max_keyword = 1

        raw_keyword_scores = [
            self.keyword_score(query, doc)
            for doc in candidate_docs
        ]

        if raw_keyword_scores:
            max_keyword = max(max(raw_keyword_scores), 1)

        for doc, sem_score, kw_score in zip(candidate_docs, semantic_scores, raw_keyword_scores):
            normalized_kw = kw_score / max_keyword
            final_score = alpha * sem_score + (1 - alpha) * normalized_kw

            results.append({
                "doc": doc,
                "score": float(final_score),
                "semantic_score": float(sem_score),
                "keyword_score": float(normalized_kw),
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]
    

INITIAL_QUERIES = [
    "đặc điểm nổi bật về tính cách, mệnh, thân",
    "điểm mạnh điểm yếu nổi bật",
    "sự nghiệp học vấn tài chính tình duyên sức khỏe",
]

def retrieve_initial_highlights(retriever, chart_index):
    all_results = []

    for query in INITIAL_QUERIES:
        results = retriever.search(
            query=query,
            chart_index=chart_index,
            top_k=6,
            alpha=0.7
        )
        all_results.extend(results)

    seen = set()
    unique_docs = []

    for item in all_results:
        doc_id = item["doc"]["id"]
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(item["doc"])

    return unique_docs[:12]