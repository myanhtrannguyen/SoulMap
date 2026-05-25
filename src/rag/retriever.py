import json
import math
from collections import Counter

from rag.normalize import normalize_text


MODE_FIXED_SIMILARITY = "fixed_similarity"
MODE_FIXED_STRUCTURED_SIMILARITY = "fixed_structured_similarity"
MODE_FIXED_STRUCTURED_KEYWORD = "fixed_structured_keyword"

MODE_ALIASES = {
    "fixed": MODE_FIXED_SIMILARITY,
    "similarity": MODE_FIXED_SIMILARITY,
    "fixed_only": MODE_FIXED_SIMILARITY,
    "combined_similarity": MODE_FIXED_STRUCTURED_SIMILARITY,
    "hybrid_similarity": MODE_FIXED_STRUCTURED_SIMILARITY,
    "hybrid": MODE_FIXED_STRUCTURED_KEYWORD,
    "keyword": MODE_FIXED_STRUCTURED_KEYWORD,
    "bm25": MODE_FIXED_STRUCTURED_KEYWORD,
}


def load_rag_documents(path: str) -> list[dict]:
    docs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                docs.append(doc)

    return docs


def normalize_mode(mode: str | None) -> str:
    mode = mode or MODE_FIXED_STRUCTURED_KEYWORD
    mode = MODE_ALIASES.get(mode, mode)

    valid_modes = {
        MODE_FIXED_SIMILARITY,
        MODE_FIXED_STRUCTURED_SIMILARITY,
        MODE_FIXED_STRUCTURED_KEYWORD,
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported retrieval mode: {mode}. Valid modes: {sorted(valid_modes)}")

    return mode


def doc_matches_chart(doc: dict, chart_index: dict | None) -> bool:
    if not chart_index:
        return False

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


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [token for token in normalized.split("_") if token]


def flatten_keyword_value(value) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(flatten_keyword_value(item) for item in value)

    if isinstance(value, dict):
        return " ".join(
            f"{flatten_keyword_value(key)} {flatten_keyword_value(item)}"
            for key, item in value.items()
        )

    return str(value)


def keyword_text_for_doc(doc: dict) -> str:
    if doc.get("doc_type") == "fixed_chunk":
        return doc.get("chunk_text", "")

    return " ".join(
        [
            flatten_keyword_value(doc.get("topic")),
            flatten_keyword_value(doc.get("star_id")),
            flatten_keyword_value(doc.get("condition")),
        ]
    )


def bm25_scores(query: str, docs: list[dict], k1: float = 1.5, b: float = 0.75) -> list[float]:
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens or not docs:
        return [0.0 for _ in docs]

    doc_tokens = [tokenize_for_bm25(keyword_text_for_doc(doc)) for doc in docs]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avg_doc_len = sum(doc_lengths) / max(len(doc_lengths), 1)

    doc_freq = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    total_docs = len(docs)
    scores = []

    for tokens, doc_len in zip(doc_tokens, doc_lengths):
        term_freq = Counter(tokens)
        score = 0.0

        for token in query_tokens:
            if token not in term_freq:
                continue

            df = doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            tf = term_freq[token]
            denom = tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
            score += idf * ((tf * (k1 + 1)) / denom)

        scores.append(score)

    return scores


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    max_score = max(scores)
    if max_score <= 0:
        return [0.0 for _ in scores]

    return [score / max_score for score in scores]


def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Không import được sentence_transformers hoặc dependency của nó. "
            "Hãy cài dependency còn thiếu, ví dụ: pip install packaging sentence-transformers"
        ) from exc

    return SentenceTransformer(model_name)


def semantic_scores(query_embedding, candidate_embeddings):
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Thiếu numpy để tính cosine similarity.") from exc

    query_vector = np.asarray(query_embedding)[0]
    candidates = np.asarray(candidate_embeddings)
    return candidates @ query_vector


class HybridRetriever:
    def __init__(
        self,
        docs: list[dict] | None = None,
        fixed_docs: list[dict] | None = None,
        mode: str = MODE_FIXED_STRUCTURED_KEYWORD,
        model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
    ):
        self.structured_docs = docs or []
        self.fixed_docs = fixed_docs or []
        self.mode = normalize_mode(mode)
        self.model = load_sentence_transformer(model_name)

        self.structured_embeddings = self._encode_docs(self.structured_docs)
        self.fixed_embeddings = self._encode_docs(self.fixed_docs)
        self.structured_keyword_embeddings = self._encode_docs(
            self.structured_docs,
            text_fn=keyword_text_for_doc,
        )
        self.fixed_keyword_embeddings = self._encode_docs(
            self.fixed_docs,
            text_fn=keyword_text_for_doc,
        )

    def _encode_docs(self, docs: list[dict], text_fn=None):
        if not docs:
            return None

        text_fn = text_fn or (lambda doc: doc["chunk_text"])
        texts = [text_fn(doc) for doc in docs]
        return self.model.encode(texts, normalize_embeddings=True)

    def _structured_candidates(self, chart_index: dict | None) -> tuple[list[dict], list[int]]:
        matched_indexes = [
            idx
            for idx, doc in enumerate(self.structured_docs)
            if doc_matches_chart(doc, chart_index)
        ]

        if not matched_indexes:
            matched_indexes = list(range(len(self.structured_docs)))

        return [self.structured_docs[idx] for idx in matched_indexes], matched_indexes

    def _combined_candidate_embeddings(self, structured_indexes: list[int]):
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("Thiếu numpy để ghép embeddings.") from exc

        embeddings = []

        if self.fixed_embeddings is not None:
            embeddings.append(self.fixed_embeddings)

        if structured_indexes and self.structured_embeddings is not None:
            embeddings.append(self.structured_embeddings[structured_indexes])

        if not embeddings:
            return None

        if len(embeddings) == 1:
            return embeddings[0]

        return np.concatenate(embeddings, axis=0)

    def _combined_candidate_keyword_embeddings(self, structured_indexes: list[int]):
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("Thiếu numpy để ghép keyword embeddings.") from exc

        embeddings = []

        if self.fixed_keyword_embeddings is not None:
            embeddings.append(self.fixed_keyword_embeddings)

        if structured_indexes and self.structured_keyword_embeddings is not None:
            embeddings.append(self.structured_keyword_embeddings[structured_indexes])

        if not embeddings:
            return None

        if len(embeddings) == 1:
            return embeddings[0]

        return np.concatenate(embeddings, axis=0)

    def _semantic_results(
        self,
        query: str,
        docs: list[dict],
        embeddings,
        source: str,
        top_k: int,
        indexes: list[int] | None = None,
    ) -> list[dict]:
        if not docs or embeddings is None:
            return []

        selected_embeddings = embeddings[indexes] if indexes is not None else embeddings
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        scores = semantic_scores(query_embedding, selected_embeddings)

        results = []
        for doc, sem_score in zip(docs, scores):
            results.append(
                {
                    "doc": doc,
                    "score": float(sem_score),
                    "semantic_score": float(sem_score),
                    "bm25_score": 0.0,
                    "source": source,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _fixed_similarity_search(self, query: str, top_k: int) -> list[dict]:
        return self._semantic_results(
            query=query,
            docs=self.fixed_docs,
            embeddings=self.fixed_embeddings,
            source="fixed",
            top_k=top_k,
        )

    def _fixed_structured_similarity_search(
        self,
        query: str,
        chart_index: dict | None,
        top_k: int,
    ) -> list[dict]:
        structured_docs, structured_indexes = self._structured_candidates(chart_index)

        results = []
        results.extend(self._fixed_similarity_search(query, top_k=top_k))
        results.extend(
            self._semantic_results(
                query=query,
                docs=structured_docs,
                embeddings=self.structured_embeddings,
                source="structured",
                top_k=top_k,
                indexes=structured_indexes,
            )
        )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _fixed_structured_keyword_search(
        self,
        query: str,
        chart_index: dict | None,
        top_k: int,
        alpha: float,
    ) -> list[dict]:
        structured_docs, structured_indexes = self._structured_candidates(chart_index)
        candidate_docs = self.fixed_docs + structured_docs

        if not candidate_docs:
            return []

        candidate_embeddings = self._combined_candidate_embeddings(structured_indexes)
        keyword_embeddings = self._combined_candidate_keyword_embeddings(structured_indexes)
        if candidate_embeddings is None or keyword_embeddings is None:
            return []

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        scores = semantic_scores(query_embedding, candidate_embeddings)
        keyword_scores = semantic_scores(query_embedding, keyword_embeddings)

        results = []
        fixed_count = len(self.fixed_docs)

        for idx, (doc, sem_score, keyword_score) in enumerate(
            zip(candidate_docs, scores, keyword_scores)
        ):
            final_score = alpha * sem_score + (1 - alpha) * keyword_score
            source = "fixed" if idx < fixed_count else "structured"

            results.append(
                {
                    "doc": doc,
                    "score": float(final_score),
                    "semantic_score": float(sem_score),
                    "keyword_score": float(keyword_score),
                    "bm25_score": 0.0,
                    "source": source,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search(
        self,
        query: str,
        chart_index: dict | None = None,
        top_k: int = 8,
        alpha: float = 0.75,
        mode: str | None = None,
    ) -> list[dict]:
        active_mode = normalize_mode(mode or self.mode)

        if active_mode == MODE_FIXED_SIMILARITY:
            return self._fixed_similarity_search(query, top_k=top_k)

        if active_mode == MODE_FIXED_STRUCTURED_SIMILARITY:
            return self._fixed_structured_similarity_search(
                query=query,
                chart_index=chart_index,
                top_k=top_k,
            )

        return self._fixed_structured_keyword_search(
            query=query,
            chart_index=chart_index,
            top_k=top_k,
            alpha=alpha,
        )


INITIAL_QUERIES = [
    "đặc điểm nổi bật về tính cách, mệnh, thân",
    "điểm mạnh điểm yếu nổi bật",
    "sự nghiệp học vấn tài chính tình duyên sức khỏe",
]


def retrieve_initial_highlights(retriever, chart_index, mode: str | None = None):
    all_results = []

    for query in INITIAL_QUERIES:
        results = retriever.search(
            query=query,
            chart_index=chart_index,
            top_k=6,
            alpha=0.7,
            mode=mode,
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
