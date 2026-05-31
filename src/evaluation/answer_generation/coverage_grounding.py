from evaluation.feature_extraction import extract_chart_features, extract_features_from_text, extract_structured_doc_features, vocab


# INITIAL SUMMARY FEATURES

def extract_summary_features(initial_summary, vocab):
    return extract_features_from_text(
        initial_summary,
        vocab
    )


# FIXED CHUNK FEATURES

def extract_fixed_chunk_features(doc, vocab):
    chunk_text = doc.get("chunk_text", "")

    return extract_features_from_text(chunk_text, vocab)


# BUILD RETRIEVED FEATURE SET

def build_structured_feature_set(retrieved_docs):
    all_features = set()

    for doc in retrieved_docs:
        if ("palace_name" in doc or "star_id" in doc):
            feats = extract_structured_doc_features(doc)

            all_features.update(feats)

    return all_features

def build_retrieved_feature_set(retrieved_docs, vocab=vocab):
    all_features = set()

    for doc in retrieved_docs:

        # structured docs
        if ("palace_name" in doc or "star_id" in doc):
            feats = extract_structured_doc_features(doc)

            all_features.update(feats)

        # fixed chunks
        else:
            feats = extract_fixed_chunk_features(doc, vocab)

            all_features.update(feats)

    return all_features


# BUILD ALLOWED CONTEXT

def build_allowed_context_features(houses_chart, tuvi_chart, user_chart, initial_summary, retrieved_docs, vocab):
    chart_features = extract_chart_features(houses_chart, tuvi_chart, user_chart)

    summary_features = extract_summary_features(initial_summary, vocab)

    retrieved_features = build_retrieved_feature_set(retrieved_docs, vocab)

    allowed_features = (chart_features | summary_features | retrieved_features)

    return allowed_features


# ANSWER FEATURES

def extract_answer_features(answer, vocab):
    return extract_features_from_text(answer, vocab)


# COVERAGE

def compute_coverage_score(answer, retrieved_docs, vocab=vocab):
    expected_features = build_structured_feature_set(retrieved_docs)

    answer_features = extract_answer_features(answer, vocab)

    matched = expected_features & answer_features

    coverage_score = (
        len(matched) / len(expected_features)
        if expected_features else None
    )

    return {
        "coverage_score": coverage_score,
        "expected_count": len(expected_features),
        "matched_count": len(matched),
        "matched_features": sorted(list(matched)),
        "missing_features": sorted(list(expected_features - matched)),
        "answer_features": sorted(list(answer_features))
    }


# GROUNDING

def compute_strict_grounding_score(answer, retrieved_docs, vocab=vocab):
    retrieved_features = build_retrieved_feature_set(retrieved_docs, vocab)

    answer_features = extract_answer_features(answer, vocab)

    grounded = answer_features & retrieved_features

    unsupported = answer_features - retrieved_features

    grounding_score = (
        len(grounded) / len(answer_features)
        if answer_features else 0
    )

    return {
        "grounding_score": round(grounding_score, 4),
        "answer_feature_count": len(answer_features),
        "grounded_count": len(grounded),
        "unsupported_count": len(unsupported),
        "grounded_features": sorted(list(grounded)),
        "unsupported_features": sorted(list(unsupported)),
        "retrieved_features": sorted(list(retrieved_features)),
    }

def compute_prompt_grounding_score(answer, houses_chart, tuvi_chart, user_chart, initial_summary, retrieved_docs, vocab=vocab):
    allowed_features = (build_allowed_context_features(houses_chart, tuvi_chart, user_chart, initial_summary, retrieved_docs, vocab))

    answer_features = extract_answer_features(answer, vocab)

    grounded = answer_features & allowed_features

    unsupported = answer_features - allowed_features

    grounding_score = (
        len(grounded) / len(answer_features)
        if answer_features else 0
    )

    return {
        "grounding_score": round(grounding_score, 4),
        "answer_feature_count": len(answer_features),
        "grounded_count": len(grounded),
        "unsupported_count": len(unsupported),
        "grounded_features": sorted(list(grounded)),
        "unsupported_features": sorted(list(unsupported)),
        "retrieved_features": sorted(list(allowed_features)),
    }