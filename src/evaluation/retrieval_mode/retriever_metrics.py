from evaluation.answer_generation.coverage_grounding import build_retrieved_feature_set


def feature_recall(expected_features, retrieved_features):
    expected = set(map(tuple, expected_features))

    if not expected:
        return 1.0

    matched = len(expected & retrieved_features)

    return matched / len(expected)


def query_success(expected_features, retrieved_features):
    expected = set(map(tuple, expected_features))

    return expected.issubset(retrieved_features)


def missing_features(expected_features, retrieved_features):
    expected = set(map(tuple, expected_features))

    return expected - retrieved_features


def reciprocal_rank(expected_features, retrieved_docs):
    expected = set(map(tuple, expected_features))

    for rank, doc in enumerate(retrieved_docs, start=1):
        doc_features = build_retrieved_feature_set([doc])

        if expected & doc_features:
            return 1.0 / rank

    return 0.0


def hit_at_k(expected_features, retrieved_docs, k):
    expected = set(map(tuple, expected_features))

    docs = retrieved_docs[:k]

    features = build_retrieved_feature_set(docs)

    return len(expected & features) > 0

def all_hit_at_k(expected_features, docs, k):
    features = build_retrieved_feature_set(docs[:k])

    expected = set(map(tuple, expected_features))

    return expected.issubset(features)