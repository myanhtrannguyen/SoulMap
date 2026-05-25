from __future__ import annotations

from typing import Iterable
import unicodedata
import re


def normalize_text(text: str) -> str:
    if not text:
        return ""

    if isinstance(text, list):
        text = ", ".join(text)

    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )
    text = text.replace("đ", "d")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)

    return text


def make_palace_feature(palace_id: str) -> str:
    return f"palace:{normalize_text(palace_id)}"


def make_star_feature(palace_id: str, star_id: str) -> str:
    return f"star:{normalize_text(palace_id)}:{normalize_text(star_id)}"


def make_brightness_feature(
    palace_id: str,
    star_id: str,
    brightness: str,
) -> str:
    return (
        f"brightness:{normalize_text(palace_id)}:"
        f"{normalize_text(star_id)}:{normalize_text(brightness)}"
    )


def make_zodiac_feature(palace_id: str, zodiac: str) -> str:
    return f"zodiac:{normalize_text(palace_id)}:{normalize_text(zodiac)}"


def make_tuan_triet_feature(palace_id: str, item: str) -> str:
    return f"tuan_triet:{normalize_text(palace_id)}:{normalize_text(item)}"


def make_dai_han_feature(palace_id: str, dai_han) -> str:
    return f"dai_han:{normalize_text(palace_id)}:{normalize_text(str(dai_han))}"


def make_required_feature(palace_id: str, star_id: str) -> str:
    return f"required:{normalize_text(palace_id)}:{normalize_text(star_id)}"


def extract_features_from_docs(
    retrieved_docs: list[dict],
    chart_index: dict,
) -> set[str]:
    features = set()

    for doc in retrieved_docs:
        palace_id = normalize_text(doc.get("palace_id", ""))

        if not palace_id:
            continue

        if palace_id not in chart_index:
            continue

        palace_data = chart_index[palace_id]

        # palace
        features.add(make_palace_feature(palace_id))

        # zodiac
        zodiac = palace_data.get("zodiac_sign")
        if zodiac:
            features.add(make_zodiac_feature(palace_id, zodiac))

        # dai_han
        dai_han = palace_data.get("dai_han")
        if dai_han is not None:
            features.add(make_dai_han_feature(palace_id, dai_han))

        # tuan triet
        for item in palace_data.get("tuan_triet", []):
            features.add(make_tuan_triet_feature(palace_id, item))

        # main star from doc
        star_id = normalize_text(doc.get("star_id", ""))

        if star_id:
            features.add(make_star_feature(palace_id, star_id))

            brightness = palace_data.get("brightness", {}).get(star_id)

            if brightness:
                features.add(
                    make_brightness_feature(
                        palace_id,
                        star_id,
                        brightness,
                    )
                )

        # required stars
        required_stars = doc.get("required_stars", {})

        for req_palace, req_stars in required_stars.items():
            req_palace = normalize_text(req_palace)

            if req_palace not in chart_index:
                continue

            user_stars = set(chart_index[req_palace]["all_stars"])

            for star in req_stars:
                star = normalize_text(star)

                if star in user_stars:
                    features.add(
                        make_required_feature(req_palace, star)
                    )

    return features


def feature_to_keywords(feature: str) -> list[str]:

    parts = feature.split(":")
    feature_type = parts[0]

    if feature_type == "palace":
        raw_keywords = [parts[1]]

    elif feature_type == "star":
        raw_keywords = [parts[1], parts[2]]

    elif feature_type == "brightness":
        raw_keywords = [
            parts[1],
            parts[2],
            parts[3],
        ]

    elif feature_type == "zodiac":
        raw_keywords = [parts[1], parts[2]]

    elif feature_type == "tuan_triet":
        raw_keywords = [parts[1], parts[2]]

    elif feature_type == "dai_han":
        raw_keywords = [parts[1], parts[2]]

    elif feature_type == "required":
        raw_keywords = [parts[1], parts[2]]

    else:
        raw_keywords = []

    return [
        normalize_text(x)
        for x in raw_keywords
    ]



def extract_mentioned_features(
    answer: str,
    candidate_features: Iterable[str],
) -> set[str]:
    normalized_answer = normalize_text(answer)

    mentioned = set()

    for feature in candidate_features:
        keywords = feature_to_keywords(feature)

        if not keywords:
            continue

        matched = True

        for kw in keywords:
            if kw not in normalized_answer:
                matched = False
                break

        if matched:
            mentioned.add(feature)

    return mentioned


def compute_coverage_score(
    answer: str,
    retrieved_docs: list[dict],
    chart_index: dict,
) -> dict:

    input_features = extract_features_from_docs(
        retrieved_docs=retrieved_docs,
        chart_index=chart_index,
    )

    mentioned_features = extract_mentioned_features(
        answer=answer,
        candidate_features=input_features,
    )

    coverage = 0.0

    if input_features:
        coverage = len(mentioned_features) / len(input_features)

    return {
        "coverage_score": round(coverage, 4),
        "num_input_features": len(input_features),
        "num_mentioned_features": len(mentioned_features),
        "input_features": sorted(input_features),
        "mentioned_features": sorted(mentioned_features),
        "missing_features": sorted(
            input_features - mentioned_features
        ),
    }