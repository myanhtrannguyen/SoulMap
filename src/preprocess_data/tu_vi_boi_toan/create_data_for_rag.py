import os
import json
import glob
from uuid import uuid4

INPUT_DIR = "data/data_process/tu_vi_boi_toan/cung"
OUTPUT_RAG = "data/data_process/tu_vi_boi_toan/rag_documents.jsonl"
OUTPUT_TOPIC = "data/data_process/tu_vi_boi_toan/topic_mapping.json"


# =========================
# MAP DESCRIPTION -> TOPICS
# =========================

TOPIC_KEYWORDS = {
    "tình_duyên": [
        "vợ chồng", "hôn nhân", "tình duyên",
        "phu thê", "đào hoa", "kết hôn"
    ],

    "sự_nghiệp": [
        "công danh", "sự nghiệp", "quan lộc",
        "thăng tiến", "nghề nghiệp", "việc làm", "nhà cửa", "cơ nghiệp", "sản nghiệp"
    ],

    "học_vấn": [
        "học", "thi cử", "trí tuệ",
        "kiến thức", "bằng cấp"
    ],

    "tài_chính": [
        "tài bạch", "tiền bạc", "tài chính",
        "giàu", "thu nhập"
    ],

    "sức_khỏe": [
        "tật ách", "bệnh", "sức khỏe",
        "ốm đau", "tai nạn", "tuổi thọ"
    ],

    "gia_đình": [
        "phụ mẫu", "cha mẹ", "anh chị em",
        "gia đình", "huynh đệ", "nhà"
    ],

    "con_cái": [
        "tử tức", "con cái"
    ],

    "bạn_bè": [
        "nô bộc", "bạn bè", "quan hệ"
    ],

    "di_chuyển": [
        "thiên di", "xuất ngoại", "đi xa"
    ]
}


# =====================================
# DETECT TOPIC FROM DESCRIPTION/TEXT
# =====================================

def detect_topics(text):
    text = text.lower()

    matched_topics = []
    dem = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            count = text.count(kw)
            if count > 0:
                dem[topic] = dem.get(topic, 0) + count
                if dem[topic] >= 2:  # Nếu đã có tổng số keyword khớp >= 3, coi như chủ đề này được xác định
                    if topic not in matched_topics:
                        matched_topics.append(topic)
                break

    return matched_topics


# =====================================
# BUILD TOPIC -> PALACES MAPPING
# =====================================

topic_to_palaces = {}


# =====================================
# CREATE RAG DOCUMENTS
# =====================================

rag_docs = []

json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))

for file_path in json_files:

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for palace_id, palace_data in data.items():

        palace_name = palace_data.get("name", palace_id).replace("cung_", "", 1)
        source_content = palace_data.get("source_content", "")

        # detect topics from description
        topics = detect_topics(source_content)

        # build topic mapping
        for topic in topics:
            topic_to_palaces.setdefault(topic, set()).add(palace_id)

        stars = palace_data.get("stars", [])

        for star in stars:

            star_id = star.get("star_id")
            if isinstance(star_id, list):
                star_id = [s.replace("_", " ") for s in star_id]
            context_type = star.get("context_type")

            required_stars = star.get("sao_toa_thu_tai_cung_menh", [])

            rules = star.get("rules", {})

            for condition, interpretation in rules.items():
                
                add_condition = ""

                if palace_id == "Cung_no_boc" and required_stars:
                    add_condition = "; cung mệnh có sao " + (", ".join(required_stars) if required_stars else "") + " " + context_type.replace("menh_status_", "").replace("_", " ")

                if palace_id == "Cung_menh":
                    condition = f"Cung {palace_name.replace('_', ' ')} có sao {", ".join(star_id) if isinstance(star_id, list) else star_id.replace('_', ' ')} luận theo {context_type.replace("_", " ")}"
                else : 
                    condition = f"Cung {palace_name.replace('_', ' ')} có sao {", ".join(star_id) if isinstance(star_id, list) else star_id.replace('_', ' ')} {condition.replace('_', ' ') if condition != "rule_text" else ''}{add_condition}"

                chunk_text = f"""
Chủ đề: {", ".join([topic.replace('_', ' ') for topic in topics]) if topics else "Không xác định"}
Cung {palace_name.replace('_', ' ')}
Điều kiện: {condition}

Luận giải:
{"\n".join(interpretation) if isinstance(interpretation, list) else interpretation}
""".strip()

                doc = {
                    "id": str(uuid4()),

                    "topic": topics,

                    "palace_id": palace_id,
                    "palace_name": palace_name,

                    "star_id": star_id,

                    "required_stars": {"cung_menh": required_stars},

                    "context_type": context_type,

                    "condition": condition,

                    "interpretation": interpretation,

                    "chunk_text": chunk_text
                }

                rag_docs.append(doc)


# =====================================
# SAVE JSONL
# =====================================

with open(OUTPUT_RAG, "w", encoding="utf-8") as f:

    for doc in rag_docs:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")


# =====================================
# SAVE TOPIC MAP
# =====================================

topic_mapping = {
    topic: sorted(list(palaces))
    for topic, palaces in topic_to_palaces.items()
}

with open(OUTPUT_TOPIC, "w", encoding="utf-8") as f:
    json.dump(topic_mapping, f, ensure_ascii=False, indent=2)


print(f"Created {len(rag_docs)} RAG documents")
print(f"Saved topic mapping")