import re
import unicodedata
from evaluation.canonicalize import canonicalize_star_id, canonicalize_house


vocab = {
    'stars': {'thiên_hư', 'tam_thai', 'hoa_cái', 'thái_dương', 'phá_toái', 'vũ_khúc', 'thiên_quan', 'nguyệt_đức', 'hồng_loan', 'thiên_khốc', 'lưu_niên_văn_tinh', 'đào_hoa', 'thiếu_dương', 'cô_thần', 'phong_cáo', 'tuế_phá', 'bác_sỹ', 'long_trì', 'thiên_giải', 'kiếp_sát', 'thất_sát', 'linh_tinh', 'quốc_ấn', 'hóa_kỵ', 'hóa_lộc', 'long_đức', 'thiên_phúc', 'đường_phù', 'thiên_đồng', 'bát_tọa', 'thiên_hình', 'thiên_khôi', 'tiểu_hao', 'quả_tú', 'quan_phủ', 'quan_phù', 'văn_xương', 'phi_liêm', 'phá_quân', 'địa_giải', 'lộc_tồn', 'thanh_long', 'địa_võng', 'tang_môn', 'thiên_việt', 'thái_tuế', 'phượng_các', 'hỷ_thần', 'thiên_sứ', 'phúc_đức', 'văn_khúc', 'đại_hao', 'tả_phù', 'cự_môn', 'thiên_thương', 'bệnh_phù', 'thái_âm', 'thiên_la', 'địa_kiếp', 'đẩu_quân', 'tấu_thư', 'thiên_không', 'thiếu_âm', 'tử_vi', 'thiên_tướng', 'thiên_hỷ', 'thiên_phủ', 'tử_phù', 'lực_sỹ', 'ân_quang', 'thai_phụ', 'giải_thần', 'kình_dương', 'thiên_đức', 'thiên_cơ', 'địa_không', 'điếu_khách', 'phục_binh', 'bạch_hổ', 'hỏa_tinh', 'hóa_quyền', 'thiên_lương', 'liêm_trinh', 'trực_phù', 'hóa_khoa', 'thiên_y', 'thiên_mã', 'thiên_trù', 'đà_la', 'thiên_quý', 'lưu_hà', 'tham_lang', 'tướng_quân', 'thiên_riêu', 'hữu_bật'},
    'houses': {'điền_trạch', 'tài_bạch', 'thiên_di', 'mệnh', 'phu_thê', 'huynh_đệ', 'nô_bộc', 'phụ_mẫu', 'phúc_đức', 'tật_ách', 'tử_tức', 'quan_lộc'},
    #'can': {'mậu', 'bính', 'giáp', 'ất', 'quý', 'tân', 'nhâm', 'kỷ', 'canh', 'đinh'},
    #'chi': {'tuất', 'dần', 'thìn', 'sửu', 'thân', 'dậu', 'ngọ', 'mão', 'tý', 'tỵ', 'mùi', 'hợi'},
    #'brightness': {'bình', 'hãm', 'đắc', 'miếu', 'vượng'},
    #'trang_sinh': {'thai', 'lâm_quan', 'suy', 'mộc_dục', 'dưỡng', 'tràng_sinh', 'tuyệt', 'mộ', 'đế_vượng', 'bệnh', 'tử', 'quan_đới'},
    #'gender': {'nam', 'nữ'},
    #'am_duong': {'âm', 'dương'},
    #'element': {'kim', 'mộc', 'thủy', 'hỏa', 'thổ'}
}


# NORMALIZE

def normalize(text: str) -> str:
    if not text:
        return ""

    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text)

    return text


# EXTRACT FEATURES FROM TEXT

def extract_features_from_text(text, vocab=vocab):
    text_norm = normalize(text)

    found = set()

    for category, terms in vocab.items():
        for term in terms:
            pattern = rf"(?<![^\W_]){re.escape(term)}(?![^\W_])"

            if re.search(pattern, text_norm):
                found.add((category, term))

    return found


# CHART FEATURES

def extract_chart_features(houses_chart, tuvi_chart, user_chart):
    features = set()

    # houses chart
    for house in houses_chart:

        # house name
        if house.get("house_topic"):
            features.update(
                canonicalize_house(house["house_topic"])
            )

        # zodiac
        if house.get("zodiac_sign"):
            features.add((
                "chi",
                normalize(house["zodiac_sign"])
            ))

        # main stars
        for star in house.get("chinh_tinh", []):

            if star.get("name"):
                features.update(
                    canonicalize_star_id(star["name"])
                )

            if star.get("brightness"):
                features.add((
                    "brightness",
                    normalize(star["brightness"])
                ))

        # side stars
        for star in house.get("phu_tinh", []):
            features.update(
                canonicalize_star_id(star)
            )

        # trang sinh
        if house.get("vong_trang_sinh"):
            features.add((
                "trang_sinh",
                normalize(house["vong_trang_sinh"])
            ))

    # tuvi chart
    if tuvi_chart:

        # elements
        if tuvi_chart.get("ban_menh", {}).get("element"):
            features.add((
                "element",
                normalize(tuvi_chart["ban_menh"]["element"])
            ))

        if tuvi_chart.get("cuc", {}).get("element"):
            features.add((
                "element",
                normalize(tuvi_chart["cuc"]["element"])
            ))

        # menh chu
        if tuvi_chart.get("menh_chu"):
            features.add((
                "stars",
                normalize(tuvi_chart["menh_chu"])
            ))

        # than chu
        if tuvi_chart.get("than_chu"):
            features.add((
                "stars",
                normalize(tuvi_chart["than_chu"])
            ))

    # user chart
    if user_chart:

        # gender
        if user_chart.get("gender"):
            features.add((
                "gender",
                normalize(user_chart["gender"])
            ))

        # âm/dương gender
        if user_chart.get("am_duong_gender"):
            text = normalize(user_chart["am_duong_gender"])

            if "âm" in text:
                features.add(("am_duong", "âm"))

            if "dương" in text:
                features.add(("am_duong", "dương"))

        # can chi
        lunar_year = (user_chart.get("dob_lunar", {}).get("year", ""))

        lunar_features = extract_features_from_text(lunar_year, vocab)

        features.update(lunar_features)

    return features


# STRUCTURED DOC FEATURES

def extract_structured_doc_features(doc):
    features = set()

    # palace
    if doc.get("palace_name"):
        features.update(canonicalize_house(doc["palace_name"]))

    # main star
    if doc.get("star_id"):
        features.update(canonicalize_star_id(doc["star_id"]))

    # required stars
    required = doc.get("required_stars", {})

    for _, stars in required.items():
        for star in stars:
            features.update(canonicalize_star_id(star))

    return features