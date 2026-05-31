import json
import re
import unicodedata

# NORMALIZE

def normalize(text: str) -> str:
    if not text:
        return ""

    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


# CANONICAL VOCAB

CANONICAL_STARS = {
    "tử_vi",
    "thiên_cơ",
    "thái_âm",
    "thái_dương",
    "vũ_khúc",
    "thiên_đồng",
    "liêm_trinh",
    "tham_lang",
    "cự_môn",
    "thiên_tướng",
    "thiên_lương",
    "thất_sát",
    "phá_quân",
    "thiên_phủ",

    "hóa_lộc",
    "hóa_quyền",
    "hóa_khoa",
    "hóa_kỵ",

    "văn_xương",
    "văn_khúc",

    "tả_phù",
    "hữu_bật",

    "lộc_tồn",
    "thiên_mã",

    "kình_dương",
    "đà_la",
    "hỏa_tinh",
    "linh_tinh",
    "địa_không",
    "địa_kiếp",

    "thiên_hình",
    "thiên_khốc",
    "thiên_hư",
    "thiên_riêu",
    "quả_tú",
    "cô_thần",

    "đào_hoa",
    "hồng_loan",
    "thiên_hỷ",

    "long_trì",
    "phượng_các",

    "thai_phụ",
    "phong_cáo",
    "tam_thai",
    "bát_tọa",

    "thiên_khôi",
    "thiên_việt",

    "thiên_quan",
    "thiên_phúc",
    "ân_quang",
    "thiên_quý",

    "giải_thần",
    "thiên_giải",
    "địa_giải",

    "thanh_long",
    "hỷ_thần",

    "quan_phù",
    "quan_phủ",

    "bạch_hổ",
    "tang_môn",
    "điếu_khách",

    "thiên_y",
    "thiên_trù",

    "thiếu_dương",
    "thiếu_âm",

    "quốc_ấn",
    "đường_phù",

    "tấu_thư",

    "thiên_la",
    "địa_võng",

    "lưu_hà",

    "kiếp_sát",

    "đại_hao",
    "tiểu_hao",

    "thiên_không",

    "phục_binh",

    "tướng_quân",

    "bệnh_phù",

    "thiên_tài",
    "thiên_thọ",

    "trực_phù",

    "thái_tuế",

    "tuế_phá",

    "thiên_đức",
    "nguyệt_đức",
    "long_đức",
    "phúc_đức",

    "bác_sỹ",
    "lực_sỹ",

    "phi_liêm",

    "đẩu_quân",

    "thiên_sứ",
    "thiên_thương",
}


# ALIASES

ALIASES = {
    "tử": "tử_vi",
    "cơ": "thiên_cơ",
    "nguyệt": "thái_âm",
    "nhật": "thái_dương",
    "vũ": "vũ_khúc",
    "đồng": "thiên_đồng",
    "liêm": "liêm_trinh",
    "tham": "tham_lang",
    "cự": "cự_môn",
    "tướng": "thiên_tướng",
    "lương": "thiên_lương",
    "sát": "thất_sát",
    "phá": "phá_quân",

    "không": "địa_không",
    "kiếp": "địa_kiếp",

    "hỏa": "hỏa_tinh",
    "linh": "linh_tinh",

    "kỵ": "hóa_kỵ",
    "khoa": "hóa_khoa",
    "quyền": "hóa_quyền",
    "lộc": "hóa_lộc",

    "xương": "văn_xương",
    "khúc": "văn_khúc",

    "tả": "tả_phù",
    "hữu": "hữu_bật",

    "riêu": "thiên_riêu",
    "hình": "thiên_hình",

    "khôi": "thiên_khôi",
    "việt": "thiên_việt",

    "mã": "thiên_mã",

    "đào": "đào_hoa",
    "hồng": "hồng_loan",

    "phủ": "thiên_phủ",

    "hổ": "bạch_hổ",

    "hao": "đại_hao",

    "qủa": "quả_tú",
    "quả": "quả_tú",

    "qúy": "thiên_quý",
    "quý": "thiên_quý",

    "khốc": "thiên_khốc",
    "hư": "thiên_hư",

    "long": "thanh_long",
    "phượng": "phượng_các",

    "quang": "ân_quang",

    "phúc": "phúc_đức",

    "tuế": "thái_tuế",

    "trực": "trực_phù",

    "cái": "hoa_cái",
}

IGNORED_TOKENS = {
    "tràng",
    "sinh",
    "tràng_sinh",
    "mộc",
    "mộc_dục",
    "dục",
    "quan",
    "đới",
    "quan_đới",
    "lâm",
    "lâm_quan",
    "đế",
    "vượng",
    "đế_vượng",
    "suy",
    "bệnh",
    "tử",
    "mộ",
    "tuyệt",
    "thai",
    "dưỡng",

    "đồng",
    "cung",
    "tọa",
    "thủ",
    "xung",
    "chiếu",
    "án",
    "ngữ",
    "gặp",
    "tại",
    "hay",
    "và",
    "có",
    "những",
    "trường",
    "hợp",
    "được",

    "sát_tinh",
    "sát",
    "tinh",
    "tam_không",

    "vô",
    "chính",
    "diệu",
}


# COMPOUND PATTERN DETECTOR

def canonicalize_star_id(star_id):
    results = set()

    # LIST INPUT
    if isinstance(star_id, list):
        for item in star_id:
            results.update(canonicalize_star_id(item))
        return results

    # STRING INPUT
    text = normalize(star_id)

    if not text:
        return results

    # exact canonical star
    if text in CANONICAL_STARS:
        results.add(("stars", text))
        return results

    # exact alias
    if text in ALIASES:
        results.add(("stars", ALIASES[text]))
        return results

    # TOKEN MATCHING
    tokens = text.split("_")

    for token in tokens:
        token = normalize(token)

        if not token:
            continue

        # ignore states / generic words
        if token in IGNORED_TOKENS:
            continue

        # exact star
        if token in CANONICAL_STARS:
            results.add(("stars", token))
            continue

        # alias -> canonical star
        if token in ALIASES:
            mapped = ALIASES[token]

            if mapped in CANONICAL_STARS:
                results.add(("stars", mapped))

    return results

def canonicalize_house(house_name):
    house = normalize(house_name)

    if house == "mệnh_và_cung_thân":
        return {
            ("houses", "mệnh"),
            ("houses", "cung_thân"),
        }

    return {("houses", house)}