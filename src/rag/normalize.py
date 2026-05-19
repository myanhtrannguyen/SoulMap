# core/normalize.py
import re
import unicodedata

def normalize_text(text: str) -> str:
    if isinstance(text, list):
        text = ", ".join(text)
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = text.replace("đ", "d")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w_]", "", text)
    return text


def normalize_star_name(name: str) -> str:
    return normalize_text(name)


def normalize_palace_name(name: str) -> str:
    mapping = {
        "mệnh": "cung_menh",
        "phụ mẫu": "cung_phu_mau",
        "phúc đức": "cung_phuc_duc",
        "điền trạch": "cung_dien_trach",
        "quan lộc": "cung_quan_loc",
        "nô bộc": "cung_no_boc",
        "thiên di": "cung_thien_di",
        "tật ách": "cung_tat_ach",
        "tài bạch": "cung_tai_bach",
        "tử tức": "cung_tu_tuc",
        "phu thê": "cung_phu_the",
        "huynh đệ": "cung_huynh_de",
    }
    return mapping.get(name.lower().strip(), normalize_text(name))