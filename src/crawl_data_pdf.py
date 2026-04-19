import fitz  # PyMuPDF
import re
import json
from ftfy import fix_text


# =========================
# CONFIG
# =========================

PDF_PATH = r"D:\Hust\Năm ba\NLP\prj\data_raw_pdf\tu_vi_boi_toan.pdf"
OUTPUT_JSON = "dataset.json"
FAISS_INDEX = "faiss.index"


# =========================
# STEP 2: FIX + CLEAN
# =========================

import re
from ftfy import fix_text
import unicodedata

TCVN3_FULL_MAP = {
    "Ö": "Ư", "ö": "ư",
    "Û": "Ử", "û": "ử",
    "Ù": "Ú", "ù": "ú",
    "Ï": "Ị", "ï": "ị",
    "Ê": "Ê", "ê": "ê",
    "Ô": "Ô", "ô": "ô",
    "Ñ": "Đ", "ñ": "đ",

    # vowel combinations
    "aø": "à", "aù": "á", "aû": "ả", "aõ": "ã", "aï": "ạ",
    "aày": "ầy", "aày": "ầy",
    "aày": "ầy",

    "eø": "è", "eù": "é", "eû": "ẻ", "eõ": "ẽ", "eï": "ẹ",
    "oø": "ò", "où": "ó", "oû": "ỏ", "oõ": "õ", "oï": "ọ",
    "uø": "ù", "uù": "ú", "uû": "ủ", "uõ": "ũ", "uï": "ụ",

    "AØ": "À", "AÙ": "Á", "AÛ": "Ả", "AÕ": "Ã", "AÏ": "Ạ",
    "EØ": "È", "EÙ": "É", "EÛ": "Ẻ", "EÕ": "Ẽ", "EÏ": "Ẹ",
    "OØ": "Ò", "OÙ": "Ó", "OÛ": "Ỏ", "OÕ": "Õ", "OÏ": "Ọ",
}

def fix_tcvn3(text):
    for k, v in TCVN3_FULL_MAP.items():
        text = text.replace(k, v)
    return text


def preprocess(text):
    # 1. fix unicode lỗi nhẹ
    text = fix_text(text)

    # 2. fix TCVN3
    text = fix_tcvn3(text)

    # 3. normalize unicode
    text = unicodedata.normalize("NFC", text)

    # 4. fix lỗi lặp dấu kiểu "aày"
    text = re.sub(r"aày", "ày", text)
    text = re.sub(r"aá", "á", text)
    text = text.replace("ưô", "ươ")
    text = text.replace("oà", "òa")
    # 5. Remove extra whitespace and newlines    
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# =========================
# STEP 6: SAVE JSON
# =========================

def save_json(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)



def main():
    
    PDF_PATH = r"D:\Hust\Năm ba\NLP\prj\data_raw_pdf\tu_vi_boi_toan.pdf"
    doc = fitz.open(PDF_PATH)
    print(f"📄 Total pages: {len(doc)}")
    raw_text = ""
    for page in doc:
        raw_text += page.get_text()
        
    with open("data_raw_text.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
        # print(len(raw_text))
        # print(raw_text[:500])

if __name__ == "__main__":
    main()