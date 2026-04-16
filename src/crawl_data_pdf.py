import fitz  # PyMuPDF
import re
import json
from ftfy import fix_text
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

PDF_PATH = r"/Users/trannguyenmyanh/Documents/SoulMap/data_raw_pdf/tu_vi_boi_toan.pdf"
OUTPUT_JSON = "dataset.json"
FAISS_INDEX = "faiss.index"

SAO_LIST = [
    "Tử Vi", "Thiên Phủ", "Thái Âm", "Thái Dương",
    "Hỏa Linh", "Kình Đà", "Cự Liêm", "Tham Lang",
    "Thiên Cơ", "Thiên Lương", "Thiên Đồng"
]

CUNG_LIST = [
    "Mệnh", "Phu Thê", "Tài Bạch", "Quan Lộc",
    "Phúc Đức", "Điền Trạch", "Huynh Đệ"
]

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
# STEP 3: DETECT CUNG
# =========================

def detect_cung(text):
    for cung in CUNG_LIST:
        if cung.lower() in text.lower():
            return cung
    return "Unknown"


# =========================
# STEP 4: EXTRACT SAO BLOCKS
# =========================

def extract_sao_blocks(text):
    chunks = []
    
    lines = text.split("\n")
    current_sao = None
    buffer = ""

    for line in lines:
        line_clean = line.strip()
        # print(line_clean)  # Debug: print each line to see what's being processed
        found = False
        for sao in SAO_LIST:
            if sao.lower() in line_clean.lower():
                if current_sao:
                    chunks.append((current_sao, buffer.strip()))
                current_sao = sao
                buffer = line_clean
                found = True
                break
        
        if not found:
            buffer += " " + line_clean

    if current_sao:
        chunks.append((current_sao, buffer.strip()))

    return chunks


# =========================
# STEP 5: BUILD DATASET
# =========================

def build_dataset(chunks):
    dataset = []

    for i, (sao, content) in enumerate(chunks):
        dataset.append({
            "id": f"{sao}_{i}",
            "sao": sao,
            "cung": detect_cung(content),
            "noi_dung": content,
            "nguon": "tu_vi_tong_hop_pdf"
        })

    return dataset


# =========================
# STEP 6: SAVE JSON
# =========================

def save_json(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


# =========================
# STEP 7: EMBEDDING + FAISS
# =========================

def build_faiss(dataset):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    texts = [item["noi_dung"] for item in dataset]
    
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


# =========================
# STEP 8: SAVE FAISS
# =========================

def save_faiss(index, path):
    faiss.write_index(index, path)


# =========================
# MAIN PIPELINE
# =========================

# def main():
# #     text = "TÖÛ VI THÖÏC HAØNH"
# #     print(text.encode('cp1252').decode('utf-8'))
#     print("🔹 Extracting PDF...")
#     doc = fitz.open(PDF_PATH)
#     dem = 0
#     dataset = []
#     print(f"📄 Total pages: {len(doc)}")
#     for page in doc:
#         raw_text = page.get_text()
#         dem += 1
        
#         print(f"Đang đọc trang {dem}/{len(doc)}", end="\r")
#         clean = preprocess(raw_text)
#         print(clean)
#         print(f"🔹 Extracting SAO blocks trang {dem}/{len(doc)}")
#         chunks = extract_sao_blocks(clean)
#         print(f"✅ Found {len(chunks)} chunks")
#         print("🔹 Building dataset...")
#         dataset += build_dataset(chunks)

#     print("🔹 Saving JSON...")
#     save_json(dataset, OUTPUT_JSON)

    

#     # print(len(clean))  # Debug: print first 500 characters of cleaned text
#     # print(clean)  # Debug: print first 500 characters of cleaned text
   

    

    

    

#     # print("🔹 Building FAISS index...")
#     # index, embeddings = build_faiss(dataset)

#     # print("🔹 Saving FAISS...")
#     # save_faiss(index, FAISS_INDEX)

#     print("🎉 DONE!")

import re
import json

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

ZODIAC_SIGNS = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]

def extract_zodiac_signs(text_line, prefix):
    """
    Hàm tìm kiếm 12 Địa chi nằm ngay sau các từ khóa như 'Miếu địa:'
    Ví dụ: "- Miếu địa: Tỵ, Ngọ, Dần, Thân: thông minh..." -> Trả về ['Tỵ', 'Ngọ', 'Dần', 'Thân']
    """
    start_idx = text_line.lower().find(prefix.lower()) + len(prefix)
    
    # Tìm dấu ":" tiếp theo để giới hạn vùng tìm kiếm (tránh lấy nhầm cung ở phần giải nghĩa)
    end_idx = text_line.find(':', start_idx)
    if end_idx == -1:
        end_idx = len(text_line)
        
    target_area = text_line[start_idx:end_idx]
    
    found_signs = []
    for sign in ZODIAC_SIGNS:
        # Nếu tên địa chi xuất hiện trong khoảng đã cắt
        if sign in target_area:
            found_signs.append(sign)
    return found_signs

def parse_star_dictionary_advanced(text):
    stars_db = []
    
    # Cắt văn bản theo mẫu "3.1.", "3.2."...
    blocks = re.split(r'(?:^|\n)3\.\d+\.\s+', text)
    
    for block in blocks[1:]: # Bỏ qua phần giới thiệu đầu tiên
        lines = block.split('\n')
        if not lines: continue
        
        header = clean_text(lines[0])
        
        # 1. Khởi tạo Document trống đúng chuẩn Schema
        star_doc = {
            "star_id": "",
            "name": "",
            "type": "Chính diệu", # Fallback mặc định
            "element": "",
            "polarity": "",
            "direction": "",
            "core_characteristics": [],
            "positions": {
                "mieu_dia": [],
                "vuong_dia": [],
                "dac_dia": [],
                "binh_hoa": [],
                "ham_dia": []
            }
        }
        
        # 2. Xử lý Header: Bóc tách Tên, Phương vị, Âm/Dương, Ngũ hành, Loại sao
        # Pattern: [Tên sao] ( [Các thuộc tính cách nhau bởi dấu phẩy] )
        header_match = re.search(r'([A-ZĐ][\w\s]+?)\s*\((.*?)\)', header)
        if header_match:
            star_doc["name"] = header_match.group(1).strip()
            star_doc["star_id"] = star_doc["name"].lower().replace(" ", "_")
            
            attrs_raw = header_match.group(2).split(',')
            for attr in attrs_raw:
                attr = attr.strip()
                
                # Phân loại phương vị (Ví dụ: Nam Bắc Đẩu tinh)
                if "Đẩu tinh" in attr or "Thiên" in attr:
                    star_doc["direction"] = attr
                    
                # Phân loại tính chất sao (Ví dụ: Đế tinh, Tù tinh, Phúc tinh)
                elif "tinh" in attr and "Đẩu" not in attr:
                    star_doc["type"] = attr
                    
                # Phân tích Âm/Dương và Ngũ hành (Ví dụ: "Dương Thổ", "Âm Kim đới Hỏa")
                else:
                    if "Dương" in attr: star_doc["polarity"] = "Dương"
                    elif "Âm" in attr: star_doc["polarity"] = "Âm"
                    
                    if "Kim" in attr: star_doc["element"] = "Kim"
                    elif "Mộc" in attr: star_doc["element"] = "Mộc"
                    elif "Thủy" in attr: star_doc["element"] = "Thủy"
                    elif "Hỏa" in attr: star_doc["element"] = "Hỏa"
                    elif "Thổ" in attr: star_doc["element"] = "Thổ"
        else:
            # Xử lý fallback nếu header không có dấu ngoặc đơn
            star_doc["name"] = header.strip()
            star_doc["star_id"] = star_doc["name"].lower().replace(" ", "_")

        # 3. Phân tích nội dung bên dưới (Core characteristics & Positions)
        for line in lines[1:]:
            clean_line = clean_text(line)
            line_lower = clean_line.lower()
            
            # Bóc tách "Chủ..."
            if clean_line.startswith("- Chủ") or clean_line.startswith("• Chủ"):
                chars_str = clean_line.split("Chủ")[1].strip()
                # Split bằng dấu phẩy và dọn dẹp khoảng trắng
                star_doc["core_characteristics"] = [c.strip() for c in chars_str.split(',') if c.strip()]
            
            # Bóc tách 12 Cung (Vị trí)
            elif "miếu địa:" in line_lower:
                star_doc["positions"]["mieu_dia"] = extract_zodiac_signs(clean_line, "miếu địa:")
            elif "vượng địa:" in line_lower:
                star_doc["positions"]["vuong_dia"] = extract_zodiac_signs(clean_line, "vượng địa:")
            elif "đắc địa:" in line_lower:
                star_doc["positions"]["dac_dia"] = extract_zodiac_signs(clean_line, "đắc địa:")
            elif "bình hòa:" in line_lower:
                star_doc["positions"]["binh_hoa"] = extract_zodiac_signs(clean_line, "bình hòa:")
            elif "hãm địa:" in line_lower:
                star_doc["positions"]["ham_dia"] = extract_zodiac_signs(clean_line, "hãm địa:")
                
        stars_db.append(star_doc)
        
    return stars_db


def parse_star_interpretations(text):
    """
    Trích xuất luận giải chi tiết từ Phần 4 (hoặc các phần luận giải Cung)
    Mẫu: "4.2.1. Tử Vi" sau đó là các thẻ "☯ Đại cương", "☯ Nam mệnh"...
    """
    interpretations_db = []
    
    # Cắt văn bản ra thành từng khối sao dựa trên mục 4.2.x.
    blocks = re.split(r'4\.2\.\d+\.\s+', text)
    
    for block in blocks[1:]: # Bỏ qua phần giới thiệu đầu tiên
        lines = block.split('\n')
        star_name = clean_text(lines[0].strip())
        star_id = star_name.lower().replace(" ", "_")
        
        current_context = ""
        current_rules = []
        rule_text = ""
        for line in lines[1:]:
            # line = line.strip()
            if not line:
                continue
            # if star_id == "hỏa,_linh":
            #     print(line)
            if "☯" in line and current_rules:
                if star_id == "hỏa,_linh":
                    print("star_id:", line)
                    print(current_rules)
                interpretations_db.append({
                    "interpretation_id": f"{star_id}_{current_context}",
                    "star_id": star_id,
                    "context_type": current_context,
                    "rules": current_rules
                })
                current_rules = []

            # Nhận diện Context [cite: 782, 792, 798, 801]
            if "☯ Đại cương" in line:
                current_context = "general"
            elif "☯ Nam mệnh" in line:
                current_context = "nam_menh"
            elif "☯ Nữ mệnh" in line:
                current_context = "nu_menh"
            elif "☯ Phú giải" in line:
                current_context = "phu_giai"
            elif current_context == "phu_giai" and line:
                # Lưu câu phú nguyên bản 
                current_rules.append({
                    "condition": "Câu Phú",
                    "effect": line
                })
            # Nhận diện các dòng luận giải bắt đầu bằng dấu '+' hoặc '-' [cite: 782, 783]
            elif line.startswith("+") or line.startswith("-"):
                if rule_text:
                    current_rules.append({
                        "condition": "Mặc định", # Cần NLP thêm để tách "Gặp Xương Khúc -> ..."
                        "effect": rule_text
                    })
                rule_text = clean_text(line[1:])
            else:
                rule_text += " " + clean_text(line)

                
            
            
    return interpretations_db

def main():
    # Đọc file dữ liệu đã extract
    # try:
    #     with open("tuvi_data.txt", "r", encoding="utf-8") as f:
    #         full_text = f.read()
    # except FileNotFoundError:
    #     print("Vui lòng lưu nội dung vào file tuvi_data.txt trước khi chạy.")
    #     return
    PDF_PATH = r"/Users/trannguyenmyanh/Documents/SoulMap/data_raw_pdf/tu_vi_tong_hop.pdf"
    doc = fitz.open(PDF_PATH)
    print(f"📄 Total pages: {len(doc)}")
    raw_text = ""
    for page in doc:
        raw_text += page.get_text()

    with open("data_raw_text.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
        print(len(raw_text))
        # print(raw_text[:500])

    # print("Đang xử lý dữ liệu...")
    
    # # 1. Trích xuất danh sách sao
    # stars_data = parse_star_dictionary_advanced(raw_text)
    # with open("stars_db.json", "w", encoding="utf-8") as f:
    #     json.dump(stars_data, f, ensure_ascii=False, indent=2)
    # print(f"Đã trích xuất thành công {len(stars_data)} vì sao vào stars_db.json")

    # # 2. Trích xuất luận giải sao tại Mệnh/Thân
    # interpretations_data = parse_star_interpretations(raw_text)
    # with open("interpretations_db.json", "w", encoding="utf-8") as f:
    #     json.dump(interpretations_data, f, ensure_ascii=False, indent=2)
    # print(f"Đã trích xuất thành công {len(interpretations_data)} khối luận giải vào interpretations_db.json")


if __name__ == "__main__":
    main()