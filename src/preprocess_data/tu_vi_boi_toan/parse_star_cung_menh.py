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
            },
            "all_content": (block) # Lưu nguyên văn nội dung để tham khảo sau này
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


def parse_star_interpretations_cung_menh(text):
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
        current_rules = {}
        rule_text = ""
        cau_phu = ""
        for line in lines[1:]:
            # line = line.strip()
            if not line or re.match(r'^\s*-\s*\d+\s*-\s*$', line.lower()) :
                continue
            # if star_id == "hỏa,_linh":
            #     print(line)
            if "☯" in line and current_rules:
                # if star_id == "hỏa,_linh":
                #     print("star_id:", line)
                #     print(current_rules)
                if "rule_text" not in current_rules:
                    current_rules["rule_text"] = []
                        
                current_rules["rule_text"].append(rule_text)
    
                interpretations_db.append({
                    "star_id": star_id,
                    "context_type": current_context,
                    "rules": current_rules
                })
                current_context = ""
                current_rules = {}
                rule_text = ""
                cau_phu = ""

            # Nhận diện Context [cite: 782, 792, 798, 801]
            if "☯" in line:
                current_context = "_".join(clean_text(line.strip()).lower().split(" ")[1:])
            elif current_context == "phú_giải" :
                
                # Lưu câu phú nguyên bản 
                
                if line[0].lower() != line[0]:
                    # print(line)
                    
                    if cau_phu:
                        if "Câu_Phú" not in current_rules:
                            current_rules["Câu_Phú"] = []

                        current_rules["Câu_Phú"].append(cau_phu)
                        
                    cau_phu = clean_text(line)
                else:
                    cau_phu += " " + clean_text(line)
                
            # Nhận diện các dòng luận giải bắt đầu bằng dấu '+' hoặc '-' [cite: 782, 783]
            elif line.startswith("+") or line.startswith("-"):
                
                if rule_text:
                    if "rule_text" not in current_rules:
                        current_rules["rule_text"] = []
                        
                    current_rules["rule_text"].append(rule_text)
                    
                rule_text = clean_text(line[1:])
            else:
                rule_text += " " + clean_text(line)
        
        if cau_phu:
            if "Câu_Phú" not in current_rules:
                current_rules["Câu_Phú"] = []

            current_rules["Câu_Phú"].append(cau_phu)
            
        if current_rules:
            interpretations_db.append({
                "star_id": star_id,
                "context_type": current_context,
                "rules": current_rules
            })
            current_rules = {}
  
    return {
        "Cung_menh": {
            "name": "cung_mệnh_và_cung_thân",
            "description": "Luận giải chi tiết về các sao tại cung Mệnh và Thân, bao gồm các bối cảnh như Đại cương, Nam mệnh, Nữ mệnh, v.v... Các luận giải được phân loại theo từng sao và từng bối cảnh cụ thể.",
            "stars": interpretations_db,
            "source_content": re.search(r'4\.\s+NHẬN\s+ĐỊNH\s+KHÁI\s+QUÁT\s+VỀ\s+CUNG\s+MỆNH\s+VÀ\s+CUNG\s+THÂN(.*?)(?=5\.\s+CUNG\s+PHỤ\s+MẪU|\Z)', text, re.DOTALL).group(1).strip()  # Lưu nguyên văn nội dung để tham khảo sau này
        }
    }

def main():
    # Đọc file dữ liệu đã extract
    # try:
    #     with open("tuvi_data.txt", "r", encoding="utf-8") as f:
    #         full_text = f.read()
    # except FileNotFoundError:
    #     print("Vui lòng lưu nội dung vào file tuvi_data.txt trước khi chạy.")
    #     return
    try:
        with open(r"D:\Hust\Năm ba\NLP\prj\data\data_process\tu_vi_boi_toan\tu_vi_boi_toan_raw_text.txt", "r", encoding="utf-8") as f:
            full_text = f.read()
    except FileNotFoundError:
        print("Vui lòng lưu nội dung vào file data_raw_text.txt.")
        return
    
    # 1. Trích xuất danh sách sao
    stars_data = parse_star_dictionary_advanced(full_text)
    with open("data/data_process/tu_vi_boi_toan/stars_db.json", "w", encoding="utf-8") as f:
        json.dump(stars_data, f, ensure_ascii=False, indent=2)
    print(f"Đã trích xuất thành công {len(stars_data)} vì sao vào stars_db.json")

    # 2. Trích xuất luận giải sao tại Mệnh/Thân
    cung = re.split(r'\d+\.\s+', full_text)
    print(cung[0])
    # print(type(full_text))
    interpretations_data_cung_menh = parse_star_interpretations_cung_menh(full_text[full_text.find("4. NHẬN ĐỊNH KHÁI QUÁT VỀ CUNG MỆNH VÀ CUNG THÂN"): full_text.rfind("5. CUNG PHỤ MẪU")])
    with open("data/data_process/tu_vi_boi_toan/cung/cung_menh_va_cung_than_db.json", "w", encoding="utf-8") as f:
        json.dump(interpretations_data_cung_menh, f, ensure_ascii=False, indent=2)
    print(f"Đã trích xuất thành công {len(interpretations_data_cung_menh)} khối luận giải vào interpretations_db.json")


if __name__ == "__main__":
    main()