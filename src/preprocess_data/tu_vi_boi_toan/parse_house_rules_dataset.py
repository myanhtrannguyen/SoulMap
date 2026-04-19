import re
import json

# Tập hợp 12 Địa chi để so khớp vị trí (Location)
ZODIAC_SIGNS = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]

# Bộ từ điển ánh xạ tên gọi tắt của các sao trong sách ra tên đầy đủ
STAR_MAPPING = {
    "Phủ": "Thiên Phủ", "Tướng": "Thiên Tướng", "Sát": "Thất Sát", 
    "Phá": "Phá Quân", "Tham": "Tham Lang", "Đồng": "Thiên Đồng", 
    "Nhật": "Thái Dương", "Nguyệt": "Thái Âm", "Cơ": "Thiên Cơ", 
    "Cự": "Cự Môn", "Lương": "Thiên Lương", "Vũ": "Vũ Khúc", 
    "Liêm": "Liêm Trinh", "Kình": "Kình Dương", "Đà": "Đà La",
    "Hỏa": "Hỏa Tinh", "Linh": "Linh Tinh", "Không": "Địa Không", 
    "Kiếp": "Địa Kiếp", "Khoa": "Hóa Khoa", "Quyền": "Hóa Quyền", 
    "Lộc": "Hóa Lộc", "Kỵ": "Hóa Kỵ"
}

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_condition_info(condition_str):
    """
    Bóc tách Location và Companion Stars từ vế điều kiện
    Ví dụ: "Đơn thủ tại Ngọ, Phủ đồng cung" -> loc: ["Ngọ"], stars: ["Thiên Phủ"]
    """
    location = []
    companion_stars = []
    
    # 1. Tìm kiếm địa chi (Vị trí)
    for sign in ZODIAC_SIGNS:
        # Dùng \b để tìm đúng từ, tránh nhầm "Mùi" trong từ khác
        if re.search(rf'\b{sign}\b', condition_str, re.IGNORECASE):
            location.append(sign)
            
    if not location:
        location = ["any"]
        
    # 2. Tìm kiếm tên sao đồng cung / hội hợp
    for short_name, full_name in STAR_MAPPING.items():
        if re.search(rf'\b{short_name}\b', condition_str, re.IGNORECASE) or \
           re.search(rf'\b{full_name}\b', condition_str, re.IGNORECASE):
            companion_stars.append(full_name)
            
    return location, companion_stars

def parse_house_interpretations(text):
    database = []
    
    # Tìm kiếm tiêu đề các Cung (Bắt đầu từ phần 5 đến 15 trong sách)
    # Pattern: Xuống dòng + Số + Chấm + Tên Cung (Ví dụ: "14. THÊ THIẾP (PHU QUÂN)")
    house_pattern = r'\n(\d{1,2})\.\s+([A-ZĐ\s\(\)]+)\n'
    house_splits = list(re.finditer(house_pattern, text))
    
    for i in range(len(house_splits)):
        house_match = house_splits[i]
        house_name_raw = house_match.group(2).strip()
        house_id_str = clean_text(house_name_raw).lower().replace(" ", "_").replace("(", "").replace(")", "")
        
        # Giới hạn nội dung của Cung hiện tại
        start_idx = house_match.end()
        end_idx = house_splits[i+1].start() if i + 1 < len(house_splits) else len(text)
        house_text = text[start_idx:end_idx]
        
        # Tìm các sao chính trong Cung (Ví dụ: "14.1. Tử Vi")
        star_pattern = r'\n\d{1,2}\.\d{1,2}\.\s+([A-ZĐ][\w\s]+)'
        star_splits = list(re.finditer(star_pattern, house_text))
        
        for j in range(len(star_splits)):
            star_match = star_splits[j]
            star_name = clean_text(star_match.group(1))
            primary_star_id = star_name.lower().replace(" ", "_")
            
            # Cắt đoạn text thuộc về sao này
            s_start = star_match.end()
            s_end = star_splits[j+1].start() if j + 1 < len(star_splits) else len(house_text)
            star_text = house_text[s_start:s_end]
            
            rules = []
            
            # Cắt các gạch đầu dòng (quy tắc luận giải)
            # Dùng regex lookahead (?=[\-\+]) để chẻ văn bản mỗi khi gặp dấu - hoặc + ở đầu dòng
            lines = re.split(r'\n(?=[\-\+])', star_text)
            
            for line in lines:
                line = line.strip()
                if not line.startswith("-") and not line.startswith("+"):
                    continue
                
                # Bỏ dấu "-" hoặc "+"
                line_content = line[1:].strip()
                
                # Cắt bởi dấu hai chấm ":" để tách Điều kiện và Luận giải
                if ":" in line_content:
                    parts = line_content.split(":", 1)
                    condition_str = parts[0].strip()
                    interpretation = clean_text(parts[1])
                    
                    locs, companions = extract_condition_info(condition_str)
                    
                    rules.append({
                        "location": locs,
                        "companion_stars": companions,
                        "interpretation": interpretation
                    })
                else:
                    # Nếu câu không có dấu ":" (áp dụng chung không kèm điều kiện vị trí)
                    rules.append({
                        "location": ["any"],
                        "companion_stars": [],
                        "interpretation": clean_text(line_content)
                    })
                    
            if rules:
                database.append({
                    "house_rule_id": f"{house_id_str}_{primary_star_id}",
                    "house_name": clean_text(house_name_raw),
                    "primary_star_id": primary_star_id,
                    "rules": rules
                })
                
    return database

def main():
    try:
        with open(r"D:\Hust\Năm ba\NLP\prj\data_process\data_raw_text.txt", "r", encoding="utf-8") as f:
            full_text = f.read()
    except FileNotFoundError:
        print("Vui lòng lưu nội dung vào file data_raw_text.txt.")
        return

    house_rules_data = parse_house_interpretations(full_text)
    
    with open("data_process/house_rules_db.json", "w", encoding="utf-8") as f:
        json.dump(house_rules_data, f, ensure_ascii=False, indent=2)
        
    print(f"Đã xuất thành công {len(house_rules_data)} bộ quy tắc theo Cung ra file house_rules_db.json!")

if __name__ == "__main__":
    main()