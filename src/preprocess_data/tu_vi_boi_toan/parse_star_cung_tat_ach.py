import re
import json

def create_id(text):
    """Chuẩn hóa text thành ID: chữ thường, gạch dưới, bỏ dấu câu."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', '_', text.strip())
    return text

def parse_tat_ach_section(raw_text):
    # 1. Khoanh vùng chương 11. CUNG TẬT ÁCH
    section_pattern = r'11\.\s+CUNG\s+TẬT\s+ÁCH.*?\n(.*?)(?=12\.\s+CUNG TÀI BẠCH|\Z)'
    section_match = re.search(section_pattern, raw_text, re.DOTALL | re.IGNORECASE)
    
    if not section_match:
        return {"error": "Không tìm thấy chương 11. CUNG TẬT ÁCH"}
    
    cung_name = section_match.group(0).split('\n')[0].strip()  # Lấy dòng tiêu đề để xác định tên cung
    cung_name = create_id(cung_name)  # Chuẩn hóa tên cung thành ID
    section_text = section_match.group(1).strip()

    # 2. Lấy Description (đoạn đầu cho đến trước 11.1)
    desc_match = re.search(r'^(.*?)(?=11\.1\.)', section_text, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else ""

    # 3. Tìm các tiểu mục chính (11.1 và 11.2) để xác định context_type
    # Tách văn bản thành 2 phần: Cứu giải và Tác họa
    sub_sections = re.split(r'(11\.[12]\.\s+.*?)\n', section_text)
    
    stars_list = []
    current_context = None

    for i in range(1, len(sub_sections), 2):
        header = sub_sections[i]
        content = sub_sections[i+1]
        
        # Xác định context_type dựa trên tiêu đề tiểu mục
        if "cứu giải" in header.lower():
            current_context = "sao_cuu_giai"
        elif "tác họa" in header.lower():
            current_context = "sao_tac_hoa"

        # Tìm các khối sao trong tiểu mục này (VD: 11.1.1. Tử Vi)
        star_blocks = re.finditer(r'11\.[123]\.\d+\.\s+([^\n-]+)(?:\s*-\s*|\n)(.*?)(?=11\.[123]\.\d+\.|\Z)', content, re.DOTALL)
        
        for block in star_blocks:
            star_name = block.group(1).strip()
            raw_content = block.group(2).strip()
            
            star_obj = {
                "star_id": create_id(star_name),
                "context_type": current_context,
                "rules": {}
            }
            
            common_text = []
            # Tách rules theo dấu gạch ngang hoặc xuống dòng
            # Lưu ý: Ưu tiên dấu gạch ngang vì dữ liệu của bạn thường dùng nó để phân tách ý
            rule_parts = re.split(r'\s*-\s*|\n(?=-)', raw_content)
            
            for part in rule_parts:
                part = part.strip()
                if not part or "Vân Đằng Thái Thứ Lang" in part: continue
                
                # Loại bỏ gạch đầu dòng nếu có
                clean_part = re.sub(r'^[-+]\s*', '', part)
                
                if ':' in clean_part:
                    key_raw, val = clean_part.split(':', 1)
                    star_obj["rules"][create_id(key_raw)] = val.strip()
                else:
                    common_text.append(clean_part)
            
            if common_text:
                rules_with_common = {"rule_text": " ".join(common_text).strip()}
                rules_with_common.update(star_obj["rules"])
                star_obj["rules"] = rules_with_common
                
            stars_list.append(star_obj)

    return {
        "Cung_tat_ach": {
            "name": cung_name,
            "description": description.replace('\n', ' '),
            "stars": stars_list,
            "source_content": section_text  # Lưu nguyên văn nội dung để tham khảo sau này
        }
    }

def main():

    try:
        with open(r"D:\Hust\Năm ba\NLP\prj\data\data_process\tu_vi_boi_toan\tu_vi_boi_toan_raw_text.txt", "r", encoding="utf-8") as f:
            full_text = f.read()
    except FileNotFoundError:
        print("Vui lòng lưu nội dung vào file tu_vi_boi_toan_raw_text.txt.")
        return

    parsed_data = parse_tat_ach_section(full_text)
    
    with open("data/data_process/tu_vi_boi_toan/cung/cung_tat_ach_db.json", "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)
        
    print(f"Đã xuất thành công dữ liệu Cung TậT ÁCH ra file cung_tat_ach_db.json!")

if __name__ == "__main__":
    main()
# Thực thi với dữ liệu text bạn cung cấp
# data = parse_tat_ach_section(raw_input_text)