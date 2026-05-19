import re
import json

def create_id(text):
    """Chuẩn hóa text thành dạng key: chữ thường, thay khoảng trắng bằng gạch dưới, bỏ dấu câu."""
    text = text.lower().strip()
    text = re.sub(r'[\(\)\.\,\-]', '', text)
    text = re.sub(r'\d+', '', text).strip()
    text = re.sub(r'\s+', '_', text)
    return text

def parse_tu_tuc_section(full_text):
    
    # 1. Khoanh vùng chương 13. CUNG TỬ TỨC
    section_pattern = r'13\.\s+CUNG\s+TỬ\s+TỨC.*?\n(.*?)(?=13\.1\.\s+|\Z)'
    # 2. Lấy Description (đoạn đầu cho đến trước 13.1)
    section_match = re.search(section_pattern, full_text, re.DOTALL | re.IGNORECASE)
    description = section_match.group(1).strip() if section_match else ""

    if not section_match:
        return {"error": "Không tìm thấy chương 13. CUNG TỬ TỨC"}
    
    cung_name = section_match.group(0).split('\n')[0].strip()  # Lấy dòng tiêu đề để xác định tên cung
    cung_name = create_id(cung_name)  # Chuẩn hóa tên cung thành ID
    
    # Giới hạn vùng tìm kiếm: từ mục "13.2. Nhận định ảnh hưởng các sao" đến mục "14. THÊ THIẾP"
    section_match = re.search(r'13\.2\.\s+Nhận định ảnh hưởng các sao(.*?)(?=14\.\s+CUNG THÊ THIẾP \(PHU QUÂN\)|\Z)', full_text, re.DOTALL)
    if not section_match:
        print("Không tìm thấy phần 13.2. Cung Tử Tức.")
        return {"Cung_tu_tuc": []}
        
    section_text = section_match.group(1)
    
    # Tìm các block sao (VD: "13.2.1. Tử Vi")
    star_blocks = re.finditer(r'13\.[23]\.(\d+)\.\s+([^\n]+)\n(.*?)(?=13\.[23]\.\d+\.|\Z)', section_text, re.DOTALL)
    
    stars_list = []
    
    for block in star_blocks:

        star_name = block.group(2).strip()
        content = block.group(3).strip()
        
        star_id = create_id(star_name)
        
        star_obj = {
            "star_id": star_id,
            "context_type": None, # Có thể map thêm dựa vào vị trí nếu cần
            "rules": {}
        }
        
        common_text = []
        # Tách rules theo dấu gạch ngang hoặc xuống dòng
        # Lưu ý: Ưu tiên dấu gạch ngang vì dữ liệu của bạn thường dùng nó để phân tách ý
        rule_parts = re.split(r'\s*[-+]\s*|\n(?=[-+])', content)
        
        for part in rule_parts:
            part = part.strip()
            if not part or "Vân Đằng Thái Thứ Lang" in part: continue
            
            # Loại bỏ gạch đầu dòng nếu có
            clean_part = re.sub(r'^[-+]\s*', '', part)
            
            if ':' in clean_part:
                key_raw, val = clean_part.split(':', 1)
                if val.strip():  # Chỉ thêm rule nếu có giá trị sau dấu ":"
                    star_obj["rules"][create_id(key_raw)] = val.strip()
                else:
                    common_text.append(clean_part)  # Nếu không có giá trị, coi như là text chung   
            else:
                common_text.append(clean_part)
        
        if common_text:
            rules_with_common = {"rule_text": " ".join(common_text).strip()}
            rules_with_common.update(star_obj["rules"])
            star_obj["rules"] = rules_with_common
            
        stars_list.append(star_obj)
        
    return {
        "Cung_tu_tuc": {
            "name": cung_name,
            "description": description,
            "stars": stars_list,
            "source_content": re.search(r'13\.\s+CUNG\s+TỬ\s+TỨC(.*?)(?=14\.\s+CUNG THÊ THIẾP \(PHU QUÂN\)|\Z)', full_text, re.DOTALL).group(1).strip()  # Lưu nguyên văn nội dung để tham khảo sau này
        }
    }

def main():

    try:
        with open(r"D:\Hust\Năm ba\NLP\prj\data\data_process\tu_vi_boi_toan\tu_vi_boi_toan_raw_text.txt", "r", encoding="utf-8") as f:
            full_text = f.read()
    except FileNotFoundError:
        print("Vui lòng lưu nội dung vào file tu_vi_boi_toan_raw_text.txt.")
        return

    parsed_data = parse_tu_tuc_section(full_text)
    
    with open("data/data_process/tu_vi_boi_toan/cung/cung_tu_tuc_db.json", "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)
        
    print(f"Đã xuất thành công dữ liệu Cung Tử Tức ra file cung_tu_tuc_db.json!")

if __name__ == "__main__":
    main()