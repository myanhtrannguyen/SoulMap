import re
import json

def create_id(text):
    """Chuẩn hóa text thành dạng ID: chữ thường, gạch dưới, không dấu câu."""
    text = text.lower().strip()
    text = re.sub(r'[\(\)\.\,\-]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', '_', text.strip())
    return text

def parse_quan_loc_section(full_text):

    text = re.sub(r'\\s*', '', full_text)

    # 1. Khoanh vùng phần 8. CUNG QUAN LỘC
    # Dùng Regex linh hoạt để bắt tiêu đề chương
    section_pattern = r'8\.\s+CUNG QUAN LỘC.*?\n(.*?)(?=9\.\s+CUNG NÔ BỘC|\Z)'
    section_match = re.search(section_pattern, text, re.DOTALL | re.IGNORECASE)

    if not section_match:
        return {"error": "Không tìm thấy chương 8. CUNG QUAN LỘC"}
    
    cung_name = section_match.group(0).split('\n')[0].strip()  # Lấy dòng tiêu đề để xác định tên cung
    cung_name = create_id(cung_name)  # Chuẩn hóa tên cung thành ID
    section_text = section_match.group(1).strip()

    # 2. Lấy mô tả đầu chương (Description)
    # Thường là đoạn văn đầu tiên trước khi vào danh sách các sao
    desc_match = re.search(r'^(.*?)(?=8\.1\.|8\.2\.|[A-ZĐ]{2,})', section_text, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else ""
    description = re.sub(r'\s+', ' ', description)  # Chuẩn hóa khoảng trắng

    # 3. Tìm các khối sao (Star Blocks) 
    # Cấu trúc: 8.x. Tên Sao
    star_blocks = re.finditer(r'8\.\d+\.\s+([^\n]+)\n(.*?)(?=8\.\d+\.|\Z)', section_text, re.DOTALL)

    stars_list = []
    
    for block in star_blocks:

        star_name = block.group(1).strip()
        content = block.group(2).strip()
        
        star_id = create_id(star_name)
        
        star_obj = {
            "star_id": star_id,
            "context_type": None, # Có thể map thêm dựa vào vị trí nếu cần
            "rules": {}
        }
        
        common_text = []
        # Tách rules theo dấu gạch ngang hoặc xuống dòng
        # Lưu ý: Ưu tiên dấu gạch ngang vì dữ liệu của bạn thường dùng nó để phân tách ý
        rule_parts = re.split(r'\s*-\s*|\n(?=-)', content)
        
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
        "Cung_quan_loc": {
            "name": cung_name,
            "description": description,
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

    parsed_data = parse_quan_loc_section(full_text)
    
    with open("data/data_process/tu_vi_boi_toan/cung/cung_quan_loc_db.json", "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)
        
    print(f"Đã xuất thành công dữ liệu Cung Quan Lộc ra file cung_quan_loc_db.json!")

if __name__ == "__main__":
    main()