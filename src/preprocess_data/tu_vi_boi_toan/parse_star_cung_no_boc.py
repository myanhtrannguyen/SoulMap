import re
import json

def create_id(text):
    """Chuẩn hóa text thành ID: chữ thường, gạch dưới, bỏ dấu câu."""
    if not text: return "unknown"
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text).strip()
    text = re.sub(r'\s+', '_', text)
    return text

def get_star_list(text):
    """Chuyển chuỗi các sao cách nhau bằng dấu phẩy thành List ID."""
    # Tách theo dấu phẩy hoặc chữ "và"
    raw_stars = re.split(r'[,]|và', text)
    star_list = []
    for s in raw_stars:
        s_clean = s.strip()
        if s_clean:
            star_list.append(create_id(s_clean))
    return star_list

def parse_no_boc_91_v2(raw_text):
    # Khoanh vùng mục 9.1
    section_91_match = re.search(r'9\.1\. Kết hợp nhận định(.*?)(?=9\.2\. Phụ đoán|\Z)', raw_text, re.DOTALL)
    if not section_91_match:
        return {"error": "Không tìm thấy mục 9.1"}
    
    content = section_91_match.group(1).strip()
    # Chia theo các khối "Mệnh - Nô Bộc"
    blocks = re.split(r'Mệnh\s+Nô\s+Bộc', content)
    
    results = []
    for block in blocks:
        if not block.strip(): continue
        
        # 1. Bóc tách thông tin Mệnh
        header_match = re.match(r'(.*?)(sáng sủa\ntốt đẹp|mờ ám\nxấu xa)', block, re.DOTALL | re.IGNORECASE)
        if not header_match: continue
        
        raw_menh_stars = header_match.group(1).strip()
        menh_status = header_match.group(2).strip()

        # 2. Tìm các gạch đầu dòng (Nhóm sao tại Nô Bộc)
        star_groups = re.finditer(r'-\s+([^\n+]+)\n(.*?)(?=-\s+|\Z)', block, re.DOTALL)
        
        for sg in star_groups:
            no_boc_star_raw = sg.group(1).strip()
            sub_content = sg.group(2).strip()
            
            # Cấu trúc schema theo yêu cầu
            star_obj = {
                "star_id": get_star_list(no_boc_star_raw), # Có thể có nhiều sao tại Nô Bộc, nên để dạng list
                "sao_toa_thu_tai_cung_menh": get_star_list(raw_menh_stars), # Thêm trường list các sao Mệnh
                "context_type": f"menh_status_{create_id(menh_status)}",
                "rules": {}
            }
            
            # 3. Tìm các dấu "+" (Trạng thái của Nô Bộc)
            rules_matches = re.finditer(r'\+\s+(sáng sủa tốt đẹp|mờ ám xấu xa|Sáng sủa tốt đẹp|Mờ ám xấu xa):\s*(.*?)(?=\+\s+|\Z)', sub_content, re.DOTALL)
            
            for rm in rules_matches:
                status_key = create_id(rm.group(1))
                star_obj["rules"][status_key] = rm.group(2).strip().replace('\n', ' ')
            
            results.append(star_obj)

    return results

def parse_no_boc_section(raw_text):
    # 1. Khoanh vùng chương 9. CUNG NÔ BỘC
    section_pattern = r'9\.\s+CUNG\s+NÔ\s+BỘC.*?\n(.*?)(?=10\.\s+CUNG THIÊN DI|\Z)'
    section_match = re.search(section_pattern, raw_text, re.DOTALL | re.IGNORECASE)
    
    if not section_match:
        return {"error": "Không tìm thấy chương 9. CUNG NÔ BỘC"}
        
    section_text = section_match.group(1).strip()

    # 2. Lấy Description (đoạn đầu cho đến trước 9.1)
    desc_match = re.search(r'^(.*?)(?=9\.1\.)', section_text, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else ""

    # 3. Xử lý mục 9.1: Kết hợp nhận định (Matrix Mệnh - Nô Bộc)
    # Phần này ta sẽ đưa vào một cấu trúc riêng là "combinations"
    combinations = []
    # Tách các khối Mệnh/Nô Bộc (dựa trên các cụm từ khóa chính)
    # Ghi chú: Do dữ liệu text bảng phức tạp, ta sẽ bóc tách các nhóm chính
    
    # 4. Xử lý mục 9.2: Phụ đoán (Các sao lẻ)
    stars_list = []
    phu_doan_match = re.search(r'9\.2\. Phụ đoán(.*?)$', section_text, re.DOTALL)
    if phu_doan_match:
        phu_doan_text = phu_doan_match.group(1).strip()
        # Tìm các khối sao 9.2.x
        star_blocks = re.finditer(r'9\.2\.\d+\.\s+([^\n-]+)\n(.*?)(?=9\.2\.\d+\.|\Z)', phu_doan_text, re.DOTALL)
        
        for block in star_blocks:
            star_name = block.group(1).strip()
            content = block.group(2).strip()
            
            star_obj = {
                "star_id": create_id(star_name),
                "context_type": "phu_doan",
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
                    star_obj["rules"][create_id(key_raw)] = val.strip()
                else:
                    common_text.append(clean_part)
            
            if common_text:
                rules_with_common = {"rule_text": " ".join(common_text).strip()}
                rules_with_common.update(star_obj["rules"])
                star_obj["rules"] = rules_with_common
                
            stars_list.append(star_obj)

    return {
        "Cung_no_boc": {
            "name": create_id(section_match.group(0).split('\n')[0].strip()), # Lấy dòng tiêu đề để xác định tên cung
            "description": description,
            "stars": parse_no_boc_91_v2(raw_text) + stars_list, # Kết hợp cả phần 9.1 vào danh sách sao, nhưng có context_type riêng để phân biệt
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

    parsed_data = parse_no_boc_section(full_text)
    
    with open("data/data_process/tu_vi_boi_toan/cung/cung_no_boc_db.json", "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)
        
    print(f"Đã xuất thành công dữ liệu Cung NÔ BỘC ra file cung_no_boc_db.json!")

if __name__ == "__main__":
    main()
# Để demo, tôi sẽ giả định kết quả JSON dựa trên data bạn gửi