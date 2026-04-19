import re
import json
import unicodedata

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def create_id(text):
    """Chuyển đổi chuỗi tiếng Việt có dấu thành ID không dấu, phân cách bằng gạch dưới"""
    text = clean_text(text).lower()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.replace(' ', '_')

def split_condition_interpretation(text, category_desc):
    """
    Hàm heuristic để tách Điều kiện và Lời luận giải từ một đoạn văn.
    """
    text = clean_text(text)
    
    # Các từ khóa thường dùng để chuyển ý sang kết quả/luận giải
    split_keywords = [" nên ", " tất ", " chắc chắn là ", " là người "]
    
    for kw in split_keywords:
        if kw in text:
            parts = text.split(kw, 1)
            condition = parts[0].strip()
            interpretation = kw.strip().capitalize() + " " + parts[1].strip()
            return condition, interpretation

    # Nếu không có từ khóa phân tách rõ ràng và câu ngắn, 
    # ta đưa toàn bộ vào điều kiện và gán interpretation theo ý nghĩa của nhóm Cách cục.
    return text, f"Thuộc {category_desc}."

def parse_cach_cuc(text):
    database = []
    
    # 1. Cắt riêng phần "19. PHÂN CỤC" đến phần "20. NHẬN XÉT..."
    start_match = re.search(r'19\.\s+PHÂN CỤC', text)
    end_match = re.search(r'20\.\s+NHẬN XÉT SỐ MỆNH', text)
    
    if not start_match or not end_match:
        print("Không tìm thấy giới hạn phần Phân Cục.")
        return []
        
    phan_cuc_text = text[start_match.end():end_match.start()]
    
    # 2. Tìm các Nhóm Cách cục (19.1. Phú cục, 19.2. Qúy cục...)
    # Pattern: 19.\d+. Tên nhóm (Chú thích nhóm)
    category_pattern = r'19\.\d+\.\s+([A-ZĐ][\w\s]+?)\s*(?:\((.*?)\))?\n'
    cat_splits = list(re.finditer(category_pattern, phan_cuc_text))
    
    for i in range(len(cat_splits)):
        cat_match = cat_splits[i]
        category_name = clean_text(cat_match.group(1))
        category_meaning = clean_text(cat_match.group(2)) if cat_match.group(2) else category_name
        
        # Trích xuất nội dung của nhóm này
        c_start = cat_match.end()
        c_end = cat_splits[i+1].start() if i + 1 < len(cat_splits) else len(phan_cuc_text)
        category_text = phan_cuc_text[c_start:c_end]
        
        # 3. Tìm các Cách cục chi tiết bên trong nhóm (Ví dụ: 19.1.1. Tài Ấm Giáp Ấn)
        item_pattern = r'19\.\d+\.\d+\.\s+([A-ZĐ][\w\s,]+)\n'
        item_splits = list(re.finditer(item_pattern, category_text))
        
        for j in range(len(item_splits)):
            item_match = item_splits[j]
            cach_cuc_name = clean_text(item_match.group(1))
            cach_cuc_id = create_id(cach_cuc_name)
            
            # Trích xuất đoạn văn mô tả Cách cục
            i_start = item_match.end()
            i_end = item_splits[j+1].start() if j + 1 < len(item_splits) else len(category_text)
            description_text = category_text[i_start:i_end].strip()
            
            # Bỏ qua các mục tham chiếu chéo (Ví dụ: "Coi Qúy cục 19.2.13. dưới đây.")
            if "Coi Qúy cục" in description_text or "Coi Phú cục" in description_text:
                continue
                
            # Tách Điều kiện và Giải đoán
            condition, interpretation = split_condition_interpretation(description_text, f"nhóm {category_name} ({category_meaning})")
            
            database.append({
                "cach_cuc_id": cach_cuc_id,
                "name": cach_cuc_name,
                "category": category_name,
                "conditions": condition,
                "interpretation": interpretation
            })
            
    # Thêm thủ công phần "17.4. Phi thường cách" vì nó nằm ngoài mục 19
    phi_thuong_cach_text = """
    - Mệnh: Tử, Phủ, Vũ, Tướng hội hợp, tất cả đều nhập miếu, vượng hay đắc địa.
    - Thân: Sát, Phá, Liêm, Tham hội hợp, tất cả cũng đều nhập miếu, vượng hay đắc địa.
    - Mệnh, Thân lại được thêm sự phù tá của các sao đắc địa là Tả, Hữu, Khôi, Việt, Xương, Khúc, Long, Phượng, Hồng Đào, Khoa, Quyền, Lộc và Kình, Đà, Không, Kiếp, Hình, Hổ.
    """
    database.append({
        "cach_cuc_id": "phi_thuong_cach",
        "name": "Phi Thường Cách",
        "category": "Phi thường cách",
        "conditions": clean_text(phi_thuong_cach_text),
        "interpretation": "Phú qúy đến tột bực, uy quyền hiến hách, có danh tiếng lưu lại ngàn thu."
    })
            
    return database

def main():
    try:
        with open(r"D:\Hust\Năm ba\NLP\prj\data_process\tu_vi_boi_toan\tu_vi_boi_toan_raw_text.txt", "r", encoding="utf-8") as f:
            full_text = f.read()
    except FileNotFoundError:
        print("Vui lòng lưu nội dung vào file data_raw_text.txt.")
        return

    cach_cuc_data = parse_cach_cuc(full_text)
    
    with open("data_process/tu_vi_boi_toan/cach_cuc_db.json", "w", encoding="utf-8") as f:
        json.dump(cach_cuc_data, f, ensure_ascii=False, indent=2)
        
    print(f"Đã xuất thành công {len(cach_cuc_data)} Cách Cục ra file cach_cuc_db.json!")

if __name__ == "__main__":
    main()