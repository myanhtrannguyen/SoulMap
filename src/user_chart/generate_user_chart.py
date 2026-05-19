import json
import os

import re
from build_user import build_user
from build_tuvi_chart import build_tuvi_chart
from build_houses_chart import build_houses_chart

def generate_user_chart(full_name, gender, dob_solar_str):
    user = build_user(full_name, gender, dob_solar_str)
    tuvi_chart = build_tuvi_chart(user)
    houses_chart = build_houses_chart(user, tuvi_chart)

    return user, tuvi_chart, houses_chart

def create_id(text):
    """Chuẩn hóa text thành dạng key: chữ thường, thay khoảng trắng bằng gạch dưới, bỏ dấu câu."""
    text = text.lower().strip()
    text = re.sub(r'[\(\)\.\,\-]', '', text)
    text = re.sub(r'\d+', '', text).strip()
    text = re.sub(r'\s+', '_', text)
    return text

if __name__ == "__main__":
    user, tuvi_chart, houses_chart = generate_user_chart(
        full_name="Trần Lê Hạ Đan",
        gender="Nữ",
        dob_solar_str="2005-07-21T10:20:00"
    )
    output_dir = f"data/data_user/{create_id(user['full_name'])}"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "user_chart.json"), "w", encoding="utf-8") as f:
        json.dump(user, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "tuvi_chart.json"), "w", encoding="utf-8") as f:
        json.dump(tuvi_chart, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "houses_chart.json"), "w", encoding="utf-8") as f:
        json.dump(houses_chart, f, ensure_ascii=False, indent=2)