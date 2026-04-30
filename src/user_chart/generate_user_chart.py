import json
from build_user import build_user
from build_tuvi_chart import build_tuvi_chart
from build_houses_chart import build_houses_chart

def generate_user_chart(full_name, gender, dob_solar_str):
    user = build_user(full_name, gender, dob_solar_str)
    tuvi_chart = build_tuvi_chart(user)
    houses_chart = build_houses_chart(user, tuvi_chart)

    return user, tuvi_chart, houses_chart

if __name__ == "__main__":
    user, tuvi_chart, houses_chart = generate_user_chart(
        full_name="Nguyễn Thu Huyền",
        gender="Nữ",
        dob_solar_str="2005-12-15T05:20:00"
    )

    with open("data/data_user/user_chart.json", "w", encoding="utf-8") as f:
        json.dump(user, f, ensure_ascii=False, indent=2)

    with open("data/data_user/tuvi_chart.json", "w", encoding="utf-8") as f:
        json.dump(tuvi_chart, f, ensure_ascii=False, indent=2)

    with open("data/data_user/houses_chart.json", "w", encoding="utf-8") as f:
        json.dump(houses_chart, f, ensure_ascii=False, indent=2)