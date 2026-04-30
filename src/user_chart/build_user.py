from datetime import datetime
from lunarcalendar import Converter, Solar

CAN = ["Giáp", "Ất", "Bính", "Đinh", "Mậu", "Kỷ", "Canh", "Tân", "Nhâm", "Quý"]
CHI = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]
DUONG_CAN = ["Giáp", "Bính", "Mậu", "Canh", "Nhâm"]


def get_can_chi_year(year):
    can = CAN[(year + 6) % 10]
    chi = CHI[(year + 8) % 12]
    return f"{can} {chi}"


def get_lunar_hour(hour):
    index = ((hour + 1) // 2) % 12
    return CHI[index]


def solar_to_lunar(dt):
    solar = Solar(dt.year, dt.month, dt.day)
    lunar = Converter.Solar2Lunar(solar)

    lunar_year_can_chi = get_can_chi_year(lunar.year)

    return {
        "dob_lunar": {
            "year": lunar_year_can_chi,
            "month": lunar.month,
            "day": lunar.day,
            "hour": get_lunar_hour(dt.hour)
        }
    }


def get_am_duong_gender(can, gender):
    if can in DUONG_CAN:
        am_duong = "Dương"
    else:
        am_duong = "Âm"
    
    return f"{am_duong} {gender}"


def build_user(full_name, gender, dob_solar_str):
    dt = datetime.fromisoformat(dob_solar_str)

    lunar_data = solar_to_lunar(dt)["dob_lunar"]

    can = lunar_data["year"].split()[0]

    am_duong_gender = get_am_duong_gender(can, gender)

    return {
        "user_id": "u_123",     # placeholder
        "full_name": full_name,
        "gender": gender,
        "dob_solar": dob_solar_str,
        "dob_lunar": lunar_data,
        "am_duong_gender": am_duong_gender
    }