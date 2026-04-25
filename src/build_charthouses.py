ZODIAC_ORDER = [
    "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi",
    "Thân", "Dậu", "Tuất", "Hợi", "Tý", "Sửu"
]

HOUR_INDEX = {
    "Tý": 0, "Sửu": 1, "Dần": 2, "Mão": 3,
    "Thìn": 4, "Tỵ": 5, "Ngọ": 6, "Mùi": 7,
    "Thân": 8, "Dậu": 9, "Tuất": 10, "Hợi": 11
}

HOUSE_ORDER = [
    "Mệnh", "Phụ Mẫu", "Phúc Đức", "Điền Trạch",
    "Quan Lộc", "Nô Bộc", "Thiên Di", "Tật Ách",
    "Tài Bạch", "Tử Tức", "Phu Thê", "Huynh Đệ"
]

BRIGHTNESS_TABLE = {
    "Tử Vi": {
        "Tỵ": "miếu", "Ngọ": "miếu", "Dần": "miếu", "Thân": "miếu",
        "Thìn": "vượng", "Tuất": "vượng",
        "Sửu": "đắc", "Mùi": "đắc",
        "Hợi": "bình", "Tý": "bình", "Mão": "bình", "Dậu": "bình"
    },
    "Liêm Trinh": {
        "Thìn": "miếu", "Tuất": "miếu",
        "Tý": "vượng", "Ngọ": "vượng", "Dần": "vượng", "Thân": "vượng",
        "Sửu": "đắc", "Mùi": "đắc",
        "Tỵ": "hãm", "Hợi": "hãm", "Mão": "hãm", "Dậu": "hãm"
    },
    "Thiên Đồng": {
        "Dần": "miếu", "Thân": "miếu",
        "Tý": "vượng",
        "Mão": "đắc", "Tỵ": "đắc", "Hợi": "đắc",
        "Ngọ": "hãm", "Dậu": "hãm", "Thìn": "hãm",
        "Tuất": "hãm", "Sửu": "hãm", "Mùi": "hãm"
    },
    "Vũ Khúc": {
        "Thìn": "miếu", "Tuất": "miếu", "Sửu": "miếu", "Mùi": "miếu",
        "Dần": "vượng", "Thân": "vượng", "Tý": "vượng", "Ngọ": "vượng",
        "Mão": "đắc", "Dậu": "đắc",
        "Tỵ": "hãm", "Hợi": "hãm"
    },
    "Thái Dương": {
        "Tỵ": "miếu", "Ngọ": "miếu",
        "Dần": "vượng", "Mão": "vượng", "Thìn": "vượng",
        "Sửu": "đắc", "Mùi": "đắc",
        "Thân": "hãm", "Dậu": "hãm", "Tuất": "hãm", "Hợi": "hãm", "Tý": "hãm"
    },
    "Thiên Cơ": {
        "Thìn": "miếu", "Tuất": "miếu", "Mão": "miếu", "Dậu": "miếu",
        "Tỵ": "vượng", "Thân": "vượng",
        "Tý": "đắc", "Ngọ": "đắc", "Sửu": "đắc", "Mùi": "đắc",
        "Dần": "hãm", "Hợi": "hãm"
    },
    "Thiên Phủ": {
        "Dần": "miếu", "Thân": "miếu", "Tý": "miếu", "Ngọ": "miếu",
        "Thìn": "vượng", "Tuất": "vượng",
        "Tỵ": "đắc", "Hợi": "đắc", "Mùi": "đắc",
        "Mão": "bình", "Dậu": "bình", "Sửu": "bình"
    },
    "Thái Âm": {
        "Dậu": "miếu", "Tuất": "miếu", "Hợi": "miếu",
        "Thân": "vượng", "Tý": "vượng",
        "Sửu": "đắc", "Mùi": "đắc",
        "Dần": "hãm", "Mão": "hãm", "Thìn": "hãm", "Tỵ": "hãm", "Ngọ": "hãm"
    },
    "Tham Lang": {
        "Sửu": "miếu", "Mùi": "miếu",
        "Thìn": "vượng", "Tuất": "vượng",
        "Dần": "đắc", "Thân": "đắc",
        "Tỵ": "hãm", "Hợi": "hãm", "Tý": "hãm",
        "Ngọ": "hãm", "Mão": "hãm", "Dậu": "hãm"
    },
    "Cự Môn": {
        "Mão": "miếu", "Dậu": "miếu",
        "Tý": "vượng", "Ngọ": "vượng", "Dần": "vượng",
        "Thân": "đắc", "Hợi": "đắc",
        "Thìn": "hãm", "Tuất": "hãm", "Sửu": "hãm",
        "Mùi": "hãm", "Tỵ": "hãm"
    },
    "Thiên Tướng": {
        "Dần": "miếu", "Thân": "miếu",
        "Thìn": "vượng", "Tuất": "vượng", "Tý": "vượng", "Ngọ": "vượng",
        "Sửu": "đắc", "Mùi": "đắc", "Tỵ": "đắc", "Hợi": "đắc",
        "Mão": "hãm", "Dậu": "hãm"
    },
    "Thiên Lương": {
        "Ngọ": "miếu", "Thìn": "miếu", "Tuất": "miếu",
        "Tý": "vượng", "Mão": "vượng", "Dần": "vượng", "Thân": "vượng",
        "Sửu": "đắc", "Mùi": "đắc",
        "Dậu": "hãm", "Tỵ": "hãm", "Hợi": "hãm"
    },
    "Thất Sát": {
        "Dần": "miếu", "Thân": "miếu", "Tý": "miếu", "Ngọ": "miếu",
        "Tỵ": "vượng", "Hợi": "vượng",
        "Sửu": "đắc", "Mùi": "đắc",
        "Mão": "hãm", "Dậu": "hãm", "Thìn": "hãm", "Tuất": "hãm"
    },
    "Phá Quân": {
        "Tý": "miếu", "Ngọ": "miếu",
        "Sửu": "vượng", "Mùi": "vượng",
        "Thìn": "đắc", "Tuất": "đắc",
        "Mão": "hãm", "Dậu": "hãm", "Dần": "hãm",
        "Thân": "hãm", "Tỵ": "hãm", "Hợi": "hãm"
    }
}

LOC_TON_TABLE = {
    "Giáp": "Dần",
    "Ất": "Mão",
    "Bính": "Tỵ",
    "Đinh": "Ngọ",
    "Mậu": "Tỵ",
    "Kỷ": "Ngọ",
    "Canh": "Thân",
    "Tân": "Dậu",
    "Nhâm": "Hợi",
    "Quý": "Tý"
}

HOA_LINH_START = {
    ("Dần","Ngọ","Tuất"): ("Sửu", "Mão"),
    ("Thân","Tý","Thìn"): ("Dần", "Tuất"),
    ("Tỵ","Dậu","Sửu"): ("Mão", "Tuất"),
    ("Hợi","Mão","Mùi"): ("Dần", "Tuất")
}

KHOI_VIET_TABLE = {
    ("Giáp", "Mậu"): ("Sửu", "Mùi"),
    ("Ất", "Kỷ"): ("Tý", "Thân"),
    ("Canh", "Tân"): ("Ngọ", "Dần"),
    ("Bính", "Đinh"): ("Hợi", "Dậu"),
    ("Nhâm", "Quý"): ("Mão", "Tỵ")
}

TRANG_SINH_START = {
    "Thủy": "Thân",
    "Mộc": "Hợi",
    "Kim": "Tỵ",
    "Thổ": "Thân",
    "Hỏa": "Dần"
}

TRANG_SINH_STARS = [
    "Tràng Sinh",
    "Mộc Dục",
    "Quan Đới",
    "Lâm Quan",
    "Đế Vượng",
    "Suy",
    "Bệnh",
    "Tử",
    "Mộ",
    "Tuyệt",
    "Thai",
    "Dưỡng"
]

TUAN_GROUPS = [
    (("Giáp Tý", "Quý Dậu"), ("Tuất", "Hợi")),
    (("Giáp Tuất", "Quý Mùi"), ("Thân", "Dậu")),
    (("Giáp Thân", "Quý Tỵ"), ("Ngọ", "Mùi")),
    (("Giáp Ngọ", "Quý Mão"), ("Thìn", "Tỵ")),
    (("Giáp Thìn", "Quý Sửu"), ("Dần", "Mão")),
    (("Giáp Dần", "Quý Hợi"), ("Tý", "Sửu"))
]

CAN = ["Giáp","Ất","Bính","Đinh","Mậu","Kỷ","Canh","Tân","Nhâm","Quý"]
CHI = ["Tý","Sửu","Dần","Mão","Thìn","Tỵ","Ngọ","Mùi","Thân","Dậu","Tuất","Hợi"]

TRIET_TABLE = {
    ("Giáp", "Kỷ"): ("Thân", "Dậu"),
    ("Ất", "Canh"): ("Mùi", "Ngọ"),
    ("Bính", "Tân"): ("Thìn", "Tỵ"),
    ("Đinh", "Nhâm"): ("Dần", "Mão"),
    ("Mậu", "Quý"): ("Tý", "Sửu")
}


def build_empty_houses(chart_id, cung_an_menh):
    start_index = ZODIAC_ORDER.index(cung_an_menh)

    houses = []

    for i in range(12):
        zodiac = ZODIAC_ORDER[(start_index + i) % 12]
        topic = HOUSE_ORDER[i]

        houses.append({
            "house_id": f"h_{i+1:03}",
            "chart_id": chart_id,
            "house_topic": topic,
            "zodiac_sign": zodiac,
            "chinh_tinh": [],
            "phu_tinh_tot": [],
            "phu_tinh_xau": [],
            "vong_trang_sinh": None,
            "tuan_triet": [],
            "dai_han": None
        })

    return houses


def an_tu_vi(lunar_day, cuc_number):
    pos = lunar_day % cuc_number
    if pos == 0:
        pos = cuc_number

    index = (pos - 1)

    return ZODIAC_ORDER[index]


def an_thien_phu(zodiac):
    idx = ZODIAC_ORDER.index(zodiac)
    return ZODIAC_ORDER[(idx + 6) % 12]


def move(zodiac, steps):
    idx = ZODIAC_ORDER.index(zodiac)
    return ZODIAC_ORDER[(idx + steps) % 12]


def an_14_chinh_tinh(tu_vi_pos):
    stars = {}

    # Nhóm Tử Vi
    stars["Tử Vi"] = tu_vi_pos
    stars["Liêm Trinh"] = move(tu_vi_pos, 4)
    stars["Thiên Đồng"] = move(tu_vi_pos, 7)
    stars["Vũ Khúc"] = move(tu_vi_pos, 8)
    stars["Thái Dương"] = move(tu_vi_pos, 9)
    stars["Thiên Cơ"] = move(tu_vi_pos, 11)

    # Nhóm Thiên Phủ
    thien_phu = an_thien_phu(tu_vi_pos)

    stars["Thiên Phủ"] = thien_phu
    stars["Thái Âm"] = move(thien_phu, 1)
    stars["Tham Lang"] = move(thien_phu, 2)
    stars["Cự Môn"] = move(thien_phu, 3)
    stars["Thiên Tướng"] = move(thien_phu, 4)
    stars["Thiên Lương"] = move(thien_phu, 5)
    stars["Thất Sát"] = move(thien_phu, 6)
    stars["Phá Quân"] = move(thien_phu, 10)

    return stars


def assign_stars_to_houses(stars):
    houses = {z: [] for z in ZODIAC_ORDER}

    for star, zodiac in stars.items():
        houses[zodiac].append(star)

    return houses


def get_brightness(star, zodiac):
    return BRIGHTNESS_TABLE[star][zodiac]


def an_loc_ton(lunar_year_str):
    can = lunar_year_str.split()[0]

    return LOC_TON_TABLE[can]


def an_kinh_duong_da_la(loc_ton_pos):
    idx = ZODIAC_ORDER.index(loc_ton_pos)

    kinh_duong = ZODIAC_ORDER[(idx + 1) % 12]
    da_la = ZODIAC_ORDER[(idx - 1) % 12]

    return kinh_duong, da_la


def an_dia_kiep_khong(lunar_hour):
    hour_idx = HOUR_INDEX[lunar_hour]

    start_idx = ZODIAC_ORDER.index("Hợi")

    dia_kiep = ZODIAC_ORDER[(start_idx + hour_idx) % 12]
    dia_khong = ZODIAC_ORDER[(start_idx - hour_idx) % 12]

    return dia_kiep, dia_khong


def get_hoa_linh_start(chi):
    for group, value in HOA_LINH_START.items():
        if chi in group:
            return value
        

def an_hoa_linh(lunar_year_str, lunar_hour, am_duong_gender):
    chi = lunar_year_str.split()[1]
    hour_idx = HOUR_INDEX[lunar_hour]

    hoa_start, linh_start = get_hoa_linh_start(chi)

    hoa_idx = ZODIAC_ORDER.index(hoa_start)
    linh_idx = ZODIAC_ORDER.index(linh_start)

    is_thuan = am_duong_gender in ["Dương Nam", "Âm Nữ"]

    if is_thuan:
        hoa = ZODIAC_ORDER[(hoa_idx + hour_idx) % 12]
        linh = ZODIAC_ORDER[(linh_idx - hour_idx) % 12]
    else:
        hoa = ZODIAC_ORDER[(hoa_idx - hour_idx) % 12]
        linh = ZODIAC_ORDER[(linh_idx + hour_idx) % 12]

    return hoa, linh


def get_luc_sat(lunar_year_str, lunar_hour, loc_ton_pos, am_duong_gender):
    kinh_duong, da_la = an_kinh_duong_da_la(loc_ton_pos)
    dia_kiep, dia_khong = an_dia_kiep_khong(lunar_hour)
    hoa_tinh, linh_tinh = an_hoa_linh(lunar_year_str, lunar_hour, am_duong_gender)

    return {
        "Kình Dương": kinh_duong,
        "Đà La": da_la,
        "Địa Kiếp": dia_kiep,
        "Địa Không": dia_khong,
        "Hỏa Tinh": hoa_tinh,
        "Linh Tinh": linh_tinh
    }


def an_ta_huu(lunar_month):
    month_steps = (lunar_month - 1) % 12

    ta_start = ZODIAC_ORDER.index("Thìn")
    ta_phu = ZODIAC_ORDER[(ta_start + month_steps) % 12]

    huu_start = ZODIAC_ORDER.index("Tuất")
    huu_bat = ZODIAC_ORDER[(huu_start - month_steps) % 12]

    return ta_phu, huu_bat


def an_xuong_khuc(lunar_hour):
    hour_idx = HOUR_INDEX[lunar_hour]

    xuong_start = ZODIAC_ORDER.index("Tuất")
    van_xuong = ZODIAC_ORDER[(xuong_start - hour_idx) % 12]

    khuc_start = ZODIAC_ORDER.index("Thìn")
    van_khuc = ZODIAC_ORDER[(khuc_start + hour_idx) % 12]

    return van_xuong, van_khuc


def an_long_phuong(lunar_year_str):
    chi = lunar_year_str.split()[1]
    year_idx = HOUR_INDEX[chi]

    long_start = ZODIAC_ORDER.index("Thìn")
    long_tri = ZODIAC_ORDER[(long_start + year_idx) % 12]

    phuong_start = ZODIAC_ORDER.index("Tuất")
    phuong_cac = ZODIAC_ORDER[(phuong_start - year_idx) % 12]

    return long_tri, phuong_cac


def an_khoi_viet(lunar_year_str):
    can = lunar_year_str.split()[0]

    for group, (khoi, viet) in KHOI_VIET_TABLE.items():
        if can in group:
            return khoi, viet
        

def get_cat_tinh(lunar_year_str, lunar_month, lunar_hour):
    ta_phu, huu_bat = an_ta_huu(lunar_month)
    van_xuong, van_khuc = an_xuong_khuc(lunar_hour)
    long_tri, phuong_cac = an_long_phuong(lunar_year_str)
    thien_khoi, thien_viet = an_khoi_viet(lunar_year_str)

    return {
        "Tả Phụ": ta_phu,
        "Hữu Bật": huu_bat,
        "Văn Xương": van_xuong,
        "Văn Khúc": van_khuc,
        "Long Trì": long_tri,
        "Phượng Cát": phuong_cac,
        "Thiên Khôi": thien_khoi,
        "Thiên Việt": thien_viet
    }


def an_trang_sinh(cuc_element, am_duong_gender):
    start_cung = TRANG_SINH_START[cuc_element]
    start_idx = ZODIAC_ORDER.index(start_cung)

    is_thuan = am_duong_gender in ["Dương Nam", "Âm Nữ"]

    result = {}

    for i, star in enumerate(TRANG_SINH_STARS):
        if is_thuan:
            idx = (start_idx + i) % 12
        else:
            idx = (start_idx - i) % 12

        cung = ZODIAC_ORDER[idx]
        result[cung] = star

    return result


def sexagenary_index(can, chi):
    can_idx = CAN.index(can)
    chi_idx = CHI.index(chi)

    for i in range(60):
        if i % 10 == can_idx and i % 12 == chi_idx:
            return i
        

def get_tuan(lunar_year_str):
    can, chi = lunar_year_str.split()
    idx = sexagenary_index(can, chi)

    group_idx = idx // 10

    return list(TUAN_GROUPS[group_idx][1])


def get_triet(lunar_year_str):
    can = lunar_year_str.split()[0]

    for group, cung in TRIET_TABLE.items():
        if can in group:
            return list(cung)
        

def get_tuan_triet(lunar_year_str):
    return {
        "tuan": get_tuan(lunar_year_str),
        "triet": get_triet(lunar_year_str)
    }


def an_dai_han(cung_an_menh, cuc_number, am_duong_gender):
    start_idx = ZODIAC_ORDER.index(cung_an_menh)

    is_thuan = am_duong_gender in ["Dương Nam", "Âm Nữ"]

    result = {}

    for i in range(12):
        if is_thuan:
            idx = (start_idx + i) % 12
        else:
            idx = (start_idx - i) % 12

        cung = ZODIAC_ORDER[idx]

        age = cuc_number + i * 10

        result[cung] = age

    return result


def build_chart_houses(user, chart):
    lunar = user["dob_lunar"]

    lunar_year = lunar["year"]
    lunar_month = lunar["month"]
    lunar_day = lunar["day"]
    lunar_hour = lunar["hour"]

    am_duong_gender = user["am_duong_gender"]

    cung_an_menh = chart["cung_an_menh"]

    chart_id = chart["chart_id"]

    # 1. Tạo khung 12 cung
    houses = build_empty_houses(chart_id, cung_an_menh)

    # 2. Chính tinh
    tu_vi_pos = an_tu_vi(lunar_day, chart["cuc"]["number"])
    stars = an_14_chinh_tinh(tu_vi_pos)
    star_map = assign_stars_to_houses(stars)

    for house in houses:
        zodiac = house["zodiac_sign"]

        for star in star_map[zodiac]:
            house["chinh_tinh"].append({
                "star_id": star,
                "brightness": get_brightness(star, zodiac)
            })

    # 3. Hung tinh
    loc_ton = an_loc_ton(lunar_year)
    luc_sat = get_luc_sat(lunar_year, lunar_hour, loc_ton, am_duong_gender)

    for star, zodiac in luc_sat.items():
        for house in houses:
            if house["zodiac_sign"] == zodiac:
                house["phu_tinh_xau"].append(star)

    # 4. Cát tinh
    cat_tinh = get_cat_tinh(lunar_year, lunar_month, lunar_hour)

    for star, zodiac in cat_tinh.items():
        for house in houses:
            if house["zodiac_sign"] == zodiac:
                house["phu_tinh_tot"].append(star)

    # 5. Tràng sinh
    trang_sinh_map = an_trang_sinh(chart["cuc"]["element"], am_duong_gender)

    for house in houses:
        zodiac = house["zodiac_sign"]
        house["vong_trang_sinh"] = trang_sinh_map.get(zodiac)

    # 6. Tuần - Triệt
    tuan_triet = get_tuan_triet(lunar_year)

    for house in houses:
        zodiac = house["zodiac_sign"]

        if zodiac in tuan_triet["tuan"]:
            house["tuan_triet"].append("Tuần")

        if zodiac in tuan_triet["triet"]:
            house["tuan_triet"].append("Triệt")

    # 7. Đại hạn
    dai_han_map = an_dai_han(
        cung_an_menh,
        chart["cuc"]["number"],
        am_duong_gender
    )

    for house in houses:
        zodiac = house["zodiac_sign"]
        house["dai_han"] = dai_han_map[zodiac]

    return houses


user = {'user_id': 'u_123', 'full_name': 'Nguyễn Văn A', 'gender': 'Nam', 'dob_solar': '1990-05-15T08:30:00', 'dob_lunar': {'year': 'Canh Ngọ', 'month': 4, 'day': 21, 'hour': 'Thìn'}, 'am_duong_gender': 'Dương Nam'}
tuvichart = {'chart_id': 'chart_456', 'user_id': 'u_123', 'ban_menh': {'element': 'Thổ', 'name': 'Lộ Bàng Thổ'}, 'cung_an_menh': 'Sửu', 'cuc': {'element': 'Hỏa', 'name': 'Hỏa Lục Cục', 'number': 6}, 'menh_chu': 'Cự Môn', 'than_chu': 'Linh Tinh', 'am_duong_thuan_nghich': 'Nghịch lý', 'menh_cuc_sinh_khac': 'Thổ khắc Thủy', 'than_cu': 'Tài Bạch'}
charthouses = build_chart_houses(user, tuvichart)

print(charthouses)