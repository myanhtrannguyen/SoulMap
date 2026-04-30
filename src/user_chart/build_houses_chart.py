# Lập template cho mỗi ô

CAN = ["Giáp","Ất","Bính","Đinh","Mậu","Kỷ","Canh","Tân","Nhâm","Quý"]

CHI = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]

HOUSE_ORDER = [
    "Mệnh", "Phụ Mẫu", "Phúc Đức", "Điền Trạch",
    "Quan Lộc", "Nô Bộc", "Thiên Di", "Tật Ách",
    "Tài Bạch", "Tử Tức", "Phu Thê", "Huynh Đệ"
]

HOUR_INDEX = {
    "Tý": 0, "Sửu": 1, "Dần": 2, "Mão": 3,
    "Thìn": 4, "Tỵ": 5, "Ngọ": 6, "Mùi": 7,
    "Thân": 8, "Dậu": 9, "Tuất": 10, "Hợi": 11
}

def move(zodiac, steps):
    idx = CHI.index(zodiac)
    return CHI[(idx + steps) % 12]

def build_empty_houses(chart_id, cung_an_menh):
    start_index = CHI.index(cung_an_menh)

    houses = []

    for i in range(12):
        zodiac = CHI[(start_index + i) % 12]
        topic = HOUSE_ORDER[i]

        houses.append({
            "house_id": f"h_{i+1:03}",
            "chart_id": chart_id,
            "house_topic": topic,
            "zodiac_sign": zodiac,
            "chinh_tinh": [],
            "phu_tinh": [],
            "vong_trang_sinh": None,
            "tuan_triet": [],
            "dai_han": None
        })

    return houses


# Xác định chính tinh

#   1. Tử Vi tinh hệ

CUNG_CUC = {
    2: HOUR_INDEX["Sửu"], 3: HOUR_INDEX["Thìn"], 
    4: HOUR_INDEX["Hợi"], 5: HOUR_INDEX["Ngọ"] , 
    6: HOUR_INDEX["Dậu"]
}

def an_tu_vi(lunar_day, cuc_number):
    start = CUNG_CUC[cuc_number] 

    pos = lunar_day % cuc_number

    if pos == 0:
        pos = cuc_number

    pos = (lunar_day + cuc_number - pos) // cuc_number
    index = int(start + pos - 1) % 12

    return CHI[index]

def an_tu_vi_tinh_he(tu_vi):
    stars = {}

    stars["Tử Vi"] = tu_vi
    stars["Liêm Trinh"] = move(tu_vi, 4)
    stars["Thiên Đồng"] = move(tu_vi, 7)
    stars["Vũ Khúc"] = move(tu_vi, 8)
    stars["Thái Dương"] = move(tu_vi, 9)
    stars["Thiên Cơ"] = move(tu_vi, 11)

    return stars

#   2. Thiên Phủ tinh hệ

def an_thien_phu_tinh_he(tu_vi):
    stars = {}

    thien_phu = move(tu_vi, 6)

    stars["Thiên Phủ"] = thien_phu
    stars["Thái Âm"] = move(thien_phu, 1)
    stars["Tham Lang"] = move(thien_phu, 2)
    stars["Cự Môn"] = move(thien_phu, 3)
    stars["Thiên Tướng"] = move(thien_phu, 4)
    stars["Thiên Lương"] = move(thien_phu, 5)
    stars["Thất Sát"] = move(thien_phu, 6)
    stars["Phá Quân"] = move(thien_phu, 10)

    return stars

#   Xác định độ sáng của 14 chính tinh

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

def get_brightness(star, zodiac):
    return BRIGHTNESS_TABLE[star][zodiac]


# Xác định phụ tinh

#   3. Thái Tuế tinh hệ

def an_thai_tue_tinh_he(chi):
    stars = {}

    stars["Thái Tuế"] = chi
    stars["Thiếu Dương"] = move(chi, 1)
    stars["Tang Môn"] = move(chi, 2)
    stars["Thiếu Âm"] = move(chi, 3)
    stars["Quan Phù"] = move(chi, 4)
    stars["Tử Phù"] = move(chi, 5)
    stars["Tuế Phá"] = move(chi, 6)
    stars["Long Đức"] = move(chi, 7)
    stars["Bạch Hổ"] = move(chi, 8)
    stars["Phúc Đức"] = move(chi, 9)
    stars["Điếu Khách"] = move(chi, 10)
    stars["Trực Phù"] = move(chi, 11)

    return stars

#   4. Lộc Tồn tinh hệ

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

def an_loc_ton_tinh_he(can, am_duong_gender):
    loc_ton = LOC_TON_TABLE[can]

    d = 1 if am_duong_gender in ["Dương Nam", "Âm Nữ"] else -1

    stars = {}

    stars["Lộc Tồn"] = loc_ton
    stars["Lực Sỹ"] = move(loc_ton, 1 * d)
    stars["Thanh Long"] = move(loc_ton, 2 * d)
    stars["Tiểu Hao"] = move(loc_ton, 3 * d)
    stars["Tướng Quân"] = move(loc_ton, 4 * d)
    stars["Tấu Thư"] = move(loc_ton, 5 * d)
    stars["Phi Liêm"] = move(loc_ton, 6 * d)
    stars["Hỷ Thần"] = move(loc_ton, 7 * d)
    stars["Bệnh Phù"] = move(loc_ton, 8 * d)
    stars["Đại Hao"] = move(loc_ton, 9 * d)
    stars["Phục Binh"] = move(loc_ton, 10 * d)
    stars["Quan Phù"] = move(loc_ton, 11 * d)

    return stars

#   6. Bộ sao Lục Sát

#       6.1. Kình Dương, Đà La

def an_kinh_duong_da_la(loc_ton):
    kinh_duong = move(loc_ton, 1)
    da_la = move(loc_ton, -1)

    return {
        "Kình Dương": kinh_duong,
        "Đà La": da_la
    }

#       6.2. Địa Không, Địa Kiếp

def an_dia_khong_kiep(lunar_hour):
    hour_idx = HOUR_INDEX[lunar_hour]

    start = "Hợi"

    dia_kiep = move(start, hour_idx)
    dia_khong = move(start, -hour_idx)

    return {
        "Địa Kiếp": dia_kiep,
        "Địa Không": dia_khong
    }

#       6.3. Hỏa Tinh, Linh Tinh

HOA_LINH_START = {
    ("Dần","Ngọ","Tuất"): ("Sửu", "Mão"),
    ("Thân","Tý","Thìn"): ("Dần", "Tuất"),
    ("Tỵ","Dậu","Sửu"): ("Mão", "Tuất"),
    ("Hợi","Mão","Mùi"): ("Dần", "Tuất")
}

def get_hoa_linh_start(chi):
    for group, value in HOA_LINH_START.items():
        if chi in group:
            return value
        
def an_hoa_linh(chi, lunar_hour, am_duong_gender):
    hour_idx = HOUR_INDEX[lunar_hour]

    hoa_start, linh_start = get_hoa_linh_start(chi)

    is_thuan = am_duong_gender in ["Dương Nam", "Âm Nữ"]

    if is_thuan:
        hoa = move(hoa_start, hour_idx)
        linh = move(linh_start, -hour_idx)
    else:
        hoa = move(hoa_start, -hour_idx)
        linh = move(linh_start, hour_idx)

    return {
        "Hỏa Tinh": hoa,
        "Linh Tinh": linh
    }

#   7. Bộ sao Tả Hữu

def an_ta_huu(lunar_month):
    steps = (lunar_month - 1)

    ta_phu = move("Thìn", steps)
    huu_bat = move("Tuất", -steps)

    return {
        "Tả Phụ": ta_phu,
        "Hữu Bật": huu_bat
    }

#   8. Bộ sao Xương Khúc

def an_xuong_khuc(lunar_hour):
    steps = HOUR_INDEX[lunar_hour]

    van_xuong = move("Tuất", -steps)
    van_khuc = move("Thìn", steps)

    return {
        "Văn Xương": van_xuong,
        "Văn Khúc": van_khuc
    }

#   9. Bộ sao Long Phượng

def an_long_phuong(chi):
    steps = CHI.index(chi)

    long_tri = move("Thìn", steps)
    phuong_cac = move("Tuất", -steps)

    return {
        "Long Trì": long_tri,
        "Phượng Các": phuong_cac
    }

#   10. Bộ sao Khôi Việt

KHOI_VIET_TABLE = {
    ("Giáp", "Mậu"): ("Sửu", "Mùi"),
    ("Ất", "Kỷ"): ("Tý", "Thân"),
    ("Canh", "Tân"): ("Ngọ", "Dần"),
    ("Bính", "Đinh"): ("Hợi", "Dậu"),
    ("Nhâm", "Quý"): ("Mão", "Tỵ")
}

def an_khoi_viet(can):
    for group, (khoi, viet) in KHOI_VIET_TABLE.items():
        if can in group:
            return {
                "Thiên Khôi": khoi,
                "Thiên Việt": viet
            }
        
#   11. Bộ sao Khốc Hư

def an_khoc_hu(chi):
    steps = CHI.index(chi)

    thien_khoc = move("Ngọ", -steps)
    thien_hu = move("Ngọ", steps)

    return {
        "Thiên Khốc": thien_khoc,
        "Thiên Hư": thien_hu
    }

#   12. Bộ sao Thai Tọa

def an_thai_toa(lunar_day, ta_phu, huu_bat):
    steps = lunar_day - 1

    tam_thai = move(ta_phu, steps)
    bat_toa = move(huu_bat, -steps)

    return {
        "Tam Thai": tam_thai,
        "Bát Tọa": bat_toa
    }

#   13. Bộ sao Quang Quý

def an_quang_quy(lunar_day, van_xuong, van_khuc):
    steps = lunar_day - 1

    an_quang = move(van_xuong, steps)
    an_quang = move(an_quang, -1)

    thien_quy = move(van_khuc, -steps)
    thien_quy = move(thien_quy, -1)

    return {
        "Ân Quang": an_quang,
        "Thiên Quý": thien_quy
    }

#   14. Bộ sao Thiên, Nguyệt Đức

def an_thien_nguyet_duc(chi):
    steps = CHI.index(chi)

    thien_duc = move("Dậu", steps)
    nguyet_duc = move("Tỵ", steps)

    return {
        "Thiên Đức": thien_duc,
        "Nguyệt Đức": nguyet_duc
    }

#   15. Bộ sao Hình, Riêu, Y

def an_hinh_rieu_y(lunar_month):
    steps = lunar_month - 1

    thien_hinh = move("Dậu", steps)
    thien_rieu = move("Sửu", steps)
    thien_y = thien_rieu

    return {
        "Thiên Hình": thien_hinh,
        "Thiên Riêu": thien_rieu,
        "Thiên Y": thien_y
    }

#   16. Bộ sao Hồng Hỷ

def an_hong_hy(chi):
    steps = CHI.index(chi)

    hong_loan = move("Mão", -steps)
    thien_hy = move(hong_loan, 6)

    return {
        "Hồng Loan": hong_loan,
        "Thiên Hỷ": thien_hy
    }

#   17. Bộ sao Ấn Phù

def an_an_phu(loc_ton):
    quoc_an = move(loc_ton, 8)
    duong_phu = move(loc_ton, -7)

    return {
        "Quốc Ấn": quoc_an,
        "Đường Phù": duong_phu
    }

#   18. Bộ sao Thiên Địa, Giải Thần

def an_thien_dia_giai(lunar_month, phuong_cac):
    steps = lunar_month - 1

    thien_giai = move("Thân", steps)
    dia_giai = move("Mùi", steps)
    giai_than = phuong_cac

    return {
        "Thiên Giải": thien_giai,
        "Địa Giải": dia_giai,
        "Giải Thần": giai_than
    }

#   19. Bộ sao Thai Cáo

def an_thai_cao(lunar_hour):
    steps = HOUR_INDEX[lunar_hour]

    thai_phu = move("Ngọ", steps)
    phong_cao = move("Dần", steps)

    return {
        "Thai Phụ": thai_phu,
        "Phong Cáo": phong_cao
    }

#   20. Bộ sao Tài Thọ

def an_tai_tho(chi, cung_menh, cung_than, houses):
    steps = CHI.index(chi)

    for h in houses:
        if h["zodiac_sign"] == cung_menh:
            menh_zodiac = h["zodiac_sign"]

        if h["house_topic"] == cung_than:
            than_zodiac = h["zodiac_sign"]

    thien_tai = move(menh_zodiac, steps)
    thien_tho = move(than_zodiac, steps)

    return {
        "Thiên Tài": thien_tai,
        "Thiên Thọ": thien_tho
    }

#   21. Bộ sao Thương Sứ

def an_thuong_su(houses):
    result = {}

    for house in houses:
        if house["house_topic"] == "Nô Bộc":
            result["Thiên Thương"] = house["zodiac_sign"]

        if house["house_topic"] == "Tật Ách":
            result["Thiên Sứ"] = house["zodiac_sign"]

    return result

#   22. Bộ sao La Võng

def an_la_vong():
    return {
        "Thiên La": "Thìn",
        "Địa Võng": "Tuất"
    }

#   23. Bộ sao Tứ Hóa

TU_HOA_TABLE = {
    "Giáp": ("Liêm Trinh", "Phá Quân", "Vũ Khúc", "Thái Dương"),
    "Ất": ("Thiên Cơ", "Thiên Lương", "Tử Vi", "Thái Âm"),
    "Bính": ("Thiên Đồng", "Thiên Cơ", "Văn Xương", "Liêm Trinh"),
    "Đinh": ("Thái Âm", "Thiên Đồng", "Thiên Cơ", "Cự Môn"),
    "Mậu": ("Tham Lang", "Thái Âm", "Hữu Bật", "Thiên Cơ"),
    "Kỷ": ("Vũ Khúc", "Tham Lang", "Thiên Lương", "Văn Khúc"),
    "Canh": ("Thái Dương", "Vũ Khúc", "Thái Âm", "Thiên Đồng"),
    "Tân": ("Cự Môn", "Thiên Lương", "Văn Khúc", "Văn Xương"),
    "Nhâm": ("Thiên Lương", "Tử Vi", "Tả Phụ", "Vũ Khúc"),
    "Quý": ("Phá Quân", "Cự Môn", "Thái Âm", "Tham Lang")
}

def an_tu_hoa(can, star_positions):
    loc, quyen, khoa, ky = TU_HOA_TABLE[can]

    return {
        "Hóa Lộc": star_positions[loc],
        "Hóa Quyền": star_positions[quyen],
        "Hóa Khoa": star_positions[khoa],
        "Hóa Kỵ": star_positions[ky]
    }

#   24. Bộ sao Quan Phúc

QUAN_PHUC_TABLE = {
    "Giáp": ("Mùi", "Dậu"),
    "Ất": ("Thìn", "Thân"),
    "Bính": ("Tỵ", "Tý"),
    "Đinh": ("Dần", "Hợi"),
    "Mậu": ("Mão", "Mão"),
    "Kỷ": ("Dậu", "Dần"),
    "Canh": ("Hợi", "Ngọ"),
    "Tân": ("Dậu", "Tỵ"),
    "Nhâm": ("Tuất", "Ngọ"),
    "Quý": ("Ngọ", "Tỵ")
}

def an_quan_phuc(can):
    thien_quan, thien_phuc = QUAN_PHUC_TABLE[can]

    return {
        "Thiên Quan": thien_quan,
        "Thiên Phúc": thien_phuc
    }

#   25. Bộ sao Cô Quả

CO_QUA_TABLE = {
    ("Hợi", "Tý", "Sửu"): ("Dần", "Tuất"),
    ("Dần", "Mão", "Thìn"): ("Tỵ", "Sửu"),
    ("Tỵ", "Ngọ", "Mùi"): ("Thân", "Thìn"),
    ("Thân", "Dậu", "Tuất"): ("Hợi", "Mùi")
}

def an_co_qua(chi):
    for group, (co_than, qua_tu) in CO_QUA_TABLE.items():
        if chi in group:
            return {
                "Cô Thần": co_than,
                "Quả Tú": qua_tu
            }
        
#   26. Sao Đào Hoa

DAO_HOA_TABLE = {
    ("Tỵ", "Dậu", "Sửu"): "Ngọ",
    ("Thân", "Tý", "Thìn"): "Dậu",
    ("Hợi", "Mão", "Mùi"): "Tý",
    ("Dần", "Ngọ", "Tuất"): "Mão"
}

def an_dao_hoa(chi):
    for group, dao_hoa in DAO_HOA_TABLE.items():
        if chi in group:
            return dao_hoa
        
#   27. Sao Thiên Mã

THIEN_MA_TABLE = {
    ("Tỵ", "Dậu", "Sửu"): "Hợi",
    ("Thân", "Tý", "Thìn"): "Dần",
    ("Hợi", "Mão", "Mùi"): "Tỵ",
    ("Dần", "Ngọ", "Tuất"): "Thân"
}

def an_thien_ma(chi):
    for group, thien_ma in THIEN_MA_TABLE.items():
        if chi in group:
            return thien_ma
        
#   28. Sao Kiếp Sát

KIEP_SAT_TABLE = {
    ("Tỵ", "Dậu", "Sửu"): "Dần",
    ("Dần", "Ngọ", "Tuất"): "Hợi",
    ("Hợi", "Mão", "Mùi"): "Thân",
    ("Thân", "Tý", "Thìn"): "Tỵ"
}

def an_kiep_sat(chi):
    for group, kiep_sat in KIEP_SAT_TABLE.items():
        if chi in group:
            return kiep_sat
        
#   29. Sao Phá Toái

PHA_TOAI_TABLE = {
    ("Tý", "Ngọ", "Mão", "Dậu"): "Tỵ",
    ("Dần", "Thân", "Tỵ", "Hợi"): "Dậu",
    ("Thìn", "Tuất", "Sửu", "Mùi"): "Sửu"
}

def an_pha_toai(chi):
    for group, pha_toai in PHA_TOAI_TABLE.items():
        if chi in group:
            return pha_toai

#   30. Sao Hoa Cái

HOA_CAI_TABLE = {
    ("Tỵ", "Dậu", "Sửu"): "Sửu",
    ("Dần", "Ngọ", "Tuất"): "Tuất",
    ("Hợi", "Mão", "Mùi"): "Mùi",
    ("Thân", "Tý", "Thìn"): "Thìn"
}

def an_hoa_cai(lunar_year_str):
    chi = lunar_year_str.split()[1]

    for group, hoa_cai in HOA_CAI_TABLE.items():
        if chi in group:
            return hoa_cai
        
#   31. Sao Lưu Hà

LUU_HA_TABLE = {
    "Giáp": "Dậu", "Ất": "Tuất", "Bính": "Mùi", "Đinh": "Thìn",
    "Mậu": "Tỵ", "Kỷ": "Ngọ", "Canh": "Thân", "Tân": "Mão",
    "Nhâm": "Hợi", "Quý": "Dần"
}

def an_luu_ha(can):
    return LUU_HA_TABLE[can]

#   32. Sao Thiên Trù

THIEN_TRU_TABLE = {
    "Giáp": "Tỵ", "Ất": "Ngọ", "Bính": "Tý", "Đinh": "Tỵ",
    "Mậu": "Ngọ", "Kỷ": "Thân", "Canh": "Dần", "Tân": "Ngọ",
    "Nhâm": "Dậu", "Quý": "Tuất"
}

def an_thien_tru(can):
    return THIEN_TRU_TABLE[can]

#   33. Sao Lưu Niên Văn Tinh

LUU_NIEN_VAN_TINH_TABLE = {
    "Giáp": "Tỵ", "Ất": "Ngọ", "Bính": "Thân", "Đinh": "Dậu",
    "Mậu": "Thân", "Kỷ": "Dậu", "Canh": "Hợi", "Tân": "Tý",
    "Nhâm": "Dần", "Quý": "Mão"
}

def an_luu_nien_van_tinh(lunar_year_str):
    can = lunar_year_str.split()[0]
    return LUU_NIEN_VAN_TINH_TABLE[can]

#   34. Sao Bác Sỹ

def an_bac_sy(loc_ton):
    return loc_ton

#   35. Sao Đẩu Quân

def an_dau_quan(thai_tue, lunar_month, lunar_hour):
    month_steps = lunar_month - 1
    dau_quan = move(thai_tue, -month_steps)

    hour_steps = HOUR_INDEX[lunar_hour]
    dau_quan = move(dau_quan, hour_steps)

    return dau_quan

#   36. Sao Thiên Không

def an_thien_khong(thai_tue):
    return move(thai_tue, 1)


# Xác định vòng Tràng Sinh

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

def an_trang_sinh(cuc_element, am_duong_gender):
    start_cung = TRANG_SINH_START[cuc_element]
    start_idx = CHI.index(start_cung)

    is_thuan = am_duong_gender in ["Dương Nam", "Âm Nữ"]

    result = {}

    for i, star in enumerate(TRANG_SINH_STARS):
        if is_thuan:
            idx = (start_idx + i) % 12
        else:
            idx = (start_idx - i) % 12

        cung = CHI[idx]
        result[cung] = star

    return result


# Xác định Tuần - Triệt

#   1. Tuần Trung không vong

TUAN_GROUPS = [
    (("Giáp Tý", "Quý Dậu"), ("Tuất", "Hợi")),
    (("Giáp Tuất", "Quý Mùi"), ("Thân", "Dậu")),
    (("Giáp Thân", "Quý Tỵ"), ("Ngọ", "Mùi")),
    (("Giáp Ngọ", "Quý Mão"), ("Thìn", "Tỵ")),
    (("Giáp Thìn", "Quý Sửu"), ("Dần", "Mão")),
    (("Giáp Dần", "Quý Hợi"), ("Tý", "Sửu"))
]

def sexagenary_index(can, chi):
    can_idx = CAN.index(can)
    chi_idx = CHI.index(chi)

    for i in range(60):
        if i % 10 == can_idx and i % 12 == chi_idx:
            return i
            
def an_tuan_trung(lunar_year_str):
    can, chi = lunar_year_str.split()
    idx = sexagenary_index(can, chi)

    group_idx = idx // 10

    return list(TUAN_GROUPS[group_idx][1])

#   2. Triệt Lộ không vong

TRIET_TABLE = {
    ("Giáp", "Kỷ"): ("Thân", "Dậu"),
    ("Ất", "Canh"): ("Mùi", "Ngọ"),
    ("Bính", "Tân"): ("Thìn", "Tỵ"),
    ("Đinh", "Nhâm"): ("Dần", "Mão"),
    ("Mậu", "Quý"): ("Tý", "Sửu")
}

def an_triet_lo(lunar_year_str):
    can = lunar_year_str.split()[0]

    for group, cung in TRIET_TABLE.items():
        if can in group:
            return list(cung)
        

# Xác định đại hạn

def an_dai_han(cung_an_menh, cuc_number, am_duong_gender):
    start_idx = CHI.index(cung_an_menh)

    is_thuan = am_duong_gender in ["Dương Nam", "Âm Nữ"]

    result = {}

    for i in range(12):
        if is_thuan:
            idx = (start_idx + i) % 12
        else:
            idx = (start_idx - i) % 12

        cung = CHI[idx]

        age = cuc_number + i * 10

        result[cung] = age

    return result


# Hoàn thành lá số

def build_houses_chart(user, tuvichart):
    # 1. Khởi tạo
    chart_id = tuvichart["chart_id"]
    houses = build_empty_houses(chart_id, tuvichart["cung_an_menh"])

    lunar = user["dob_lunar"]
    lunar_day = lunar["day"]
    lunar_month = lunar["month"]
    lunar_hour = lunar["hour"]
    lunar_year_str = lunar["year"]

    can = lunar_year_str.split()[0]
    chi = lunar_year_str.split()[1]

    am_duong_gender = user["am_duong_gender"]
    cuc_number = tuvichart["cuc"]["number"]

    # 2. Chính tinh
    tu_vi = an_tu_vi(lunar_day, cuc_number)

    stars = {}
    stars.update(an_tu_vi_tinh_he(tu_vi))
    stars.update(an_thien_phu_tinh_he(tu_vi))

    # 3. Phụ tinh
    stars.update(an_thai_tue_tinh_he(chi))

    loc_ton_he = an_loc_ton_tinh_he(can, am_duong_gender)
    stars.update(loc_ton_he)

    stars.update(an_kinh_duong_da_la(loc_ton_he["Lộc Tồn"]))
    stars.update(an_dia_khong_kiep(lunar_hour))
    stars.update(an_hoa_linh(chi, lunar_hour, am_duong_gender))

    ta_huu = an_ta_huu(lunar_month)
    stars.update(ta_huu)

    xuong_khuc = an_xuong_khuc(lunar_hour)
    stars.update(xuong_khuc)

    stars.update(an_long_phuong(chi))
    stars.update(an_khoi_viet(can))
    stars.update(an_khoc_hu(chi))

    stars.update(an_thai_toa(lunar_day, ta_huu["Tả Phụ"], ta_huu["Hữu Bật"]))
    stars.update(an_quang_quy(lunar_day, xuong_khuc["Văn Xương"], xuong_khuc["Văn Khúc"]))

    stars.update(an_thien_nguyet_duc(chi))
    stars.update(an_hinh_rieu_y(lunar_month))
    stars.update(an_hong_hy(chi))
    stars.update(an_an_phu(loc_ton_he["Lộc Tồn"]))

    phuong_cac = stars["Phượng Các"]
    stars.update(an_thien_dia_giai(lunar_month, phuong_cac))

    stars.update(an_thai_cao(lunar_hour))

    stars.update(an_tai_tho(chi, tuvichart["cung_an_menh"], tuvichart["than_cu"], houses))

    stars.update(an_la_vong())

    stars.update(an_tu_hoa(can, stars))
    stars.update(an_quan_phuc(can))
    stars.update(an_co_qua(chi))

    stars["Đào Hoa"] = an_dao_hoa(chi)
    stars["Thiên Mã"] = an_thien_ma(chi)
    stars["Kiếp Sát"] = an_kiep_sat(chi)
    stars["Phá Toái"] = an_pha_toai(chi)
    stars["Hoa Cái"] = an_hoa_cai(lunar_year_str)

    stars["Lưu Hà"] = an_luu_ha(can)
    stars["Thiên Trù"] = an_thien_tru(can)
    stars["Lưu Niên Văn Tinh"] = an_luu_nien_van_tinh(lunar_year_str)

    stars["Bác Sỹ"] = an_bac_sy(loc_ton_he["Lộc Tồn"])

    thai_tue = stars["Thái Tuế"]
    stars["Đẩu Quân"] = an_dau_quan(thai_tue, lunar_month, lunar_hour)
    stars["Thiên Không"] = an_thien_khong(thai_tue)

    # 4. Tràng Sinh
    trang_sinh_map = an_trang_sinh(
        tuvichart["cuc"]["element"],
        am_duong_gender
    )

    # 5. Tuần - Triệt
    tuan = an_tuan_trung(lunar_year_str)
    triet = an_triet_lo(lunar_year_str)

    # 6. Đại hạn
    dai_han_map = an_dai_han(
        tuvichart["cung_an_menh"],
        cuc_number,
        am_duong_gender
    )

    # 7. Đổ sao vào từng cung
    for house in houses:
        zodiac = house["zodiac_sign"]

        # Chính tinh
        for star, pos in stars.items():
            if pos == zodiac:
                if star in BRIGHTNESS_TABLE:
                    house["chinh_tinh"].append({
                        "name": star,
                        "brightness": get_brightness(star, zodiac)
                    })
                else:
                    house["phu_tinh"].append(star)

        # Tràng sinh
        house["vong_trang_sinh"] = trang_sinh_map.get(zodiac)

        # Tuần - Triệt
        if zodiac in tuan:
            house["tuan_triet"].append("Tuần")

        if zodiac in triet:
            house["tuan_triet"].append("Triệt")

        # Đại hạn
        house["dai_han"] = dai_han_map[zodiac]

    # 8. Sao đặc biệt theo cung
    thuong_su = an_thuong_su(houses)

    for house in houses:
        if house["zodiac_sign"] == thuong_su.get("Thiên Thương"):
            house["phu_tinh"].append("Thiên Thương")

        if house["zodiac_sign"] == thuong_su.get("Thiên Sứ"):
            house["phu_tinh"].append("Thiên Sứ")

    return houses