# rag/chart_utils.py
from rag.normalize import normalize_star_name, normalize_palace_name

def build_chart_index(houses_chart: list[dict]) -> dict:
    chart_index = {}

    for house in houses_chart:
        palace_id = normalize_palace_name(house["house_topic"])

        chinh_tinh = [
            normalize_star_name(star["name"])
            for star in house.get("chinh_tinh", [])
        ]

        phu_tinh = [
            normalize_star_name(star)
            for star in house.get("phu_tinh", [])
        ]

        all_stars = chinh_tinh + phu_tinh

        chart_index[palace_id] = {
            "house_topic": house["house_topic"],
            "zodiac_sign": house.get("zodiac_sign"),
            "chinh_tinh": chinh_tinh,
            "phu_tinh": phu_tinh,
            "all_stars": all_stars,
            "brightness": {
                normalize_star_name(star["name"]): star.get("brightness")
                for star in house.get("chinh_tinh", [])
            },
            "tuan_triet": house.get("tuan_triet", []),
            "dai_han": house.get("dai_han"),
        }

    return chart_index