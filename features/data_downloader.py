# -*- coding: utf-8 -*-
"""
features/data_downloader.py — Tải dữ liệu thực tế từ Spoonacular API
=====================================================================
Script tải nguyên liệu + công thức THỰC TẾ từ Spoonacular API,
lưu vào CSV fallback files.

Chạy trên Colab:
    python features/data_downloader.py --api-key YOUR_SPOONACULAR_KEY

Hoặc import trong notebook:
    from features.data_downloader import download_all_data
    download_all_data(api_key="YOUR_KEY", data_dir="data")

CSVs tạo ra:
    data/ingredients.csv   — nguyên liệu + giá USD + calo + category
    data/recipes.csv       — công thức + nguyên liệu + steps + cuisine
"""

import os
import csv
import json
import sys
import time
from typing import List, Dict, Any, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

SPOONACULAR_BASE = "https://api.spoonacular.com"
USD_TO_VND = 25000

# Danh sách nguyên liệu phổ biến cho ẩm thực quốc tế (tên tiếng Anh)
COMMON_INGREDIENTS = [
    # Protein
    "beef", "pork", "chicken", "shrimp", "salmon", "tofu", "egg",
    "ground beef", "chicken breast", "bacon", "tuna",
    # Vegetables
    "tomato", "onion", "garlic", "ginger", "carrot", "potato",
    "bell pepper", "mushroom", "spinach", "broccoli", "cabbage",
    "cucumber", "corn", "lettuce", "green onion", "chili pepper",
    # Grains & Pasta
    "rice", "spaghetti", "bread", "flour", "noodles", "tortilla",
    # Dairy & Cheese
    "butter", "milk", "mozzarella cheese", "parmesan cheese", "cream",
    # Sauces & Condiments
    "soy sauce", "fish sauce", "olive oil", "sesame oil", "vinegar",
    "tomato sauce", "sriracha", "wasabi", "gochujang", "lime juice",
    # Spices
    "salt", "black pepper", "sugar", "cumin", "paprika", "basil",
    "cilantro", "oregano", "cinnamon",
    # Asian specific
    "nori", "kimchi", "coconut milk", "lemongrass", "bean sprouts",
    # Mexican
    "avocado", "jalapeño", "black beans", "cilantro",
]

# Truy vấn tìm recipe theo cuisine
CUISINE_QUERIES = [
    ("vietnamese", "pho"),
    ("vietnamese", "banh mi"),
    ("vietnamese", "spring rolls"),
    ("italian", "spaghetti bolognese"),
    ("italian", "pizza margherita"),
    ("italian", "risotto"),
    ("japanese", "sushi"),
    ("japanese", "ramen"),
    ("japanese", "teriyaki"),
    ("korean", "bibimbap"),
    ("korean", "kimchi stew"),
    ("korean", "bulgogi"),
    ("mexican", "tacos"),
    ("mexican", "burrito"),
    ("mexican", "guacamole"),
    ("european", "caesar salad"),
    ("european", "beef stew"),
    ("european", "mushroom soup"),
]


def fetch_ingredient_info(name: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Tìm và lấy thông tin chi tiết nguyên liệu từ Spoonacular."""
    try:
        # Bước 1: Tìm ingredient ID
        resp = requests.get(
            f"{SPOONACULAR_BASE}/food/ingredients/search",
            params={"apiKey": api_key, "query": name, "number": 1},
            timeout=10
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None

        ing_id = results[0]["id"]
        ing_name = results[0]["name"]

        # Bước 2: Lấy chi tiết (giá, calo, category)
        resp2 = requests.get(
            f"{SPOONACULAR_BASE}/food/ingredients/{ing_id}/information",
            params={"apiKey": api_key, "amount": 1, "unit": "serving"},
            timeout=10
        )
        resp2.raise_for_status()
        info = resp2.json()

        # Extract giá (US cents → USD)
        cost_cents = info.get("estimatedCost", {}).get("value", 0)
        cost_usd = cost_cents / 100.0

        # Extract calo
        calories = 0
        for nutrient in info.get("nutrition", {}).get("nutrients", []):
            if nutrient.get("name") == "Calories":
                calories = nutrient.get("amount", 0)
                break

        category = info.get("aisle", "other")

        return {
            "id": ing_id,
            "name": ing_name,
            "price_usd": round(cost_usd, 2),
            "price_vnd": round(cost_usd * USD_TO_VND),
            "calories": round(calories),
            "category": category,
            "unit": info.get("unit", "g"),
        }
    except Exception as e:
        print(f"   {name}: {e}")
        return None


def fetch_recipe(cuisine: str, query: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Tìm và lấy thông tin chi tiết công thức từ Spoonacular."""
    try:
        resp = requests.get(
            f"{SPOONACULAR_BASE}/recipes/complexSearch",
            params={
                "apiKey": api_key,
                "query": query,
                "cuisine": cuisine,
                "number": 1,
                "addRecipeInformation": True,
                "fillIngredients": True,
                "addRecipeNutrition": True,
            },
            timeout=15
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None

        r = results[0]

        # Extract ingredients
        ingredients = []
        for ing in r.get("extendedIngredients", [])[:10]:
            ingredients.append({
                "name": ing.get("name", ""),
                "amount": ing.get("amount", 1),
                "unit": ing.get("unit", ""),
            })

        # Extract steps
        steps = []
        for inst in r.get("analyzedInstructions", []):
            for step in inst.get("steps", [])[:8]:
                steps.append(step.get("step", ""))

        # Extract nutrition
        calories = 0
        for n in r.get("nutrition", {}).get("nutrients", []):
            if n.get("name") == "Calories":
                calories = n.get("amount", 0)
                break

        cuisine_vn_map = {
            "vietnamese": "Việt Nam", "italian": "Ý", "japanese": "Nhật Bản",
            "korean": "Hàn Quốc", "mexican": "Mexico", "european": "Phương Tây",
        }

        return {
            "id": r.get("id", 0),
            "name": r.get("title", ""),
            "cuisine": cuisine_vn_map.get(cuisine.lower(), cuisine),
            "image": r.get("image", ""),
            "servings": r.get("servings", 4),
            "ready_in_minutes": r.get("readyInMinutes", 45),
            "calories": round(calories),
            "ingredients": ingredients,
            "steps": steps,
            "tags": [t for t in (r.get("dishTypes", []) + r.get("cuisines", []))[:5]],
            "source": "spoonacular",
            "source_url": r.get("sourceUrl", ""),
        }
    except Exception as e:
        print(f"   {cuisine}/{query}: {e}")
        return None


def download_all_data(api_key: str, data_dir: str = "data"):
    """
    Tải toàn bộ dữ liệu thực tế từ Spoonacular API.
    Lưu vào CSV files trong data_dir.
    """
    os.makedirs(data_dir, exist_ok=True)

    if not HAS_REQUESTS:
        print(" Cần cài đặt requests: pip install requests")
        return

    # === TẢI NGUYÊN LIỆU ===
    print("\n Đang tải nguyên liệu từ Spoonacular API...")
    ingredients = []
    for i, name in enumerate(COMMON_INGREDIENTS):
        print(f"  [{i+1}/{len(COMMON_INGREDIENTS)}] {name}...", end=" ")
        info = fetch_ingredient_info(name, api_key)
        if info:
            ingredients.append(info)
            print(f" ${info['price_usd']} ({info['price_vnd']:,}đ)")
        else:
            print("")
        time.sleep(0.3)  # Rate limit

    # Lưu CSV
    ing_csv = os.path.join(data_dir, "ingredients.csv")
    with open(ing_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "name", "price_usd", "price_vnd", "calories", "category", "unit"
        ])
        writer.writeheader()
        writer.writerows(ingredients)
    print(f"\n Đã lưu {len(ingredients)} nguyên liệu → {ing_csv}")

    # Lưu JSON (backup)
    ing_json = os.path.join(data_dir, "ingredients.json")
    with open(ing_json, 'w', encoding='utf-8') as f:
        json.dump(ingredients, f, ensure_ascii=False, indent=2)

    # === TẢI CÔNG THỨC ===
    print("\n Đang tải công thức từ Spoonacular API...")
    recipes = []
    for i, (cuisine, query) in enumerate(CUISINE_QUERIES):
        print(f"  [{i+1}/{len(CUISINE_QUERIES)}] {cuisine}/{query}...", end=" ")
        recipe = fetch_recipe(cuisine, query, api_key)
        if recipe:
            recipes.append(recipe)
            print(f" {recipe['name']}")
        else:
            print("")
        time.sleep(0.5)  # Rate limit

    # Lưu JSON
    rec_json = os.path.join(data_dir, "recipes.json")
    with open(rec_json, 'w', encoding='utf-8') as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    print(f"\n Đã lưu {len(recipes)} công thức → {rec_json}")

    # Lưu CSV
    rec_csv = os.path.join(data_dir, "recipes.csv")
    with open(rec_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "name", "cuisine", "servings", "calories", "ingredients_json", "steps_json"
        ])
        writer.writeheader()
        for r in recipes:
            writer.writerow({
                "id": r["id"],
                "name": r["name"],
                "cuisine": r["cuisine"],
                "servings": r.get("servings", 4),
                "calories": r.get("calories", 0),
                "ingredients_json": json.dumps(r["ingredients"], ensure_ascii=False),
                "steps_json": json.dumps(r["steps"], ensure_ascii=False),
            })
    print(f" Đã lưu {len(recipes)} công thức → {rec_csv}")

    print(f"\n Hoàn tất! {len(ingredients)} nguyên liệu + {len(recipes)} công thức")
    return ingredients, recipes


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--api-key":
        print("Usage: python features/data_downloader.py --api-key YOUR_SPOONACULAR_KEY")
        sys.exit(1)
    download_all_data(api_key=sys.argv[2], data_dir="data")
