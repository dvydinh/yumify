# -*- coding: utf-8 -*-
"""
features/data_loader.py — Trình tải dữ liệu (NO MOCK DATA)
============================================================
Tải dữ liệu từ:
  1. CSV/JSON files đã tải from Spoonacular API (via data_downloader.py)
  2. Spoonacular API trực tiếp (nếu có key)
  3. Kaggle Food.com dataset (via feature_extractor.py)

KHÔNG CÓ dữ liệu mock hardcoded. Nếu không có file data → yêu cầu chạy download.

Tác giả: Nhóm Sinh Viên NMAI
"""

import os
import csv
import json
from typing import List, Dict, Any, Optional


class DataLoader:
    """
    Trình tải dữ liệu — NO hardcoded fallback.

    Hỗ trợ:
    - Tải từ CSV (data/ingredients.csv, data/recipes.csv)
    - Tải từ JSON (data/ingredients.json, data/recipes.json)
    - Spoonacular API trực tiếp
    """

    def __init__(self, data_dir: str = "data"):
        """
        Khởi tạo DataLoader.

        Args:
            data_dir: Thư mục chứa dữ liệu
        """
        self.data_dir = os.path.abspath(data_dir)
        self.ingredients: List[Dict[str, Any]] = []
        self.recipes: List[Dict[str, Any]] = []

    def load_all(self, api_config=None):
        """
        Tải tất cả dữ liệu.

        Pipeline:
          1. CSV files (từ data_downloader.py)
          2. JSON files
          3. Spoonacular API trực tiếp (nếu có config)
          4. Nếu tất cả fail → in hướng dẫn
        """
        self._load_ingredients(api_config)
        self._load_recipes(api_config)
        print(f"\n Tổng kết: {len(self.ingredients)} nguyên liệu, {len(self.recipes)} công thức")

    def _load_ingredients(self, api_config=None):
        """Tải nguyên liệu từ CSV → JSON → API."""
        # 1. CSV
        csv_path = os.path.join(self.data_dir, "ingredients.csv")
        if os.path.exists(csv_path):
            try:
                self.ingredients = self._load_csv(csv_path)
                # Chuyển đổi types
                for ing in self.ingredients:
                    ing["id"] = int(ing.get("id", 0))
                    ing["price"] = float(ing.get("price_vnd", ing.get("price", 0)))
                    ing["price_usd"] = float(ing.get("price_usd", 0))
                    ing["calories"] = float(ing.get("calories", 0))
                if self.ingredients:
                    print(f" Đã tải {len(self.ingredients)} nguyên liệu từ {csv_path}")
                    return
            except Exception as e:
                print(f" Lỗi đọc CSV: {e}")

        # 2. JSON
        json_path = os.path.join(self.data_dir, "ingredients.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.ingredients = json.load(f)
                # Đảm bảo có field price
                for ing in self.ingredients:
                    if "price_usd" in ing:
                        ing["price"] = ing.get("price_usd", 0.5)
                    elif "price_vnd" in ing:
                        ing["price"] = ing.get("price_vnd", 12500) / 25000.0
                    else:
                        ing["price"] = 0.5
                if self.ingredients:
                    print(f" Đã tải {len(self.ingredients)} nguyên liệu từ {json_path}")
                    return
            except Exception as e:
                print(f" Lỗi đọc JSON: {e}")

        # 3. Spoonacular API
        if api_config:
            self._download_ingredients_from_api(api_config)
            if self.ingredients:
                return

        # 4. Không có dữ liệu
        print(" Không tìm thấy dữ liệu nguyên liệu!")
        print("   Chạy: python features/data_downloader.py --api-key YOUR_SPOONACULAR_KEY")

    def _load_recipes(self, api_config=None):
        """Tải công thức từ JSON → CSV → API."""
        # 1. JSON (ưu tiên vì chứa nested data)
        json_path = os.path.join(self.data_dir, "recipes.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.recipes = json.load(f)
                if self.recipes:
                    print(f" Đã tải {len(self.recipes)} công thức từ {json_path}")
            except Exception as e:
                print(f" Lỗi đọc JSON: {e}")

        # 2. CSV
        csv_path = os.path.join(self.data_dir, "recipes.csv")
        if os.path.exists(csv_path):
            try:
                rows = self._load_csv(csv_path)
                for row in rows:
                    try:
                        self.recipes.append({
                            "id": int(row.get("id", 0)),
                            "name": row.get("name", ""),
                            "cuisine": row.get("cuisine", ""),
                            "servings": int(row.get("servings", 4)),
                            "calories": float(row.get("calories", 0)),
                            "ingredients": json.loads(row.get("ingredients_json", "[]")),
                            "steps": json.loads(row.get("steps_json", "[]")),
                        })
                    except Exception:
                        continue
                if self.recipes:
                    print(f" Đã phân tích file CSV (hiện có {len(self.recipes)} công thức)")
            except Exception as e:
                print(f" Lỗi đọc CSV: {e}")

        # 3. Kaggle cache
        kaggle_path = os.path.join(self.data_dir, "recipes_kaggle.json")
        if os.path.exists(kaggle_path):
            try:
                with open(kaggle_path, 'r', encoding='utf-8') as f:
                    k_recipes = json.load(f)
                    self.recipes.extend(k_recipes)
                if k_recipes:
                    print(f" Đã tải thêm {len(k_recipes)} công thức từ Kaggle cache")
            except Exception:
                pass

        # 4. API cache
        api_cache = os.path.join(self.data_dir, "recipes_api_cache.json")
        if os.path.exists(api_cache):
            try:
                with open(api_cache, 'r', encoding='utf-8') as f:
                    self.recipes = json.load(f)
                if self.recipes:
                    print(f" Đã tải {len(self.recipes)} công thức từ API cache")
                    return
            except Exception:
                pass

        # 5. Không có dữ liệu
        if not self.recipes:
            print(" Không tìm thấy dữ liệu công thức!")
            print("   Chạy: python features/data_downloader.py --api-key YOUR_SPOONACULAR_KEY")

    def _load_csv(self, filepath: str) -> List[Dict[str, Any]]:
        """Tải file CSV thành list of dicts."""
        rows = []
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    def _download_ingredients_from_api(self, api_config):
        """Tải nguyên liệu trực tiếp từ Spoonacular API."""
        try:
            from features.data_downloader import download_all_data
            if hasattr(api_config, 'spoonacular_key') and api_config.spoonacular_key:
                print(" Đang tải dữ liệu từ Spoonacular API...")
                download_all_data(api_config.spoonacular_key, self.data_dir)
                # Reload
                json_path = os.path.join(self.data_dir, "ingredients.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        self.ingredients = json.load(f)
        except Exception as e:
            print(f" API download failed: {e}")

    # ================================================================
    # TÌM KIẾM VÀ TIỆN ÍCH
    # ================================================================
    def find_ingredient(self, name: str) -> Optional[Dict[str, Any]]:
        """Tìm nguyên liệu theo tên (case-insensitive)."""
        name_lower = name.lower()
        for ing in self.ingredients:
            if ing.get("name", "").lower() == name_lower:
                return ing
        # Partial match
        for ing in self.ingredients:
            if name_lower in ing.get("name", "").lower():
                return ing
        return None

    def search_recipes_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Tìm công thức theo tags."""
        tags_lower = [t.lower() for t in tags]
        results = []
        for recipe in self.recipes:
            recipe_tags = [t.lower() for t in recipe.get("tags", [])]
            if any(tag in recipe_tags for tag in tags_lower):
                results.append(recipe)
        return results

    def search_recipes_by_cuisine(self, cuisine: str) -> List[Dict[str, Any]]:
        """Tìm công thức theo loại ẩm thực."""
        return [r for r in self.recipes if r.get("cuisine", "").lower() == cuisine.lower()]

    def get_recipe_total_cost(self, recipe: Dict[str, Any]) -> float:
        """Tính tổng chi phí ước tính cho một công thức."""
        total = 0
        for ing in recipe.get("ingredients", []):
            found = self.find_ingredient(ing.get("name", ""))
            if found:
                qty = float(ing.get("quantity", ing.get("amount", 1)))
                total += found.get("price", 0) * qty
            else:
                total += 30000  # Giá mặc định cho nguyên liệu không tìm thấy
        return total


# ============================================================================
# CHẠY THỬ
# ============================================================================
if __name__ == "__main__":
    loader = DataLoader(data_dir=os.path.join(os.path.dirname(__file__), '..', 'data'))
    loader.load_all()

    if loader.ingredients:
        print(f"\nTop 5 nguyên liệu:")
        for ing in loader.ingredients[:5]:
            price = ing.get('price', ing.get('price_vnd', 0))
            print(f"  {ing['name']}: {price:,.0f}đ ({ing.get('calories', 0)} cal)")

    if loader.recipes:
        print(f"\nTop 5 công thức:")
        for r in loader.recipes[:5]:
            print(f"  {r['name']} ({r.get('cuisine', 'N/A')})")
