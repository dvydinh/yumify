# -*- coding: utf-8 -*-
"""
features/feature_extractor.py — TF-IDF Feature Extraction (ML Pillar)
=====================================================================
Trụ cột AI: Machine Learning

Tải dataset công thức thực tế và trích xuất đặc trưng TF-IDF.
Hỗ trợ:
  - Tải dataset từ Spoonacular API hoặc URL công khai
  - Xây dựng TF-IDF vocabulary từ nguyên liệu và hướng dẫn
  - Cosine similarity cho recipe matching
  - Lưu/tải embeddings (.npy hoặc JSON)
  - Lọc theo ẩm thực (cuisine filtering)

Fallback: Sử dụng dữ liệu cục bộ khi API không khả dụng.

Tác giả: Nhóm Sinh Viên NMAI
"""

import os
import json
import math
import re
import csv
import subprocess
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from features.api_client import (
        spoonacular_search_recipes, APIConfig,
        CUISINE_VN_TO_EN, translate_ingredient
    )
    HAS_API = True
except ImportError:
    HAS_API = False

# URL dataset công khai — Food.com Recipes (Kaggle)
# Dataset gốc: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
# File chính: RAW_recipes.csv (~60MB, 230k+ công thức)
KAGGLE_DATASET_SLUG = "shuyangli94/food-com-recipes-and-user-interactions"
KAGGLE_CSV_FILENAME = "RAW_recipes.csv"


# ============================================================================
# CẤU TRÚC DỮ LIỆU
# ============================================================================

@dataclass
class SimilarityMatch:
    """Kết quả so khớp TF-IDF."""
    recipe: Dict[str, Any]
    similarity: float
    rank: int


# ============================================================================
# TẢI DATASET THỰC TẾ
# ============================================================================

def _download_kaggle_csv(data_dir: str) -> Optional[str]:
    """
    Tải dataset Food.com từ Kaggle.

    Ưu tiên sử dụng kagglehub (có sẵn trên Google Colab, không cần auth
    cho public datasets). Fallback: wget/curl từ mirror nếu kagglehub fail.

    Dataset: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

    Returns:
        Đường dẫn file CSV nếu thành công, None nếu thất bại.
    """
    csv_path = os.path.join(data_dir, "recipes_kaggle.csv")
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
        print(f" Dataset Kaggle đã có: {csv_path}")
        return csv_path

    os.makedirs(data_dir, exist_ok=True)

    # === Phương pháp 1: kagglehub (có sẵn trên Colab) ===
    try:
        import kagglehub
        print(f" Đang tải dataset từ Kaggle ({KAGGLE_DATASET_SLUG})...")
        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
        # Tìm file RAW_recipes.csv trong thư mục đã tải
        raw_csv = os.path.join(dataset_path, KAGGLE_CSV_FILENAME)
        if os.path.exists(raw_csv):
            import shutil
            shutil.copy2(raw_csv, csv_path)
            print(f" Đã tải dataset Kaggle Food.com: {csv_path}")
            return csv_path
        # Tìm đệ quy nếu cấu trúc thư mục khác
        for root, dirs, files in os.walk(dataset_path):
            for fname in files:
                if fname.lower() == KAGGLE_CSV_FILENAME.lower():
                    import shutil
                    shutil.copy2(os.path.join(root, fname), csv_path)
                    print(f" Đã tải dataset Kaggle Food.com: {csv_path}")
                    return csv_path
        print(" kagglehub: không tìm thấy RAW_recipes.csv")
    except ImportError:
        print(" kagglehub chưa cài (sẽ thử wget)...")
    except Exception as e:
        print(f" kagglehub lỗi: {e}")

    # === Phương pháp 2: wget/curl từ mirror ===
    mirror_url = f"https://raw.githubusercontent.com/dvydinh/yumify/main/data/recipes_kaggle.csv"
    for cmd in [
        ["wget", "-q", "-O", csv_path, mirror_url],
        ["curl", "-sL", "-o", csv_path, mirror_url],
    ]:
        try:
            subprocess.run(cmd, check=True, timeout=30, capture_output=True)
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
                print(f" Đã tải dataset từ mirror: {csv_path}")
                return csv_path
        except Exception:
            continue

    return None


# ============================================================================
# HEURISTIC QUANTITY ESTIMATION
# ============================================================================
# Kaggle Food.com CSV does not include ingredient weights.
# Instead of hardcoding quantity=1 (which makes all AI calorie/cost
# calculations meaningless), we estimate realistic default quantities
# based on ingredient type, using a category-keyword mapping derived
# from ingredients.json.
#
# Reference: USDA Standard Portion Sizes
#   - Meat/Poultry: ~200g per serving
#   - Seafood: ~150g per serving
#   - Vegetables/Produce: ~100g per serving
#   - Grains/Carbs: ~150g per serving
#   - Dairy/Cheese: ~50g per serving
#   - Spices/Seasonings: ~5g per serving
#   - Oils/Liquids: ~15ml per serving
#   - Default (unknown): ~80g

_QUANTITY_CATEGORY_MAP = {
    # Meat & Poultry (200g default)
    "meat": [
        "beef", "pork", "chicken", "duck", "lamb", "turkey", "veal",
        "bacon", "sausage", "ham", "steak", "ground", "mince", "chuck",
    ],
    # Seafood (150g default)
    "seafood": [
        "shrimp", "fish", "salmon", "tuna", "crab", "squid", "clam",
        "oyster", "lobster", "scallop", "anchov", "prawn", "mussel",
    ],
    # Vegetables & Produce (100g default)
    "produce": [
        "tomato", "onion", "garlic", "carrot", "potato", "pepper",
        "mushroom", "spinach", "broccoli", "cabbage", "lettuce",
        "celery", "cucumber", "corn", "bean", "pea", "zucchini",
        "eggplant", "avocado", "lemon", "lime", "ginger",
    ],
    # Grains & Carbs (150g default)
    "grain": [
        "rice", "pasta", "noodle", "bread", "flour", "tortilla",
        "couscous", "quinoa", "oat", "cereal", "fettuccine", "penne",
        "spaghetti", "udon", "ramen",
    ],
    # Dairy (50g default)
    "dairy": [
        "cheese", "mozzarella", "parmesan", "cream", "butter", "milk",
        "yogurt", "sour cream", "creme", "gruyère",
    ],
    # Spices & Seasonings (5g default)
    "spice": [
        "salt", "pepper", "cumin", "paprika", "turmeric", "cinnamon",
        "oregano", "basil", "thyme", "rosemary", "bay", "chili",
        "cardamom", "anise", "mustard", "wasabi", "chilli",
        "coriander", "mint", "parsley", "dill", "sage",
    ],
    # Sauces & Liquids (15ml default)
    "sauce": [
        "oil", "vinegar", "sauce", "soy", "wine", "sake", "broth",
        "stock", "water", "juice", "honey", "syrup", "ketchup",
    ],
    # Sugars (30g default)
    "sugar": [
        "sugar", "brown sugar", "caster", "icing", "molasses",
    ],
    # Eggs (1 unit ≈ 60g)
    "egg": [
        "egg",
    ],
}

_QUANTITY_DEFAULTS = {
    "meat": 200,
    "seafood": 150,
    "produce": 100,
    "grain": 150,
    "dairy": 50,
    "spice": 5,
    "sauce": 15,
    "sugar": 30,
    "egg": 60,
}


def _estimate_ingredient_quantity(ingredient_name: str) -> int:
    """
    Estimate a realistic default quantity (grams) for an ingredient.

    Uses keyword matching against a category map derived from
    ingredients.json categories and USDA standard portion sizes.

    Args:
        ingredient_name: Ingredient name string from CSV.

    Returns:
        Estimated quantity in grams.
    """
    name_lower = ingredient_name.lower()

    for category, keywords in _QUANTITY_CATEGORY_MAP.items():
        for keyword in keywords:
            if keyword in name_lower:
                return _QUANTITY_DEFAULTS[category]

    # Default for unknown ingredients
    return 80


def _parse_kaggle_csv(csv_path: str, max_recipes: int = 5000) -> List[Dict[str, Any]]:
    """
    Parse Food.com CSV thành danh sách recipe dicts.

    Cột CSV (RAW_recipes.csv):
        name, id, minutes, contributor_id, submitted, tags, nutrition, n_steps,
        steps, description, ingredients, n_ingredients
    """
    recipes = []
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_recipes:
                    break
                try:
                    # Parse tags (list dạng string)
                    tags_str = row.get('tags', '[]')
                    tags = [t.strip("' ") for t in tags_str.strip('[]').split(',')][:5]

                    # Parse ingredients — with HEURISTIC QUANTITY ESTIMATION
                    # Instead of hardcoded quantity=1 (which makes all AI
                    # calorie/cost calculations meaningless), we estimate
                    # realistic quantities based on ingredient category.
                    ing_str = row.get('ingredients', '[]')
                    ing_list = [i.strip("' ") for i in ing_str.strip('[]').split(',')]
                    ingredients = [
                        {
                            "name": name.strip(),
                            "quantity": _estimate_ingredient_quantity(name.strip()),
                            "unit": "g"
                        }
                        for name in ing_list[:10] if name.strip()
                    ]

                    # Parse steps
                    steps_str = row.get('steps', '[]')
                    steps = [s.strip("' ") for s in steps_str.strip('[]').split("',")][:8]

                    # Parse nutrition [calories, fat, sugar, sodium, protein, sat_fat, carbs]
                    nutr_str = row.get('nutrition', '[]')
                    nutr_vals = [float(x.strip()) for x in nutr_str.strip('[]').split(',')]
                    calories = nutr_vals[0] if nutr_vals else 0

                    # Detect cuisine from tags
                    cuisine = _detect_cuisine_from_tags(tags)

                    recipes.append({
                        "id": int(row.get('id', i)),
                        "name": row.get('name', f'Recipe {i}'),
                        "cuisine": cuisine,
                        "tags": tags,
                        "ingredients": ingredients,
                        "steps": steps,
                        "calories": calories,
                        "servings": 4,
                        "source": "kaggle_foodcom"
                    })
                except Exception:
                    continue

        print(f" Parsed {len(recipes)} recipes từ Kaggle CSV")
    except Exception as e:
        print(f" Lỗi parse CSV: {e}")

    return recipes


def _detect_cuisine_from_tags(tags: List[str]) -> str:
    """Nhận diện cuisine từ tags dataset."""
    tags_lower = [t.lower() for t in tags]
    cuisine_map = {
        "italian": "Ý", "japanese": "Nhật Bản", "korean": "Hàn Quốc",
        "mexican": "Mexico", "vietnamese": "Việt Nam", "thai": "Thái Lan",
        "chinese": "Trung Quốc", "indian": "Ấn Độ", "french": "Pháp",
        "european": "Phương Tây", "american": "Phương Tây",
        "mediterranean": "Phương Tây", "greek": "Phương Tây",
    }
    for tag in tags_lower:
        for key, value in cuisine_map.items():
            if key in tag:
                return value
    return "Quốc tế"


def download_recipe_dataset(
    data_dir: str = "data",
    config: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Tải dataset công thức thực tế.

    Pipeline tải (4 lớp):
      0. Kaggle Food.com CSV (wget/gdown) — DATASET THỰC TẾ
      1. Spoonacular API (nếu có key)
      2. File JSON cục bộ (data/recipes.json)
      3. API cache

    Args:
        data_dir: Thư mục lưu dữ liệu
        config: APIConfig (optional)

    Returns:
        Danh sách công thức
    """
    recipes = []

    # === Bước 0: Tải Kaggle Food.com dataset (REAL DATA) ===
    kaggle_cache = os.path.join(data_dir, "recipes_kaggle.json")
    if os.path.exists(kaggle_cache):
        try:
            with open(kaggle_cache, 'r', encoding='utf-8') as f:
                recipes = json.load(f)
            print(f" Đã tải {len(recipes)} recipes từ Kaggle cache")
            return recipes
        except Exception:
            pass

    csv_path = _download_kaggle_csv(data_dir)
    if csv_path:
        recipes = _parse_kaggle_csv(csv_path, max_recipes=5000)
        if recipes:
            try:
                with open(kaggle_cache, 'w', encoding='utf-8') as f:
                    json.dump(recipes, f, ensure_ascii=False, indent=2)
                print(f" Đã cache {len(recipes)} Kaggle recipes → {kaggle_cache}")
            except Exception:
                pass
            return recipes

    # === Bước 1: Spoonacular API ===
    if HAS_API and config and hasattr(config, 'has_spoonacular') and config.has_spoonacular():
        print(" Đang tải recipes từ Spoonacular API...")
        cuisines = ["italian", "japanese", "korean", "mexican", "vietnamese"]
        for cuisine in cuisines:
            try:
                results = spoonacular_search_recipes(
                    query="popular", cuisine=cuisine, number=5, config=config
                )
                for r in results:
                    recipes.append({
                        "id": r.get("id", 0),
                        "name": r.get("title", ""),
                        "cuisine": cuisine.capitalize(),
                        "image": r.get("image", ""),
                        "tags": [],
                        "ingredients": [
                            {"name": ing.get("name", ""), "quantity": 1, "unit": "phần"}
                            for ing in r.get("extendedIngredients", [])[:8]
                        ],
                        "steps": [r.get("summary", "")][:5],
                        "source": "spoonacular"
                    })
            except Exception as e:
                print(f"   {cuisine}: {e}")
        if recipes:
            cache_path = os.path.join(data_dir, "recipes_api_cache.json")
            try:
                os.makedirs(data_dir, exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(recipes, f, ensure_ascii=False, indent=2)
                print(f" Đã cache {len(recipes)} recipes từ API")
            except Exception:
                pass
            return recipes

    # === Bước 2: File JSON cục bộ ===
    json_path = os.path.join(data_dir, 'recipes.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                recipes = json.load(f)
            print(f" Đã tải {len(recipes)} recipes từ {json_path}")
            return recipes
        except Exception as e:
            print(f" Lỗi đọc {json_path}: {e}")

    # === Bước 3: API cache ===
    cache_path = os.path.join(data_dir, "recipes_api_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                recipes = json.load(f)
            print(f" Đã tải {len(recipes)} recipes từ API cache")
            return recipes
        except Exception:
            pass

    print(" Không tìm thấy dataset. Vui lòng kiểm tra kết nối mạng.")
    return []


# ============================================================================
# BỘ TRÍCH XUẤT ĐẶC TRƯNG TF-IDF
# ============================================================================
class RecipeFeatureExtractor:
    """
    Bộ trích xuất đặc trưng TF-IDF cho công thức nấu ăn.

    Pipeline:
        1. Tokenize: tách từ tiếng Việt (hỗ trợ Unicode)
        2. Build vocabulary: từ tất cả recipes
        3. Compute TF-IDF: cho mỗi recipe document
        4. Cosine similarity: so khớp truy vấn

    Công thức:
        TF(t,d) = f(t,d) / max{f(t',d)}
        IDF(t) = log(N / |{d ∈ D : t ∈ d}|)
        TF-IDF(t,d) = TF(t,d) × IDF(t)
        cos(q,d) = (q·d) / (||q|| × ||d||)
    """

    def __init__(self):
        """Khởi tạo extractor."""
        self.vocabulary: Dict[str, int] = {}   # word → index
        self.idf: Dict[str, float] = {}        # word → IDF score
        self.tfidf_matrix: List[List[float]] = []
        self.recipes: List[Dict[str, Any]] = []
        self._stopwords = {
            "và", "của", "cho", "các", "với", "trong", "một", "có",
            "là", "được", "sẵn", "để", "lên", "ra", "vào", "rồi",
            "bước", "thêm", "khi", "nếu", "thì", "này", "đó",
            "the", "a", "an", "and", "or", "of", "to", "in", "for",
            "with", "on", "at", "by", "from", "up", "as", "into",
        }

    def _tokenize(self, text: str) -> List[str]:
        """
        Tách từ tiếng Việt và tiếng Anh.

        - Chuyển lowercase
        - Giữ chữ cái Unicode (tiếng Việt)
        - Loại stopwords
        - Loại từ quá ngắn (< 2 ký tự)
        """
        text = text.lower()
        tokens = re.findall(r'[a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+', text)
        return [t for t in tokens if len(t) >= 2 and t not in self._stopwords]

    def _recipe_to_text(self, recipe: Dict[str, Any]) -> str:
        """Chuyển recipe thành văn bản phẳng cho TF-IDF."""
        parts = [recipe.get("name", "")]
        for ing in recipe.get("ingredients", []):
            if isinstance(ing, dict):
                parts.append(ing.get("name", ""))
            elif isinstance(ing, str):
                parts.append(ing)
        parts.extend(recipe.get("tags", []))
        parts.append(recipe.get("cuisine", ""))
        for step in recipe.get("steps", []):
            parts.append(step)
        return " ".join(parts)

    def fit(self, recipes: List[Dict[str, Any]]):
        """
        Xây dựng TF-IDF vocabulary và ma trận từ danh sách recipes.

        Args:
            recipes: Danh sách công thức
        """
        self.recipes = recipes
        N = len(recipes)
        if N == 0:
            return

        # Bước 1: Tokenize tất cả documents
        docs = []
        for recipe in recipes:
            text = self._recipe_to_text(recipe)
            tokens = self._tokenize(text)
            docs.append(tokens)

        # Bước 2: Build vocabulary + IDF
        doc_freq: Dict[str, int] = defaultdict(int)
        all_words = set()

        for tokens in docs:
            unique_tokens = set(tokens)
            for word in unique_tokens:
                doc_freq[word] += 1
            all_words.update(unique_tokens)

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.idf = {}
        for word in all_words:
            self.idf[word] = math.log(N / (doc_freq[word] + 1)) + 1  # smoothed IDF

        # Bước 3: Compute TF-IDF matrix
        self.tfidf_matrix = []
        for tokens in docs:
            tf_counts = Counter(tokens)
            max_tf = max(tf_counts.values()) if tf_counts else 1
            vector = [0.0] * len(self.vocabulary)
            for word, count in tf_counts.items():
                if word in self.vocabulary:
                    tf = count / max_tf
                    vector[self.vocabulary[word]] = tf * self.idf.get(word, 1.0)
            self.tfidf_matrix.append(vector)

        print(f" TF-IDF: {len(self.vocabulary)} features, {N} documents")

    def _vectorize_query(self, query: str) -> List[float]:
        """Chuyển truy vấn thành TF-IDF vector."""
        tokens = self._tokenize(query)
        tf_counts = Counter(tokens)
        max_tf = max(tf_counts.values()) if tf_counts else 1
        vector = [0.0] * len(self.vocabulary)
        for word, count in tf_counts.items():
            if word in self.vocabulary:
                tf = count / max_tf
                vector[self.vocabulary[word]] = tf * self.idf.get(word, 1.0)
        return vector

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Tính cosine similarity giữa 2 vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def find_similar(
        self,
        query: str,
        top_k: int = 5,
        cuisine_filter: Optional[str] = None
    ) -> List[SimilarityMatch]:
        """
        Tìm công thức tương tự nhất với truy vấn.

        Args:
            query: Truy vấn người dùng (tiếng Việt)
            top_k: Số kết quả trả về
            cuisine_filter: Lọc theo ẩm thực (optional)

        Returns:
            Danh sách SimilarityMatch sắp xếp theo similarity giảm dần
        """
        if not self.tfidf_matrix:
            return []

        query_vec = self._vectorize_query(query)

        scores = []
        for idx, doc_vec in enumerate(self.tfidf_matrix):
            recipe = self.recipes[idx]
            if cuisine_filter:
                r_cuisine = recipe.get("cuisine", "")
                if r_cuisine != cuisine_filter:
                    continue
            sim = self._cosine_similarity(query_vec, doc_vec)
            scores.append((sim, idx))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (sim, idx) in enumerate(scores[:top_k], 1):
            results.append(SimilarityMatch(
                recipe=self.recipes[idx],
                similarity=sim,
                rank=rank
            ))
        return results

    def save_embeddings(self, filepath: str):
        """Lưu TF-IDF embeddings ra file."""
        if HAS_NUMPY:
            np_path = filepath if filepath.endswith('.npy') else filepath + '.npy'
            np.save(np_path, np.array(self.tfidf_matrix))
            # Lưu vocabulary song song
            vocab_path = np_path.replace('.npy', '_vocab.json')
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(self.vocabulary, f, ensure_ascii=False)
            print(f" Đã lưu embeddings: {np_path} ({len(self.tfidf_matrix)} docs)")
        else:
            json_path = filepath if filepath.endswith('.json') else filepath + '.json'
            data = {
                "vocabulary": self.vocabulary,
                "idf": self.idf,
                "matrix": self.tfidf_matrix,
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            print(f" Đã lưu embeddings (JSON): {json_path}")

    def load_embeddings(self, filepath: str) -> bool:
        """Tải TF-IDF embeddings từ file."""
        if HAS_NUMPY:
            np_path = filepath if filepath.endswith('.npy') else filepath + '.npy'
            if os.path.exists(np_path):
                self.tfidf_matrix = np.load(np_path).tolist()
                vocab_path = np_path.replace('.npy', '_vocab.json')
                if os.path.exists(vocab_path):
                    with open(vocab_path, 'r', encoding='utf-8') as f:
                        self.vocabulary = json.load(f)
                print(f" Đã tải embeddings từ {np_path}")
                return True

        json_path = filepath if filepath.endswith('.json') else filepath + '.json'
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.vocabulary = data.get("vocabulary", {})
            self.idf = data.get("idf", {})
            self.tfidf_matrix = data.get("matrix", [])
            print(f" Đã tải embeddings từ {json_path}")
            return True

        return False


# ============================================================================
# CHẠY THỬ
# ============================================================================
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    recipes = download_recipe_dataset(data_dir)

    if recipes:
        ext = RecipeFeatureExtractor()
        ext.fit(recipes)

        # Test similarity search
        test_queries = [
            "phở bò thịt bò",
            "pasta cheese italian",
            "sushi cá hồi Nhật",
            "kimchi tương ớt Hàn Quốc",
        ]
        for query in test_queries:
            matches = ext.find_similar(query, top_k=3)
            print(f"\n Query: '{query}'")
            for m in matches:
                print(f"   {m.rank}. {m.recipe['name']} (sim={m.similarity:.3f})")

        # Lưu embeddings
        emb_path = os.path.join(data_dir, "tfidf_embeddings")
        ext.save_embeddings(emb_path)
    else:
        print(" Không có dữ liệu để xử lý.")
