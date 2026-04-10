# -*- coding: utf-8 -*-
"""
features/api_client.py — Unified API Client
================================================
Integrates 3 external APIs with JSON fallback for each endpoint.

APIs:
  1. Spoonacular — Search recipes, ingredients, prices, nutrition
  2. OpenWeatherMap — Current weather -> affects food suggestions
  3. HuggingFace Inference — Mistral-7B (text), SDXL (image)

Fallback: All APIs have local data fallbacks when:
  - API keys are not configured
  - Rate limits are exceeded
  - Network errors occur

"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# CONFIGURATION & CONSTANTS
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

@dataclass
class APIConfig:
    """API keys configuration."""
    spoonacular_key: str = ""
    openweathermap_key: str = ""
    huggingface_key: str = ""
    cache_enabled: bool = True

    def has_spoonacular(self) -> bool:
        return bool(self.spoonacular_key)

    def has_weather(self) -> bool:
        return bool(self.openweathermap_key)

    def has_huggingface(self) -> bool:
        return bool(self.huggingface_key)

# SIMPLE CACHE
_api_cache: Dict[str, Any] = {}

def _cache_get(key: str) -> Optional[Any]:
    return _api_cache.get(key)

def _cache_set(key: str, value: Any):
    _api_cache[key] = value

# SPOONACULAR API
SPOONACULAR_BASE = "https://api.spoonacular.com"

def spoonacular_search_recipes(
    query: str,
    cuisine: str = "",
    number: int = 5,
    config: Optional[APIConfig] = None
) -> List[Dict[str, Any]]:
    """Search recipes via Spoonacular API."""
    cache_key = f"spoon_search_{query}_{cuisine}_{number}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if config and config.has_spoonacular() and HAS_REQUESTS:
        try:
            params = {
                "apiKey": config.spoonacular_key,
                "query": query,
                "number": number,
                "addRecipeInformation": True,
                "fillIngredients": True,
            }
            if cuisine:
                params["cuisine"] = cuisine

            resp = requests.get(
                f"{SPOONACULAR_BASE}/recipes/complexSearch",
                params=params, timeout=10
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            _cache_set(cache_key, results)
            return results
        except Exception as e:
            print(f" Spoonacular search failed: {e}. Using fallback.")

    return _fallback_recipes(query, cuisine, number)

def spoonacular_get_recipe_info(
    recipe_id: int,
    config: Optional[APIConfig] = None
) -> Dict[str, Any]:
    """Get detailed recipe info from Spoonacular."""
    cache_key = f"spoon_recipe_{recipe_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if config and config.has_spoonacular() and HAS_REQUESTS:
        try:
            resp = requests.get(
                f"{SPOONACULAR_BASE}/recipes/{recipe_id}/information",
                params={"apiKey": config.spoonacular_key, "includeNutrition": True},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            _cache_set(cache_key, data)
            return data
        except Exception as e:
            print(f" Spoonacular recipe info failed: {e}. Using fallback.")

    return {}

def spoonacular_get_price(
    ingredients: List[str],
    config: Optional[APIConfig] = None
) -> Dict[str, float]:
    """Estimate ingredient prices (USD) via Spoonacular."""
    cache_key = f"spoon_price_{'_'.join(sorted(ingredients))}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    prices = {}

    if config and config.has_spoonacular() and HAS_REQUESTS:
        try:
            ingredient_list = "\\n".join(ingredients)
            resp = requests.post(
                f"{SPOONACULAR_BASE}/recipes/parseIngredients",
                params={"apiKey": config.spoonacular_key},
                data={"ingredientList": ingredient_list, "servings": 1},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data:
                name = item.get("name", "")
                cost = item.get("estimatedCost", {})
                usd_price = cost.get("value", 0) / 100  # cents to USD
                prices[name] = usd_price
            _cache_set(cache_key, prices)
            return prices
        except Exception as e:
            print(f" Spoonacular pricing failed: {e}. Using fallback.")

    return _fallback_prices(ingredients)

def spoonacular_get_ingredient_details(
    ingredients: List[str],
    config: Optional[APIConfig] = None
) -> Dict[str, Dict[str, float]]:
    """
    Estimate ingredient prices (USD) and calories via Spoonacular.
    Returns dict: {name: {'price': usd, 'calories': kcal}}
    """
    if not ingredients:
        return {}
        
    cache_key = f"spoon_details_{'_'.join(sorted(ingredients))}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    details = {}
    if config and config.has_spoonacular() and HAS_REQUESTS:
        try:
            ingredient_list = "\\n".join(ingredients)
            resp = requests.post(
                f"{SPOONACULAR_BASE}/recipes/parseIngredients",
                params={"apiKey": config.spoonacular_key, "includeNutrition": "true"},
                data={"ingredientList": ingredient_list, "servings": 1},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data:
                name = item.get("name", "")
                
                # Calculate price in USD
                cost = item.get("estimatedCost", {})
                usd_price = cost.get("value", 0) / 100  # cents to USD
                
                # Statistically Unbiased Calorie Fallback
                calories = None
                nutrition = item.get("nutrition", {})
                if nutrition:
                    nutrients = nutrition.get("nutrients", [])
                    for n in nutrients:
                        if n.get("name") == "Calories":
                            calories = n.get("amount")
                            break
                            
                if calories is None:
                    # Compute mean from DB
                    try:
                        ing_path = os.path.join(DATA_DIR, 'ingredients.json')
                        with open(ing_path, 'r', encoding='utf-8') as db_f:
                            db_data = json.load(db_f)
                            all_cals = [i.get("calories", 50) for i in db_data if "calories" in i]
                            calories = sum(all_cals) / len(all_cals) if all_cals else 100.0
                    except:
                        calories = 100.0
                            
                details[name] = {"price": usd_price, "calories": calories}
                
            _cache_set(cache_key, details)
            return details
        except Exception as e:
            print(f" Spoonacular details failed: {e}")

    # PURE GENERATIVE FALLBACK: Return empty domains for unresolvable ingredients 
    # to enforce strict tracking of True optimal search costs.
    _cache_set(cache_key, details)
    return details

# OPENWEATHERMAP API
WEATHER_BASE = "https://api.openweathermap.org/data/2.5"

WEATHER_FOOD_MAP = {
    "hot": {
        "suggest": ["salad", "sushi", "cold noodles", "smoothie"],
        "avoid": ["hotpot", "stew", "ramen"],
        "suggestion": "Hot weather — Suggest refreshing, light dishes"
    },
    "cold": {
        "suggest": ["hotpot", "ramen", "stew", "soup", "pho"],
        "avoid": ["salad", "cold dishes"],
        "suggestion": "Cold weather — Suggest warm, hearty dishes"
    },
    "rain": {
        "suggest": ["soup", "pho", "ramen", "porridge"],
        "avoid": [],
        "suggestion": "Rainy weather — Suggest hot soups"
    },
    "normal": {
        "suggest": [],
        "avoid": [],
        "suggestion": "Normal weather conditions"
    }
}

@dataclass
class WeatherInfo:
    """Weather information."""
    temperature: float = 28.0      # Celsius
    description: str = "clear sky"
    humidity: int = 70
    city: str = "Ho Chi Minh City"
    food_suggestion: str = ""
    category: str = "normal"       # hot, cold, rain, normal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": f"{self.temperature}°C",
            "description": self.description,
            "humidity": f"{self.humidity}%",
            "city": self.city,
            "suggestion": self.food_suggestion,
        }

def get_weather(
    city: str = "Ho Chi Minh City",
    config: Optional[APIConfig] = None
) -> WeatherInfo:
    """Get current weather from OpenWeatherMap."""
    cache_key = f"weather_{city}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    info = WeatherInfo(city=city)

    if config and config.has_weather() and HAS_REQUESTS:
        try:
            resp = requests.get(
                f"{WEATHER_BASE}/weather",
                params={
                    "q": city,
                    "appid": config.openweathermap_key,
                    "units": "metric",
                    "lang": "en"
                },
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            info.temperature = data["main"]["temp"]
            info.description = data["weather"][0].get("description", "")
            info.humidity = data["main"].get("humidity", 70)
        except Exception as e:
            print(f" Weather API failed: {e}. Using default values.")

    if info.temperature >= 33:
        info.category = "hot"
    elif info.temperature <= 20:
        info.category = "cold"
    elif "rain" in info.description.lower():
        info.category = "rain"
    else:
        info.category = "normal"

    weather_map = WEATHER_FOOD_MAP.get(info.category, WEATHER_FOOD_MAP["normal"])
    info.food_suggestion = weather_map["suggestion"]

    _cache_set(cache_key, info)
    return info

# HUGGINGFACE INFERENCE API
HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_TEXT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HF_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

def hf_generate_text(
    prompt: str,
    config: Optional[APIConfig] = None,
    max_tokens: int = 500
) -> str:
    """Generate text via HuggingFace Serverless Inference API (Mistral-7B)."""
    if config and config.has_huggingface() and HAS_REQUESTS:
        try:
            headers = {"Authorization": f"Bearer {config.huggingface_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
            resp = requests.post(
                f"{HF_API_BASE}/{HF_TEXT_MODEL}",
                headers=headers, json=payload, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "")
        except Exception as e:
            print(f" HuggingFace text gen failed: {e}")

    return _fallback_recipe_text(prompt)

def hf_generate_image(
    prompt: str,
    config: Optional[APIConfig] = None
) -> Optional[bytes]:
    """Generate recipe image via HuggingFace Serverless Inference API (SDXL)."""
    if config and config.has_huggingface() and HAS_REQUESTS:
        try:
            headers = {"Authorization": f"Bearer {config.huggingface_key}"}
            payload = {"inputs": prompt}
            resp = requests.post(
                f"{HF_API_BASE}/{HF_IMAGE_MODEL}",
                headers=headers, json=payload, timeout=120
            )
            resp.raise_for_status()
            if resp.headers.get("content-type", "").startswith("image"):
                return resp.content
        except Exception as e:
            print(f" HuggingFace image gen failed: {e}")

    return None

# FALLBACK DATA

def _fallback_recipes(query: str, cuisine: str, number: int) -> List[Dict[str, Any]]:
    """Fallback: load recipes from local JSON."""
    try:
        recipes_path = os.path.join(DATA_DIR, 'recipes.json')
        if os.path.exists(recipes_path):
            with open(recipes_path, 'r', encoding='utf-8') as f:
                all_recipes = json.load(f)

            if cuisine:
                filtered = [r for r in all_recipes if r.get("cuisine", "").lower() == cuisine.lower()]
                if filtered:
                    all_recipes = filtered

            query_lower = query.lower()
            scored = []
            for r in all_recipes:
                score = 0
                name = r.get("name", "").lower()
                tags = [t.lower() for t in r.get("tags", [])]
                if query_lower in name:
                    score += 10
                for word in query_lower.split():
                    if word in name:
                        score += 3
                    if word in " ".join(tags):
                        score += 2
                scored.append((score, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [r for _, r in scored[:number]]
    except Exception:
        pass
    return []

def _fallback_prices(ingredients: List[str]) -> Dict[str, float]:
    """Fallback: lookup prices directly from JSON DB without hardcoding."""
    prices = {}
    try:
        ing_path = os.path.join(DATA_DIR, 'ingredients.json')
        if os.path.exists(ing_path):
            with open(ing_path, 'r', encoding='utf-8') as f:
                db = json.load(f)
            name_map = {item["name"].lower(): item["price"] for item in db}
            for ing in ingredients:
                if ing.lower() in name_map:
                    prices[ing] = name_map[ing.lower()]
    except Exception:
        pass
    return prices

def _fallback_recipe_text(prompt: str) -> str:
    """Fallback text if offline."""
    return (
        " Recipe generated by AI system (offline mode):\n\n"
        "For an optimal generative experience, please configure the HuggingFace API key.\n\n"
        "Currently using local database fallback."
    )

# FORMATTING UTILS
def format_usd(amount: float) -> str:
    """Format USD currency beautifully."""
    return f"${amount:,.2f}"

if __name__ == "__main__":
    config = APIConfig()
    recipes = spoonacular_search_recipes("pasta", "Italian", 3, config)
    print(f"Recipes found: {len(recipes)}")
    prices = spoonacular_get_price(["beef", "tomato"], config)
    print(f"Prices: {prices}")
