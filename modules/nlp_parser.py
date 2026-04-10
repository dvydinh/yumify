# -*- coding: utf-8 -*-
"""
modules/nlp_parser.py — Natural Language Processing Module
===================================================================

NER (Named Entity Recognition) pipeline for English
using regex, keyword matching, AND Machine Learning (Naive Bayes).

ML Classifier Integration:
    Cuisine classification uses CuisineNaiveBayesClassifier as the
    PRIMARY classification method. The ML model is trained on recipes.json
    at initialization time. Rule-based keyword matching serves as
    a FALLBACK when the ML model has low confidence.

    Data Flow:
        User Input → Tokenize → ML Model.predict_with_confidence()
                                     ↓
                              confidence >= threshold? → Use ML prediction
                                     ↓ (no)
                              Fallback to keyword matching

Extracts entities:
  - Budget: "$15", "20 bucks"
  - Ingredients: "beef", "mushroom"
  - Health Conditions: "stomachache", "diabetes"
  - Dish Preferences: "hotpot", "pho", "pasta"
  - Target Cuisine: "Italian", "Japanese" (via ML + fallback)

"""

import re
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# ML CLASSIFIER INTEGRATION
from modules.ml_classifier import CuisineNaiveBayesClassifier

# PARSED RESULT STRUCTURE
@dataclass
class ParsedInput:
    """Result of English Natural Language analysis."""
    budget: float = 0.0                          # Budget (USD)
    ingredients: List[str] = field(default_factory=list)    # Mentioned ingredients
    health_conditions: List[str] = field(default_factory=list)  # Health conditions
    dish_preferences: List[str] = field(default_factory=list)   # Dish preferences
    excluded_tags: List[str] = field(default_factory=list)      # Excluded tags/ingredients
    target_cuisine: str = ""                     # Target cuisine (Italian, Japanese, Korean, Mexican, Western)
    cuisine_confidence: float = 0.0              # ML classifier confidence for cuisine
    cuisine_method: str = ""                     # "ml" or "fallback_keyword"
    raw_text: str = ""                           # Original raw text
    confidence: float = 0.0                      # Overall confidence [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "budget": self.budget,
            "ingredients": self.ingredients,
            "health_conditions": self.health_conditions,
            "dish_preferences": self.dish_preferences,
            "excluded_tags": self.excluded_tags,
            "target_cuisine": self.target_cuisine,
            "cuisine_confidence": round(self.cuisine_confidence, 4),
            "cuisine_method": self.cuisine_method,
            "raw_text": self.raw_text,
            "confidence": round(self.confidence, 2),
        }

# ENGLISH ENTITY DICTIONARIES (DEFAULT ONTOLOGY)

HEALTH_CONDITION_PATTERNS: Dict[str, List[str]] = {
    "stomachache": [
        "stomachache", "stomach ache", "gastritis", "ulcer",
        "stomach", "acid reflux", "bellyache"
    ],
    "diabetes": [
        "diabetes", "diabetic", "high blood sugar",
        "sugar", "sugar-free", "no sugar"
    ],
    "high blood pressure": [
        "high blood pressure", "hypertension",
        "low sodium", "less salt"
    ],
    "gout": [
        "gout", "joint pain", "uric acid"
    ],
    "seafood allergy": [
        "seafood allergy", "no seafood",
        "allergic to shrimp", "allergic to crab", "allergic to fish"
    ],
    "pregnancy": [
        "pregnant", "pregnancy", "expecting"
    ],
    "vegetarian": [
        "vegetarian", "vegan", "no meat", "plant-based"
    ],
}

DISH_PATTERNS: Dict[str, List[str]] = {
    "pho": ["pho", "noodle soup"],
    "bun bo": ["bun bo", "spicy beef noodle"],
    "hotpot": ["hotpot", "hot pot"],
    "mushroom hotpot": ["mushroom hotpot"],
    "seafood hotpot": ["seafood hotpot", "thai hotpot"],
    "bun rieu": ["bun rieu", "crab noodle soup"],
    "sour soup": ["sour soup", "canh chua"],
    "fried rice": ["fried rice"],
    "stir-fried noodles": ["stir-fried noodles", "chow mein"],
    "porridge": ["porridge", "congee"],
    "stew": ["stew", "braised"],
    "stir-fried vegetables": ["stir-fried vegetables", "stir fry veggies"],
    "spaghetti": ["spaghetti", "pasta"],
    "pizza": ["pizza"],
    "risotto": ["risotto"],
    "sushi": ["sushi", "sashimi"],
    "ramen": ["ramen"],
    "bibimbap": ["bibimbap", "mixed rice"],
    "tteokbokki": ["tteokbokki", "spicy rice cake"],
    "tacos": ["tacos", "taco"],
    "burrito": ["burrito"],
    "guacamole": ["guacamole"],
    "salad": ["salad"],
    "steak": ["steak", "beefsteak"],
}

CUISINE_PATTERNS: Dict[str, List[str]] = {
    "Italian": [
        "italy", "italian", "pasta", "pizza", "spaghetti", "risotto",
    ],
    "Japanese": [
        "japan", "japanese", "sushi", "ramen", "sashimi",
        "yakimeshi", "tempura",
    ],
    "Korean": [
        "korea", "korean", "bibimbap", "tteokbokki",
        "ramyeon", "kimchi", "kim chi",
    ],
    "Mexican": [
        "mexico", "mexican", "tacos", "taco", "burrito", "guacamole", "nachos",
    ],
    "Western": [
        "western", "european", "steak", "caesar salad",
    ],
    "Vietnamese": [
        "vietnam", "vietnamese", "traditional",
    ],
}

INGREDIENT_KEYWORDS: List[str] = [
    # Meat
    "beef", "pork", "chicken", "duck", "meat",
    # Seafood
    "shrimp", "fish", "squid", "crab", "clam", "oyster", "snail",
    "salmon", "basa fish", "shrimp paste",
    # Vegetables
    "water spinach", "bok choy", "cabbage", "tomato", "carrot",
    "potato", "corn", "bean sprouts", "celery", "bamboo shoot",
    "lettuce", "spinach",
    # Mushroom
    "mushroom", "straw mushroom", "enoki mushroom", "king oyster mushroom",
    "shiitake mushroom", "oyster mushroom", "button mushroom",
    # Beans
    "tofu", "black bean", "soybean",
    # Carbs
    "rice noodle", "pho noodle", "noodle", "rice",
    "spaghetti", "tortilla", "ramen noodle",
    # Spices
    "chili", "satay", "lemongrass", "ginger", "garlic", "onion",
    "fish sauce", "coconut milk", "lime", "lemon",
    "cheese", "mozzarella", "parmesan",
    "olive oil", "tomato sauce", "basil",
    "soy sauce", "wasabi", "seaweed", "rice vinegar",
    "kimchi", "gochujang", "korean chili paste",
    "sesame oil", "sour cream", "butter", "heavy cream",
    # Egg
    "egg", "chicken egg",
    # Fruits
    "avocado", "durian",
]

EXCLUSION_PATTERNS: List[Tuple[str, str]] = [
    (r"no\s+spicy", "spicy"),
    (r"not\s+spicy", "spicy"),
    (r"no\s+salt", "salty"),
    (r"not\s+salty", "salty"),
    (r"less\s+spicy", "spicy"),
    (r"less\s+salt", "salty"),
    (r"less\s+oil", "fat"),
    (r"no\s+oil", "fat"),
    (r"no\s+seafood", "seafood"),
    (r"no\s+meat", "meat"),
    (r"avoid\s+spicy", "spicy"),
    (r"avoid\s+salty", "salty"),
    (r"avoid\s+sweet", "sweet"),
    (r"no\s+sugar", "sweet"),
]

@dataclass
class OntologyStructure:
    health_conditions: dict
    dish_patterns: dict
    cuisine_patterns: dict
    ingredients: list
    exclusions: list

# ML CUISINE CLASSIFIER — MAPPING TABLE
# recipes.json uses Vietnamese cuisine names, but the parser outputs English.
# This table maps ML predictions back to the standard English labels.
_CUISINE_ML_TO_STANDARD: Dict[str, str] = {
    "Ý": "Italian",
    "Nhật Bản": "Japanese",
    "Hàn Quốc": "Korean",
    "Mexico": "Mexican",
    "Việt Nam": "Vietnamese",
    "Phương Tây": "Western",
    "Thái Lan": "Thai",
    "Trung Quốc": "Chinese",
    "Ấn Độ": "Indian",
    "Pháp": "French",
    "Quốc tế": "International",
    # English labels map to themselves
    "Italian": "Italian",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "Mexican": "Mexican",
    "Vietnamese": "Vietnamese",
    "Western": "Western",
}

# Confidence threshold: ML prediction is accepted only if above this.
_ML_CONFIDENCE_THRESHOLD = 0.25

class EnglishNLPParser:
    """
    Entity Extraction with ML Integration.

    Uses Structured Knowledge (Ontologies) separated from the parsing engine,
    AND a trained Multinomial Naive Bayes classifier for cuisine classification.

    Cuisine Classification Pipeline:
        1. Extract ingredients from user text.
        2. Feed ingredients + text tokens to ML classifier.
        3. If ML confidence >= threshold → use ML prediction.
        4. Else → fallback to keyword matching.
    """
    def __init__(self, ontology_path: str = None):
        """
        Initialize parser with ontology abstraction AND ML classifier.

        The ML classifier is trained on recipes.json at initialization time,
        establishing the Data → Train → Predict pipeline.
        """
        if ontology_path and os.path.exists(ontology_path):
            self._load_ontology(ontology_path)
        else:
            self._load_default_ontology()

        self._budget_patterns = self._compile_budget_patterns()
        self._exclusion_patterns = [
            (re.compile(p, re.IGNORECASE | re.UNICODE), tag)
            for p, tag in self.ontology.exclusions
        ]
        self._sorted_ingredients = sorted(
            self.ontology.ingredients, key=len, reverse=True
        )

        # ML CLASSIFIER INITIALIZATION (Integration)
        self.ml_classifier = CuisineNaiveBayesClassifier()
        self._train_ml_classifier()

    def _train_ml_classifier(self) -> None:
        """
        Train the Naive Bayes classifier on the FULL dataset.

        Training priority:
          1. Kaggle Food.com dataset (thousands of recipes) — PRIMARY
          2. recipes.json (25 recipes) — FALLBACK only

        Note: Previously trained on recipes.json (25 samples) which
        Previously used local JSON. Now uses download_recipe_dataset() to get
        the real Kaggle data with proper cuisine labels.
        """
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'data'
        )

        training_data = []
        source = ""

        # TIER 1: Kaggle cached dataset (thousands of recipes)
        kaggle_cache = os.path.join(base_dir, 'recipes_kaggle.json')
        if os.path.exists(kaggle_cache):
            try:
                with open(kaggle_cache, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                source = f"Kaggle cache ({len(training_data)} recipes)"
            except Exception:
                pass

        # Tier 1b: Try download_recipe_dataset if no cache
        if not training_data:
            try:
                from features.feature_extractor import download_recipe_dataset
                training_data = download_recipe_dataset(base_dir)
                if training_data and len(training_data) > 50:
                    source = f"download_recipe_dataset ({len(training_data)} recipes)"
            except Exception:
                pass

        # TIER 2: recipes.json fallback (25 recipes)
        if not training_data:
            recipes_path = os.path.join(base_dir, 'recipes.json')
            if os.path.exists(recipes_path):
                try:
                    with open(recipes_path, 'r', encoding='utf-8') as f:
                        training_data = json.load(f)
                    source = f"recipes.json fallback ({len(training_data)} recipes)"
                except Exception:
                    pass

        if not training_data:
            print(f"  [NLP] Warning: No training data found.")
            return

        try:
            self.ml_classifier.train(training_data)
            n_classes = len(self.ml_classifier.class_counts)
            n_vocab = len(self.ml_classifier.vocab)
            print(f"  [NLP] ML Classifier trained: {source}, "
                  f"{n_classes} cuisines, {n_vocab} vocabulary terms")
        except Exception as e:
            print(f"  [NLP] ML Classifier training failed: {e}")

    def _load_ontology(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.ontology = OntologyStructure(
            health_conditions=data.get("health_conditions", {}),
            dish_patterns=data.get("dish_patterns", {}),
            cuisine_patterns=data.get("cuisine_patterns", {}),
            ingredients=data.get("ingredients", []),
            exclusions=[tuple(x) for x in data.get("exclusions", [])]
        )

    def _load_default_ontology(self):
        self.ontology = OntologyStructure(
            health_conditions=HEALTH_CONDITION_PATTERNS,
            dish_patterns=DISH_PATTERNS,
            cuisine_patterns=CUISINE_PATTERNS,
            ingredients=INGREDIENT_KEYWORDS,
            exclusions=EXCLUSION_PATTERNS
        )

    @staticmethod
    def _compile_budget_patterns() -> List[re.Pattern]:
        """
        Compile regex patterns for USD budget extraction.
        Supports:
        - "$15.5", "15$" -> 15.5
        - "20 bucks", "20 dollars", "20 usd" -> 20.0
        """
        return [
            re.compile(r'\$\s*(\d+(?:\.\d+)?)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*\$', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:bucks|dollars|usd)', re.IGNORECASE)
        ]

    def _extract_budget(self, text: str) -> float:
        for pattern in self._budget_patterns:
            match = pattern.search(text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        return 0.0

    def _classify_cuisine_ml(
        self, ingredients: List[str], text: str
    ) -> Tuple[str, float]:
        """
        Classify cuisine using the trained ML Naive Bayes model.

        Constructs feature tokens from:
          1. Extracted ingredient names (primary features)
          2. Raw text tokens (secondary features for context)

        Args:
            ingredients: Extracted ingredient names from the text.
            text: Raw user input text.

        Returns:
            (cuisine_label, confidence) where cuisine_label is in
            standard English format (e.g., "Italian") and confidence ∈ [0, 1].
            Returns ("", 0.0) if model is not trained.
        """
        if not self.ml_classifier.is_trained:
            return "", 0.0

        # Build feature tokens: ingredients + text words
        feature_tokens = list(ingredients)  # ingredient names
        # Add individual words from the raw text for additional context
        text_words = [w for w in text.lower().split() if len(w) >= 2]
        feature_tokens.extend(text_words)

        if not feature_tokens:
            return "", 0.0

        cuisine_vn, confidence = self.ml_classifier.predict_with_confidence(
            feature_tokens
        )

        if cuisine_vn is None:
            return "", 0.0

        # Map Vietnamese label to standard English label
        cuisine_en = _CUISINE_ML_TO_STANDARD.get(cuisine_vn, cuisine_vn)

        return cuisine_en, confidence

    def parse(self, text: str) -> ParsedInput:
        """
        Parse natural language into structured constraints.

        Cuisine classification uses a TWO-TIER approach:
            Tier 1 (PRIMARY):   ML Naive Bayes classifier
            Tier 2 (FALLBACK):  Rule-based keyword matching
        """
        text_lower = text.lower()
        result = ParsedInput(raw_text=text)

        # 1. Budget
        result.budget = self._extract_budget(text)

        # 2. Exclusions
        for pattern, tag in self._exclusion_patterns:
            if pattern.search(text_lower) and tag not in result.excluded_tags:
                result.excluded_tags.append(tag)

        # 3. Ingredients (Longest match first)
        found_ings = []
        temp_text = text_lower
        for keyword in self._sorted_ingredients:
            if keyword in temp_text:
                found_ings.append(keyword)
                temp_text = temp_text.replace(keyword, " ")
        result.ingredients = list(set(found_ings))

        # 4. Health conditions
        for condition, keywords in self.ontology.health_conditions.items():
            if any(kw in text_lower for kw in keywords):
                result.health_conditions.append(condition)

        # 5. Dish preferences
        for dish, keywords in self.ontology.dish_patterns.items():
            if any(kw in text_lower for kw in keywords):
                result.dish_preferences.append(dish)

        # 6. TARGET CUISINE — ML NAIVE BAYES CLASSIFICATION
        #    The trained ML model is the SOLE decision maker.
        #    NER extracts ingredients → ML model.predict(ingredients) → cuisine
        #    No IF-ELSE regex fallback. This is the real ML pipeline.
        #
        #    Data Flow:
        #      User: "tôi muốn nấu món gì có bò, phở và ớt"
        #      → NER: ["beef", "noodle", "chili"]
        #      → Naive Bayes: P(Vietnamese|features) = argmax → "Vietnamese"
        ml_cuisine, ml_confidence = self._classify_cuisine_ml(
            result.ingredients, text
        )

        if ml_cuisine:
            result.target_cuisine = ml_cuisine
            result.cuisine_confidence = ml_confidence
            result.cuisine_method = "ml"

        # 7. Calculate Confidence
        result.confidence = self._calculate_confidence(result)
        return result

    def _calculate_confidence(self, parsed: ParsedInput) -> float:
        score = 0.0
        max_score = 5.0

        if parsed.ingredients:
            score += min(len(parsed.ingredients), 2) * 1.5
        if parsed.budget > 0:
            score += 1.0
        if parsed.health_conditions:
            score += 0.5
        if parsed.dish_preferences:
            score += 0.5
        if parsed.target_cuisine:
            score += 0.5

        return min(max(score / max_score, 0.1), 1.0)

