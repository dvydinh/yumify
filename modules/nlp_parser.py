# -*- coding: utf-8 -*-
"""
modules/nlp_parser.py — Natural Language Processing Module
===================================================================
AI Pillar: Machine Learning (ML) & Knowledge Representation
[CS188] Chapter 8: Structured Knowledge & Ontologies

Implements an NER (Named Entity Recognition) pipeline for English
using regex and keyword matching.

Extracts entities:
  - Budget: "$15", "20 bucks"
  - Ingredients: "beef", "mushroom"
  - Health Conditions: "stomachache", "diabetes"
  - Dish Preferences: "hotpot", "pho", "pasta"
  - Target Cuisine: "Italian", "Japanese"

Tác giả: Nhóm Sinh Viên NMAI
"""

import re
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# ============================================================================
# PARSED RESULT STRUCTURE
# ============================================================================
@dataclass
class ParsedInput:
    """Result of English Natural Language analysis."""
    budget: float = 0.0                          # Budget (USD)
    ingredients: List[str] = field(default_factory=list)    # Mentioned ingredients
    health_conditions: List[str] = field(default_factory=list)  # Health conditions
    dish_preferences: List[str] = field(default_factory=list)   # Dish preferences
    excluded_tags: List[str] = field(default_factory=list)      # Excluded tags/ingredients
    target_cuisine: str = ""                     # Target cuisine (Italian, Japanese, Korean, Mexican, Western)
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
            "raw_text": self.raw_text,
            "confidence": round(self.confidence, 2),
        }

# ============================================================================
# ENGLISH ENTITY DICTIONARIES (DEFAULT ONTOLOGY)
# ============================================================================

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

class EnglishNLPParser:
    """
    [CS188] Ontology-based Entity Extraction.
    Uses Structured Knowledge (Ontologies) separated from the parsing engine.
    """
    def __init__(self, ontology_path: str = None):
        """
        [CS188] Initialize parser with ontology abstraction.
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

    def parse(self, text: str) -> ParsedInput:
        """Parse natural language into structured constraints."""
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

        # 6. Target cuisine
        for cuisine, keywords in self.ontology.cuisine_patterns.items():
            if any(kw in text_lower for kw in keywords):
                result.target_cuisine = cuisine
                break

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
