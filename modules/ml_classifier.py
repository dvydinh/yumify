# -*- coding: utf-8 -*-
"""
modules/ml_classifier.py — Multinomial Naive Bayes Cuisine Classifier
=====================================================================
AI Pillar: Machine Learning (L.O.2.2)
[CS188] Chapter 20: Statistical Learning — Naive Bayes

Implements a Multinomial Naive Bayes classifier for cuisine classification
based on ingredient word features. This module is ACTIVELY INTEGRATED
into the NLP parsing pipeline (nlp_parser.py) to classify user intent.

Data Flow:  User Input → NLP Tokenize → ML Model.predict() → Cuisine Decision

Laplace Smoothing (Add-1):
    P(word | class) = (count(word, class) + 1) / (N_class + |V|)

Tác giả: Nhóm Sinh Viên NMAI
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict


class CuisineNaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier for cuisine classification.

    This classifier is trained on recipe data (ingredients → cuisine label)
    and used in the NLP pipeline to classify user queries into cuisine
    categories. The decision is based on P(Cuisine | words) computed
    via Bayes' Rule with Laplace smoothing.

    Integration Point:
        nlp_parser.py calls predict_with_confidence() during parse(),
        using the ML prediction as the PRIMARY cuisine classifier.
        Rule-based keyword matching serves only as a FALLBACK.
    """

    def __init__(self):
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.vocab: set = set()
        self.word_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_docs: int = 0
        self.is_trained: bool = False

    def train(self, recipes: List[Dict[str, Any]]) -> None:
        """
        Train the Naive Bayes model from a list of recipes.

        Each recipe must have 'cuisine' (label) and 'ingredients' (features).
        Words are extracted from ingredient names via whitespace tokenization.

        Args:
            recipes: List of recipe dicts with 'cuisine' and 'ingredients' keys.
        """
        # Reset state for re-training
        self.class_counts = defaultdict(int)
        self.vocab = set()
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_docs = 0

        for r in recipes:
            cuisine = r.get("cuisine", "Khác")
            self.class_counts[cuisine] += 1
            self.total_docs += 1

            for ing in r.get("ingredients", []):
                # preprocess: lowercase and split words
                name = ing.get("name", "").lower() if isinstance(ing, dict) else str(ing).lower()
                for word in name.split():
                    if len(word) >= 2:  # Skip single-char tokens
                        self.vocab.add(word)
                        self.word_counts[cuisine][word] += 1

        self.is_trained = True

    def _compute_log_posteriors(self, words: List[str]) -> Dict[str, float]:
        """
        Compute log P(Cuisine | words) for all cuisine classes.

        Uses log-space arithmetic to prevent underflow:
            log P(C|W) ∝ log P(C) + Σ log P(wᵢ|C)

        Laplace smoothing ensures no zero probabilities.

        Args:
            words: Tokenized input words (lowercased).

        Returns:
            Dict mapping cuisine → log posterior (unnormalized).
        """
        vocab_size = len(self.vocab)
        log_posteriors = {}

        for cuisine, count in self.class_counts.items():
            # Prior: log P(Cuisine)
            log_prob = math.log(count / self.total_docs)

            # Total words in this class
            total_words_in_class = sum(self.word_counts[cuisine].values())

            # Likelihood: Σ log P(word | Cuisine) with Laplace smoothing
            for word in words:
                count_word = self.word_counts[cuisine].get(word, 0)
                prob_word = (count_word + 1) / (total_words_in_class + vocab_size)
                log_prob += math.log(prob_word)

            log_posteriors[cuisine] = log_prob

        return log_posteriors

    def predict(self, ingredients_names: List[str]) -> Optional[str]:
        """
        Predict cuisine for a set of ingredient names.

        Args:
            ingredients_names: List of ingredient name strings.

        Returns:
            Predicted cuisine label, or None if not trained.
        """
        if not self.is_trained:
            return None

        words = []
        for name in ingredients_names:
            words.extend(name.lower().split())

        if not words:
            return None

        log_posteriors = self._compute_log_posteriors(words)

        if not log_posteriors:
            return None

        return max(log_posteriors, key=log_posteriors.get)

    def predict_with_confidence(
        self, ingredients_names: List[str]
    ) -> Tuple[Optional[str], float]:
        """
        Predict cuisine with a confidence score (normalized posterior).

        This is the PRIMARY method called by nlp_parser.py for cuisine
        classification in the integrated pipeline.

        Confidence is computed as the normalized posterior probability:
            P(C_best | words) = exp(log_best) / Σ exp(log_Cᵢ)
        using the log-sum-exp trick for numerical stability.

        Args:
            ingredients_names: List of ingredient name strings or raw
                               text tokens from user input.

        Returns:
            (predicted_cuisine, confidence) where confidence ∈ [0, 1].
            Returns (None, 0.0) if model is not trained or input is empty.
        """
        if not self.is_trained:
            return None, 0.0

        words = []
        for name in ingredients_names:
            words.extend(name.lower().split())

        if not words:
            return None, 0.0

        log_posteriors = self._compute_log_posteriors(words)

        if not log_posteriors:
            return None, 0.0

        # Find best class
        best_cuisine = max(log_posteriors, key=log_posteriors.get)
        best_log = log_posteriors[best_cuisine]

        # Normalize using log-sum-exp for numerical stability
        max_log = max(log_posteriors.values())
        log_sum = max_log + math.log(
            sum(math.exp(lp - max_log) for lp in log_posteriors.values())
        )
        confidence = math.exp(best_log - log_sum)

        return best_cuisine, confidence

    def predict_top_k(
        self, ingredients_names: List[str], k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Return top-k cuisine predictions with confidence scores.

        Args:
            ingredients_names: List of ingredient name strings.
            k: Number of top predictions to return.

        Returns:
            List of (cuisine, confidence) tuples sorted by confidence desc.
        """
        if not self.is_trained:
            return []

        words = []
        for name in ingredients_names:
            words.extend(name.lower().split())

        if not words:
            return []

        log_posteriors = self._compute_log_posteriors(words)

        if not log_posteriors:
            return []

        # Normalize all posteriors
        max_log = max(log_posteriors.values())
        log_sum = max_log + math.log(
            sum(math.exp(lp - max_log) for lp in log_posteriors.values())
        )

        results = [
            (cuisine, math.exp(lp - log_sum))
            for cuisine, lp in log_posteriors.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def evaluate(self, test_recipes: List[Dict[str, Any]]) -> float:
        """Evaluate accuracy on a test set."""
        if not self.is_trained or not test_recipes:
            return 0.0

        correct = 0
        for r in test_recipes:
            true_cuisine = r.get("cuisine", "")
            ing_names = [
                ing.get("name", "") if isinstance(ing, dict) else str(ing)
                for ing in r.get("ingredients", [])
            ]
            pred_cuisine = self.predict(ing_names)
            if pred_cuisine == true_cuisine:
                correct += 1

        return correct / len(test_recipes)
