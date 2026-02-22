# -*- coding: utf-8 -*-
"""
modules/ml_classifier.py
========================
Module Học máy (Machine Learning) - Yêu cầu (E) của bài tập lớn.
Sử dụng thuật toán Multinomial Naive Bayes để phân loại ẩm thực (Cuisine)
dựa trên tập hợp các nguyên liệu của công thức.
"""

import math
from collections import defaultdict

class CuisineNaiveBayesClassifier:
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.vocab = set()
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_docs = 0
        self.is_trained = False

    def train(self, recipes):
        """
        Huấn luyện mô hình Naive Bayes phân loại cuisine từ list recipes.
        Mỗi recipe là dict có 'cuisine' và 'ingredients'.
        """
        for r in recipes:
            cuisine = r.get("cuisine", "Khác")
            self.class_counts[cuisine] += 1
            self.total_docs += 1
            
            for ing in r.get("ingredients", []):
                # preprocess: lowercase and split words
                name = ing.get("name", "").lower()
                for word in name.split():
                    self.vocab.add(word)
                    self.word_counts[cuisine][word] += 1
                    
        self.is_trained = True

    def predict(self, ingredients_names):
        """
        Dự đoán Cuisine cho một tập các nguyên liệu đầu vào.
        Sử dụng Laplace smoothing (tránh xác suất 0).
        """
        if not self.is_trained:
            return "Chưa huấn luyện"
            
        best_cuisine = None
        max_log_prob = -float('inf')
        
        words = []
        for name in ingredients_names:
            words.extend(name.lower().split())
            
        vocab_size = len(self.vocab)
        
        for cuisine, count in self.class_counts.items():
            # Xác suất tiên nghiệm (Prior) P(Cuisine)
            log_prob = math.log(count / self.total_docs)
            
            # Tổng số từ trong cuisine này
            total_words_in_class = sum(self.word_counts[cuisine].values())
            
            # Xác suất hậu nghiệm (Likelihood) P(word | Cuisine)
            for word in words:
                # Laplace smoothing: (count + 1) / (N + V)
                count_word = self.word_counts[cuisine].get(word, 0)
                prob_word = (count_word + 1) / (total_words_in_class + vocab_size)
                log_prob += math.log(prob_word)
                
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_cuisine = cuisine
                
        return best_cuisine

    def evaluate(self, test_recipes):
        """Đánh giá độ chính xác (Accuracy) trên tập test."""
        if not self.is_trained or not test_recipes:
            return 0.0
            
        correct = 0
        for r in test_recipes:
            true_cuisine = r.get("cuisine", "")
            ing_names = [ing.get("name", "") for ing in r.get("ingredients", [])]
            pred_cuisine = self.predict(ing_names)
            if pred_cuisine == true_cuisine:
                correct += 1
                
        return correct / len(test_recipes)
