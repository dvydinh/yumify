# -*- coding: utf-8 -*-
"""
modules/bayes_risk.py — Expert Bayesian Network and Expected Utility
====================================================================
AI Pillar: Bayesian Network & Expected Utility Theory (L.O.2.2)
[CS188] Chapter 13-16: Quantifying Uncertainty & Making Simple Decisions

Implements an Expert-driven Bayesian Network using Subjectivist
Probabilities and Expected Utility Theory.

Tác giả: Nhóm Sinh Viên NMAI
"""

import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================================
# CẤU TRÚC DỮ LIỆU
# ============================================================================

@dataclass
class BayesResult:
    """
    Kết quả đánh giá từ Expert-driven Bayesian Network.

    Attributes:
        preference_score: P(Like | Evidence) ∈ [0, 1] — Posterior probability
        risk_score: P(DigestiveRisk | Ingredients) ∈ [0, 1]
        risk_level: Discretized risk: "thấp", "trung bình", "cao"
        preference_label: Discretized preference label
        feature_contributions: Đóng góp của từng Chance Node vào posterior
        risk_factors: Chi tiết các yếu tố rủi ro phát hiện được
        explanation: Giải thích kết quả bằng ngôn ngữ tự nhiên
    """
    preference_score: float = 0.0
    risk_score: float = 0.0
    risk_level: str = "low"
    preference_label: str = "medium"
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preference_probability": round(self.preference_score, 4),
            "preference_label": self.preference_label,
            "risk_probability": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "feature_contributions": {
                k: round(v, 4) for k, v in self.feature_contributions.items()
            },
            "risk_factors": self.risk_factors,
            "explanation": self.explanation,
        }


# ============================================================================
# CONDITIONAL PROBABILITY TABLES (CPTs)
# Expert-elicited Subjectivist Probabilities
# ============================================================================

# P(feature | Like) — Likelihood table
# Xác suất có điều kiện: quan sát thấy feature khi người dùng THÍCH món
# Được gán bởi chuyên gia dinh dưỡng dựa trên khảo sát ẩm thực Việt Nam
EXPERT_CPT_LIKE: Dict[str, float] = {
    # Food Categories — P(category | Like)
    "meat": 0.75,
    "seafood": 0.65,
    "mushroom": 0.70,
    "vegetable": 0.60,
    "bean": 0.55,
    "carb": 0.80,
    "spice": 0.85,

    # Cuisine Style Nodes — P(style | Like)
    "traditional": 0.85,
    "fast": 0.70,
    "healthy": 0.65,
    "vegetarian": 0.50,
    "spicy": 0.60,
    "non-spicy": 0.70,
    "light": 0.60,
    "easy to digest": 0.75,
    "nutritious": 0.70,
    "cheap": 0.75,
    "classic": 0.80,
    "luxury": 0.72,
    "fresh": 0.68,
    "warm": 0.65,

    # Cuisine Origin Nodes — P(origin | Like)
    "italian": 0.72,
    "japanese": 0.70,
    "korean": 0.68,
    "mexican": 0.65,
    "western": 0.67,

    # Price Range Nodes — P(price_bucket | Like)
    "cheap_price": 0.80,
    "medium_price": 0.75,
    "high_price": 0.50,

    # Health Compatibility Node
    "health_compatible": 0.90,
    "health_incompatible": 0.20,
}

# P(feature | ¬Like) — Complement CPT
# Xác suất quan sát feature khi người dùng KHÔNG THÍCH món
EXPERT_CPT_NOT_LIKE: Dict[str, float] = {
    "meat": 0.50, "seafood": 0.40, "mushroom": 0.35, "vegetable": 0.45,
    "bean": 0.40, "carb": 0.55, "spice": 0.60,
    "traditional": 0.40, "fast": 0.50, "healthy": 0.35, "vegetarian": 0.30,
    "spicy": 0.45, "non-spicy": 0.50, "light": 0.40, "easy to digest": 0.45,
    "nutritious": 0.40, "cheap": 0.55, "classic": 0.45, "luxury": 0.38,
    "fresh": 0.42, "warm": 0.40,
    "italian": 0.42, "japanese": 0.40, "korean": 0.38, "mexican": 0.35, "western": 0.40,
    "cheap_price": 0.60, "medium_price": 0.55, "high_price": 0.40,
    "health_compatible": 0.50, "health_incompatible": 0.60,
}

# Prior: P(Like) — Xác suất tiên nghiệm
# Reflects base rate of user satisfaction (expert estimate)
PRIOR_LIKE = 0.6
PRIOR_NOT_LIKE = 0.4


# ============================================================================
# DIGESTIVE RISK CPTs (Expert-elicited Risk Factors)
# ============================================================================

# P(Risk | Ingredient/Tag) — Conditional risk probabilities
# Mỗi entry biểu diễn một Chance Node trong Bayesian Network cho rủi ro
DIGESTIVE_RISK_CPT: Dict[str, Dict[str, Any]] = {
    # English keys to match Kaggle dataset ingredient names
    "spicy":          {"prior_risk": 0.35, "description": "Capsaicin irritates gastric mucosa"},
    "chili":          {"prior_risk": 0.40, "description": "High-concentration capsaicin irritant"},
    "hot pepper":     {"prior_risk": 0.40, "description": "Capsaicin causes mucosal inflammation"},
    "chili paste":    {"prior_risk": 0.45, "description": "Oil-chili mix difficult to digest"},
    "hot sauce":      {"prior_risk": 0.30, "description": "Moderate capsaicin in sauce form"},
    "sriracha":       {"prior_risk": 0.30, "description": "Fermented chili sauce with garlic"},
    "cayenne":        {"prior_risk": 0.38, "description": "Concentrated capsaicin powder"},
    "jalapeno":       {"prior_risk": 0.32, "description": "Moderate capsaicin content"},
    "seafood":        {"prior_risk": 0.25, "description": "Risk of marine protein allergy"},
    "shrimp":         {"prior_risk": 0.28, "description": "Common shellfish allergen"},
    "oil":            {"prior_risk": 0.20, "description": "Fat slows gut motility"},
    "butter":         {"prior_risk": 0.18, "description": "Saturated fat causes bloating"},
    "cream":          {"prior_risk": 0.20, "description": "High fat dairy difficult to digest"},
    "fat":            {"prior_risk": 0.20, "description": "Heavy fat content causes bloating"},
    "coconut milk":   {"prior_risk": 0.25, "description": "Saturated fat from coconut"},
    "coconut cream":  {"prior_risk": 0.25, "description": "High saturated fat content"},
    "gluten":         {"prior_risk": 0.20, "description": "Gluten intolerance trigger"},
    "wheat":          {"prior_risk": 0.18, "description": "Contains gluten proteins"},
    "milk":           {"prior_risk": 0.18, "description": "Lactose intolerance trigger"},
    "cheese":         {"prior_risk": 0.15, "description": "Lactose and casein content"},
    "gochujang":      {"prior_risk": 0.35, "description": "Korean chili paste, highly spicy"},
    "wasabi":         {"prior_risk": 0.30, "description": "Isothiocyanate mucosal irritant"},
    "curry":          {"prior_risk": 0.25, "description": "Spice blend can irritate stomach"},
    "pepper":         {"prior_risk": 0.15, "description": "Piperine mild gastric irritant"},
}

# ============================================================================
# SYNERGY RISK CPT — Hidden Causes for Noisy-OR (replaces illegal multipliers)
# ============================================================================
# When multiple risk factors co-occur, their SYNERGY creates an additional
# hidden cause in the Noisy-OR model. Instead of multiplying P(Risk) by a
# constant (which violates Kolmogorov), we treat the synergy as an
# INDEPENDENT hidden cause with its own P(Risk_synergy).
#
# The combined Noisy-OR then becomes:
#   P(Risk) = 1 - ∏(1 - P_i) × ∏(1 - P_synergy_j)
#
# This naturally satisfies P ∈ [0, 1] without any min() clamping.
# Reference: Henrion (1989), "Some Practical Issues in Constructing
#            Belief Networks", UAI'89.
SYNERGY_RISK_CPT: List[Dict[str, Any]] = [
    {
        "factors": {"spicy", "oil"},
        "synergy_risk": 0.40,
        "reason": "Capsaicin + oil causes dual mucosal irritation"
    },
    {
        "factors": {"chili", "oil"},
        "synergy_risk": 0.45,
        "reason": "Chili + oil increases reflux risk"
    },
    {
        "factors": {"chili paste", "oil"},
        "synergy_risk": 0.50,
        "reason": "Chili paste contains both capsaicin and fat"
    },
    {
        "factors": {"seafood", "spicy"},
        "synergy_risk": 0.35,
        "reason": "Spicy seafood heightens allergic response risk"
    },
    {
        "factors": {"seafood", "chili"},
        "synergy_risk": 0.38,
        "reason": "Seafood + capsaicin compounds digestive irritation"
    },
    {
        "factors": {"milk", "spicy"},
        "synergy_risk": 0.30,
        "reason": "Casein + capsaicin hard to break down in stomach"
    },
    {
        "factors": {"cream", "chili"},
        "synergy_risk": 0.32,
        "reason": "Dairy fat + capsaicin causes intestinal cramping"
    },
    {
        "factors": {"butter", "spicy"},
        "synergy_risk": 0.30,
        "reason": "Saturated fat + spice amplifies gastric distress"
    },
]

# ============================================================================
# HEALTH-CONDITIONAL RISK CPT — P(Risk | factor, health_condition)
# ============================================================================
# Instead of multiplying prior_risk * 1.5 (which can exceed 1.0 or violate
# probability axioms), we define explicit conditional probabilities.
# P(Risk | factor, condition) is directly specified by domain experts.
HEALTH_CONDITIONAL_RISK_CPT: Dict[str, Dict[str, float]] = {
    # P(Risk | factor, stomachache / gastritis)
    "stomachache": {
        "spicy": 0.70, "chili": 0.75, "chili paste": 0.80,
        "hot sauce": 0.55, "cayenne": 0.72, "jalapeno": 0.60,
        "oil": 0.45, "fat": 0.45, "butter": 0.40,
        "gochujang": 0.65, "wasabi": 0.60, "curry": 0.50,
    },
    "đau dạ dày": {
        "spicy": 0.70, "chili": 0.75, "chili paste": 0.80,
        "oil": 0.45, "fat": 0.45,
        "gochujang": 0.65, "wasabi": 0.60,
    },
    # P(Risk | factor, gout)
    "gout": {
        "seafood": 0.70, "shrimp": 0.72,
    },
    # P(Risk | factor, gluten intolerance)
    "gluten intolerance": {
        "gluten": 0.85, "wheat": 0.80,
    },
    # P(Risk | factor, lactose intolerance)
    "lactose intolerance": {
        "milk": 0.80, "cheese": 0.70, "cream": 0.75,
    },
    # P(Risk | factor, diabetes)
    "diabetes": {
        "sugar": 0.60, "honey": 0.50,
    },
}


# ============================================================================
# BAYESIAN RECIPE EVALUATOR
# Expert-driven Bayesian Network + Expected Utility
# ============================================================================

class BayesianRecipeEvaluator:
    """
    Bộ đánh giá công thức ăn sử dụng Expert-driven Bayesian Network
    và Expected Utility Theory.

    Mô-đun này KHÔNG phải là Machine Learning. Các xác suất được gán
    (assigned) bởi chuyên gia dinh dưỡng theo phương pháp Expert
    Elicitation (Subjectivist Probability — de Finetti), không phải
    được học (learned) từ tập dữ liệu huấn luyện.

    Mô hình Bayesian Network:

        [FoodCategory]──┐
        [CuisineStyle]──┤
        [PriceRange]────┼──→ [UserPreference] ──→ [ExpectedUtility]
        [HealthMatch]───┘                              ↑
                                                       │
        [SpicyLevel]────┐                              │
        [FatContent]────┼──→ [DigestiveRisk] ──────────┘
        [Allergens]─────┘

    Hai chức năng chính:

    1. Posterior Preference Inference:
       P(Like | e₁, e₂, ..., eₙ) sử dụng Bayes' Rule trong log-space

    2. Risk Assessment qua Extended Noisy-OR Model:
       P(Risk) = 1 - ∏(1 - P(Riskᵢ)) × ∏(1 - P(Synergy_j))
       Synergy hidden causes model conditional dependence.
    """

    def __init__(self):
        """Khởi tạo Bayesian Network với Expert-elicited CPTs."""
        self.cpt_like = EXPERT_CPT_LIKE
        self.cpt_not_like = EXPERT_CPT_NOT_LIKE
        self.prior_like = PRIOR_LIKE
        self.prior_not_like = PRIOR_NOT_LIKE
        self.risk_cpt = DIGESTIVE_RISK_CPT
        self.synergy_cpt = SYNERGY_RISK_CPT
        self.health_risk_cpt = HEALTH_CONDITIONAL_RISK_CPT

    # ------------------------------------------------------------------
    # Feature Extraction (Evidence Nodes)
    # ------------------------------------------------------------------
    def _extract_evidence(
        self,
        recipe: Dict[str, Any],
        total_cost: float,
        budget: float,
        health_match: bool,
        user_preferred_tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        Trích xuất Evidence Nodes (quan sát) từ công thức.

        Chuyển đổi thông tin thô của recipe thành tập các biến
        quan sát (observed variables) trong Bayesian Network.
        Mỗi evidence node tương ứng với một entry trong CPT.

        Args:
            recipe: Công thức nấu ăn (dict)
            total_cost: Tổng chi phí (VNĐ)
            budget: Ngân sách tối đa (VNĐ)
            health_match: Có phù hợp điều kiện sức khỏe không
            user_preferred_tags: Tags người dùng ưa thích

        Returns:
            Danh sách evidence node IDs có trong CPTs
        """
        evidence = []

        # 1. Tags → Style/Category evidence nodes
        for tag in recipe.get("tags", []):
            tag_lower = tag.lower()
            if tag_lower in self.cpt_like:
                evidence.append(tag_lower)

        # 2. Ingredient categories → Food Category evidence nodes
        ingredient_categories = {
            "meat": ["meat", "bò", "heo", "gà", "lợn", "vịt", "beef", "pork", "chicken"],
            "seafood": ["tôm", "cá", "mực", "cua", "sò", "shrimp", "fish", "salmon"],
            "vegetable": ["vegetable", "salad", "cải", "bắp cải", "cà chua", "spinach", "broccoli"],
            "mushroom": ["mushroom", "mushroom"],
            "carb": ["cơm", "bún", "phở", "mì", "pasta", "rice", "noodle", "bread"],
        }

        recipe_ings = " ".join(
            i.get("name", "") for i in recipe.get("ingredients", [])
        ).lower()

        for category, keywords in ingredient_categories.items():
            if any(kw in recipe_ings for kw in keywords):
                if category not in evidence:
                    evidence.append(category)

        # 3. Price Range → Price Node
        if total_cost > 0 and budget > 0:
            ratio = total_cost / budget
            if ratio < 0.4:
                evidence.append("cheap_price")
            elif ratio < 0.8:
                evidence.append("medium_price")
            else:
                evidence.append("high_price")

        # 4. Health match → Health Compatibility Node
        evidence.append("health_compatible" if health_match else "health_incompatible")

        # 5. User preferred tags
        if user_preferred_tags:
            for tag in user_preferred_tags:
                tag_lower = tag.lower()
                if tag_lower in self.cpt_like and tag_lower not in evidence:
                    evidence.append(tag_lower)

        return evidence

    # ------------------------------------------------------------------
    # Posterior Preference Inference (Bayes' Rule in Log-space)
    # ------------------------------------------------------------------
    def predict_preference(
        self,
        recipe: Dict[str, Any],
        total_cost: float = 0,
        budget: float = 200000,
        health_match: bool = True,
        user_preferred_tags: Optional[List[str]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Tính Posterior Probability P(Like | Evidence) qua Bayes' Rule.

        Áp dụng Conditional Independence Assumption (tương tự Naive Bayes
        nhưng với Expert-assigned CPTs thay vì data-learned parameters):

            P(Like | e₁,...,eₙ) ∝ P(Like) × ∏ᵢ P(eᵢ | Like)

        Sử dụng log-space arithmetic để tránh floating-point underflow:
            log P(Like | E) = log P(Like) + Σ log P(eᵢ | Like) - log P(E)

        Default probability (Laplace-style prior) = 0.5 được sử dụng
        cho evidence nodes không có trong CPT.

        Args:
            recipe: Công thức nấu ăn
            total_cost: Tổng chi phí
            budget: Ngân sách
            health_match: Phù hợp sức khỏe
            user_preferred_tags: Tags ưa thích

        Returns:
            (posterior_probability, feature_contributions)
        """
        evidence = self._extract_evidence(
            recipe, total_cost, budget, health_match, user_preferred_tags
        )

        if not evidence:
            return self.prior_like, {}

        # Log-space computation to prevent numerical underflow
        # log P(Like | E) ∝ log P(Like) + Σ log P(eᵢ | Like)
        log_like = math.log(self.prior_like)
        log_not_like = math.log(self.prior_not_like)
        contributions = {}

        for e in evidence:
            # CPT lookup with default prior of 0.5 (maximum entropy default)
            p_e_given_like = self.cpt_like.get(e, 0.5)
            p_e_given_not_like = self.cpt_not_like.get(e, 0.5)

            log_like += math.log(max(p_e_given_like, 1e-10))
            log_not_like += math.log(max(p_e_given_not_like, 1e-10))

            # Track individual contribution (log-likelihood ratio)
            contributions[e] = p_e_given_like

        # Normalize using log-sum-exp trick for numerical stability
        max_log = max(log_like, log_not_like)
        log_sum = max_log + math.log(
            math.exp(log_like - max_log) + math.exp(log_not_like - max_log)
        )

        posterior = math.exp(log_like - log_sum)
        posterior = max(0.0, min(1.0, posterior))

        return posterior, contributions

    # ------------------------------------------------------------------
    # Digestive Risk Assessment (Independent Risk Model)
    # ------------------------------------------------------------------
    def assess_risk(
        self,
        recipe: Dict[str, Any],
        health_conditions: Optional[List[str]] = None,
        ingredients_db: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Đánh giá rủi ro tiêu hóa sử dụng Extended Noisy-OR Model.

        Mô hình xác suất (Noisy-OR mở rộng với Synergy Hidden Causes):

        Bước 1 — Independent Noisy-OR:
            survival_product = ∏ᵢ (1 - P(Riskᵢ))
            Mỗi yếu tố rủi ro i là một nguyên nhân độc lập.

        Bước 2 — Synergy Hidden Causes (Conditional Dependence):
            Khi 2 yếu tố (A, B) xuất hiện đồng thời, sự hiệp đồng (synergy)
            của chúng được mô hình hóa như một nguyên nhân ẩn (hidden cause)
            với xác suất độc lập P_synergy:
                survival_product *= (1 - P_synergy)

            Điều này tương đương với việc thêm một node cha
            mới vào Bayesian Network:
                [Spicy]──┐   [Dầu]───┐   [Synergy(Spicy∩Dầu)]─┐
                       └───┴────────┴──────────────────┘
                              [DigestiveRisk]

        Kết quả: P(Risk) = 1 - survival_product
        Luôn hội tụ trong [0, 1] một cách tự nhiên theo lý thuyết Noisy-OR,
        không cần min(0.95, ...) để ép kiểu.

        Ref: Henrion (1989), "Some Practical Issues in Constructing
             Belief Networks", UAI'89.

        Args:
            recipe: Công thức nấu ăn
            health_conditions: Tình trạng sức khỏe
            ingredients_db: Unused (API compatibility)

        Returns:
            (risk_probability, identified_risk_factors)
        """
        # Build text representation of recipe for matching
        recipe_text_parts = []
        for ing in recipe.get("ingredients", []):
            recipe_text_parts.append(ing.get("name", "").lower())
        for tag in recipe.get("tags", []):
            recipe_text_parts.append(tag.lower())
        recipe_text = " ".join(recipe_text_parts)

        # ============================================================
        # STEP 1: Detect individual risk factors (Independent Causes)
        # ============================================================
        found_risks = []
        detected_factor_names = set()
        survival_product = 1.0  # ∏(1 - P(risk_i))

        for factor, info in self.risk_cpt.items():
            if factor in recipe_text:
                risk_prob = info["prior_risk"]

                # Health Condition CPT: P(Risk | factor, condition)
                if health_conditions:
                    for cond in health_conditions:
                        cond_cpt = self.health_risk_cpt.get(cond, {})
                        if factor in cond_cpt:
                            risk_prob = cond_cpt[factor]

                survival_product *= (1.0 - risk_prob)
                detected_factor_names.add(factor)
                found_risks.append({
                    "yếu_tố": factor,
                    "xác_suất": round(risk_prob, 3),
                    "mô_tả": info["description"]
                })

        # ============================================================
        # STEP 2: Synergy Hidden Causes (Noisy-OR Extension)
        # ============================================================
        # When co-occurring factors create additional risk beyond their
        # independent contributions, model the synergy as a HIDDEN CAUSE
        # with its own probability. Multiply (1 - P_synergy) into the
        # survival product. This is mathematically sound Noisy-OR.
        for synergy in self.synergy_cpt:
            if synergy["factors"].issubset(detected_factor_names):
                synergy_risk = synergy["synergy_risk"]
                survival_product *= (1.0 - synergy_risk)
                found_risks.append({
                    "yếu_tố": " + ".join(sorted(synergy["factors"])),
                    "xác_suất": round(synergy_risk, 3),
                    "mô_tả": f"Synergy (hidden cause): {synergy['reason']}"
                })

        # Final risk: P(Risk) = 1 - survival_product
        # Naturally in [0, 1] — no clamping needed.
        final_risk = 1.0 - survival_product

        return final_risk, found_risks

    # ------------------------------------------------------------------
    # Combined Evaluation (Expected Utility)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        recipe: Dict[str, Any],
        user_conditions: Optional[List[str]] = None,
        total_cost: float = 0,
        budget: float = 200000,
        health_match: bool = True,
        ingredients_db: Optional[List[Dict[str, Any]]] = None
    ) -> BayesResult:
        """
        Đánh giá toàn diện công thức bằng Expected Utility Theory.

        Pipeline đánh giá:
            1. Posterior Inference: P(Like | Evidence) via Bayes' Rule
            2. Risk Assessment: P(Risk | Ingredients) via Independent Model
            3. Expected Utility: EU = P(Like) × U(Like) - λ × P(Risk)

        Decision Theory (Ch. 16, Russell & Norvig):
            Chọn action a* = argmax_a EU(a)
            EU(recipe) = P(Like|E) × U_positive - Risk_penalty

        Args:
            recipe: Công thức nấu ăn
            user_conditions: Tình trạng sức khỏe
            total_cost: Tổng chi phí
            budget: Ngân sách
            health_match: Phù hợp sức khỏe
            ingredients_db: Unused (API compatibility)

        Returns:
            BayesResult chứa posterior, risk, và expected utility explanation
        """
        result = BayesResult()

        # === Bước 1: Posterior Preference via Bayesian Inference ===
        pref_score, contributions = self.predict_preference(
            recipe, total_cost, budget, health_match
        )
        result.preference_score = pref_score
        result.feature_contributions = contributions

        # Discretize posterior into human-readable label
        if pref_score >= 0.75:
            result.preference_label = "cao"
        elif pref_score >= 0.50:
            result.preference_label = "trung bình"
        else:
            result.preference_label = "thấp"

        # === Bước 2: Risk Assessment via Independent Risk Model ===
        risk_score, risk_factors = self.assess_risk(
            recipe, user_conditions
        )
        result.risk_score = risk_score
        result.risk_factors = risk_factors

        # Discretize risk
        if risk_score >= 0.6:
            result.risk_level = "cao"
        elif risk_score >= 0.3:
            result.risk_level = "trung bình"
        else:
            result.risk_level = "thấp"

        # === Bước 3: Expected Utility Explanation ===
        recipe_name = recipe.get("name", "Món ăn")

        # Risk-adjusted utility
        # EU = P(Like) - λ × P(Risk), where λ = risk_aversion_coefficient
        RISK_AVERSION = 0.3
        expected_utility = pref_score - RISK_AVERSION * risk_score

        explanation_lines = [
            f"Đánh giá Bayesian cho '{recipe_name}':",
            f"  Posterior P(Like|E) = {pref_score:.4f} ({result.preference_label})",
            f"  Risk P(Risk|I) = {risk_score:.4f} ({result.risk_level})",
            f"  Expected Utility EU = {expected_utility:.4f}",
        ]

        if risk_factors:
            explanation_lines.append(f"  Phát hiện {len(risk_factors)} yếu tố rủi ro tiêu hóa")
            for rf in risk_factors[:3]:
                explanation_lines.append(
                    f"    - {rf['yếu_tố']}: P={rf['xác_suất']:.3f} ({rf['mô_tả']})"
                )

        if expected_utility >= 0.5:
            explanation_lines.append("  Khuyến nghị: Phù hợp — Expected Utility cao")
        elif expected_utility >= 0.2:
            explanation_lines.append("  Khuyến nghị: Chấp nhận được — cần lưu ý rủi ro")
        else:
            explanation_lines.append("  Khuyến nghị: Cân nhắc thay đổi — EU thấp")

        result.explanation = "\n".join(explanation_lines)

        return result


# ============================================================================
# CHAY THU
# ============================================================================
if __name__ == "__main__":
    evaluator = BayesianRecipeEvaluator()

    sample_recipe = {
        "name": "Bún bò Huế",
        "tags": ["traditional", "spicy", "nutritious"],
        "ingredients": [
            {"name": "thịt bò", "quantity": 0.3},
            {"name": "bún", "quantity": 0.5},
            {"name": "ớt", "quantity": 0.05},
            {"name": "sa tế", "quantity": 0.02},
        ],
    }

    result = evaluator.evaluate(
        recipe=sample_recipe,
        user_conditions=["đau dạ dày"],
        total_cost=85000,
        budget=150000,
        health_match=True
    )

    print("=" * 60)
    print("Demo: Expert-driven Bayesian Network Evaluation")
    print("=" * 60)
    print(f"\n  Món: {sample_recipe['name']}")
    print(f"\n{result.explanation}")
    print(f"\n  Chi tiết:")
    print(f"    P(Like | Evidence) = {result.preference_score:.4f}")
    print(f"    P(Risk | Ingredients) = {result.risk_score:.4f}")
