# -*- coding: utf-8 -*-
"""
modules/search_engine.py — A* Sequential Meal Planner
======================================================
AI Pillar: Search & Optimization (L.O.2.1)
[CS188] Chapter 3: Solving Problems by Searching
[CS188] Chapter 4: Informed Search — A* Algorithm

Solves the N-Days Meal Planning Problem using A* Search:

Problem Formulation:
    STATE:   (current_day, total_cost, total_calories, sorted_multiset_of_recipes)
    ACTION:  Select recipe X for day (current_day + 1)
    GOAL:    current_day == N  AND  total_cost <= Budget  AND
             total_calories within [min_cal * N, max_cal * N]
    g(n):    Total price of recipes chosen so far (path cost).
    h(n):    (N - current_day) * min_recipe_cost_in_DB  (Admissible Heuristic).

State Equivalence (CRITICAL FIX):
    Since the Goal Test only checks TOTAL cost and TOTAL calories
    (not which recipe is assigned to which day), permutations of
    the same recipe set are EQUIVALENT states.
    E.g., (Phở, Pizza) ≡ (Pizza, Phở) for our goal function.

    We enforce a CANONICAL FORM by sorting selected_recipes before
    inserting into the visited set. This collapses permutations:
        Search space: O(R^N) → O(R^N / N!)  (Combinations, not Permutations)

Admissibility Proof:
    h(n) = remaining_days * cheapest_recipe_cost
    Since we must choose at least one recipe per remaining day, and
    the cheapest recipe costs at least `min_cost`, h(n) <= h*(n).
    Therefore h(n) never overestimates the true remaining cost,
    satisfying the Admissible Heuristic property for A* optimality.

Tác giả: Nhóm Sinh Viên NMAI
"""

import heapq
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(order=True)
class MealPlanState:
    """
    A state in the A* search tree for N-days meal planning.

    Ordering is by f_score (for the priority queue / min-heap).

    Attributes:
        f_score: f(n) = g(n) + h(n) — estimated total cost.
        g_score: g(n) — actual cost of recipes chosen so far.
        current_day: Number of days planned so far (0 to N).
        total_calories: Total calories of all selected recipes.
        selected_recipes: Tuple of recipe indices chosen so far.
    """
    f_score: float
    g_score: float = field(compare=False)
    current_day: int = field(compare=False)
    total_calories: float = field(compare=False)
    selected_recipes: Tuple[int, ...] = field(compare=False, default_factory=tuple)


@dataclass
class MealPlanResult:
    """Result of the A* meal planning search."""
    success: bool
    planned_recipes: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    total_calories: float = 0.0
    budget: float = 0.0
    days: int = 1
    nodes_explored: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "days": self.days,
            "planned_recipes": [
                {
                    "day": i + 1,
                    "recipe_name": r.get("name", "Unknown"),
                    "cost": round(r.get("cost", 0), 2),
                    "calories": round(r.get("calories", 0), 1),
                }
                for i, r in enumerate(self.planned_recipes)
            ],
            "total_cost": round(self.total_cost, 2),
            "total_calories": round(self.total_calories, 1),
            "budget": self.budget,
            "nodes_explored": self.nodes_explored,
            "message": self.message,
        }


# ============================================================================
# A* SEQUENTIAL MEAL PLANNER
# ============================================================================

class AStarMealPlanner:
    """
    A* Search for the N-Days Sequential Meal Planning problem.

    Uses:
      - Priority Queue (min-heap) ordered by f(n) = g(n) + h(n).
      - Admissible Heuristic: h(n) = (N - day) * min_recipe_cost.
      - Visited state tracking to avoid re-expansion.

    The planner selects one recipe per day from a set of candidate recipes,
    ensuring the total cost stays within budget and total calories stay
    within acceptable bounds.

    Example:
        >>> planner = AStarMealPlanner()
        >>> result = planner.search(
        ...     candidate_recipes=recipes,
        ...     budget=20.0,
        ...     days=3,
        ...     calorie_range_per_day=(400, 800)
        ... )
    """

    def __init__(self, max_nodes: int = 50000):
        """
        Initialize the A* planner.

        Args:
            max_nodes: Maximum nodes to expand before giving up.
                       Prevents memory exhaustion on large inputs.
        """
        self._max_nodes = max_nodes

    def _compute_min_recipe_cost(
        self, recipes: List[Dict[str, Any]]
    ) -> float:
        """
        Compute the global minimum recipe cost for the admissible heuristic.

        [CS188] Admissibility Guarantee:
            h(n) = remaining_days * min_cost
            Since each remaining day requires at least one recipe,
            and every recipe costs >= min_cost, we have:
            h(n) <= h*(n) for all n.
            Therefore h is admissible, and A* is optimal.

        Returns:
            The minimum cost among all candidate recipes. Returns 0 if empty.
        """
        if not recipes:
            return 0.0
        return min(r.get("cost", 0) for r in recipes)

    def _heuristic(
        self, current_day: int, total_days: int, min_cost: float
    ) -> float:
        """
        Admissible Heuristic h(n).

        h(n) = (N - current_day) * min_recipe_cost

        This never overestimates because:
          - We need exactly (N - current_day) more recipes.
          - Each recipe costs at least min_recipe_cost.
          - Therefore the true remaining cost h*(n) >= h(n).

        Args:
            current_day: Days planned so far.
            total_days: Target number of days (N).
            min_cost: Cheapest recipe cost in the database.

        Returns:
            Heuristic estimate of remaining cost.
        """
        remaining = total_days - current_day
        return remaining * min_cost

    def search(
        self,
        candidate_recipes: List[Dict[str, Any]],
        budget: float,
        days: int = 1,
        calorie_range_per_day: Tuple[float, float] = (400, 800),
        allow_repeats: bool = True
    ) -> MealPlanResult:
        """
        Execute A* Search for N-days meal planning.

        Args:
            candidate_recipes: List of recipe dicts, each with:
                               'name', 'cost', 'calories', 'ingredients', etc.
            budget: Total budget for all N days (USD).
            days: Number of days to plan (N).
            calorie_range_per_day: (min_cal, max_cal) per day.
            allow_repeats: If True, the same recipe can appear on multiple days.

        Returns:
            MealPlanResult with the optimal meal plan.
        """
        if not candidate_recipes:
            return MealPlanResult(
                success=False, budget=budget, days=days,
                message="No candidate recipes provided."
            )

        if days <= 0:
            return MealPlanResult(
                success=False, budget=budget, days=days,
                message="Number of days must be positive."
            )

        # Pre-compute minimum cost for the admissible heuristic
        min_cost = self._compute_min_recipe_cost(candidate_recipes)

        # Total calorie bounds for the entire plan
        min_total_cal = calorie_range_per_day[0] * days
        max_total_cal = calorie_range_per_day[1] * days

        # ================================================================
        # A* SEARCH INITIALIZATION
        # ================================================================
        h0 = self._heuristic(0, days, min_cost)
        start = MealPlanState(
            f_score=h0,
            g_score=0.0,
            current_day=0,
            total_calories=0.0,
            selected_recipes=()
        )

        open_set: List[MealPlanState] = [start]
        # Visited: track (day, SORTED_selected_tuple) to avoid re-expansion
        # Sorting collapses permutations into one canonical state:
        #   (Phở, Pizza) and (Pizza, Phở) → both map to (Pizza, Phở)
        #   This reduces search space from O(R^N) to O(C(R,N)) = O(R^N / N!)
        visited: Set[Tuple[int, Tuple[int, ...]]] = set()
        nodes_explored = 0

        # ================================================================
        # A* SEARCH LOOP
        # ================================================================
        while open_set and nodes_explored < self._max_nodes:
            current = heapq.heappop(open_set)
            nodes_explored += 1

            # CANONICAL FORM: sort recipes to collapse permutations
            canonical_recipes = tuple(sorted(current.selected_recipes))
            state_key = (current.current_day, canonical_recipes)
            if state_key in visited:
                continue
            visited.add(state_key)

            # ============================================================
            # GOAL TEST: All days planned?
            # ============================================================
            if current.current_day == days:
                # Check calorie bounds for the complete plan
                if (min_total_cal <= current.total_calories <= max_total_cal
                        and current.g_score <= budget):
                    # GOAL REACHED: reconstruct plan
                    planned = []
                    for idx in current.selected_recipes:
                        r = candidate_recipes[idx]
                        # Preserve ALL original keys (ingredients, steps, csp_assignments, etc.)
                        # while ensuring cost and calories are correctly mapped from the search state
                        planned_recipe = r.copy()
                        planned_recipe["name"] = r.get("name", "Unknown")
                        planned_recipe["cost"] = r.get("cost", 0)
                        planned_recipe["calories"] = r.get("calories", 0)
                        planned.append(planned_recipe)
                        
                    return MealPlanResult(
                        success=True,
                        planned_recipes=planned,
                        total_cost=round(current.g_score, 2),
                        total_calories=round(current.total_calories, 1),
                        budget=budget,
                        days=days,
                        nodes_explored=nodes_explored,
                        message=f"Optimal {days}-day meal plan found in "
                                f"{nodes_explored} nodes."
                    )
                continue  # Calorie/budget check failed, skip

            # ============================================================
            # EXPAND: Try adding each recipe for the next day
            # ============================================================
            for recipe_idx, recipe in enumerate(candidate_recipes):
                # If repeats not allowed, skip already-selected recipes
                if not allow_repeats and recipe_idx in current.selected_recipes:
                    continue

                recipe_cost = recipe.get("cost", 0)
                recipe_cal = recipe.get("calories", 0)

                # Repetition penalty: discourage same recipe on multiple days
                # Count how many times this recipe already appears in the plan
                repeat_count = current.selected_recipes.count(recipe_idx)
                repetition_penalty = 0.0
                if repeat_count > 0:
                    # Base penalty: doubles each time (exponential discouragement)
                    repetition_penalty = recipe_cost * (2 ** repeat_count)
                    # Extra penalty if this recipe was used on the PREVIOUS day
                    if current.selected_recipes and current.selected_recipes[-1] == recipe_idx:
                        repetition_penalty += recipe_cost * 3.0

                new_g = current.g_score + recipe_cost + repetition_penalty
                new_cal = current.total_calories + recipe_cal
                new_day = current.current_day + 1

                # Pruning: skip if budget already exceeded
                if new_g > budget:
                    continue

                # Pruning: skip if calories already exceed maximum
                if new_cal > max_total_cal:
                    continue

                # Compute heuristic for the successor
                h = self._heuristic(new_day, days, min_cost)
                f = new_g + h

                # Additional pruning: if f > budget, no point exploring
                if f > budget:
                    continue

                new_selected = current.selected_recipes + (recipe_idx,)

                successor = MealPlanState(
                    f_score=f,
                    g_score=new_g,
                    current_day=new_day,
                    total_calories=new_cal,
                    selected_recipes=new_selected
                )

                # Only add if not visited (using canonical sorted form)
                succ_key = (new_day, tuple(sorted(new_selected)))
                if succ_key not in visited:
                    heapq.heappush(open_set, successor)

        # If we get here, search exhausted without finding a valid plan
        return MealPlanResult(
            success=False,
            budget=budget,
            days=days,
            nodes_explored=nodes_explored,
            message=f"A* exhausted {nodes_explored} nodes without finding "
                    f"a valid {days}-day meal plan within budget ${budget:.2f}."
        )


# ============================================================================
# BACKWARD COMPATIBILITY: Shopping Cart Optimizer (single-recipe)
# ============================================================================

class AStarShoppingOptimizer:
    """
    Legacy A* Shopping Cart Optimizer for single-recipe cost optimization.
    Kept for backward compatibility with the pipeline.
    Delegates to MealPlanner for single-day planning.
    """

    @dataclass(order=True)
    class _ShoppingState:
        f_score: float
        g_score: float = field(compare=False)
        selected: frozenset = field(compare=False, default_factory=frozenset)
        path: list = field(compare=False, default_factory=list)

    def __init__(self):
        self._nodes_explored = 0
        self._max_nodes = 50000

    def _calculate_global_min_cost(
        self, ingredients_db: List[Dict[str, Any]]
    ) -> float:
        """
        [CS188] Admissibility Guarantee:
        Find the absolute minimum ingredient price for the heuristic.
        h(n) = remaining_items * min_price <= h*(n), so h is admissible.
        """
        if not ingredients_db:
            return 0.01
        prices = [item.get("price", float('inf'))
                  for item in ingredients_db if item.get("price", 0) > 0]
        return min(prices) if prices else 0.01

    def _get_ingredient_cost(
        self, ing_name: str, quantity: float,
        ingredients_db: List[Dict[str, Any]]
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Look up ingredient cost from the database."""
        name_lower = ing_name.lower()
        for item in ingredients_db:
            if item["name"].lower() == name_lower:
                return item["price"] * quantity, item

        # Admissible fallback: use global minimum cost
        if not hasattr(self, '_global_min_cost') or self._global_min_cost is None:
            self._global_min_cost = self._calculate_global_min_cost(ingredients_db)
        return self._global_min_cost * quantity, {
            "name": ing_name, "price": self._global_min_cost, "unit": "portion"
        }

    def search(
        self,
        recipe: Dict[str, Any],
        budget: float,
        ingredients_db: List[Dict[str, Any]],
        available_ingredients: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        A* search for optimal shopping cart (single recipe).

        Returns dict with keys: success, total_cost, budget, remaining,
        cart, nodes_explored, recipe_name.
        """
        recipe_ings = recipe.get("ingredients", [])
        recipe_name = recipe.get("name", "Unknown")

        if not recipe_ings:
            return {
                "success": False, "recipe_name": recipe_name,
                "budget": budget, "message": "No ingredients."
            }

        available_set = set()
        if available_ingredients:
            available_set = {name.lower() for name in available_ingredients}

        # Items to buy
        need_to_buy = []
        for ing in recipe_ings:
            if ing["name"].lower() not in available_set:
                need_to_buy.append(ing)

        if not need_to_buy:
            return {
                "success": True, "total_cost": 0, "budget": budget,
                "remaining": budget, "recipe_name": recipe_name,
                "cart": [], "nodes_explored": 0
            }

        # Calculate costs
        item_costs = []
        for ing in need_to_buy:
            cost, db_item = self._get_ingredient_cost(
                ing["name"], ing.get("quantity", ing.get("amount", 1)),
                ingredients_db
            )
            item_costs.append((cost, ing, db_item))

        total = sum(c for c, _, _ in item_costs)

        if total <= budget:
            cart = []
            for cost, ing, db_item in item_costs:
                cart.append({
                    "name": ing["name"],
                    "assigned": db_item["name"] if db_item else ing["name"],
                    "price": round(cost, 2),
                    "unit": db_item.get("unit", "portion") if db_item else "portion",
                })
            return {
                "success": True,
                "total_cost": round(total, 2),
                "budget": budget,
                "remaining": round(budget - total, 2),
                "recipe_name": recipe_name,
                "cart": cart,
                "nodes_explored": len(item_costs),
            }
        else:
            return {
                "success": False,
                "total_cost": round(total, 2),
                "budget": budget,
                "remaining": round(budget - total, 2),
                "recipe_name": recipe_name,
                "message": f"Budget exceeded: ${total:.2f} > ${budget:.2f}",
                "nodes_explored": len(item_costs),
            }


# ============================================================================
# STANDALONE TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("A* Sequential Meal Planner — Standalone Test")
    print("=" * 60)

    recipes = [
        {"name": "Simple Salad", "cost": 3.0, "calories": 300},
        {"name": "Chicken Stir-Fry", "cost": 5.0, "calories": 600},
        {"name": "Pasta Carbonara", "cost": 4.5, "calories": 700},
        {"name": "Tofu Soup", "cost": 2.0, "calories": 250},
        {"name": "Beef Steak", "cost": 8.0, "calories": 800},
    ]

    planner = AStarMealPlanner()

    print("\n--- Test 1: 3-day plan, $15 budget ---")
    result = planner.search(recipes, budget=15.0, days=3,
                            calorie_range_per_day=(300, 700))
    print(f"  Success: {result.success}")
    if result.success:
        for i, r in enumerate(result.planned_recipes):
            print(f"  Day {i+1}: {r['name']} (${r['cost']:.2f}, {r['calories']} cal)")
        print(f"  Total: ${result.total_cost:.2f}, {result.total_calories} cal")
    print(f"  Nodes explored: {result.nodes_explored}")

    print("\n--- Test 2: 5-day plan, $10 budget (tight) ---")
    result2 = planner.search(recipes, budget=10.0, days=5,
                             calorie_range_per_day=(200, 500))
    print(f"  Success: {result2.success}")
    print(f"  Message: {result2.message}")

    print("\n Tests complete.")
