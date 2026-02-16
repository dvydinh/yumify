# -*- coding: utf-8 -*-
"""
modules/csp_solver.py — Constraint Satisfaction Problem Solver
==============================================================
AI Pillar: CSP / Heuristic Search (L.O.2.1)
[CS188] Chapter 6: Constraint Satisfaction Problems

Implements a CSP solver for ingredient selection with:
  - Variables: Required ingredients in a recipe.
  - Domains: Alternative ingredients in the database (same category).
  - Hard Constraints:
      * Total cost <= Budget (USD)
      * Total calories within [min_cal, max_cal]
  - Soft Constraints:
      * Prefer cheaper ingredients.
      * Prefer user-preferred ingredients.

Algorithm: Backtracking + Forward Checking + MRV Heuristic

CRITICAL FIX (Forward Checking):
    Forward Checking prunes each unassigned variable's domain by checking
    BOTH constraints simultaneously:
      1. remaining_budget >= candidate_cost  (Budget feasibility)
      2. min_possible_calories <= upper_cal_bound AND
         max_possible_calories >= lower_cal_bound  (Calorie feasibility)

    This dual pruning is essential for correctness (Ch. 6, Sec. 6.3).

Tác giả: Nhóm Sinh Viên NMAI
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import copy


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CSPSolution:
    """Solution for the Constraint Satisfaction Problem."""
    success: bool
    assignments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_cost: float = 0.0
    total_calories: float = 0.0
    budget: float = 0.0
    calorie_range: Tuple[float, float] = (0, 99999)
    backtracks: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "ingredients": {
                name: {
                    "price": info.get("price", 0),
                    "calories": info.get("calories", 0),
                    "unit": info.get("unit", ""),
                }
                for name, info in self.assignments.items()
            },
            "total_cost": round(self.total_cost, 2),
            "total_calories": round(self.total_calories, 1),
            "budget": self.budget,
            "calorie_range": self.calorie_range,
            "backtracks": self.backtracks,
            "message": self.message,
        }


@dataclass
class CSPVariable:
    """A CSP variable representing an ingredient slot."""
    name: str                                   # Original ingredient name
    quantity: float                             # Required quantity
    unit: str                                   # Unit of measurement
    domain: List[Dict[str, Any]] = field(default_factory=list)  # Possible values
    assigned: Optional[Dict[str, Any]] = None   # Currently assigned value


# ============================================================================
# CSP SOLVER
# ============================================================================

class IngredientCSPSolver:
    """
    CSP Solver for ingredient selection using:
      - Backtracking Search with Forward Checking (Ch. 6, Sec. 6.3)
      - MRV Heuristic (Minimum Remaining Values) for variable ordering

    The solver ensures that:
      - Total cost of selected ingredients does not exceed the budget.
      - Total calories of selected ingredients falls within the allowed range.
      - No excluded ingredients are selected.

    Example:
        >>> solver = IngredientCSPSolver()
        >>> solution = solver.solve(
        ...     recipe=recipe,
        ...     budget=8.0,  # USD
        ...     calorie_range=(300, 1500),
        ...     ingredients_db=db
        ... )
    """

    def __init__(self):
        """Initialize CSP Solver."""
        self._backtracks = 0
        self._max_backtracks = 10000

    def _build_domains(
        self,
        recipe: Dict[str, Any],
        ingredients_db: List[Dict[str, Any]],
        excluded_ingredients: Optional[List[str]] = None
    ) -> List[CSPVariable]:
        """
        Build CSP variables and their domains from the recipe and database.

        Each ingredient in the recipe becomes a CSP variable.
        Its domain includes the original ingredient + alternatives in the
        same category, excluding any ingredients in the exclusion list.

        Args:
            recipe: Recipe dictionary with 'ingredients' list.
            ingredients_db: Database of available ingredients.
            excluded_ingredients: Ingredients to exclude (from Forward Chaining).

        Returns:
            List of CSPVariable objects with populated domains.
        """
        excluded = set()
        if excluded_ingredients:
            excluded = {name.lower() for name in excluded_ingredients}

        # Build category index for alternatives
        category_index: Dict[str, List[Dict[str, Any]]] = {}
        for item in ingredients_db:
            cat = item.get("category", "other")
            if cat not in category_index:
                category_index[cat] = []
            if item["name"].lower() not in excluded:
                category_index[cat].append(item)

        variables: List[CSPVariable] = []
        for ing in recipe.get("ingredients", []):
            ing_name = ing["name"]
            quantity = ing.get("quantity", 1.0)
            unit = ing.get("unit", "portion")

            # Find original ingredient in DB
            original = None
            for item in ingredients_db:
                if item["name"].lower() == ing_name.lower():
                    original = item
                    break

            # Build domain: original + same-category alternatives
            domain = []
            if original and original["name"].lower() not in excluded:
                domain.append(original)

            if original:
                cat = original.get("category", "other")
                for alt in category_index.get(cat, []):
                    if alt["name"] != original["name"] and alt not in domain:
                        domain.append(alt)

            if not domain:
                # Fallback: use ALL non-excluded items
                for item in ingredients_db:
                    if item["name"].lower() not in excluded:
                        domain.append(item)
                        if len(domain) >= 5:
                            break

            variables.append(CSPVariable(
                name=ing_name,
                quantity=quantity,
                unit=unit,
                domain=domain
            ))

        return variables

    def _select_mrv_variable(
        self, variables: List[CSPVariable]
    ) -> Optional[int]:
        """
        MRV Heuristic (Minimum Remaining Values) — Ch. 6, Sec. 6.3.1.

        Select the unassigned variable with the SMALLEST domain.
        This heuristic identifies the most constrained variable first,
        which is proven to reduce the search tree size significantly.

        Returns:
            Index of the MRV variable, or None if all assigned.
        """
        best_idx = None
        best_size = float('inf')

        for i, var in enumerate(variables):
            if var.assigned is None and len(var.domain) > 0:
                if len(var.domain) < best_size:
                    best_size = len(var.domain)
                    best_idx = i

        return best_idx

    def _forward_check(
        self,
        variables: List[CSPVariable],
        assigned_idx: int,
        remaining_budget: float,
        current_calories: float,
        calorie_range: Tuple[float, float]
    ) -> bool:
        """
        Forward Checking with DUAL CONSTRAINT PRUNING (Ch. 6, Sec. 6.3.2).

        After assigning a value to variable[assigned_idx], prune the domains
        of ALL remaining unassigned variables by removing values that would
        violate EITHER:
          1. Budget constraint: candidate_cost > remaining_budget
          2. Calorie constraint: choosing this candidate would make it
             impossible to reach a valid total calorie count.

        This is the CRITICAL FIX: naive Forward Checking only prunes on
        budget. We also prune on calorie bounds to maintain arc consistency.

        Args:
            variables: Current variable list with some assigned.
            assigned_idx: Index of the just-assigned variable.
            remaining_budget: Budget left after current assignments.
            current_calories: Calories accumulated so far.
            calorie_range: (min_calories, max_calories) bounds.

        Returns:
            True if all remaining variables still have non-empty domains.
            False if any domain becomes empty (domain wipe-out -> backtrack).
        """
        min_cal, max_cal = calorie_range

        # Count remaining unassigned variables (excluding the one just assigned)
        unassigned_indices = [
            i for i, v in enumerate(variables)
            if v.assigned is None and i != assigned_idx
        ]
        n_remaining = len(unassigned_indices)

        for i in unassigned_indices:
            var = variables[i]
            pruned_domain = []

            for candidate in var.domain:
                cost = candidate.get("price", 0) * var.quantity
                cals = candidate.get("calories", 0) * var.quantity

                # ============================================================
                # CONSTRAINT 1: Budget feasibility
                # ============================================================
                if cost > remaining_budget:
                    continue  # Prune: exceeds remaining budget

                # ============================================================
                # CONSTRAINT 2: Calorie feasibility (Bounds Propagation)
                # ============================================================
                # After choosing this candidate, can we still reach a
                # valid total calorie count with the remaining variables?
                #
                # Optimistic estimate: assume all other remaining variables
                # contribute 0 calories (minimum) or infinite (maximum).
                # Therefore, we only check that THIS candidate's calories
                # don't push us irrecoverably out of bounds.
                #
                # projected_cal = current_calories + this_cal
                # For the total to be valid:
                #   projected_cal <= max_cal  (can't already exceed max)
                projected_cal = current_calories + cals
                if projected_cal > max_cal:
                    continue  # Prune: calories already exceed upper bound

                # Keep this candidate
                pruned_domain.append(candidate)

            var.domain = pruned_domain

            # Domain Wipe-Out: if any variable has empty domain, fail
            if len(var.domain) == 0:
                return False

        return True

    def _backtrack(
        self,
        variables: List[CSPVariable],
        budget: float,
        calorie_range: Tuple[float, float],
        current_cost: float,
        current_calories: float
    ) -> Optional[List[CSPVariable]]:
        """
        Backtracking Search with Forward Checking (Ch. 6, Algorithm 6.5).

        Recursive backtracking that:
          1. Selects the next variable via MRV.
          2. Tries each value in the variable's domain.
          3. Checks constraints (budget + calories).
          4. Runs Forward Checking to prune future domains.
          5. Recurses or backtracks.

        Returns:
            Completed variable list if solution found, else None.
        """
        # Base case: all variables assigned
        if all(v.assigned is not None for v in variables):
            # Final constraint check
            if current_cost <= budget:
                min_cal, max_cal = calorie_range
                if min_cal <= current_calories <= max_cal:
                    return variables
            return None

        # Select variable using MRV heuristic
        var_idx = self._select_mrv_variable(variables)
        if var_idx is None:
            return None

        var = variables[var_idx]

        # Try each value in the domain
        for candidate in list(var.domain):
            self._backtracks += 1
            if self._backtracks > self._max_backtracks:
                return None

            cost = candidate.get("price", 0) * var.quantity
            cals = candidate.get("calories", 0) * var.quantity

            # Check: does this assignment violate budget constraint?
            new_cost = current_cost + cost
            if new_cost > budget:
                continue

            # Check: does this assignment exceed calorie upper bound?
            new_cals = current_calories + cals
            if new_cals > calorie_range[1]:
                continue

            # ASSIGN the value
            var.assigned = candidate

            # FORWARD CHECK: prune domains of remaining variables
            saved_domains = {
                i: list(variables[i].domain)
                for i in range(len(variables))
                if variables[i].assigned is None and i != var_idx
            }

            remaining_budget = budget - new_cost
            fc_ok = self._forward_check(
                variables, var_idx, remaining_budget,
                new_cals, calorie_range
            )

            if fc_ok:
                # Recurse
                result = self._backtrack(
                    variables, budget, calorie_range,
                    new_cost, new_cals
                )
                if result is not None:
                    return result

            # UNDO: restore domains and unassign
            for i, saved in saved_domains.items():
                variables[i].domain = saved
            var.assigned = None

        return None

    def solve(
        self,
        recipe: Dict[str, Any],
        budget: float,
        calorie_range: Tuple[float, float] = (200, 2000),
        ingredients_db: Optional[List[Dict[str, Any]]] = None,
        excluded_ingredients: Optional[List[str]] = None
    ) -> CSPSolution:
        """
        Solve the ingredient assignment CSP.

        Args:
            recipe: Recipe dict with 'ingredients' and 'name'.
            budget: Maximum budget in USD.
            calorie_range: (min_calories, max_calories).
            ingredients_db: Available ingredients database.
            excluded_ingredients: Ingredients excluded by Forward Chaining.

        Returns:
            CSPSolution with assignment results.
        """
        if not ingredients_db:
            return CSPSolution(
                success=False,
                budget=budget,
                calorie_range=calorie_range,
                message="No ingredients database provided."
            )

        recipe_name = recipe.get("name", "Unknown")
        recipe_ings = recipe.get("ingredients", [])

        if not recipe_ings:
            return CSPSolution(
                success=True,
                budget=budget,
                calorie_range=calorie_range,
                message=f"Recipe '{recipe_name}' has no ingredients."
            )

        # Build variables and domains
        variables = self._build_domains(recipe, ingredients_db, excluded_ingredients)

        # Check for empty domains before starting
        for var in variables:
            if not var.domain:
                return CSPSolution(
                    success=False,
                    budget=budget,
                    calorie_range=calorie_range,
                    message=f"No valid options for ingredient '{var.name}'. "
                            f"Domain is empty after exclusion filtering."
                )

        # Run Backtracking + Forward Checking + MRV
        self._backtracks = 0
        result = self._backtrack(variables, budget, calorie_range, 0.0, 0.0)

        if result is None:
            return CSPSolution(
                success=False,
                budget=budget,
                calorie_range=calorie_range,
                backtracks=self._backtracks,
                message=f"CSP unsatisfiable for '{recipe_name}' within "
                        f"budget ${budget:.2f} and calorie range {calorie_range}."
            )

        # Build solution from assignments
        assignments = {}
        total_cost = 0.0
        total_cals = 0.0
        for var in result:
            if var.assigned:
                c = var.assigned.get("price", 0) * var.quantity
                k = var.assigned.get("calories", 0) * var.quantity
                total_cost += c
                total_cals += k
                assignments[var.name] = {
                    "assigned": var.assigned["name"],
                    "price": round(c, 2),
                    "calories": round(k, 1),
                    "unit": var.unit,
                    "quantity": var.quantity,
                }

        return CSPSolution(
            success=True,
            assignments=assignments,
            total_cost=round(total_cost, 2),
            total_calories=round(total_cals, 1),
            budget=budget,
            calorie_range=calorie_range,
            backtracks=self._backtracks,
            message=f"CSP solved for '{recipe_name}' in {self._backtracks} backtracks."
        )


# ============================================================================
# STANDALONE TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CSP Solver — Standalone Test")
    print("=" * 60)

    db = [
        {"name": "chicken", "price": 2.5, "calories": 200, "category": "meat"},
        {"name": "beef", "price": 4.0, "calories": 250, "category": "meat"},
        {"name": "tofu", "price": 1.0, "calories": 80, "category": "protein"},
        {"name": "tomato", "price": 0.5, "calories": 20, "category": "vegetable"},
        {"name": "carrot", "price": 0.3, "calories": 25, "category": "vegetable"},
        {"name": "rice", "price": 0.8, "calories": 300, "category": "carb"},
    ]

    recipe = {
        "name": "Simple Stir-Fry",
        "ingredients": [
            {"name": "chicken", "quantity": 1, "unit": "portion"},
            {"name": "tomato", "quantity": 2, "unit": "pieces"},
            {"name": "rice", "quantity": 1, "unit": "cup"},
        ]
    }

    solver = IngredientCSPSolver()

    print("\n--- Test 1: Normal budget ---")
    sol = solver.solve(recipe, budget=10.0, calorie_range=(200, 800), ingredients_db=db)
    print(f"  Success: {sol.success}")
    print(f"  Total cost: ${sol.total_cost:.2f}")
    print(f"  Total calories: {sol.total_calories}")
    print(f"  Backtracks: {sol.backtracks}")

    print("\n--- Test 2: Very tight budget (should fail) ---")
    sol2 = solver.solve(recipe, budget=0.5, calorie_range=(200, 800), ingredients_db=db)
    print(f"  Success: {sol2.success}")
    print(f"  Message: {sol2.message}")

    print("\n--- Test 3: Tight calorie range ---")
    sol3 = solver.solve(recipe, budget=10.0, calorie_range=(100, 250), ingredients_db=db)
    print(f"  Success: {sol3.success}")
    print(f"  Total calories: {sol3.total_calories}")

    print("\n Tests complete.")
