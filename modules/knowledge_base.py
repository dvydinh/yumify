# -*- coding: utf-8 -*-
"""
modules/knowledge_base.py — Forward Chaining Inference Engine
==============================================================

Production System (Rule-Based Expert System).

Architecture (Russell & Norvig, Ch. 7-8):
    - fact set: A Python `set` of string atoms (propositions).
    - Rule Base: A list of Horn Clauses loaded from `data/rules.json`.
      Each rule: antecedents (set) => consequents (set).
    - Inference Engine: Forward Chaining — data-driven, iterating
      until a Fixed Point (no new facts can be derived).

"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set, FrozenSet
from dataclasses import dataclass, field

# DATA STRUCTURES (Ch. 7 — Propositional Logic)

@dataclass(frozen=True)
class HornClause:
    """
    A Horn Clause (Definite Clause) for Forward Chaining.

    Logical Structure:
        antecedent_1 ∧ antecedent_2 ∧ ... ∧ antecedent_n  →  consequent_1 ∧ ... ∧ consequent_m

    The clause is "fired" when ALL antecedents are present in fact set.
    Upon firing, ALL consequents are added to fact set.

    Attributes:
        rule_id: Unique identifier for tracing/debugging.
        antecedents: Frozenset of string atoms (premises).
        consequents: Frozenset of string atoms (conclusions).
    """
    rule_id: str
    antecedents: FrozenSet[str]
    consequents: FrozenSet[str]

    def is_satisfied(self, working_memory: Set[str]) -> bool:
        """Check if ALL antecedents are a SUBSET of fact set."""
        return self.antecedents.issubset(working_memory)

    def __repr__(self) -> str:
        ant = " ∧ ".join(sorted(self.antecedents))
        con = " ∧ ".join(sorted(self.consequents))
        return f"Rule[{self.rule_id}]: ({ant}) -> ({con})"

@dataclass
class InferenceResult:
    """
    Result of the Forward Chaining inference process.

    Attributes:
        working_memory: The final set of all known facts after inference.
        fired_rules: Ordered list of rule_ids that were activated.
        derived_facts: Facts that were NOT in the initial set (new knowledge).
    """
    working_memory: Set[str] = field(default_factory=set)
    fired_rules: List[str] = field(default_factory=list)
    derived_facts: Set[str] = field(default_factory=set)

    def get_facts_by_prefix(self, prefix: str) -> List[str]:
        """
        Extract fact arguments by prefix convention.

        Convention: facts are stored as "prefix:argument" strings.
        Example: "exclude_ingredient:chili" -> returns "chili"
                 "warning:Avoid spicy food" -> returns "Avoid spicy food"
        """
        results = []
        for fact in self.working_memory:
            if fact.startswith(prefix + ":"):
                results.append(fact[len(prefix) + 1:])
        return sorted(results)

    @property
    def excluded_ingredients(self) -> List[str]:
        return self.get_facts_by_prefix("exclude_ingredient")

    @property
    def excluded_tags(self) -> List[str]:
        return self.get_facts_by_prefix("exclude_tag")

    @property
    def warnings(self) -> List[str]:
        return self.get_facts_by_prefix("warning")

    @property
    def recommendations(self) -> List[str]:
        return self.get_facts_by_prefix("recommend")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "excluded_ingredients": self.excluded_ingredients,
            "excluded_tags": self.excluded_tags,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "fired_rules": self.fired_rules,
            "total_derived_facts": len(self.derived_facts),
        }

# RULE BASE LOADER (reads from data/rules.json)

def load_rule_base(rules_path: str) -> List[HornClause]:
    """
    Load the Rule Base from a JSON file.

    Expected JSON format (array of objects):
    [
      {
        "rule_id": "R_STOMACH",
        "antecedents": ["condition:stomachache"],
        "consequents": ["exclude_ingredient:chili", "exclude_tag:spicy", ...]
      },
      ...
    ]

    The engine treats antecedents and consequents as OPAQUE STRING ATOMS.
    No parsing or interpretation of the string content occurs here.

    Args:
        rules_path: Absolute or relative path to the rules JSON file.

    Returns:
        A list of HornClause objects.
    """
    clauses: List[HornClause] = []
    if not os.path.exists(rules_path):
        print(f"  [KB] Warning: Rule base not found at {rules_path}")
        return clauses

    with open(rules_path, 'r', encoding='utf-8') as f:
        raw_rules = json.load(f)

    for entry in raw_rules:
        clause = HornClause(
            rule_id=entry.get("rule_id", "UNKNOWN"),
            antecedents=frozenset(entry.get("antecedents", [])),
            consequents=frozenset(entry.get("consequents", []))
        )
        clauses.append(clause)

    return clauses

# FORWARD CHAINING INFERENCE ENGINE (Ch. 7 — Russell & Norvig, Fig. 7.15)

class ForwardChainingEngine:
    """
    Forward Chaining / Data-Driven Inference Engine.

    Algorithm (Russell & Norvig, Fig. 7.15 — adapted for Production Systems):

        FUNCTION Forward-Chain(KB, initial_facts):
            working_memory ← initial_facts
            REPEAT:
                new_facts_added ← FALSE
                FOR EACH rule IN KB:
                    IF rule.antecedents ⊆ working_memory AND
                       rule.consequents ⊄ working_memory:
                        working_memory ← working_memory ∪ rule.consequents
                        Mark rule as fired
                        new_facts_added ← TRUE
            UNTIL new_facts_added = FALSE  (Fixed Point reached)
            RETURN working_memory

    Properties:
        - Sound: Only derives facts entailed by KB ∪ initial_facts.
        - Complete: Derives ALL facts entailable from Horn Clauses.
        - Termination: Guaranteed because the set of possible facts is finite.

    IMPORTANT: This class contains ZERO domain-specific strings.
    All knowledge resides in the Rule Base (rules.json).
    """

    def __init__(self, rule_base: Optional[List[HornClause]] = None):
        """
        Initialize the engine with an optional pre-loaded rule base.

        Args:
            rule_base: List of HornClause objects. If None, must be
                       provided when calling `infer()`.
        """
        self.rule_base: List[HornClause] = rule_base or []

    def infer(
        self,
        initial_facts: Set[str],
        rule_base: Optional[List[HornClause]] = None,
        max_iterations: int = 100
    ) -> InferenceResult:
        """
        Execute Forward Chaining inference.

        Args:
            initial_facts: Set of string atoms representing known facts.
            rule_base: Optional override for the rule base.
            max_iterations: Safety limit to prevent infinite loops
                            (should never be reached with a finite KB).

        Returns:
            InferenceResult containing the final fact set,
            list of fired rules, and newly derived facts.
        """
        kb = rule_base if rule_base is not None else self.rule_base
        working_memory: Set[str] = set(initial_facts)
        fired_rule_ids: List[str] = []
        fired_set: Set[str] = set()  # For O(1) lookup

        iteration = 0
        changed = True

        # MAIN LOOP: Iterate until Fixed Point (no new facts derived)
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for clause in kb:
                # Skip already-fired rules (each rule fires at most once
                # per initial fact set, since we don't retract facts)
                if clause.rule_id in fired_set:
                    continue

                # CHECK: Are ALL antecedents satisfied by fact set?
                # This is a pure SET SUBSET operation — no domain logic.
                if clause.is_satisfied(working_memory):
                    # Compute NEW consequents not yet in fact set
                    new_facts = clause.consequents - working_memory

                    if new_facts:
                        # FIRE THE RULE: Add consequents to fact set
                        working_memory |= new_facts
                        fired_rule_ids.append(clause.rule_id)
                        fired_set.add(clause.rule_id)
                        changed = True

        # Compute derived facts (everything that wasn't in the initial set)
        derived = working_memory - initial_facts

        return InferenceResult(
            working_memory=working_memory,
            fired_rules=fired_rule_ids,
            derived_facts=derived,
        )

# DIETARY KNOWLEDGE BASE (Facade wrapping the Engine)

class DietaryKnowledgeBase:
    """
    High-level facade that wraps the ForwardChainingEngine for the
    culinary domain.

    This class:
      1. Loads the rule base from `data/rules.json` (once, at init).
      2. Converts user-provided health conditions into initial facts.
      3. Runs the engine and returns structured results.

    The facade itself is domain-AWARE (it knows about the fact
    naming convention like "condition:X"), but the ENGINE it wraps
    is .

    Example:
        >>> kb = DietaryKnowledgeBase()
        >>> result = kb.evaluate(health_conditions=["stomachache"])
        >>> "chili" in result.excluded_ingredients
        True
    """

    def __init__(self, rules_path: Optional[str] = None):
        """
        Initialize by loading the external rule base.

        Args:
            rules_path: Path to rules.json. Defaults to data/rules.json
                        relative to the project root.
        """
        if rules_path is None:
            base_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'data'
            )
            rules_path = os.path.join(base_dir, 'rules.json')

        self.rule_base = load_rule_base(rules_path)
        self.engine = ForwardChainingEngine(self.rule_base)

    def evaluate(
        self,
        health_conditions: Optional[List[str]] = None,
        preferences: Optional[List[str]] = None,
        excluded_tags_from_nlp: Optional[List[str]] = None,
    ) -> InferenceResult:
        """
        Run Forward Chaining given user health conditions and preferences.

        Workflow:
            1. Convert health_conditions -> facts: {"condition:X", ...}
            2. Convert preferences -> facts: {"preferences:X", ...}
            3. Convert NLP-excluded tags -> facts: {"exclude_tag:X", ...}
            4. Run ForwardChainingEngine.infer(initial_facts)
            5. Return structured InferenceResult.

        Args:
            health_conditions: e.g., ["stomachache", "diabetes"]
            preferences: e.g., ["vegetarian"]
            excluded_tags_from_nlp: e.g., ["spicy", "salty"]

        Returns:
            InferenceResult with excluded ingredients, tags, warnings, etc.
        """
        initial_facts: Set[str] = set()

        # --- NLP → KB key normalisation ---
        # NLP may output different strings than what rules.json expects.
        _CONDITION_MAP = {
            "hypertension":    "high blood pressure",
            "high_blood_pressure": "high blood pressure",
            "hbp":             "high blood pressure",
            "seafood_allergy": "seafood allergy",
            "seafood allergy": "seafood allergy",
        }
        # Some NLP outputs are preferences, not medical conditions.
        _PREF_ALIASES = {"vegetarian", "vegan", "halal", "kosher"}

        # Build initial fact set from user inputs
        if health_conditions:
            for cond in health_conditions:
                key = cond.strip().lower()
                # Route preference-like conditions to the preferences channel
                if key in _PREF_ALIASES:
                    initial_facts.add(f"preferences:{key}")
                    continue
                # Normalise condition key
                mapped = _CONDITION_MAP.get(key, key)
                initial_facts.add(f"condition:{mapped}")

        if preferences:
            for pref in preferences:
                initial_facts.add(f"preferences:{pref}")

        if excluded_tags_from_nlp:
            for tag in excluded_tags_from_nlp:
                initial_facts.add(f"exclude_tag:{tag}")

        # Run the Forward Chaining engine
        result = self.engine.infer(initial_facts)
        return result

    def get_exclusions(
        self,
        health_conditions: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Convenience method: get excluded ingredients and tags.

        Args:
            health_conditions: List of condition strings.

        Returns:
            (excluded_ingredients, excluded_tags)
        """
        result = self.evaluate(health_conditions=health_conditions)
        return result.excluded_ingredients, result.excluded_tags

    def filter_ingredients(
        self,
        ingredients: List[str],
        health_conditions: List[str],
        ingredients_db: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Filter ingredients: keep safe ones, remove excluded ones.

        Uses Forward Chaining to derive the exclusion list,
        then applies substring matching to filter.

        Args:
            ingredients: Original ingredient list.
            health_conditions: Health conditions from user.
            ingredients_db: (Unused, kept for API compatibility).

        Returns:
            (safe_ingredients, removed_ingredients)
        """
        excl_i, excl_t = self.get_exclusions(health_conditions)
        safe = []
        removed = []
        for ing in ingredients:
            ing_lower = ing.lower()
            is_excluded = any(e.lower() in ing_lower for e in excl_i)
            if is_excluded:
                removed.append(ing)
            else:
                safe.append(ing)
        return safe, removed

# STANDALONE TEST
if __name__ == "__main__":
    print("=" * 60)
    print("Forward Chaining Engine — Standalone Test")
    print("=" * 60)

    kb = DietaryKnowledgeBase()
    print(f"\nLoaded {len(kb.rule_base)} rules from rules.json")
    for r in kb.rule_base:
        print(f"  {r}")

    print("\n--- Test 1: Stomachache ---")
    result = kb.evaluate(health_conditions=["stomachache"])
    print(f"  Fired rules: {result.fired_rules}")
    print(f"  Excluded ingredients: {result.excluded_ingredients}")
    print(f"  Excluded tags: {result.excluded_tags}")
    print(f"  Warnings: {result.warnings}")

    print("\n--- Test 2: Diabetes + Vegetarian ---")
    result = kb.evaluate(
        health_conditions=["diabetes"],
        preferences=["vegetarian"]
    )
    print(f"  Fired rules: {result.fired_rules}")
    print(f"  Excluded ingredients: {result.excluded_ingredients}")
    print(f"  Excluded tags: {result.excluded_tags}")

    print("\n--- Test 3: No conditions (empty WM) ---")
    result = kb.evaluate()
    print(f"  Fired rules: {result.fired_rules}")
    print(f"  Derived facts: {len(result.derived_facts)}")
    assert len(result.fired_rules) == 0, "No rules should fire with empty WM"

    print("\n✓ All tests passed.")
