"""Food ontology utilities for allergen checking, recipe categorization, and nutrition."""

import json
import re
from pathlib import Path
from typing import List, Dict, Set, Optional
from dataclasses import dataclass


@dataclass
class AllergenMatch:
    ingredient: str
    allergen_type: str
    match_category: str  # 'primary', 'derivative', 'hidden_source'
    confidence: float


@dataclass
class RecipeCategory:
    cuisine_type: Optional[str] = None
    cooking_methods: List[str] = None
    dietary_restrictions: List[str] = None
    nutritional_profile: Dict[str, float] = None


class FoodOntology:
    """Lightweight food ontology for practical cooking applications."""
    
    def __init__(self, taxonomy_path: str = None):
        if taxonomy_path is None:
            taxonomy_path = Path(__file__).parent.parent / "data" / "food_taxonomy.json"
        
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)
    
    def check_allergens(self, ingredients: List[str], user_allergies: List[str]) -> List[AllergenMatch]:
        """
        Check ingredients against user allergies using ontology hierarchy.
        
        Args:
            ingredients: List of ingredient strings
            user_allergies: List of allergen types user is allergic to
        
        Returns:
            List of allergen matches found
        """
        matches = []
        allergen_data = self.taxonomy.get("allergens", {})
        
        for allergen_type in user_allergies:
            if allergen_type not in allergen_data:
                continue
                
            allergen_info = allergen_data[allergen_type]
            
            for ingredient in ingredients:
                ingredient_lower = ingredient.lower().strip()
                
                # Check primary allergens
                for primary in allergen_info.get("primary", []):
                    if self._ingredient_contains(ingredient_lower, primary):
                        matches.append(AllergenMatch(
                            ingredient=ingredient,
                            allergen_type=allergen_type,
                            match_category="primary",
                            confidence=0.95
                        ))
                
                # Check derivatives
                for derivative in allergen_info.get("derivatives", []):
                    if self._ingredient_contains(ingredient_lower, derivative):
                        matches.append(AllergenMatch(
                            ingredient=ingredient,
                            allergen_type=allergen_type,
                            match_category="derivative",
                            confidence=0.85
                        ))
                
                # Check hidden sources
                for hidden in allergen_info.get("hidden_sources", []):
                    if self._ingredient_contains(ingredient_lower, hidden):
                        matches.append(AllergenMatch(
                            ingredient=ingredient,
                            allergen_type=allergen_type,
                            match_category="hidden_source",
                            confidence=0.75
                        ))
        
        return matches
    
    def categorize_recipe(self, title: str, ingredients: List[str], instructions: str) -> RecipeCategory:
        """
        Categorize recipe based on ingredients and cooking methods.
        
        Args:
            title: Recipe title
            ingredients: List of ingredients
            instructions: Cooking instructions
            
        Returns:
            RecipeCategory with detected categories
        """
        category = RecipeCategory()
        
        # Detect cuisine type
        category.cuisine_type = self._detect_cuisine(title, ingredients, instructions)
        
        # Detect cooking methods
        category.cooking_methods = self._detect_cooking_methods(instructions)
        
        # Check dietary restrictions
        category.dietary_restrictions = self._check_dietary_restrictions(ingredients)
        
        # Basic nutritional profile
        category.nutritional_profile = self._estimate_nutrition(ingredients)
        
        return category
    
    def get_nutritional_substitutes(self, ingredient: str, target_nutrition: str) -> List[str]:
        """
        Find nutritionally similar ingredients.
        
        Args:
            ingredient: Original ingredient
            target_nutrition: Target nutritional category (protein, carbs, fats)
            
        Returns:
            List of nutritionally similar ingredients
        """
        nutrition_data = self.taxonomy.get("nutritional_categories", {})
        macronutrients = nutrition_data.get("macronutrients", {})
        
        # Find what category the original ingredient belongs to
        original_category = None
        for macro, foods in macronutrients.items():
            if any(self._ingredient_contains(ingredient.lower(), food) for food in foods):
                original_category = macro
                break
        
        if not original_category:
            return []
        
        # Return alternatives from the same nutritional category
        return macronutrients.get(target_nutrition or original_category, [])
    
    def validate_dietary_restriction(self, ingredients: List[str], restriction: str) -> Dict[str, List[str]]:
        """
        Validate if recipe meets dietary restriction.
        
        Args:
            ingredients: Recipe ingredients
            restriction: Dietary restriction (vegetarian, vegan, keto, etc.)
            
        Returns:
            Dict with 'violations' and 'concerns' lists
        """
        dietary_data = self.taxonomy.get("dietary_restrictions", {})
        restriction_info = dietary_data.get(restriction.lower(), {})
        
        violations = []
        concerns = []
        
        forbidden = restriction_info.get("forbidden", [])
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower().strip()
            
            for forbidden_item in forbidden:
                if self._ingredient_contains(ingredient_lower, forbidden_item):
                    violations.append(ingredient)
        
        return {
            "violations": violations,
            "concerns": concerns,
            "compliant": len(violations) == 0
        }
    
    def _ingredient_contains(self, ingredient: str, target: str) -> bool:
        """Check if ingredient contains target food item."""
        # Handle plurals and variations
        target_variations = [
            target,
            target + "s",
            target.rstrip("s") if target.endswith("s") else target
        ]
        
        return any(variation in ingredient for variation in target_variations)
    
    def _detect_cuisine(self, title: str, ingredients: List[str], instructions: str) -> Optional[str]:
        """Detect cuisine type from recipe components."""
        cuisine_data = self.taxonomy.get("cuisine_types", {})
        text = f"{title} {' '.join(ingredients)} {instructions}".lower()
        
        cuisine_scores = {}
        
        for cuisine_family, cuisines in cuisine_data.items():
            score = 0
            for cuisine in cuisines:
                if cuisine in text:
                    score += 1
            if score > 0:
                cuisine_scores[cuisine_family] = score
        
        if cuisine_scores:
            return max(cuisine_scores, key=cuisine_scores.get)
        return None
    
    def _detect_cooking_methods(self, instructions: str) -> List[str]:
        """Detect cooking methods from instructions."""
        methods_data = self.taxonomy.get("cooking_methods", {})
        instructions_lower = instructions.lower()
        
        detected_methods = []
        
        for method_type, methods in methods_data.items():
            for method in methods:
                if method in instructions_lower:
                    detected_methods.append(method)
        
        return list(set(detected_methods))
    
    def _check_dietary_restrictions(self, ingredients: List[str]) -> List[str]:
        """Check what dietary restrictions the recipe satisfies."""
        dietary_data = self.taxonomy.get("dietary_restrictions", {})
        satisfied_restrictions = []
        
        for restriction, rules in dietary_data.items():
            validation = self.validate_dietary_restriction(ingredients, restriction)
            if validation["compliant"]:
                satisfied_restrictions.append(restriction)
        
        return satisfied_restrictions
    
    def _estimate_nutrition(self, ingredients: List[str]) -> Dict[str, float]:
        """Estimate basic nutritional profile."""
        nutrition_data = self.taxonomy.get("nutritional_categories", {})
        macronutrients = nutrition_data.get("macronutrients", {})
        
        # Map the taxonomy keys to our profile keys
        profile = {"proteins": 0, "carbohydrates": 0, "fats": 0}
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            
            for macro, foods in macronutrients.items():
                if any(self._ingredient_contains(ingredient_lower, food) for food in foods):
                    if macro in profile:
                        profile[macro] += 1
        
        # Normalize to percentages
        total = sum(profile.values())
        if total > 0:
            profile = {k: v/total for k, v in profile.items()}
        
        return profile


class EnhancedAllergenChecker:
    """Enhanced allergen checking with severity and cross-contamination."""
    
    def __init__(self, ontology: FoodOntology):
        self.ontology = ontology
    
    def comprehensive_allergen_check(self, recipe_dict: dict, user_allergies: List[str]) -> Dict:
        """
        Comprehensive allergen analysis including severity and suggestions.
        
        Args:
            recipe_dict: Recipe with Title, Cleaned_Ingredients, Instructions
            user_allergies: User's allergen list
            
        Returns:
            Comprehensive allergen report
        """
        import ast
        
        # Parse ingredients
        ingredients = recipe_dict.get("Cleaned_Ingredients", "[]")
        if isinstance(ingredients, str):
            try:
                ingredients = ast.literal_eval(ingredients)
            except:
                ingredients = [ingredients]
        
        # Check for allergens
        allergen_matches = self.ontology.check_allergens(ingredients, user_allergies)
        
        # Categorize by severity
        critical = [m for m in allergen_matches if m.confidence >= 0.9]
        warning = [m for m in allergen_matches if 0.7 <= m.confidence < 0.9]
        caution = [m for m in allergen_matches if m.confidence < 0.7]
        
        # Generate suggestions
        suggestions = self._generate_allergen_suggestions(allergen_matches, ingredients)
        
        return {
            "safe": len(allergen_matches) == 0,
            "allergen_matches": allergen_matches,
            "severity": {
                "critical": critical,
                "warning": warning, 
                "caution": caution
            },
            "suggestions": suggestions,
            "modified_ingredients": self._suggest_substitutions(allergen_matches, ingredients)
        }
    
    def _generate_allergen_suggestions(self, matches: List[AllergenMatch], ingredients: List[str]) -> List[str]:
        """Generate helpful suggestions for allergen management."""
        suggestions = []
        
        if not matches:
            suggestions.append("✅ This recipe appears safe for your allergies!")
            return suggestions
        
        # Group by allergen type
        by_allergen = {}
        for match in matches:
            if match.allergen_type not in by_allergen:
                by_allergen[match.allergen_type] = []
            by_allergen[match.allergen_type].append(match)
        
        for allergen_type, allergen_matches in by_allergen.items():
            critical_matches = [m for m in allergen_matches if m.confidence >= 0.9]
            
            if critical_matches:
                suggestions.append(f"🚨 AVOID: Contains {allergen_type} ({', '.join([m.ingredient for m in critical_matches])})")
            else:
                suggestions.append(f"⚠️ CAUTION: May contain {allergen_type} ({', '.join([m.ingredient for m in allergen_matches])})")
        
        suggestions.append("💡 Consider asking for ingredient substitutions")
        
        return suggestions
    
    def _suggest_substitutions(self, matches: List[AllergenMatch], ingredients: List[str]) -> List[str]:
        """Suggest modified ingredient list with substitutions."""
        # This would integrate with your existing substitution system
        modified = ingredients.copy()
        
        for match in matches:
            if match.confidence >= 0.8:  # Only substitute high-confidence matches
                # You could integrate this with your FAISS-based substitution system
                modified = [ing for ing in modified if ing != match.ingredient]
                modified.append(f"{match.ingredient} → [SUBSTITUTE NEEDED]")
        
        return modified