"""Enhanced chef agent with food ontology integration."""

from .food_ontology import FoodOntology, EnhancedAllergenChecker
from .agents.chef_agent import ChefAgent
from langchain_core.messages import AIMessage
import ast


class OntologyEnhancedChefAgent(ChefAgent):
    """Chef agent enhanced with food ontology capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ontology = FoodOntology()
        self.allergen_checker = EnhancedAllergenChecker(self.ontology)
    
    def select_recipe_with_safety_check(self, user_query: str, state: dict):
        """Enhanced recipe selection with allergen and dietary checks."""
        # First, do normal recipe selection
        response = self.select_recipe(user_query, state)
        
        # If a recipe was selected, do safety checks
        selected_recipe = state.get("selected_recipe")
        if selected_recipe:
            user_profile = state.get("user_profile", {})
            safety_report = self._generate_safety_report(selected_recipe, user_profile)
            
            if safety_report["has_concerns"]:
                # Append safety warnings to response
                response += "\n\n" + safety_report["message"]
                
                # Store safety info in state for later use
                state["safety_report"] = safety_report
        
        return response
    
    def _generate_safety_report(self, recipe: dict, user_profile: dict) -> dict:
        """Generate comprehensive safety report for a recipe."""
        allergies = user_profile.get("allergies", [])
        dietary_restrictions = user_profile.get("dietary_restrictions", [])
        
        # Parse ingredients
        ingredients = recipe.get("Cleaned_Ingredients", "[]")
        if isinstance(ingredients, str):
            try:
                ingredients = ast.literal_eval(ingredients)
            except:
                ingredients = [ingredients]
        
        report = {
            "has_concerns": False,
            "allergen_issues": [],
            "dietary_issues": [],
            "message": "",
            "severity": "none"
        }
        
        # Check allergens
        if allergies:
            allergen_result = self.allergen_checker.comprehensive_allergen_check(recipe, allergies)
            if not allergen_result["safe"]:
                report["has_concerns"] = True
                report["allergen_issues"] = allergen_result["severity"]
                
                if allergen_result["severity"]["critical"]:
                    report["severity"] = "critical"
                elif allergen_result["severity"]["warning"]:
                    report["severity"] = "warning"
                else:
                    report["severity"] = "caution"
        
        # Check dietary restrictions
        for restriction in dietary_restrictions:
            validation = self.ontology.validate_dietary_restriction(ingredients, restriction)
            if not validation["compliant"]:
                report["has_concerns"] = True
                report["dietary_issues"].append({
                    "restriction": restriction,
                    "violations": validation["violations"]
                })
        
        # Generate user-friendly message
        if report["has_concerns"]:
            report["message"] = self._format_safety_message(report)
        
        return report
    
    def _format_safety_message(self, report: dict) -> str:
        """Format safety concerns into user-friendly message."""
        messages = []
        
        if report["allergen_issues"]:
            if report["severity"] == "critical":
                messages.append("🚨 CRITICAL ALLERGEN WARNING:")
                for match in report["allergen_issues"]["critical"]:
                    messages.append(f"   - {match.ingredient} contains {match.allergen_type}")
                messages.append("   This recipe is NOT SAFE for you. Please choose a different recipe.")
            
            elif report["severity"] == "warning":
                messages.append("⚠️  ALLERGEN CAUTION:")
                for match in report["allergen_issues"]["warning"]:
                    messages.append(f"   - {match.ingredient} may contain {match.allergen_type}")
                messages.append("   Please verify ingredients carefully or consider substitutions.")
        
        if report["dietary_issues"]:
            messages.append("🥗 DIETARY RESTRICTION NOTICE:")
            for issue in report["dietary_issues"]:
                violations = ", ".join(issue["violations"])
                messages.append(f"   - Not {issue['restriction']}: contains {violations}")
        
        if report["has_concerns"]:
            messages.append("\n💡 Would you like me to suggest ingredient substitutions?")
        
        return "\n".join(messages)
    
    def suggest_recipe_alternatives(self, state: dict, dietary_focus: str = None) -> str:
        """Suggest alternative recipes based on user restrictions."""
        user_profile = state.get("user_profile", {})
        
        # Use ontology to filter recipes
        safe_recipes = []
        
        # This would integrate with your existing recipe search
        # For now, return a helpful message
        message = "Based on your dietary needs, I recommend looking for recipes that:\n"
        
        allergies = user_profile.get("allergies", [])
        if allergies:
            message += f"- Are free from: {', '.join(allergies)}\n"
        
        dietary_restrictions = user_profile.get("dietary_restrictions", [])
        if dietary_restrictions:
            message += f"- Meet {', '.join(dietary_restrictions)} requirements\n"
        
        if dietary_focus:
            nutritional_subs = self.ontology.get_nutritional_substitutes("", dietary_focus)
            if nutritional_subs:
                message += f"- Focus on {dietary_focus}: {', '.join(nutritional_subs[:3])}\n"
        
        message += "\nWould you like me to search for recipes meeting these criteria?"
        
        return message
    
    def categorize_current_recipe(self, state: dict) -> str:
        """Provide recipe categorization analysis."""
        selected_recipe = state.get("selected_recipe")
        if not selected_recipe:
            return "No recipe selected for analysis."
        
        # Parse ingredients
        ingredients = selected_recipe.get("Cleaned_Ingredients", "[]")
        if isinstance(ingredients, str):
            try:
                ingredients = ast.literal_eval(ingredients)
            except:
                ingredients = [ingredients]
        
        category = self.ontology.categorize_recipe(
            selected_recipe["Title"],
            ingredients,
            selected_recipe.get("Instructions", "")
        )
        
        analysis = [
            f"📊 Recipe Analysis: {selected_recipe['Title']}",
            "",
            f"🌍 Cuisine Style: {category.cuisine_type or 'Not detected'}",
            f"👨‍🍳 Cooking Methods: {', '.join(category.cooking_methods) if category.cooking_methods else 'Basic preparation'}",
            f"🥗 Dietary Categories: {', '.join(category.dietary_restrictions) if category.dietary_restrictions else 'No specific restrictions'}",
            "",
            "🍽️ Nutritional Profile:"
        ]
        
        if category.nutritional_profile:
            for macro, percentage in category.nutritional_profile.items():
                analysis.append(f"   - {macro.title()}: {percentage:.1%}")
        
        return "\n".join(analysis)
    
    def enhanced_ingredient_check(self, state: dict) -> str:
        """Enhanced ingredient checking with ontology insights."""
        selected_recipe = state.get("selected_recipe")
        user_profile = state.get("user_profile", {})
        
        if not selected_recipe:
            return "No recipe selected for ingredient checking."
        
        # Parse ingredients
        ingredients = selected_recipe.get("Cleaned_Ingredients", "[]")
        if isinstance(ingredients, str):
            try:
                ingredients = ast.literal_eval(ingredients)
            except:
                ingredients = [ingredients]
        
        # Generate safety report
        safety_report = self._generate_safety_report(selected_recipe, user_profile)
        
        # Standard ingredient list
        message = f"Here are the ingredients for {selected_recipe['Title']}:\n\n"
        for i, ingredient in enumerate(ingredients, 1):
            message += f"{i}. {ingredient}\n"
        
        # Add safety information
        if safety_report["has_concerns"]:
            message += "\n" + safety_report["message"]
        else:
            message += "\n✅ This recipe appears safe for your dietary needs!"
        
        # Add categorization
        message += "\n\n" + self.categorize_current_recipe(state)
        
        message += "\n\nDo you have all these ingredients ready? If you need substitutions, just let me know!"
        
        return message