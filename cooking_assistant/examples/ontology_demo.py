#!/usr/bin/env python3
"""
Demo of food ontology capabilities for allergen checking, 
recipe categorization, and nutritional enhancement.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.food_ontology import FoodOntology, EnhancedAllergenChecker, RecipeCategory
import pandas as pd


def demo_allergen_checking():
    """Demonstrate comprehensive allergen checking."""
    print("🔍 ALLERGEN CHECKING DEMO")
    print("=" * 50)
    
    ontology = FoodOntology()
    checker = EnhancedAllergenChecker(ontology)
    
    # Sample recipe with potential allergens
    sample_recipe = {
        "Title": "Creamy Mushroom Pasta",
        "Cleaned_Ingredients": "['pasta', 'mushrooms', 'heavy cream', 'parmesan cheese', 'butter', 'garlic', 'wheat flour']",
        "Instructions": "Cook pasta. Sauté mushrooms in butter. Add cream and cheese."
    }
    
    user_allergies = ["dairy", "gluten"]
    
    result = checker.comprehensive_allergen_check(sample_recipe, user_allergies)
    
    print(f"Recipe: {sample_recipe['Title']}")
    print(f"User allergies: {user_allergies}")
    print(f"Safe: {result['safe']}")
    print()
    
    print("Severity Analysis:")
    for severity, matches in result['severity'].items():
        if matches:
            print(f"  {severity.upper()}: {len(matches)} matches")
            for match in matches:
                print(f"    - {match.ingredient} → {match.allergen_type} ({match.confidence:.0%})")
    print()
    
    print("Suggestions:")
    for suggestion in result['suggestions']:
        print(f"  {suggestion}")
    print()


def demo_recipe_categorization():
    """Demonstrate recipe categorization."""
    print("🏷️  RECIPE CATEGORIZATION DEMO")
    print("=" * 50)
    
    ontology = FoodOntology()
    
    recipes = [
        {
            "title": "Chicken Teriyaki Stir Fry",
            "ingredients": ["chicken breast", "soy sauce", "ginger", "garlic", "broccoli", "rice"],
            "instructions": "Stir fry chicken in wok. Add vegetables. Serve over steamed rice."
        },
        {
            "title": "Mediterranean Grilled Fish",
            "ingredients": ["fish fillet", "olive oil", "lemon", "oregano", "tomatoes", "feta cheese"],
            "instructions": "Grill fish with olive oil. Serve with tomato salad and feta."
        },
        {
            "title": "Vegan Lentil Curry", 
            "ingredients": ["red lentils", "coconut milk", "curry powder", "onions", "tomatoes"],
            "instructions": "Simmer lentils in coconut milk with spices until tender."
        }
    ]
    
    for recipe in recipes:
        category = ontology.categorize_recipe(
            recipe["title"], 
            recipe["ingredients"], 
            recipe["instructions"]
        )
        
        print(f"Recipe: {recipe['title']}")
        print(f"  Cuisine: {category.cuisine_type}")
        print(f"  Cooking methods: {category.cooking_methods}")
        print(f"  Dietary restrictions: {category.dietary_restrictions}")
        print(f"  Nutrition profile: {category.nutritional_profile}")
        print()


def demo_nutritional_enhancement():
    """Demonstrate nutritional analysis and substitutions."""
    print("🥗 NUTRITIONAL ENHANCEMENT DEMO")
    print("=" * 50)
    
    ontology = FoodOntology()
    
    # Demo nutritional substitutes
    ingredients_to_substitute = ["chicken breast", "white rice", "butter"]
    
    for ingredient in ingredients_to_substitute:
        print(f"Nutritional substitutes for '{ingredient}':")
        
        # Get substitutes for each macronutrient category
        for nutrition_type in ["proteins", "carbohydrates", "fats"]:
            substitutes = ontology.get_nutritional_substitutes(ingredient, nutrition_type)
            if substitutes:
                print(f"  {nutrition_type.title()}: {', '.join(substitutes[:5])}")
        print()
    
    # Demo dietary restriction validation
    print("Dietary Restriction Validation:")
    test_ingredients = ["chicken", "rice", "vegetables", "olive oil"]
    
    for restriction in ["vegetarian", "vegan", "keto"]:
        validation = ontology.validate_dietary_restriction(test_ingredients, restriction)
        print(f"  {restriction.title()}: {'✅ Compliant' if validation['compliant'] else '❌ Not compliant'}")
        if validation['violations']:
            print(f"    Violations: {', '.join(validation['violations'])}")
    print()


def demo_integration_with_existing_system():
    """Show how to integrate with existing cooking assistant."""
    print("🔗 INTEGRATION DEMO")
    print("=" * 50)
    
    ontology = FoodOntology()
    checker = EnhancedAllergenChecker(ontology)
    
    # Load a few recipes from your existing data
    try:
        df = pd.read_csv("../../../13k-recipes.csv").head(3)
        
        user_profile = {
            "allergies": ["nuts", "dairy"],
            "dietary_restrictions": ["vegetarian"],
            "nutritional_goals": ["high_protein"]
        }
        
        print(f"User profile: {user_profile}")
        print()
        
        for idx, recipe in df.iterrows():
            print(f"Recipe: {recipe['Title']}")
            
            # Allergen check
            allergen_result = checker.comprehensive_allergen_check(
                recipe.to_dict(), 
                user_profile["allergies"]
            )
            
            # Recipe categorization
            ingredients = recipe.get("Cleaned_Ingredients", "[]")
            try:
                import ast
                ingredients_list = ast.literal_eval(ingredients)
            except:
                ingredients_list = [ingredients]
            
            category = ontology.categorize_recipe(
                recipe["Title"],
                ingredients_list,
                recipe.get("Instructions", "")
            )
            
            # Dietary restriction check
            dietary_result = ontology.validate_dietary_restriction(
                ingredients_list,
                user_profile["dietary_restrictions"][0]
            )
            
            print(f"  ✅ Safe for allergies: {allergen_result['safe']}")
            print(f"  🥬 Vegetarian: {dietary_result['compliant']}")
            print(f"  🌍 Cuisine: {category.cuisine_type}")
            print(f"  👨‍🍳 Methods: {', '.join(category.cooking_methods) if category.cooking_methods else 'None detected'}")
            print()
            
    except FileNotFoundError:
        print("Recipe file not found. Run this from the project root directory.")


if __name__ == "__main__":
    demo_allergen_checking()
    demo_recipe_categorization()
    demo_nutritional_enhancement()
    demo_integration_with_existing_system()