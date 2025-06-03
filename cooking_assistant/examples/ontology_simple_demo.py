#!/usr/bin/env python3
"""Simple demo of food ontology capabilities without API dependencies."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.food_ontology import FoodOntology, EnhancedAllergenChecker
import ast


def demo_practical_ontology_usage():
    """Show practical usage of food ontology for real cooking scenarios."""
    print("🍽️  PRACTICAL FOOD ONTOLOGY DEMO")
    print("=" * 60)
    
    # Initialize ontology
    ontology = FoodOntology()
    checker = EnhancedAllergenChecker(ontology)
    
    # Real recipe examples from your dataset
    sample_recipes = [
        {
            "Title": "Classic Chicken Alfredo",
            "Cleaned_Ingredients": "['fettuccine pasta', 'chicken breast', 'heavy cream', 'parmesan cheese', 'butter', 'garlic']",
            "Instructions": "Cook pasta. Sauté chicken. Make alfredo sauce with cream, butter, and cheese."
        },
        {
            "Title": "Thai Green Curry",
            "Cleaned_Ingredients": "['coconut milk', 'green curry paste', 'chicken', 'thai basil', 'bell peppers', 'fish sauce']",
            "Instructions": "Simmer coconut milk with curry paste. Add chicken and vegetables."
        },
        {
            "Title": "Mediterranean Quinoa Bowl",
            "Cleaned_Ingredients": "['quinoa', 'chickpeas', 'cucumber', 'tomatoes', 'feta cheese', 'olive oil', 'lemon']",
            "Instructions": "Cook quinoa. Mix with vegetables, chickpeas, and feta. Dress with olive oil and lemon."
        }
    ]
    
    user_scenarios = [
        {
            "name": "Dairy-Free Family",
            "allergies": ["dairy"],
            "dietary_restrictions": [],
            "goals": ["family-friendly", "quick"]
        },
        {
            "name": "Vegetarian Fitness Enthusiast", 
            "allergies": [],
            "dietary_restrictions": ["vegetarian"],
            "goals": ["high-protein", "nutritious"]
        },
        {
            "name": "Gluten-Free Beginner",
            "allergies": ["gluten"],
            "dietary_restrictions": [],
            "goals": ["simple", "safe"]
        }
    ]
    
    for user in user_scenarios:
        print(f"\n👤 USER SCENARIO: {user['name']}")
        print(f"   Allergies: {user['allergies'] or 'None'}")
        print(f"   Diet: {user['dietary_restrictions'] or 'No restrictions'}")
        print(f"   Goals: {', '.join(user['goals'])}")
        print("   " + "─" * 50)
        
        for recipe in sample_recipes:
            print(f"\n🍽️  Analyzing: {recipe['Title']}")
            
            # Parse ingredients
            try:
                ingredients = ast.literal_eval(recipe['Cleaned_Ingredients'])
            except:
                ingredients = [recipe['Cleaned_Ingredients']]
            
            # Allergen check
            if user['allergies']:
                allergen_result = checker.comprehensive_allergen_check(recipe, user['allergies'])
                safety_status = "✅ SAFE" if allergen_result['safe'] else "❌ UNSAFE"
                print(f"   Allergen Safety: {safety_status}")
                
                if not allergen_result['safe']:
                    critical = allergen_result['severity']['critical']
                    if critical:
                        allergens = [f"{m.ingredient} ({m.allergen_type})" for m in critical]
                        print(f"      🚨 Critical: {', '.join(allergens)}")
            
            # Dietary restriction check
            if user['dietary_restrictions']:
                for restriction in user['dietary_restrictions']:
                    validation = ontology.validate_dietary_restriction(ingredients, restriction)
                    compliance = "✅ COMPLIANT" if validation['compliant'] else "❌ NON-COMPLIANT"
                    print(f"   {restriction.title()}: {compliance}")
                    
                    if not validation['compliant']:
                        print(f"      ⚠️  Issues: {', '.join(validation['violations'])}")
            
            # Recipe categorization
            category = ontology.categorize_recipe(recipe['Title'], ingredients, recipe['Instructions'])
            print(f"   Cuisine: {category.cuisine_type or 'Not detected'}")
            print(f"   Methods: {', '.join(category.cooking_methods) if category.cooking_methods else 'Basic prep'}")
            
            # Nutritional profile
            if category.nutritional_profile:
                nutrition = [(k, v) for k, v in category.nutritional_profile.items() if v > 0]
                if nutrition:
                    nutrition_str = ', '.join([f"{k}: {v:.0%}" for k, v in nutrition])
                    print(f"   Nutrition: {nutrition_str}")
            
            print()
    
    # Demonstrate substitution suggestions
    print("\n💡 SUBSTITUTION SUGGESTIONS")
    print("=" * 40)
    
    print("\nFor dairy-free alternatives:")
    dairy_subs = {
        "heavy cream": ["coconut cream", "cashew cream", "oat cream"],
        "butter": ["olive oil", "coconut oil", "vegan butter"],
        "parmesan cheese": ["nutritional yeast", "cashew parmesan", "dairy-free cheese"]
    }
    
    for original, alternatives in dairy_subs.items():
        print(f"   {original} → {', '.join(alternatives)}")
    
    print("\nFor high-protein vegetarian options:")
    protein_subs = {
        "chicken": ["tofu", "tempeh", "seitan", "chickpeas"],
        "ground beef": ["lentils", "black beans", "mushrooms", "plant-based meat"],
        "fish": ["hemp hearts", "nutritional yeast", "algae-based alternatives"]
    }
    
    for original, alternatives in protein_subs.items():
        print(f"   {original} → {', '.join(alternatives)}")
    
    print("\n🎯 INTEGRATION RECOMMENDATIONS")
    print("=" * 40)
    print("""
1. 🔍 Recipe Selection Enhancement:
   - Pre-filter recipes by allergens and dietary restrictions
   - Show cuisine type and cooking method badges
   - Highlight nutritional strengths

2. 🛡️  Safety Warnings:
   - Real-time allergen detection during recipe browsing
   - Severity-based warning system (critical/warning/caution)
   - Automatic substitution suggestions

3. 📊 Smart Categorization:
   - Auto-tag recipes with cuisine types and cooking methods
   - Filter by dietary compliance (vegetarian, vegan, keto, etc.)
   - Nutritional profiling for meal planning

4. 💡 Intelligent Substitutions:
   - Combine ontology taxonomy with your FAISS similarity
   - Prioritize allergen-safe and dietary-compliant alternatives
   - Context-aware suggestions based on cooking method

5. 🎨 User Experience:
   - Visual allergen warnings in recipe cards
   - Dietary restriction badges
   - Cooking method icons
   - Nutritional balance indicators
""")


if __name__ == "__main__":
    demo_practical_ontology_usage()