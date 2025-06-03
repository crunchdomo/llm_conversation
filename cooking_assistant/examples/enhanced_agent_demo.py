#!/usr/bin/env python3
"""Demo of enhanced chef agent with ontology integration."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_chef_agent import OntologyEnhancedChefAgent
from core.models import UserProfile
import pandas as pd


def demo_enhanced_chef():
    """Demonstrate enhanced chef agent capabilities."""
    print("🧑‍🍳 ENHANCED CHEF AGENT DEMO")
    print("=" * 50)
    
    # Load a sample recipe
    try:
        df = pd.read_csv("../../13k-recipes.csv")
        sample_recipe = df[df['Title'].str.contains('pasta', case=False, na=False)].iloc[0]
    except:
        # Fallback sample recipe
        sample_recipe = {
            'Title': 'Creamy Mushroom Pasta',
            'Cleaned_Ingredients': "['pasta', 'mushrooms', 'heavy cream', 'parmesan cheese', 'butter', 'garlic']",
            'Instructions': 'Cook pasta according to package directions. Sauté mushrooms in butter. Add cream and cheese. Toss with pasta.'
        }
    
    # Create user profile with restrictions
    user_profile = UserProfile(
        experience_level="beginner",
        allergies=["dairy"],
        dietary_restrictions=["vegetarian"],
        preferred_cuisine="italian"
    )
    
    # Initialize enhanced chef (mock job_id and data)
    job_id = "demo_123"
    recipes_df = pd.DataFrame([sample_recipe])  # Mock dataframe
    
    chef = OntologyEnhancedChefAgent(
        job_id=job_id,
        recipes_df=recipes_df,
        faiss_index=None,  # Mock for demo
        recipe_searcher=None,  # Mock for demo
        ingredient_substituter=None,  # Mock for demo
        trainee_experience_log=user_profile
    )
    
    # Simulate state
    state = {
        "user_profile": user_profile.model_dump(),
        "selected_recipe": sample_recipe,
        "messages": []
    }
    
    print(f"👤 User Profile:")
    print(f"   Experience: {user_profile.experience_level}")
    print(f"   Allergies: {user_profile.allergies}")
    print(f"   Dietary: {user_profile.dietary_restrictions}")
    print(f"   Cuisine: {user_profile.preferred_cuisine}")
    print()
    
    print(f"🍝 Selected Recipe: {sample_recipe['Title']}")
    print()
    
    # Test enhanced ingredient checking
    print("📋 ENHANCED INGREDIENT CHECK:")
    print("-" * 30)
    ingredient_response = chef.enhanced_ingredient_check(state)
    print(ingredient_response)
    print()
    
    # Test recipe categorization
    print("📊 RECIPE CATEGORIZATION:")
    print("-" * 30)
    categorization = chef.categorize_current_recipe(state)
    print(categorization)
    print()
    
    # Test safety report generation
    print("🛡️  SAFETY ANALYSIS:")
    print("-" * 30)
    safety_report = chef._generate_safety_report(sample_recipe, user_profile.model_dump())
    
    if safety_report["has_concerns"]:
        print("❌ Safety concerns detected:")
        print(safety_report["message"])
    else:
        print("✅ No safety concerns detected!")
    print()
    
    # Test alternative suggestions
    print("💡 ALTERNATIVE SUGGESTIONS:")
    print("-" * 30)
    alternatives = chef.suggest_recipe_alternatives(state, dietary_focus="proteins")
    print(alternatives)
    print()


if __name__ == "__main__":
    demo_enhanced_chef()