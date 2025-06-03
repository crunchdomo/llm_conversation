#!/usr/bin/env python3
"""Test conversation with any available recipe."""

import pandas as pd
from cooking_assistant.main import CookingAssistant
from cooking_assistant.scenarios import get_scenario_by_name
from cooking_assistant.core.models import UserProfile

def test_with_available_recipe():
    """Test conversation with whatever recipe is available."""
    print("Loading small sample to see what recipes we have...")
    
    # Load recipes and see what's available
    df = pd.read_csv("13k-recipes.csv")
    sample_df = df.sample(n=30, random_state=42)
    
    print("Available recipes in sample:")
    for i, recipe in enumerate(sample_df.head(10).itertuples()):
        print(f"{i+1}. {recipe.Title}")
    
    # Use the first recipe title for our test
    first_recipe = sample_df.iloc[0]
    test_query = f"I want to make {first_recipe.Title}"
    
    print(f"\nTesting with query: '{test_query}'")
    
    # Create assistant and run conversation
    assistant = CookingAssistant(sample_size=30)
    
    # Create custom user profile
    user_profile = UserProfile(
        experience_level="beginner",
        allergies=[],
        preferred_cuisine=""
    )
    
    # Run conversation
    job_id = assistant.run_conversation(
        user_query=test_query,
        user_profile=user_profile,
        visualize=False
    )
    
    print(f"\nConversation completed! Job ID: {job_id}")

if __name__ == "__main__":
    test_with_available_recipe()