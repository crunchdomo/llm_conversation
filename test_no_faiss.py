#!/usr/bin/env python3
"""Test conversation without FAISS indexing."""

import pandas as pd
from cooking_assistant.core.models import UserProfile, State
from cooking_assistant.core.agents.chef_agent import ChefAgent
from cooking_assistant.core.agents.trainee_agent import TraineeAgent
from cooking_assistant.core.graph.workflow import build_cooking_graph
from cooking_assistant.core.utils import generate_job_id
from langchain_core.messages import AIMessage

def test_minimal_conversation():
    """Test conversation flow without FAISS."""
    print("Testing minimal conversation...")
    
    # Create minimal setup
    job_id = generate_job_id()
    df = pd.read_csv("13k-recipes.csv").head(5)  # Just 5 recipes
    
    # Mock agents (without FAISS)
    class MockFAISS:
        def build_recipe_index(self, *args): pass
        def build_ingredient_index(self, *args): pass
        recipe_index = None
        ingredient_index = None
    
    class MockRecipeSearcher:
        def __init__(self, df): 
            self.df = df
        def search_by_ingredients(self, *args):
            return df.head(1)  # Return first recipe
    
    class MockIngredientSubstituter:
        def __init__(self, *args): pass
    
    # Create mock chef that just returns test responses
    chef = ChefAgent(job_id, df, MockFAISS(), MockRecipeSearcher(df), MockIngredientSubstituter())
    trainee = TraineeAgent(job_id)
    
    # Override chef respond method for testing
    original_respond = chef.respond
    def mock_respond(conversation, prompt):
        print(f"🔥 CHEF CALLED with prompt: {prompt[:50]}...")
        return f"Chef response to: {prompt[:30]}..."
    chef.respond = mock_respond
    
    # Test the workflow
    user_profile = UserProfile(experience_level="beginner")
    
    # Create state with selected recipe
    initial_state = {
        "messages": [chef.system_message],
        "phase": "ingredient_check",
        "user_profile": user_profile.model_dump(),
        "chef_agent": chef,
        "trainee_agent": trainee,
        "selected_recipe": {
            "Title": "Test Cookies", 
            "Cleaned_Ingredients": "['flour', 'sugar', 'butter']",
            "Instructions": "Preheat oven to 350F. Mix flour and sugar in a bowl. Add butter and mix well. Form into balls and place on baking sheet. Bake for 15 minutes until golden brown."
        },
        "step_idx": 0,
        "current_agent": None,
        "max_retries": 0,
        "user_query": "I want to make cookies",
        "clarified_topics": [],
        "current_recipe": None,
        "adjusted_ingredients": {},
        "validated_substitutes": [],
        "last_intent": None,
        "same_step_turns": 0
    }
    
    # Build and test graph
    graph = build_cooking_graph()
    
    print("Starting test conversation...")
    step = 0
    for output in graph.stream(initial_state):
        step += 1
        phase = output.get('phase', 'unknown')
        messages = output.get('messages', [])
        print(f"Step {step}: Phase={phase}, Messages={len(messages)}")
        
        if step > 5:  # Limit test
            break
    
    print("Test completed!")

if __name__ == "__main__":
    test_minimal_conversation()