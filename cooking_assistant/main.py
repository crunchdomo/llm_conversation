"""Main orchestration script for cooking conversation system."""

import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from langsmith import Client, traceable
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.messages import HumanMessage

from .core.models import UserProfile, State
from .core.search.faiss_index import FAISSIndex
from .core.search.recipe_search import RecipeSearcher
from .core.search.ingredient_substitution import IngredientSubstituter
from .core.agents.chef_agent import ChefAgent
from .core.agents.trainee_agent import TraineeAgent
from .core.graph.workflow import build_cooking_graph
from .core.utils import (
    save_conversation_to_json, 
    print_token_summary, 
    visualize_graph,
    generate_job_id
)
from .scenarios import get_scenario_by_name, list_all_scenarios


class CookingAssistant:
    """Main cooking assistant orchestrator."""
    
    def __init__(self, recipes_csv_path: str = "13k-recipes.csv", sample_size: int = 100):
        print(f"Loading recipes from {recipes_csv_path}...")
        self.recipes_df = pd.read_csv(recipes_csv_path)
        print(f"Loaded {len(self.recipes_df)} total recipes")
        
        # Use sample for faster testing (default to 1000 for performance)
        if sample_size and sample_size < len(self.recipes_df):
            print(f"Using sample of {sample_size} recipes for faster performance...")
            self.recipes_df = self.recipes_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        elif sample_size is None:
            # Default to 100 recipes for reasonable performance during testing
            sample_size = 100
            print(f"Using default sample of {sample_size} recipes for reasonable performance...")
            self.recipes_df = self.recipes_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        self.faiss_index = FAISSIndex()
        self.recipe_searcher = RecipeSearcher(self.recipes_df)
        self.ingredient_substituter = IngredientSubstituter(self.faiss_index)
        
        # Build FAISS indices
        print("Building FAISS indices...")
        self.faiss_index.build_recipe_index(self.recipes_df)
        self.faiss_index.build_ingredient_index(self.recipes_df)
        print(f"Recipe index: {self.faiss_index.recipe_index.ntotal} recipes")
        print(f"Ingredient index: {self.faiss_index.ingredient_index.ntotal} ingredients")
    
    def run_scenario(self, scenario_name: str, visualize: bool = False) -> str:
        """Run a predefined scenario."""
        scenario = get_scenario_by_name(scenario_name)
        print(f"\n=== Running Scenario: {scenario.name} ===")
        print(f"Description: {scenario.description}")
        print(f"User Query: {scenario.user_query}")
        
        return self.run_conversation(
            user_query=scenario.user_query,
            user_profile=scenario.user_profile,
            visualize=visualize
        )
    
    def run_conversation(self, user_query: str, user_profile: UserProfile = None, 
                        visualize: bool = False) -> str:
        """Run a cooking conversation with given parameters."""
        job_id = generate_job_id()
        
        if user_profile is None:
            user_profile = UserProfile()
        
        # Initialize agents
        chef = ChefAgent(
            job_id=job_id,
            recipes_df=self.recipes_df,
            faiss_index=self.faiss_index,
            recipe_searcher=self.recipe_searcher,
            ingredient_substituter=self.ingredient_substituter,
            trainee_experience_log=user_profile
        )
        
        trainee = TraineeAgent(
            job_id=job_id,
            trainee_experience_log=user_profile
        )
        
        # Build graph
        graph = build_cooking_graph()
        
        if visualize:
            print("\n=== Graph Structure ===")
            visualize_graph(graph)
        
        # Initialize state
        initial_state = {
            "messages": [chef.system_message],
            "phase": "introduction",
            "user_profile": user_profile.model_dump(),
            "step_idx": 0,
            "retries": 0,
            "same_step_turns": 0,
            "selected_recipe": None,
            "chef_agent": chef,
            "trainee_agent": trainee,
            "user_query": user_query,
            "clarified_topics": [],
            "validated_substitutes": [],
            "last_intent": None,
            "max_retries": 0,
            "current_agent": None,
            "current_recipe": None,
            "adjusted_ingredients": {}
        }
        
        # Run conversation
        config = {"recursion_limit": 200}
        final_state = None
        
        print(f"\n=== Starting Conversation ===")
        print(f"User Profile: {user_profile.experience_level} level, allergies: {user_profile.allergies}")
        
        try:
            message_count = len(initial_state["messages"])
            step_count = 0
            for output in graph.stream(initial_state, config=config, stream_mode="values"):
                step_count += 1
                phase = output.get('phase', 'unknown')
                print(f"\n--- Step {step_count} | Phase: {phase} ---")
                
                # Print new messages
                messages = output.get("messages", [])
                new_messages = messages[message_count:]
                
                for msg in new_messages:
                    if hasattr(msg, 'content'):
                        # Determine speaker based on message type
                        if isinstance(msg, HumanMessage):
                            speaker = "👤 Trainee"
                        else:
                            speaker = "👨‍🍳 Chef"
                        print(f"{speaker}: {msg.content}")
                        print()
                
                message_count = len(messages)
                
                if phase == "done" or step_count > 8:  # Limit to 8 steps for testing
                    if step_count > 8:
                        print("⏰ Conversation limit reached (8 steps) - stopping for demo")
                    final_state = output
                    break
        
        except Exception as e:
            print(f"Conversation error: {e}")
            final_state = output if 'output' in locals() else initial_state
        
        # Save results
        if final_state:
            filename = save_conversation_to_json(job_id, final_state, chef, trainee)
            print(f"\nConversation saved to: {filename}")
        
        # Print summary
        print_token_summary(chef, trainee)
        
        return job_id

    def run_interactive(self):
        """Run interactive cooking session."""
        print("=== Interactive Cooking Assistant ===")
        user_query = input("What would you like to cook? ")
        
        # Get user profile
        experience = input("Experience level (beginner/intermediate/advanced): ").strip() or "beginner"
        allergies_input = input("Any allergies? (comma-separated, or press enter for none): ").strip()
        allergies = [a.strip() for a in allergies_input.split(",")] if allergies_input else []
        cuisine = input("Preferred cuisine (or press enter for any): ").strip()
        
        user_profile = UserProfile(
            experience_level=experience,
            allergies=allergies,
            preferred_cuisine=cuisine
        )
        
        return self.run_conversation(user_query, user_profile, visualize=True)

    def list_scenarios(self):
        """List all available test scenarios."""
        scenarios = list_all_scenarios()
        print("Available scenarios:")
        for scenario in scenarios:
            print(f"  - {scenario}")


def main():
    """Main entry point with CLI options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cooking Assistant")
    parser.add_argument("--scenario", help="Run specific scenario")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--visualize", action="store_true", help="Show graph visualization")
    parser.add_argument("--recipes-csv", default="13k-recipes.csv", help="Path to recipes CSV")
    parser.add_argument("--sample-size", type=int, help="Use sample of N recipes for faster testing")
    
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = CookingAssistant(args.recipes_csv, sample_size=args.sample_size)
    
    if args.list_scenarios:
        assistant.list_scenarios()
    elif args.scenario:
        # Disable tracing to avoid LangSmith warnings
        assistant.run_scenario(args.scenario, visualize=args.visualize)
    elif args.interactive:
        # Disable tracing to avoid LangSmith warnings  
        assistant.run_interactive()
    else:
        print("Use --help for usage options")


if __name__ == "__main__":
    main()