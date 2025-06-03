"""Chef agent for cooking assistance and recipe guidance."""

import os
import time
from datetime import datetime
from typing import List, Optional
from anthropic import APIError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from ..models import IntentDetection, State, UserProfile


class ChefAgent:
    """AI Chef agent for recipe guidance and cooking assistance."""
    
    def __init__(self, job_id: str, recipes_df, faiss_index, recipe_searcher, 
                 ingredient_substituter, trainee_experience_log: Optional[UserProfile] = None):
        self.llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.environ["OPENAI_API_KEY"])
        self.structured_llm = ChatOpenAI(
            model="gpt-4o-mini", 
            openai_api_key=os.environ["OPENAI_API_KEY"]
        ).with_structured_output(IntentDetection)
        
        self.job_id = job_id
        self.token_cost_log = []
        self.recipes_df = recipes_df
        self.faiss_index = faiss_index
        self.recipe_searcher = recipe_searcher
        self.ingredient_substituter = ingredient_substituter
        self.trainee_experience_log = trainee_experience_log
        self.system_message = self._build_system_message()

    def detect_intent(self, user_query: str) -> IntentDetection:
        """Analyze cooking-related query and classify intent."""
        prompt = f"""Analyze this cooking-related query:
        {user_query}

        Classify the intent:
        - 'specific_recipe' if the query matches or closely resembles the name of a known recipe
        - 'similar_recipes' if requesting variations
        - 'ingredient_search' if listing ingredients they have
        - 'substitution' if asking about ingredient substitutions

        Extract recipe names or ingredients as needed."""
        return self.structured_llm.invoke(prompt)

    def select_recipe(self, user_query: str, state: State) -> str:
        """Select or find recipes based on user query."""
        try:
            intent = self.detect_intent(user_query)
            
            if intent.intent_type == "specific_recipe":
                recipe = self.recipe_searcher.get_recipe_by_name(intent.target_recipe)
                if not recipe:
                    return "Recipe not found. Please try another."
                
                state["selected_recipe"] = recipe
                return f"Selected recipe: {recipe['Title']}"
            
            elif intent.intent_type == "similar_recipes":
                similar = self.recipe_searcher.get_similar_recipes(intent.target_recipe)
                if similar.empty:
                    return "No similar recipes found."
                return f"Similar recipes: {', '.join(similar['Title'].tolist())}"
            
            elif intent.intent_type == "ingredient_search":
                matches = self.faiss_index.search_recipes_by_ingredients(
                    intent.ingredients, self.recipes_df, k=5
                )
                if matches.empty:
                    return "No recipes match those ingredients."
                return f"Matching recipes: {', '.join(matches['Title'].tolist())}"
            
            elif intent.intent_type == "substitution":
                return self._handle_substitution_request(intent, state)
            
            else:
                return "Please clarify your request"
                
        except Exception as e:
            print(f"Recipe selection error: {e}")
            return "Error finding recipes. Please try again."

    def _handle_substitution_request(self, intent: IntentDetection, state: State) -> str:
        """Handle ingredient substitution requests."""
        recipe_name = intent.target_recipe or state.get("current_recipe")
        if not recipe_name:
            return "Please select a recipe first before requesting ingredient substitutions."
            
        recipe = state.get("selected_recipe")
        if not recipe:
            recipe = self.recipe_searcher.get_recipe_by_name(recipe_name)
            if not recipe:
                return f"Could not find recipe: {recipe_name}"
        
        # Generate LLM substitutes
        llm_subs = self.ingredient_substituter.generate_llm_substitutes(
            self, intent.substitute_for, recipe['Title'], recipe['Instructions']
        )
        
        # Get FAISS substitutes
        faiss_subs = [sub for sub, _ in self.faiss_index.find_ingredient_substitutes(
            intent.substitute_for, k=10
        )]
        
        # Combine and validate
        all_candidates = list(set(llm_subs + faiss_subs))
        validated_subs = self.ingredient_substituter.validate_with_llm(
            self, intent.substitute_for, all_candidates, recipe
        )
        
        state["validated_substitutes"] = validated_subs
        return (
            f"For {intent.substitute_for} in {recipe['Title']}, the best substitutes are:\n"
            + "\n".join(f"- {sub}" for sub in validated_subs)
            + "\nWhich would you like to use?"
        )

    def _build_system_message(self) -> SystemMessage:
        """Build system message based on trainee experience."""
        if self.trainee_experience_log:
            experience_level = self.trainee_experience_log.experience_level
            allergies = self.trainee_experience_log.allergies
            preferred_cuisine = self.trainee_experience_log.preferred_cuisine
            notes = self.trainee_experience_log.notes
        else:
            experience_level = 'unknown'
            allergies = []
            preferred_cuisine = 'any'
            notes = ''

        allergy_str = ", ".join(allergies) if allergies else "none"

        experience_info = f"""
        User Profile:
        - Experience level: {experience_level}
        - Allergies: {allergy_str}
        - Preferred cuisine: {preferred_cuisine}
        - Notes: {notes}

        IMPORTANT INSTRUCTIONS:
        - NEVER suggest or proceed with any recipe or step that contains any of the user's allergens: {allergy_str}.
        - If the user requests a recipe with an allergen, gently warn them and suggest safe alternatives.
        - ALWAYS adapt your explanations to the user's experience level.
        - Before each step, check if the user is comfortable and ready to proceed.
        """

        base_instructions = """
        Please follow these instructions carefully:

        1. Introduction:
        - Introduce yourself as ChefAI.
        - Briefly explain that you're here to help with recipe preparation.

        2. Recipe Confirmation:
        - Confirm the recipe name with the user.

        3. Ingredient Recall:
        - List all the ingredients required for the recipe.
        - Ask the user if they have all the ingredients ready.

        4. Step-by-Step Guidance:
        - Provide instructions one step at a time.
        - After each step, wait for the user to say "next" or ask a question before proceeding.
        - If the user asks a question, answer it thoroughly before continuing.

        5. Completion:
        - When all steps are complete, congratulate the user and ask if they need any final advice.

        Remember to maintain a friendly and encouraging tone throughout the interaction.
        """
        
        return SystemMessage(content=experience_info + base_instructions)

    @traceable(run_type="chain", name="Chef Response", metadata={"role": "chef"})
    def respond(self, conversation: List, prompt: str) -> str:
        """Generate chef response with LLM."""
        run = get_current_run_tree()
        if run:
            run.metadata["job_id"] = self.job_id
            run.metadata["run_id"] = str(run.id)
            
        messages = [self.system_message] + conversation + [HumanMessage(content=prompt)]
        max_retries = 5
        base_delay = 1

        for attempt in range(max_retries):
            try:
                with get_openai_callback() as cb:
                    response = self.llm.invoke(messages).content.strip()
                    
                    self.token_cost_log.append({
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_tokens": cb.total_tokens,
                        "cost": cb.total_cost,
                        "timestamp": datetime.now().isoformat(),
                        "prompt": prompt
                    })
                    self._update_running_totals()
                    print(f"ChefAgent LLM call: {cb.total_tokens} tokens, ${cb.total_cost:.5f}")
                    return response
                    
            except APIError as e:
                if "overloaded_error" in str(e):
                    delay = base_delay * (2 ** attempt)
                    print(f"OverloadedError: Retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise
                    
        raise Exception("Max retries exceeded")

    def _update_running_totals(self):
        """Update cumulative token and cost totals."""
        total_tokens = 0
        total_cost = 0
        for entry in self.token_cost_log:
            total_tokens += entry["total_tokens"]
            total_cost += entry["cost"]
            entry["cumulative_tokens"] = total_tokens
            entry["cumulative_cost"] = total_cost