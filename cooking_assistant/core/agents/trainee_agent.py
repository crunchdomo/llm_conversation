"""Trainee agent for simulating cooking student interactions."""

import os
from datetime import datetime
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.callbacks import get_openai_callback
from langsmith import traceable

from ..models import State, UserProfile


class TraineeAgent:
    """AI Trainee agent for simulating cooking student behavior."""
    
    def __init__(self, job_id: str, trainee_experience_log: Optional[UserProfile] = None, 
                 conversation_mode: str = "mixed"):
        self.llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.environ["OPENAI_API_KEY"])
        self.job_id = job_id
        self.current_step = 0
        self.recipe_data = None
        self.conversation_mode = conversation_mode
        self.token_cost_log = []
        self.trainee_experience_log = trainee_experience_log or UserProfile()
        
    @traceable(run_type="chain", name="Trainee Response", metadata={"role": "trainee"})
    def generate_response(self, chef_message: str, num_steps: int) -> str:
        """Generate trainee response based on experience level."""
        experience_level = self.trainee_experience_log.experience_level
        notes = self.trainee_experience_log.notes

        # Adjust question-asking behavior based on experience
        if experience_level == "advanced":
            question_instruction = (
                "ONLY ask a question if this step is ambiguous or unusually challenging for an expert cook. "
                "If everything is clear, say 'next'. Do NOT ask about basic techniques or substitutions."
            )
        elif experience_level == "intermediate":
            question_instruction = (
                "Ask a question if you are unsure about a technique or ingredient. "
                "Otherwise, say 'next'."
            )
        else:  # beginner
            question_instruction = (
                "If you have any doubt about the technique, ingredient, or process in this step, ask ONE SHORT, direct question. "
                "Otherwise, say 'next'."
            )

        # Use notes for further customization
        if notes and "never" in notes.lower():
            question_instruction += (
                f" You have noted: {notes}. If this step involves something you have never done, ask for extra explanation."
            )

        prompt = f"""
        You are a cooking trainee with experience level: {experience_level}.
        {f'Notes: {notes}' if notes else ''}
        Your goal is to act in accordance with your experience.

        Last chef instruction: {chef_message}

        INSTRUCTIONS:
        - If you are advanced, only ask about unclear or complex steps. If everything is clear, say 'next'.
        - If you are a beginner, ask about anything you are unsure about.
        {question_instruction}
        """
        
        with get_openai_callback() as cb:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            self.token_cost_log.append({
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "cost": cb.total_cost,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt
            })
            self._update_running_totals()
            print(f"TraineeAgent LLM call: {cb.total_tokens} tokens, ${cb.total_cost:.5f}")
            return response.content.strip().lower()
        
    def confirm_ingredients(self, state: State) -> str:
        """Automatically confirm ingredient readiness."""
        try:
            ingredients_msg = state["messages"][-1].content
            allergies = state["user_profile"].get("allergies", [])
            
            # Check for allergens automatically
            has_allergen = any(allergen.lower() in ingredients_msg.lower() for allergen in allergies)
            
            if has_allergen:
                return f"I notice this recipe contains {', '.join(allergies)} which I'm allergic to. Can we find an alternative?"
            else:
                return "Yes, I have all the ingredients ready to proceed."
                
        except Exception as e:
            print(f"Error in confirm_ingredients: {e}")
            return "Yes, let's proceed with the recipe."
        
    def choose_recipe(self, recipe_list: List[str], state: State) -> str:
        """Automatically choose a recipe based on preferences."""
        try:
            preferences = state["user_profile"]
            preferred_cuisine = preferences.get("preferred_cuisine", "").lower()
            allergies = preferences.get("allergies", [])
            
            # Pick first recipe that matches preferences and avoids allergens
            for recipe in recipe_list:
                # Check for preferred cuisine
                if preferred_cuisine and preferred_cuisine in recipe.lower():
                    # Check for allergens (basic check)
                    if not any(allergen.lower() in recipe.lower() for allergen in allergies):
                        return recipe
            
            # Fallback: return first recipe
            return recipe_list[0] if recipe_list else ""
            
        except Exception as e:
            print(f"Error choosing recipe: {e}")
            return recipe_list[0] if recipe_list else ""

    def _update_running_totals(self):
        """Update cumulative token and cost totals."""
        total_tokens = 0
        total_cost = 0
        for entry in self.token_cost_log:
            total_tokens += entry["total_tokens"]
            total_cost += entry["cost"]
            entry["cumulative_tokens"] = total_tokens
            entry["cumulative_cost"] = total_cost