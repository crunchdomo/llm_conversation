from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd
from experiment_config import ExperimentConfig
from thefuzz import process
import json
import time
from datetime import datetime

class IntentDetection(BaseModel):
    intent_type: str = Field(..., description="Type of request")
    target_recipe: str | None = Field(None, description="Recipe name if specified")
    ingredients: List[str] | None = Field(None, description="List of ingredients")
    substitute_for: str | None = Field(None, description="Ingredient to substitute")
    substitute_with: str | None = Field(None, description="Replacement ingredient")

class ConfigurableChefAgent:
    def __init__(self, job_id: str, config: ExperimentConfig, recipes_df: pd.DataFrame, 
                 trainee_profile: Dict[str, Any] = None):
        self.job_id = job_id
        self.config = config
        self.recipes_df = recipes_df
        self.trainee_profile = trainee_profile or {}
        self.token_cost_log = []
        
        # Initialize LLM based on config
        self.llm = ChatOpenAI(model=config.chef_model)
        if config.use_structured_output:
            self.structured_llm = ChatOpenAI(model=config.chef_model).with_structured_output(IntentDetection)
        else:
            self.structured_llm = self.llm
            
        self.system_message = self._build_system_message()
    
    def _build_system_message(self) -> SystemMessage:
        """Build system message based on configuration"""
        base_instructions = """You are ChefAI, a cooking assistant."""
        
        # Adapt based on trainee experience level from config
        experience_level = self.config.trainee_experience_level
        if experience_level == "beginner":
            base_instructions += " Provide detailed explanations and encourage questions."
        elif experience_level == "advanced":
            base_instructions += " Be concise and focus on key techniques."
        
        # Add allergy handling if enabled
        if self.config.enable_allergy_warnings:
            allergies = self.trainee_profile.get("allergies", [])
            if allergies:
                base_instructions += f" NEVER suggest recipes with: {', '.join(allergies)}."
        
        return SystemMessage(content=base_instructions)
    
    def detect_intent(self, user_query: str) -> IntentDetection:
        """Detect user intent with optional structured output"""
        if self.config.use_structured_output:
            prompt = f"Analyze this cooking query: {user_query}"
            return self.structured_llm.invoke(prompt)
        else:
            # Fallback to simple keyword matching
            if any(word in user_query.lower() for word in ["substitute", "replace", "instead"]):
                return IntentDetection(intent_type="substitution")
            elif "ingredient" in user_query.lower():
                return IntentDetection(intent_type="ingredient_search")
            else:
                return IntentDetection(intent_type="specific_recipe", target_recipe=user_query)
    
    def select_recipe(self, user_query: str) -> Dict[str, Any]:
        """Recipe selection with configurable search methods"""
        intent = self.detect_intent(user_query)
        
        if intent.intent_type == "specific_recipe":
            return self._find_specific_recipe(intent.target_recipe)
        elif intent.intent_type == "ingredient_search":
            return self._find_recipes_by_ingredients(intent.ingredients)
        elif intent.intent_type == "substitution":
            return self._handle_substitution(intent)
        
        return {"error": "Could not understand request"}
    
    def _find_specific_recipe(self, recipe_name: str) -> Dict[str, Any]:
        """Find recipe using configured search method"""
        if self.config.use_faiss_search:
            # Use FAISS semantic search (if available)
            return self._faiss_recipe_search(recipe_name)
        else:
            # Fallback to fuzzy matching
            return self._fuzzy_recipe_search(recipe_name)
    
    def _fuzzy_recipe_search(self, recipe_name: str) -> Dict[str, Any]:
        """Fuzzy string matching fallback"""
        recipe_titles = self.recipes_df['Title'].tolist()
        match, score = process.extractOne(recipe_name, recipe_titles)
        
        if score > self.config.similarity_threshold * 100:
            recipe = self.recipes_df[self.recipes_df['Title'] == match].iloc[0].to_dict()
            return {"recipe": recipe, "method": "fuzzy", "score": score}
        return {"error": "No recipe found"}
    
    def _faiss_recipe_search(self, recipe_name: str) -> Dict[str, Any]:
        """FAISS search if enabled and available"""
        # This would use your existing FAISS implementation
        # For now, fallback to fuzzy search
        return self._fuzzy_recipe_search(recipe_name)
    
    def _handle_substitution(self, intent: IntentDetection) -> Dict[str, Any]:
        """Handle substitution based on config method"""
        method = self.config.substitution_method
        
        if method == "llm_only":
            return self._llm_substitution(intent.substitute_for)
        elif method == "faiss_only":
            return self._faiss_substitution(intent.substitute_for)
        elif method == "ontology_only":
            return self._ontology_substitution(intent.substitute_for)
        else:  # hybrid
            return self._hybrid_substitution(intent.substitute_for)
    
    def _llm_substitution(self, ingredient: str) -> Dict[str, Any]:
        """LLM-only substitution"""
        prompt = f"Suggest 3 cooking substitutes for {ingredient}. Return as comma-separated list."
        response = self.llm.invoke([HumanMessage(content=prompt)]).content
        substitutes = [s.strip() for s in response.split(",")]
        return {"substitutes": substitutes, "method": "llm_only"}
    
    def _faiss_substitution(self, ingredient: str) -> Dict[str, Any]:
        """FAISS-only substitution"""
        # Placeholder - would use your FAISS ingredient search
        return {"substitutes": [f"{ingredient}_substitute"], "method": "faiss_only"}
    
    def _ontology_substitution(self, ingredient: str) -> Dict[str, Any]:
        """Ontology-only substitution"""
        # Placeholder - would use your ontology search
        return {"substitutes": [f"{ingredient}_ontology_sub"], "method": "ontology_only"}
    
    def _hybrid_substitution(self, ingredient: str) -> Dict[str, Any]:
        """Hybrid approach combining multiple methods"""
        substitutes = []
        methods_used = []
        
        if self.config.use_llm_validation:
            llm_subs = self._llm_substitution(ingredient)["substitutes"]
            substitutes.extend(llm_subs)
            methods_used.append("llm")
        
        if self.config.use_ontology_substitution:
            ont_subs = self._ontology_substitution(ingredient)["substitutes"]
            substitutes.extend(ont_subs)
            methods_used.append("ontology")
        
        # Remove duplicates while preserving order
        unique_substitutes = list(dict.fromkeys(substitutes))
        
        return {
            "substitutes": unique_substitutes[:3],  # Top 3
            "method": "hybrid",
            "methods_used": methods_used
        }

class ConfigurableTraineeAgent:
    def __init__(self, job_id: str, config: ExperimentConfig, trainee_profile: Dict[str, Any] = None):
        self.job_id = job_id
        self.config = config
        self.trainee_profile = trainee_profile or {}
        self.token_cost_log = []
        self.clarification_count = 0
        
        self.llm = ChatOpenAI(model=config.trainee_model)
    
    def generate_response(self, chef_message: str, step_number: int) -> str:
        """Generate trainee response based on configuration"""
        experience_level = self.config.trainee_experience_level
        
        # Advanced trainees ask fewer questions
        if experience_level == "advanced":
            if "next" in chef_message.lower() or step_number > 1:
                return "next"
            else:
                return self._ask_advanced_question(chef_message)
        
        # Beginners ask more questions but respect limits
        elif experience_level == "beginner":
            if self.clarification_count >= self.config.max_clarification_turns:
                self.clarification_count = 0
                return "next"
            else:
                return self._ask_beginner_question(chef_message)
        
        # Intermediate behavior
        else:
            return self._ask_intermediate_question(chef_message)
    
    def _ask_advanced_question(self, chef_message: str) -> str:
        """Advanced trainee asks minimal, specific questions"""
        if any(word in chef_message.lower() for word in ["temperature", "time", "doneness"]):
            self.clarification_count += 1
            return "What's the exact temperature/timing for this step?"
        return "next"
    
    def _ask_beginner_question(self, chef_message: str) -> str:
        """Beginner asks more questions"""
        question_triggers = ["heat", "cook", "add", "mix", "prepare"]
        if any(trigger in chef_message.lower() for trigger in question_triggers):
            self.clarification_count += 1
            return f"Can you explain how to {question_triggers[0]} properly?"
        return "next"
    
    def _ask_intermediate_question(self, chef_message: str) -> str:
        """Intermediate level questions"""
        if self.clarification_count < 2 and any(word in chef_message.lower() for word in ["technique", "method"]):
            self.clarification_count += 1
            return "What's the best technique for this step?"
        return "next"