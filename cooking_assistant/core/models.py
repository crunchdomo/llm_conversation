"""Core data models and type definitions for the cooking assistant."""

from typing import Annotated, TypedDict, List, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class IntentDetection(BaseModel):
    """Identify user intent and extract key details"""
    intent_type: str = Field(..., description="Type of request: 'specific_recipe', 'similar_recipes', 'ingredient_search', 'substitution', 'recipe_selected'")
    target_recipe: str | None = Field(None, description="Recipe name if specified")
    ingredients: list[str] | None = Field(None, description="List of ingredients if provided")
    substitute_for: str | None = Field(None, description="Ingredient the user wants to substitute")
    substitute_with: str | None = Field(None, description="Ingredient the user wants to use instead")


class State(TypedDict):
    """LangGraph state for cooking conversations."""
    messages: Annotated[list[BaseMessage], add_messages]
    phase: Annotated[str, lambda _, x: x]
    max_retries: Annotated[int, lambda _, x: x]
    current_agent: Annotated[str, lambda _, x: x]
    user_profile: dict
    step_idx: int
    same_step_turns: int
    selected_recipe: dict | None
    chef_agent: Any  # Runtime agent instance
    trainee_agent: Any  # Runtime agent instance
    user_query: str | None
    clarified_topics: list[str]
    current_recipe: Annotated[dict | None, lambda _, x: x]
    adjusted_ingredients: Annotated[dict, lambda _, x: x]
    validated_substitutes: list[str]
    last_intent: IntentDetection | None


class UserProfile(BaseModel):
    """User profile for personalized cooking assistance."""
    experience_level: str = Field(default="beginner", description="Options: 'beginner', 'intermediate', 'advanced'")
    preferred_cuisine: str = Field(default="", description="Preferred cuisine type")
    allergies: list[str] = Field(default_factory=list, description="List of allergies")
    notes: str = Field(default="", description="Additional notes about cooking experience")


class Recipe(BaseModel):
    """Recipe data model."""
    title: str
    ingredients: list[str]
    instructions: str
    cleaned_ingredients: list[str] | None = None
    
    @classmethod
    def from_dataframe_row(cls, row) -> 'Recipe':
        """Create Recipe from pandas DataFrame row."""
        return cls(
            title=row['Title'],
            ingredients=row.get('Ingredients', []),
            instructions=row['Instructions'],
            cleaned_ingredients=row.get('Cleaned_Ingredients')
        )