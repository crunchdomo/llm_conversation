"""Recipe search utilities using fuzzy matching and semantic search."""

import re
import ast
import pandas as pd
from thefuzz import process
from typing import Optional


class RecipeSearcher:
    """Recipe search and retrieval utilities."""
    
    def __init__(self, recipes_df: pd.DataFrame):
        self.recipes_df = recipes_df
        
    def get_recipe_by_name(self, name: str, threshold: int = 80) -> Optional[dict]:
        """Find recipe by name using fuzzy matching."""
        recipe_titles = self.recipes_df['Title'].tolist()
        match, score = process.extractOne(name, recipe_titles)
        
        if score > threshold:
            return self.recipes_df[self.recipes_df['Title'] == match].iloc[0].to_dict()
        return None
    
    def get_similar_recipes(self, name: str, limit: int = 5) -> pd.DataFrame:
        """Get recipes with similar names."""
        return self.recipes_df[
            self.recipes_df['Title'].str.lower().str.contains(name.lower(), na=False)
        ].head(limit)
    
    def contains_allergen(self, ingredients: str, allergies: list[str]) -> bool:
        """Check if recipe contains any allergens."""
        try:
            ingredients_list = ast.literal_eval(ingredients)
        except Exception:
            ingredients_list = [ingredients]
            
        return any(
            allergen.lower() in " ".join(ingredients_list).lower() 
            for allergen in allergies
        )
    
    def extract_recipes_from_message(self, message: str) -> list[str]:
        """Extract recipe names from a message string."""
        match = re.search(r"Matching recipes: (.+)", message)
        if match:
            return [r.strip() for r in match.group(1).split(",") if r.strip()]
        return []
    
    def parse_recipe_steps(self, recipe_text: str) -> list[str]:
        """Parse recipe instructions into individual steps."""
        steps = [s.strip() for s in re.split(r'\n+|\d+\.', recipe_text) if s.strip()]
        return steps