"""Ingredient substitution utilities using LLM and ontology."""

import re
import numpy as np
from langchain_core.messages import HumanMessage
from typing import List


class IngredientSubstituter:
    """Handle ingredient substitution using multiple approaches."""
    
    def __init__(self, faiss_index, ontology_graph=None):
        self.faiss_index = faiss_index
        self.ontology_graph = ontology_graph
    
    def generate_llm_substitutes(self, agent, original_ingredient: str, recipe_title: str, instructions: str) -> List[str]:
        """Generate substitutes using LLM."""
        prompt = f"""Suggest 5 culinary substitutes for {original_ingredient} in '{recipe_title}':
        Recipe Context: {instructions}
        Consider:
        - Flavor compatibility
        - Texture properties
        - Cooking behavior
        Return ONLY a comma-separated list"""
        
        response = agent.llm.invoke([HumanMessage(content=prompt)]).content
        return [s.strip() for s in response.split(",") if s.strip()]
    
    def validate_formats_with_faiss(self, llm_subs: List[str], original: str, threshold: float = 0.8) -> List[str]:
        """Validate substitute formats using FAISS similarity."""
        if self.faiss_index.ingredient_index is None:
            return llm_subs[:5]
            
        valid = []
        original_formats = [
            ing for ing in self.faiss_index.unique_ingredients 
            if original in ing.lower()
        ]
        
        for sub in llm_subs:
            sub_embed = self.faiss_index.model.encode([sub], convert_to_numpy=True)
            D, I = self.faiss_index.ingredient_index.search(sub_embed, 5)
            
            for i, score in zip(I[0], D[0]):
                candidate = self.faiss_index.unique_ingredients[i]
                if any(fmt in candidate.lower() for fmt in original_formats):
                    valid.append(candidate)
                    break
                    
        return list(set(valid))[:5]
    
    def get_llm_compatibility_score(self, agent, original: str, substitute: str) -> float:
        """Get LLM compatibility score for substitution."""
        prompt = f"""Rate compatibility between {original} and {substitute} (1-10) considering:
        - Flavor profile
        - Texture when cooked
        - Common substitution practices
        Return ONLY the number"""
        
        try:
            response = agent.llm.invoke([HumanMessage(content=prompt)]).content
            return float(response.strip())
        except:
            return 5.0
    
    def hybrid_ranking(self, agent, original: str, candidates: List[str]) -> List[str]:
        """Rank substitutes using hybrid LLM + semantic similarity."""
        if not candidates:
            return []
            
        if self.faiss_index.ingredient_embeddings is None or original not in self.faiss_index.unique_ingredients:
            return candidates
            
        original_idx = self.faiss_index.unique_ingredients.index(original)
        original_embed = self.faiss_index.ingredient_embeddings[original_idx]
        
        scores = []
        for candidate in candidates:
            if candidate not in self.faiss_index.unique_ingredients:
                scores.append((candidate, 0.5))
                continue
                
            candidate_idx = self.faiss_index.unique_ingredients.index(candidate)
            faiss_score = 1 / (1 + np.linalg.norm(original_embed - self.faiss_index.ingredient_embeddings[candidate_idx]))
            
            llm_score = self.get_llm_compatibility_score(agent, original, candidate) / 10.0
            
            scores.append((candidate, 0.6 * llm_score + 0.4 * faiss_score))
        
        return [item[0] for item in sorted(scores, key=lambda x: x[1], reverse=True)]
    
    def validate_with_llm(self, chef_agent, original: str, candidates: List[str], recipe: dict) -> List[str]:
        """Validate substitutes with LLM and return top 3."""
        prompt = (
            f"Given the recipe '{recipe['Title']}', rank these as substitutes for '{original}':\n"
            f"{', '.join(candidates)}\n"
            "Provide ONLY the top 3 as a bullet list, no explanations."
        )
        response = chef_agent.llm.invoke([HumanMessage(content=prompt)]).content
        
        validated = [
            line.lstrip('-•* ').strip()
            for line in response.split('\n')
            if line.strip().startswith(('-', '*', '•'))
        ]
        return validated
    
    def apply_substitution(self, chef_agent, recipe: dict, original_ingredient: str, substitute: str, candidates: List[str]) -> str:
        """Apply substitution and update recipe instructions."""
        prompt = (
            f"Given the recipe '{recipe['Title']}', the user wants to substitute '{original_ingredient}' with '{substitute}'. "
            f"Other possible substitutes are: {', '.join(candidates)}. "
            "Update the ingredient list and instructions for this substitution. "
            "Adjust cooking times or tips if needed."
        )
        return chef_agent.llm.invoke([HumanMessage(content=prompt)]).content.strip()
    
    def extract_ingredient_names(self, candidates: List[str]) -> List[str]:
        """Extract clean ingredient names from formatted strings."""
        names = []
        for item in candidates:
            match = re.match(r"[\*\-]*\s*([A-Za-z\s]+)", item)
            if match:
                name = match.group(1).strip().lower()
                if name and name not in names:
                    names.append(name)
        return names