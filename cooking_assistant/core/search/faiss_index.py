"""FAISS indexing utilities for recipe and ingredient search."""

import os
import ast
import logging
import traceback
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# Set environment variables to avoid multiprocessing issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPS_AVAILABLE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)


class FAISSIndex:
    """FAISS indexing for recipes and ingredients."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        # Force CPU usage to avoid MPS-related segfaults on macOS
        import torch
        torch.set_num_threads(1)  # Single thread to avoid multiprocessing issues
        device = 'cpu'
        self.model = SentenceTransformer(model_name, device=device)
        self.recipe_index = None
        self.ingredient_index = None
        self.recipe_embeddings = None
        self.ingredient_embeddings = None
        self.unique_ingredients = None
        
    def build_recipe_index(self, recipes_df: pd.DataFrame, ingredient_col: str = 'Cleaned_Ingredients'):
        """Build FAISS index for recipe ingredient lists."""
        try:
            logging.info(f"Building recipe index for {len(recipes_df)} recipes...")
            
            # Safely parse ingredient lists with error handling and text simplification
            ingredient_texts = []
            for idx, ings in enumerate(recipes_df[ingredient_col]):
                try:
                    if isinstance(ings, str):
                        parsed_ings = ast.literal_eval(ings)
                        # Simplify ingredients by extracting main ingredient names
                        simplified_ings = []
                        for ing in parsed_ings[:5]:  # Only first 5 ingredients
                            # Extract main ingredient (remove measurements and prep details)
                            words = ing.split()
                            if len(words) > 1:
                                # Take last word which is often the main ingredient
                                main_ing = words[-1].strip('.,')
                                if main_ing and len(main_ing) > 2:
                                    simplified_ings.append(main_ing)
                        ingredient_texts.append(' '.join(simplified_ings))  # Space separated, not comma
                    elif isinstance(ings, list):
                        simplified_ings = [' '.join(ing.split()[1:]) for ing in ings[:10]]
                        ingredient_texts.append(', '.join(simplified_ings))
                    else:
                        ingredient_texts.append(str(ings))
                except Exception as e:
                    logging.warning(f"Error parsing ingredients at row {idx}: {e}")
                    ingredient_texts.append("")
            
            # Remove empty texts
            ingredient_texts = [t for t in ingredient_texts if t.strip()]
            
            if not ingredient_texts:
                raise ValueError("No valid ingredient texts found")
                
            logging.info(f"Encoding {len(ingredient_texts)} ingredient texts...")
            
            # Encode in very small batches to avoid memory issues
            batch_size = 10
            embeddings_list = []
            
            for i in range(0, len(ingredient_texts), batch_size):
                batch = ingredient_texts[i:i+batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                embeddings_list.append(batch_embeddings)
                logging.info(f"Encoded batch {i//batch_size + 1}/{(len(ingredient_texts)-1)//batch_size + 1}")
            
            self.recipe_embeddings = np.vstack(embeddings_list)
            embedding_dim = self.recipe_embeddings.shape[1]
            
            logging.info(f"Creating FAISS index with dimension {embedding_dim}...")
            
            # Ensure embeddings are contiguous and float32
            self.recipe_embeddings = np.ascontiguousarray(self.recipe_embeddings.astype(np.float32))
            
            self.recipe_index = faiss.IndexFlatL2(embedding_dim)
            self.recipe_index.add(self.recipe_embeddings)
            
            logging.info("Recipe index built successfully!")
            return self.recipe_index, self.recipe_embeddings
            
        except Exception as e:
            logging.error(f"Error building recipe index: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def build_ingredient_index(self, recipes_df: pd.DataFrame, ingredient_col: str = 'Cleaned_Ingredients'):
        """Build FAISS index for unique ingredients."""
        try:
            logging.info(f"Building ingredient index from {len(recipes_df)} recipes...")
            
            unique_ingredients = set()
            for idx, ingr_list in enumerate(recipes_df[ingredient_col]):
                try:
                    if isinstance(ingr_list, str):
                        parsed_list = ast.literal_eval(ingr_list)
                        unique_ingredients.update(parsed_list)
                    elif isinstance(ingr_list, list):
                        unique_ingredients.update(ingr_list)
                except Exception as e:
                    logging.warning(f"Error parsing ingredients at row {idx}: {e}")
                    continue
            
            # Filter out empty/invalid ingredients
            self.unique_ingredients = [ing for ing in unique_ingredients if isinstance(ing, str) and ing.strip()]
            
            if not self.unique_ingredients:
                raise ValueError("No valid ingredients found")
                
            logging.info(f"Found {len(self.unique_ingredients)} unique ingredients")
            
            # Encode in batches to avoid memory issues
            batch_size = 500  # Smaller batch for individual ingredients
            embeddings_list = []
            
            for i in range(0, len(self.unique_ingredients), batch_size):
                batch = self.unique_ingredients[i:i+batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                embeddings_list.append(batch_embeddings)
                logging.info(f"Encoded ingredient batch {i//batch_size + 1}/{(len(self.unique_ingredients)-1)//batch_size + 1}")
            
            self.ingredient_embeddings = np.vstack(embeddings_list)
            embedding_dim = self.ingredient_embeddings.shape[1]
            
            logging.info(f"Creating ingredient FAISS index with dimension {embedding_dim}...")
            
            # Ensure embeddings are contiguous and float32
            self.ingredient_embeddings = np.ascontiguousarray(self.ingredient_embeddings.astype(np.float32))
            
            self.ingredient_index = faiss.IndexFlatL2(embedding_dim)
            self.ingredient_index.add(self.ingredient_embeddings)
            
            logging.info("Ingredient index built successfully!")
            return self.ingredient_index, self.ingredient_embeddings, self.unique_ingredients
            
        except Exception as e:
            logging.error(f"Error building ingredient index: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def search_recipes_by_ingredients(self, ingredients: list[str], recipes_df: pd.DataFrame, k: int = 10):
        """Search recipes by ingredient similarity."""
        if self.recipe_index is None:
            raise ValueError("Recipe index not built. Call build_recipe_index first.")
            
        user_query = ', '.join(ingredients)
        user_embedding = self.model.encode([user_query], convert_to_numpy=True)
        D, I = self.recipe_index.search(user_embedding, k)
        
        results = recipes_df.iloc[I[0]].copy()
        results['similarity'] = 1 / (1 + D[0])
        return results
    
    def find_ingredient_substitutes(self, target_ingredient: str, k: int = 10):
        """Find ingredient substitutes using FAISS similarity."""
        if self.ingredient_index is None:
            raise ValueError("Ingredient index not built. Call build_ingredient_index first.")
            
        query_vec = self.model.encode([target_ingredient], convert_to_numpy=True)
        D, I = self.ingredient_index.search(query_vec, k)
        
        substitutes = [self.unique_ingredients[i] for i in I[0]]
        similarities = 1 / (1 + D[0])
        return list(zip(substitutes, similarities))