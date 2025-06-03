#!/usr/bin/env python3
"""Simplified main script to avoid import issues."""

import os
import pandas as pd
import ast
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import argparse

# Set environment variable to force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPS_AVAILABLE"] = "0"

class SimpleCookingAssistant:
    """Simplified cooking assistant for testing."""
    
    def __init__(self, sample_size=50):
        print("Loading CSV...")
        self.recipes_df = pd.read_csv("13k-recipes.csv")
        print(f"Loaded {len(self.recipes_df)} recipes")
        
        if sample_size < len(self.recipes_df):
            print(f"Using sample of {sample_size} recipes...")
            self.recipes_df = self.recipes_df.sample(n=sample_size, random_state=42)
            
        print("Loading sentence transformer (CPU only)...")
        import torch
        torch.set_num_threads(1)  # Single thread to avoid issues
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
        
        print("Building simple search index...")
        self._build_simple_index()
        
    def _build_simple_index(self):
        """Build a simple FAISS index."""
        # Simplified ingredient extraction
        ingredient_texts = []
        for _, row in self.recipes_df.iterrows():
            try:
                ings = row['Cleaned_Ingredients']
                if isinstance(ings, str):
                    parsed = ast.literal_eval(ings)
                    # Just use first 3 ingredients, simplified
                    simple_ings = []
                    for ing in parsed[:3]:
                        # Extract main ingredient word
                        words = ing.split()
                        if len(words) > 1:
                            simple_ings.append(words[-1])  # Last word often the ingredient
                    ingredient_texts.append(' '.join(simple_ings))
                else:
                    ingredient_texts.append("unknown")
            except:
                ingredient_texts.append("unknown")
        
        print(f"Encoding {len(ingredient_texts)} ingredient texts...")
        
        # Encode in very small batches
        batch_size = 10
        all_embeddings = []
        for i in range(0, len(ingredient_texts), batch_size):
            batch = ingredient_texts[i:i+batch_size]
            print(f"  Encoding batch {i//batch_size + 1}/{(len(ingredient_texts)-1)//batch_size + 1}")
            embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(embeddings)
        
        print("Combining embeddings...")
        self.embeddings = np.vstack(all_embeddings).astype(np.float32)
        
        print("Creating FAISS index...")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        
        print(f"Index built with {self.index.ntotal} recipes")
    
    def search(self, query, k=5):
        """Search for recipes."""
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        D, I = self.index.search(query_embedding, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            recipe = self.recipes_df.iloc[idx]
            results.append({
                'title': recipe['Title'],
                'distance': dist,
                'ingredients': recipe['Cleaned_Ingredients'][:100] + '...'
            })
        
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="pasta", help="Search query")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of recipes to use")
    args = parser.parse_args()
    
    try:
        assistant = SimpleCookingAssistant(args.sample_size)
        
        print(f"\nSearching for: {args.query}")
        results = assistant.search(args.query)
        
        print("\nTop results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']} (distance: {result['distance']:.3f})")
            print(f"   Ingredients: {result['ingredients']}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()