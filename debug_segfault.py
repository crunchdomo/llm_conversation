#!/usr/bin/env python3

import pandas as pd
import ast
from cooking_assistant.core.search.faiss_index import FAISSIndex

def test_with_real_data():
    """Test FAISS with actual recipe data."""
    try:
        print("Loading CSV...")
        df = pd.read_csv("13k-recipes.csv")
        print(f"Loaded {len(df)} recipes")
        
        # Take tiny sample
        sample_df = df.head(5)
        print(f"Using {len(sample_df)} recipes for test")
        
        print("Creating FAISS index...")
        faiss_index = FAISSIndex()
        
        print("Building recipe index...")
        faiss_index.build_recipe_index(sample_df)
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_real_data()