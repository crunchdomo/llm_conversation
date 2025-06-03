#!/usr/bin/env python3

import pandas as pd
import ast
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def debug_step_by_step():
    """Debug each step to find where segfault occurs."""
    try:
        print("Step 1: Loading CSV...")
        df = pd.read_csv("13k-recipes.csv")
        sample_df = df.head(3)
        print(f"Got {len(sample_df)} recipes")
        
        print("Step 2: Loading SentenceTransformer...")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
        print("Model loaded successfully")
        
        print("Step 3: Parsing ingredients...")
        ingredient_texts = []
        for idx, ings in enumerate(sample_df['Cleaned_Ingredients']):
            print(f"  Processing row {idx}...")
            try:
                if isinstance(ings, str):
                    parsed_ings = ast.literal_eval(ings)
                    simplified_ings = []
                    for ing in parsed_ings[:5]:  # Only first 5
                        main_ing = ing.split(',')[0].strip()
                        main_ing = ' '.join(main_ing.split()[1:])
                        if main_ing:
                            simplified_ings.append(main_ing)
                    text = ', '.join(simplified_ings)
                    ingredient_texts.append(text)
                    print(f"    Result: {text[:50]}...")
            except Exception as e:
                print(f"    Error: {e}")
                ingredient_texts.append("pasta")
        
        print("Step 4: Testing encoding with single text...")
        test_embedding = model.encode(["pasta, tomato"], convert_to_numpy=True)
        print(f"Single encoding works: {test_embedding.shape}")
        
        print("Step 5: Encoding all texts...")
        print(f"About to encode {len(ingredient_texts)} texts")
        embeddings = model.encode(ingredient_texts, convert_to_numpy=True, show_progress_bar=False)
        print(f"Encoding successful: {embeddings.shape}")
        
        print("Step 6: Preparing for FAISS...")
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        print(f"Array prepared: {embeddings.shape}, dtype: {embeddings.dtype}")
        
        print("Step 7: Creating FAISS index...")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        print("Index created")
        
        print("Step 8: Adding to FAISS...")
        index.add(embeddings)
        print("Added to index successfully")
        
        print("Step 9: Testing search...")
        query = model.encode(["pasta"], convert_to_numpy=True)
        query = np.ascontiguousarray(query.astype(np.float32))
        D, I = index.search(query, k=2)
        print(f"Search successful: {D}, {I}")
        
        print("All steps completed successfully!")
        
    except Exception as e:
        print(f"Error at current step: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_step_by_step()