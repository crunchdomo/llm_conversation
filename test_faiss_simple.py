#!/usr/bin/env python3

"""Simple test for FAISS functionality."""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def test_simple_faiss():
    """Test basic FAISS functionality."""
    print("Loading sentence transformer...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    print("Creating sample data...")
    texts = [
        "pasta, tomato, cheese",
        "chicken, rice, vegetables", 
        "beef, potatoes, onions",
        "fish, lemon, herbs",
        "salad, lettuce, dressing"
    ]
    
    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, convert_to_numpy=True)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("Creating FAISS index...")
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    print(f"Index has {index.ntotal} vectors")
    
    print("Testing search...")
    query_embedding = model.encode(["italian pasta"], convert_to_numpy=True)
    query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
    D, I = index.search(query_embedding, k=3)
    
    print("Search results:")
    for i, (dist, idx) in enumerate(zip(D[0], I[0])):
        print(f"  {i+1}. {texts[idx]} (distance: {dist:.3f})")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_simple_faiss()