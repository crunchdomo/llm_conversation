#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import signal
import sys

def timeout_handler(signum, frame):
    print("Timeout! SentenceTransformer is hanging.")
    sys.exit(1)

# Set a 30-second timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

try:
    print("Loading SentenceTransformer...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("Model loaded successfully!")
    
    print("Testing encoding...")
    texts = ["pasta with tomato", "chicken with rice"]
    embeddings = model.encode(texts, convert_to_numpy=True)
    print(f"Encoded {len(texts)} texts, shape: {embeddings.shape}")
    
    signal.alarm(0)  # Cancel timeout
    print("Test completed successfully!")
    
except Exception as e:
    signal.alarm(0)  # Cancel timeout
    print(f"Error: {e}")
    sys.exit(1)