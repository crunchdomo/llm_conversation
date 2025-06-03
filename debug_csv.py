#!/usr/bin/env python3

import pandas as pd
import ast

# Load and examine the CSV
df = pd.read_csv("13k-recipes.csv")
print(f"CSV shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"First few rows:")
print(df.head(2))

# Check the cleaned ingredients column
print(f"\nCleaned_Ingredients samples:")
for i in range(min(5, len(df))):
    ingredients = df.iloc[i]['Cleaned_Ingredients']
    print(f"Row {i}: {type(ingredients)} - {str(ingredients)[:100]}...")
    
    if isinstance(ingredients, str):
        try:
            parsed = ast.literal_eval(ingredients)
            print(f"  Parsed: {type(parsed)} with {len(parsed)} items")
            print(f"  Sample items: {parsed[:3] if len(parsed) > 0 else 'empty'}")
        except Exception as e:
            print(f"  Parse error: {e}")
    print()