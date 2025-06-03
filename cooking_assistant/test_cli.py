#!/usr/bin/env python3
"""
CLI entry point for running cooking conversation tests.

Usage examples:

# List available scenarios and providers
python test_cli.py --list-scenarios
python test_cli.py --list-providers

# Run specific scenario types
python test_cli.py --conversation-types specific_recipe ingredient_based
python test_cli.py --conversation-types substitution --llm-providers openai_gpt4_mini

# Run specific scenarios
python test_cli.py --scenarios specific_recipe_chicken_teriyaki ingredient_based_chicken_onions

# Run all tests with available providers
python test_cli.py --max-workers 2

# Run batch with custom name
python test_cli.py --batch-name "experiment_1" --conversation-types specific_recipe
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cooking_assistant.testing.test_runner import main

if __name__ == "__main__":
    main()