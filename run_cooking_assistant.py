#!/usr/bin/env python3
"""Clean wrapper to run cooking assistant without module warnings."""

import warnings
import sys
import os

# Suppress the specific runpy warning
warnings.filterwarnings("ignore", message=".*found in sys.modules.*", category=RuntimeWarning)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run main
from cooking_assistant.main import main

if __name__ == "__main__":
    main()