#!/usr/bin/env python3

from cooking_assistant.main import CookingAssistant
from cooking_assistant.core.models import UserProfile

# Quick test with minimal setup
assistant = CookingAssistant(sample_size=15)  # Very small sample

user_profile = UserProfile(experience_level="beginner", allergies=[])

# Test with a simple query
job_id = assistant.run_conversation(
    user_query="I want to make cookies", 
    user_profile=user_profile
)

print(f"Done! Job ID: {job_id}")