"""Cooking Assistant - A multi-agent cooking conversation system."""

from .main import CookingAssistant
from .scenarios import (
    CookingScenario, 
    get_scenario_by_name, 
    list_all_scenarios,
    ALL_SCENARIOS
)
from .core.models import UserProfile, IntentDetection, State

__version__ = "0.1.0"
__all__ = [
    "CookingAssistant",
    "CookingScenario", 
    "get_scenario_by_name",
    "list_all_scenarios",
    "ALL_SCENARIOS",
    "UserProfile",
    "IntentDetection", 
    "State"
]