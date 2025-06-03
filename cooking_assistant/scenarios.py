"""Scenario definitions for testing different cooking conversation flows."""

from typing import Dict, List, Any
from .core.models import UserProfile


class CookingScenario:
    """Base class for cooking conversation scenarios."""
    
    def __init__(self, name: str, description: str, user_profile: UserProfile, 
                 user_query: str, expected_outcomes: List[str] = None):
        self.name = name
        self.description = description
        self.user_profile = user_profile
        self.user_query = user_query
        self.expected_outcomes = expected_outcomes or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "user_profile": {
                "experience_level": self.user_profile.experience_level,
                "preferred_cuisine": self.user_profile.preferred_cuisine,
                "allergies": self.user_profile.allergies,
                "notes": self.user_profile.notes
            },
            "user_query": self.user_query,
            "expected_outcomes": self.expected_outcomes
        }


# Predefined scenarios for testing
BEGINNER_SCENARIOS = [
    CookingScenario(
        name="beginner_pasta",
        description="Beginner wants to make simple pasta",
        user_profile=UserProfile(
            experience_level="beginner",
            preferred_cuisine="Italian",
            allergies=[],
            notes="Never cooked pasta before"
        ),
        user_query="I want to make spaghetti",
        expected_outcomes=[
            "Should ask many questions about basic techniques",
            "Should provide detailed explanations",
            "Should check understanding frequently"
        ]
    ),
    
    CookingScenario(
        name="beginner_with_allergies",
        description="Beginner with nut allergies wants to bake",
        user_profile=UserProfile(
            experience_level="beginner",
            preferred_cuisine="",
            allergies=["nuts", "peanuts"],
            notes="Want to learn baking"
        ),
        user_query="I want to make chocolate chip cookies",
        expected_outcomes=[
            "Should check for nut allergies in recipe",
            "Should suggest nut-free alternatives",
            "Should provide basic baking guidance"
        ]
    )
]

INTERMEDIATE_SCENARIOS = [
    CookingScenario(
        name="intermediate_substitution",
        description="Intermediate cook needs ingredient substitution",
        user_profile=UserProfile(
            experience_level="intermediate",
            preferred_cuisine="Asian",
            allergies=[],
            notes="Comfortable with basic techniques"
        ),
        user_query="Can I use chicken instead of beef in this stir-fry?",
        expected_outcomes=[
            "Should provide substitution options",
            "Should adjust cooking times/techniques",
            "Should ask moderate number of questions"
        ]
    ),
    
    CookingScenario(
        name="intermediate_technique_focus",
        description="Intermediate cook wants to learn new technique",
        user_profile=UserProfile(
            experience_level="intermediate",
            preferred_cuisine="French",
            allergies=[],
            notes="Never made risotto before"
        ),
        user_query="I want to make mushroom risotto",
        expected_outcomes=[
            "Should focus on risotto technique",
            "Should provide technique-specific guidance",
            "Should assume basic cooking knowledge"
        ]
    )
]

ADVANCED_SCENARIOS = [
    CookingScenario(
        name="advanced_complex_dish",
        description="Advanced cook making complex dish",
        user_profile=UserProfile(
            experience_level="advanced",
            preferred_cuisine="French",
            allergies=[],
            notes="Professional chef background"
        ),
        user_query="I want to make beef wellington",
        expected_outcomes=[
            "Should provide minimal basic explanations",
            "Should focus on advanced techniques only",
            "Should rarely ask questions unless complex"
        ]
    ),
    
    CookingScenario(
        name="advanced_dietary_restrictions",
        description="Advanced cook with multiple dietary restrictions",
        user_profile=UserProfile(
            experience_level="advanced",
            preferred_cuisine="Mediterranean",
            allergies=["gluten", "dairy"],
            notes="Experienced with dietary modifications"
        ),
        user_query="I need a gluten-free, dairy-free lasagna",
        expected_outcomes=[
            "Should suggest appropriate substitutions",
            "Should assume knowledge of techniques",
            "Should focus on dietary modifications"
        ]
    )
]

SUBSTITUTION_SCENARIOS = [
    CookingScenario(
        name="protein_substitution",
        description="User wants to substitute main protein",
        user_profile=UserProfile(
            experience_level="intermediate",
            preferred_cuisine="",
            allergies=["chicken"],
            notes=""
        ),
        user_query="Can I use turkey instead of chicken in this recipe?",
        expected_outcomes=[
            "Should provide substitution guidance",
            "Should adjust cooking instructions",
            "Should consider allergy restrictions"
        ]
    ),
    
    CookingScenario(
        name="ingredient_unavailable",
        description="User missing key ingredient",
        user_profile=UserProfile(
            experience_level="beginner",
            preferred_cuisine="",
            allergies=[],
            notes=""
        ),
        user_query="I don't have heavy cream for this recipe",
        expected_outcomes=[
            "Should suggest cream substitutes",
            "Should explain how substitution affects recipe",
            "Should provide beginner-friendly alternatives"
        ]
    )
]

# All scenarios combined
ALL_SCENARIOS = {
    "beginner": BEGINNER_SCENARIOS,
    "intermediate": INTERMEDIATE_SCENARIOS,
    "advanced": ADVANCED_SCENARIOS,
    "substitution": SUBSTITUTION_SCENARIOS
}


def get_scenario_by_name(name: str) -> CookingScenario:
    """Get specific scenario by name."""
    for category in ALL_SCENARIOS.values():
        for scenario in category:
            if scenario.name == name:
                return scenario
    raise ValueError(f"Scenario '{name}' not found")


def list_all_scenarios() -> List[str]:
    """List all available scenario names."""
    scenarios = []
    for category in ALL_SCENARIOS.values():
        scenarios.extend([s.name for s in category])
    return scenarios