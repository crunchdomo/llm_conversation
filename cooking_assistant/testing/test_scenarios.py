"""Test scenarios for different conversation types and user interactions."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum


class ConversationType(Enum):
    """Types of cooking conversations to test."""
    SPECIFIC_RECIPE = "specific_recipe"          # "I want to make chicken teriyaki"
    INGREDIENT_BASED = "ingredient_based"        # "I have chicken and onions"
    GENERAL_RECIPE = "general_recipe"           # "How do I make spaghetti bolognese"
    SUBSTITUTION = "substitution"               # "Can I use turkey instead of chicken?"
    DIETARY_RESTRICTION = "dietary_restriction"  # "I need vegan alternatives"
    ALLERGEN_SAFE = "allergen_safe"             # "I'm allergic to nuts"


class LLMProvider(Enum):
    """LLM providers to test."""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT4_MINI = "openai_gpt4_mini"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    ANTHROPIC_HAIKU = "anthropic_haiku"
    XAI_GROK = "xai_grok"


@dataclass
class UserProfile:
    """User profile for testing scenarios."""
    experience_level: str = "beginner"
    allergies: List[str] = None
    dietary_restrictions: List[str] = None
    preferred_cuisine: str = ""
    notes: str = ""
    
    def __post_init__(self):
        if self.allergies is None:
            self.allergies = []
        if self.dietary_restrictions is None:
            self.dietary_restrictions = []


@dataclass
class TestScenario:
    """A complete test scenario definition."""
    name: str
    conversation_type: ConversationType
    user_query: str
    user_profile: UserProfile
    expected_outcomes: List[str]
    llm_providers: List[LLMProvider]
    max_turns: int = 10
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestScenarioBuilder:
    """Builder for creating comprehensive test scenarios."""
    
    @staticmethod
    def create_specific_recipe_scenarios() -> List[TestScenario]:
        """Test scenarios for specific recipe requests."""
        return [
            TestScenario(
                name="specific_recipe_chicken_teriyaki",
                conversation_type=ConversationType.SPECIFIC_RECIPE,
                user_query="I want to make chicken teriyaki",
                user_profile=UserProfile(experience_level="beginner"),
                expected_outcomes=[
                    "recipe_found",
                    "ingredients_listed",
                    "step_by_step_guidance",
                    "completion_confirmation"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE],
                metadata={"cuisine": "asian", "protein": "chicken", "difficulty": "easy"}
            ),
            TestScenario(
                name="specific_recipe_pasta_carbonara",
                conversation_type=ConversationType.SPECIFIC_RECIPE,
                user_query="I want to make pasta carbonara",
                user_profile=UserProfile(experience_level="intermediate"),
                expected_outcomes=[
                    "recipe_found",
                    "ingredients_listed", 
                    "technique_explanation",
                    "step_by_step_guidance"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE],
                metadata={"cuisine": "italian", "protein": "eggs", "difficulty": "medium"}
            ),
            TestScenario(
                name="specific_recipe_beef_stir_fry",
                conversation_type=ConversationType.SPECIFIC_RECIPE,
                user_query="Show me how to make beef stir fry",
                user_profile=UserProfile(experience_level="advanced"),
                expected_outcomes=[
                    "recipe_found",
                    "ingredients_listed",
                    "cooking_technique_focus",
                    "minimal_questions"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI],
                metadata={"cuisine": "asian", "protein": "beef", "difficulty": "easy"}
            )
        ]
    
    @staticmethod
    def create_ingredient_based_scenarios() -> List[TestScenario]:
        """Test scenarios for ingredient-based recipe requests."""
        return [
            TestScenario(
                name="ingredient_based_chicken_onions",
                conversation_type=ConversationType.INGREDIENT_BASED,
                user_query="I have chicken and onions, what can I make?",
                user_profile=UserProfile(experience_level="beginner"),
                expected_outcomes=[
                    "multiple_recipe_options",
                    "recipe_selection",
                    "ingredients_confirmation",
                    "cooking_guidance"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE],
                metadata={"primary_ingredients": ["chicken", "onions"], "open_ended": True}
            ),
            TestScenario(
                name="ingredient_based_vegetables_only",
                conversation_type=ConversationType.INGREDIENT_BASED,
                user_query="I have carrots, bell peppers, and mushrooms",
                user_profile=UserProfile(
                    experience_level="intermediate",
                    dietary_restrictions=["vegetarian"]
                ),
                expected_outcomes=[
                    "vegetarian_recipes_only",
                    "recipe_selection",
                    "cooking_method_discussion",
                    "step_by_step_guidance"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE],
                metadata={"primary_ingredients": ["carrots", "bell peppers", "mushrooms"], "vegetarian": True}
            ),
            TestScenario(
                name="ingredient_based_pantry_staples",
                conversation_type=ConversationType.INGREDIENT_BASED,
                user_query="I have rice, eggs, and soy sauce",
                user_profile=UserProfile(experience_level="beginner"),
                expected_outcomes=[
                    "simple_recipe_suggestions",
                    "fried_rice_option",
                    "basic_technique_explanation",
                    "encouragement"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI],
                metadata={"primary_ingredients": ["rice", "eggs", "soy sauce"], "simple": True}
            )
        ]
    
    @staticmethod
    def create_general_recipe_scenarios() -> List[TestScenario]:
        """Test scenarios for general recipe instruction requests."""
        return [
            TestScenario(
                name="general_recipe_spaghetti_bolognese",
                conversation_type=ConversationType.GENERAL_RECIPE,
                user_query="How do I make spaghetti bolognese?",
                user_profile=UserProfile(experience_level="beginner"),
                expected_outcomes=[
                    "complete_recipe_explanation",
                    "ingredient_list",
                    "detailed_steps",
                    "cooking_tips"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE],
                metadata={"cuisine": "italian", "complexity": "medium", "classic_dish": True}
            ),
            TestScenario(
                name="general_recipe_chocolate_chip_cookies",
                conversation_type=ConversationType.GENERAL_RECIPE,
                user_query="Teach me to make chocolate chip cookies",
                user_profile=UserProfile(experience_level="beginner"),
                expected_outcomes=[
                    "baking_recipe_provided",
                    "precise_measurements",
                    "baking_technique_explanation",
                    "troubleshooting_tips"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE],
                metadata={"type": "dessert", "baking": True, "beginner_friendly": True}
            ),
            TestScenario(
                name="general_recipe_chicken_curry",
                conversation_type=ConversationType.GENERAL_RECIPE,
                user_query="How do you make chicken curry?",
                user_profile=UserProfile(experience_level="intermediate"),
                expected_outcomes=[
                    "spice_explanation",
                    "cooking_technique",
                    "flavor_development",
                    "serving_suggestions"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI],
                metadata={"cuisine": "indian", "spices": True, "complexity": "medium"}
            )
        ]
    
    @staticmethod
    def create_substitution_scenarios() -> List[TestScenario]:
        """Test scenarios for ingredient substitution requests."""
        return [
            TestScenario(
                name="substitution_turkey_for_chicken",
                conversation_type=ConversationType.SUBSTITUTION,
                user_query="Can I use turkey instead of chicken in this stir fry?",
                user_profile=UserProfile(experience_level="intermediate"),
                expected_outcomes=[
                    "substitution_confirmed",
                    "cooking_adjustment_explained",
                    "recipe_adaptation",
                    "continued_guidance"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE],
                metadata={"original": "chicken", "substitute": "turkey", "recipe_context": "stir fry"}
            ),
            TestScenario(
                name="substitution_coconut_milk_for_cream",
                conversation_type=ConversationType.SUBSTITUTION,
                user_query="I don't have heavy cream, can I use coconut milk?",
                user_profile=UserProfile(
                    experience_level="beginner",
                    dietary_restrictions=["dairy-free"]
                ),
                expected_outcomes=[
                    "substitution_validated",
                    "ratio_adjustment_explained",
                    "texture_difference_noted",
                    "recipe_modification"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE],
                metadata={"original": "heavy cream", "substitute": "coconut milk", "dietary": "dairy-free"}
            ),
            TestScenario(
                name="substitution_multiple_ingredients",
                conversation_type=ConversationType.SUBSTITUTION,
                user_query="I need to replace both butter and eggs in this cookie recipe",
                user_profile=UserProfile(
                    experience_level="intermediate",
                    dietary_restrictions=["vegan"]
                ),
                expected_outcomes=[
                    "multiple_substitutions_provided",
                    "baking_chemistry_explained",
                    "vegan_alternatives_suggested",
                    "result_expectations_set"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI],
                metadata={"multiple_subs": True, "baking": True, "vegan": True}
            )
        ]
    
    @staticmethod
    def create_dietary_restriction_scenarios() -> List[TestScenario]:
        """Test scenarios for dietary restriction accommodations."""
        return [
            TestScenario(
                name="dietary_vegan_pasta_dish",
                conversation_type=ConversationType.DIETARY_RESTRICTION,
                user_query="I want to make a vegan pasta dish",
                user_profile=UserProfile(
                    experience_level="beginner",
                    dietary_restrictions=["vegan"]
                ),
                expected_outcomes=[
                    "vegan_recipe_provided",
                    "no_animal_products",
                    "nutritional_balance_considered",
                    "flavor_enhancement_tips"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE],
                metadata={"dietary": "vegan", "protein_alternatives": True}
            ),
            TestScenario(
                name="dietary_keto_dinner",
                conversation_type=ConversationType.DIETARY_RESTRICTION,
                user_query="I need a keto-friendly dinner recipe",
                user_profile=UserProfile(
                    experience_level="intermediate",
                    dietary_restrictions=["keto"]
                ),
                expected_outcomes=[
                    "low_carb_recipe",
                    "high_fat_ingredients",
                    "carb_count_awareness",
                    "keto_compliant_verification"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE],
                metadata={"dietary": "keto", "macros": True}
            ),
            TestScenario(
                name="dietary_gluten_free_bread",
                conversation_type=ConversationType.DIETARY_RESTRICTION,
                user_query="How can I make gluten-free bread?",
                user_profile=UserProfile(
                    experience_level="advanced",
                    dietary_restrictions=["gluten-free"]
                ),
                expected_outcomes=[
                    "gluten_free_flour_explanation",
                    "binding_agent_discussion",
                    "texture_expectations",
                    "baking_technique_adaptation"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI],
                metadata={"dietary": "gluten-free", "baking": True, "complex": True}
            )
        ]
    
    @staticmethod
    def create_allergen_safe_scenarios() -> List[TestScenario]:
        """Test scenarios for allergen-safe cooking."""
        return [
            TestScenario(
                name="allergen_nut_free_dessert",
                conversation_type=ConversationType.ALLERGEN_SAFE,
                user_query="I need a nut-free dessert for my kid's school party",
                user_profile=UserProfile(
                    experience_level="beginner",
                    allergies=["nuts", "tree nuts"]
                ),
                expected_outcomes=[
                    "nut_free_recipe_confirmed",
                    "cross_contamination_awareness",
                    "safe_ingredients_verified",
                    "kid_friendly_suggestions"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE],
                metadata={"allergens": ["nuts"], "context": "school_party", "safety_critical": True}
            ),
            TestScenario(
                name="allergen_dairy_egg_free_cake",
                conversation_type=ConversationType.ALLERGEN_SAFE,
                user_query="I'm allergic to dairy and eggs, can you help me make a birthday cake?",
                user_profile=UserProfile(
                    experience_level="intermediate",
                    allergies=["dairy", "eggs"]
                ),
                expected_outcomes=[
                    "allergen_free_baking_recipe",
                    "multiple_substitutions_explained",
                    "texture_expectations_managed",
                    "celebration_focus_maintained"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE],
                metadata={"allergens": ["dairy", "eggs"], "baking": True, "special_occasion": True}
            ),
            TestScenario(
                name="allergen_shellfish_free_seafood",
                conversation_type=ConversationType.ALLERGEN_SAFE,
                user_query="I love seafood but I'm allergic to shellfish, what can I cook?",
                user_profile=UserProfile(
                    experience_level="advanced",
                    allergies=["shellfish"]
                ),
                expected_outcomes=[
                    "fish_vs_shellfish_distinction",
                    "safe_seafood_options",
                    "cross_contamination_prevention",
                    "recipe_suggestions"
                ],
                llm_providers=[LLMProvider.OPENAI_GPT4_MINI],
                metadata={"allergens": ["shellfish"], "seafood_focus": True, "education": True}
            )
        ]
    
    @classmethod
    def get_all_scenarios(cls) -> List[TestScenario]:
        """Get all predefined test scenarios."""
        scenarios = []
        scenarios.extend(cls.create_specific_recipe_scenarios())
        scenarios.extend(cls.create_ingredient_based_scenarios())
        scenarios.extend(cls.create_general_recipe_scenarios())
        scenarios.extend(cls.create_substitution_scenarios())
        scenarios.extend(cls.create_dietary_restriction_scenarios())
        scenarios.extend(cls.create_allergen_safe_scenarios())
        return scenarios
    
    @classmethod
    def get_scenarios_by_type(cls, conversation_type: ConversationType) -> List[TestScenario]:
        """Get scenarios filtered by conversation type."""
        all_scenarios = cls.get_all_scenarios()
        return [s for s in all_scenarios if s.conversation_type == conversation_type]
    
    @classmethod
    def get_scenarios_by_llm(cls, llm_provider: LLMProvider) -> List[TestScenario]:
        """Get scenarios that support a specific LLM provider."""
        all_scenarios = cls.get_all_scenarios()
        return [s for s in all_scenarios if llm_provider in s.llm_providers]