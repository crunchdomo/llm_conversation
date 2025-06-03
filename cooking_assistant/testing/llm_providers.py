"""Multi-LLM provider support for testing different models."""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT4_MINI = "openai_gpt4_mini"
    OPENAI_GPT35_TURBO = "openai_gpt35_turbo"
    ANTHROPIC_CLAUDE_35_SONNET = "anthropic_claude_35_sonnet"
    ANTHROPIC_CLAUDE_35_HAIKU = "anthropic_claude_35_haiku"
    ANTHROPIC_CLAUDE_3_OPUS = "anthropic_claude_3_opus"
    XAI_GROK_BETA = "xai_grok_beta"


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    provider: LLMProvider
    model_name: str
    api_key_env: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 120
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class BaseLLMWrapper(ABC):
    """Base wrapper for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._create_client()
    
    @abstractmethod
    def _create_client(self) -> BaseChatModel:
        """Create the LLM client."""
        pass
    
    def invoke(self, messages: list[BaseMessage]) -> str:
        """Invoke the LLM with messages."""
        try:
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            raise Exception(f"LLM invocation failed for {self.config.provider.value}: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.config.provider.value
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.config.model_name


class OpenAIWrapper(BaseLLMWrapper):
    """Wrapper for OpenAI models."""
    
    def _create_client(self) -> ChatOpenAI:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {self.config.api_key_env}")
        
        return ChatOpenAI(
            model=self.config.model_name,
            openai_api_key=api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            **self.config.extra_params
        )


class AnthropicWrapper(BaseLLMWrapper):
    """Wrapper for Anthropic Claude models."""
    
    def _create_client(self) -> ChatAnthropic:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {self.config.api_key_env}")
        
        return ChatAnthropic(
            model=self.config.model_name,
            anthropic_api_key=api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            **self.config.extra_params
        )


class XAIWrapper(BaseLLMWrapper):
    """Wrapper for XAI Grok models."""
    
    def _create_client(self) -> ChatXAI:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {self.config.api_key_env}")
        
        return ChatXAI(
            model=self.config.model_name,
            xai_api_key=api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            **self.config.extra_params
        )


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    # Default configurations for each provider
    DEFAULT_CONFIGS = {
        LLMProvider.OPENAI_GPT4: LLMConfig(
            provider=LLMProvider.OPENAI_GPT4,
            model_name="gpt-4",
            api_key_env="OPENAI_API_KEY",
            temperature=0.7,
            max_tokens=2000
        ),
        LLMProvider.OPENAI_GPT4_MINI: LLMConfig(
            provider=LLMProvider.OPENAI_GPT4_MINI,
            model_name="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            temperature=0.7,
            max_tokens=2000
        ),
        LLMProvider.OPENAI_GPT35_TURBO: LLMConfig(
            provider=LLMProvider.OPENAI_GPT35_TURBO,
            model_name="gpt-3.5-turbo",
            api_key_env="OPENAI_API_KEY",
            temperature=0.7,
            max_tokens=2000
        ),
        LLMProvider.ANTHROPIC_CLAUDE_35_SONNET: LLMConfig(
            provider=LLMProvider.ANTHROPIC_CLAUDE_35_SONNET,
            model_name="claude-3-5-sonnet-20241022",
            api_key_env="ANTHROPIC_API_KEY",
            temperature=0.7,
            max_tokens=2000
        ),
        LLMProvider.ANTHROPIC_CLAUDE_35_HAIKU: LLMConfig(
            provider=LLMProvider.ANTHROPIC_CLAUDE_35_HAIKU,
            model_name="claude-3-5-haiku-20241022",
            api_key_env="ANTHROPIC_API_KEY",
            temperature=0.7,
            max_tokens=2000
        ),
        LLMProvider.ANTHROPIC_CLAUDE_3_OPUS: LLMConfig(
            provider=LLMProvider.ANTHROPIC_CLAUDE_3_OPUS,
            model_name="claude-3-opus-20240229",
            api_key_env="ANTHROPIC_API_KEY",
            temperature=0.7,
            max_tokens=2000
        ),
        LLMProvider.XAI_GROK_BETA: LLMConfig(
            provider=LLMProvider.XAI_GROK_BETA,
            model_name="grok-beta",
            api_key_env="XAI_API_KEY",
            temperature=0.7,
            max_tokens=2000
        )
    }
    
    @classmethod
    def create_provider(cls, provider: LLMProvider, custom_config: Optional[LLMConfig] = None) -> BaseLLMWrapper:
        """Create an LLM provider wrapper."""
        config = custom_config or cls.DEFAULT_CONFIGS.get(provider)
        if not config:
            raise ValueError(f"No configuration found for provider: {provider}")
        
        if provider in [LLMProvider.OPENAI_GPT4, LLMProvider.OPENAI_GPT4_MINI, LLMProvider.OPENAI_GPT35_TURBO]:
            return OpenAIWrapper(config)
        elif provider in [LLMProvider.ANTHROPIC_CLAUDE_35_SONNET, LLMProvider.ANTHROPIC_CLAUDE_35_HAIKU, LLMProvider.ANTHROPIC_CLAUDE_3_OPUS]:
            return AnthropicWrapper(config)
        elif provider == LLMProvider.XAI_GROK_BETA:
            return XAIWrapper(config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @classmethod
    def get_available_providers(cls) -> list[LLMProvider]:
        """Get list of available providers based on environment variables."""
        available = []
        
        for provider, config in cls.DEFAULT_CONFIGS.items():
            if os.environ.get(config.api_key_env):
                available.append(provider)
        
        return available
    
    @classmethod
    def validate_provider_access(cls, provider: LLMProvider) -> bool:
        """Validate if a provider is accessible (has API key)."""
        config = cls.DEFAULT_CONFIGS.get(provider)
        if not config:
            return False
        return bool(os.environ.get(config.api_key_env))


class TestableChefAgent:
    """Chef agent wrapper that can use different LLM providers for testing."""
    
    def __init__(self, llm_wrapper: BaseLLMWrapper, job_id: str, recipes_df, faiss_index=None, recipe_searcher=None, ingredient_substituter=None, trainee_experience_log=None):
        self.llm_wrapper = llm_wrapper
        self.job_id = job_id
        self.recipes_df = recipes_df
        self.faiss_index = faiss_index
        self.recipe_searcher = recipe_searcher
        self.ingredient_substituter = ingredient_substituter
        self.trainee_experience_log = trainee_experience_log
        self.token_cost_log = []
        self.system_message = self._build_system_message()
    
    def _build_system_message(self) -> SystemMessage:
        """Build system message for the chef agent."""
        experience_level = self.trainee_experience_log.get('experience_level', 'unknown') if self.trainee_experience_log else 'unknown'
        allergies = self.trainee_experience_log.get('allergies', []) if self.trainee_experience_log else []
        preferred_cuisine = self.trainee_experience_log.get('preferred_cuisine', 'any') if self.trainee_experience_log else 'any'
        notes = self.trainee_experience_log.get('notes', '') if self.trainee_experience_log else ''

        allergy_str = ", ".join(allergies) if allergies else "none"

        content = f"""
User Profile:
- Experience level: {experience_level}
- Allergies: {allergy_str}
- Preferred cuisine: {preferred_cuisine}
- Notes: {notes}

IMPORTANT INSTRUCTIONS:
- NEVER suggest or proceed with any recipe or step that contains any of the user's allergens: {allergy_str}.
- If the user requests a recipe with an allergen, gently warn them and suggest safe alternatives.
- ALWAYS adapt your explanations to the user's experience level.
- Before each step, check if the user is comfortable and ready to proceed.

Please follow these instructions carefully:

1. Introduction:
- Introduce yourself as ChefAI.
- Briefly explain that you're here to help with recipe preparation.

2. Recipe Confirmation:
- Confirm the recipe name with the user.

3. Ingredient Recall:
- List all the ingredients required for the recipe.
- Ask the user if they have all the ingredients ready.

4. Step-by-Step Guidance:
- Provide instructions one step at a time.
- After each step, wait for the user to say "next" or ask a question before proceeding.
- If the user asks a question, answer it thoroughly before continuing.

5. Completion:
- When all steps are complete, congratulate the user and ask if they need any final advice.

Remember to maintain a friendly and encouraging tone throughout the interaction.
"""
        return SystemMessage(content=content)
    
    def respond(self, conversation: list[BaseMessage], prompt: str) -> str:
        """Generate a response using the configured LLM."""
        messages = [self.system_message] + conversation + [HumanMessage(content=prompt)]
        
        try:
            response = self.llm_wrapper.invoke(messages)
            
            # Log basic info (simplified for testing)
            self.token_cost_log.append({
                "prompt": prompt,
                "response": response,
                "provider": self.llm_wrapper.get_provider_name(),
                "model": self.llm_wrapper.get_model_name(),
                "timestamp": None  # Add timestamp if needed
            })
            
            return response
        except Exception as e:
            error_msg = f"Error with {self.llm_wrapper.get_provider_name()}: {str(e)}"
            self.token_cost_log.append({
                "prompt": prompt,
                "response": None,
                "error": error_msg,
                "provider": self.llm_wrapper.get_provider_name(),
                "model": self.llm_wrapper.get_model_name(),
                "timestamp": None
            })
            raise Exception(error_msg)


class TestableTraineeAgent:
    """Trainee agent wrapper that can use different LLM providers for testing."""
    
    def __init__(self, llm_wrapper: BaseLLMWrapper, job_id: str, trainee_experience_log=None, conversation_mode="mixed"):
        self.llm_wrapper = llm_wrapper
        self.job_id = job_id
        self.trainee_experience_log = trainee_experience_log
        self.conversation_mode = conversation_mode
        self.token_cost_log = []
        self.current_step = 0
    
    def generate_response(self, chef_message: str, num_steps: int) -> str:
        """Generate trainee response using the configured LLM."""
        experience_level = self.trainee_experience_log.get('experience_level', 'beginner') if self.trainee_experience_log else 'beginner'
        notes = self.trainee_experience_log.get("notes", "") if self.trainee_experience_log else ""

        # Adjust question-asking behavior based on experience
        if experience_level == "advanced":
            question_instruction = (
                "ONLY ask a question if this step is ambiguous or unusually challenging for an expert cook. "
                "If everything is clear, say 'next'. Do NOT ask about basic techniques or substitutions."
            )
        elif experience_level == "intermediate":
            question_instruction = (
                "Ask a question if you are unsure about a technique or ingredient. "
                "Otherwise, say 'next'."
            )
        else:  # beginner
            question_instruction = (
                "If you have any doubt about the technique, ingredient, or process in this step, ask ONE SHORT, direct question. "
                "Otherwise, say 'next'."
            )

        # Optionally, use notes for further customization
        if notes and "never" in notes.lower():
            question_instruction += (
                f" You have noted: {notes}. If this step involves something you have never done, ask for extra explanation."
            )

        prompt = f"""
You are a cooking trainee with experience level: {experience_level}.
{f'Notes: {notes}' if notes else ''}
Your goal is to act in accordance with your experience.

Last chef instruction: {chef_message}

INSTRUCTIONS:
- If you are advanced, only ask about unclear or complex steps. If everything is clear, say 'next'.
- If you are a beginner, ask about anything you are unsure about.
{question_instruction}
"""
        
        try:
            response = self.llm_wrapper.invoke([HumanMessage(content=prompt)])
            
            # Log basic info
            self.token_cost_log.append({
                "prompt": prompt,
                "response": response,
                "provider": self.llm_wrapper.get_provider_name(),
                "model": self.llm_wrapper.get_model_name(),
                "timestamp": None
            })
            
            return response.strip().lower()
        except Exception as e:
            error_msg = f"Error with {self.llm_wrapper.get_provider_name()}: {str(e)}"
            self.token_cost_log.append({
                "prompt": prompt,
                "response": None,
                "error": error_msg,
                "provider": self.llm_wrapper.get_provider_name(),
                "model": self.llm_wrapper.get_model_name(),
                "timestamp": None
            })
            raise Exception(error_msg)