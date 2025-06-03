from dataclasses import dataclass
from typing import Dict, Any, Optional
import json

@dataclass
class ExperimentConfig:
    """Configuration for ablation experiments"""
    
    # Core system toggles
    use_faiss_search: bool = True
    use_ontology_substitution: bool = True
    use_llm_validation: bool = True
    use_supervisor_pattern: bool = False
    
    # Agent behavior
    trainee_experience_level: str = "beginner"  # beginner, intermediate, advanced
    chef_complexity_adaptation: bool = True
    
    # Search and retrieval
    faiss_k_recipes: int = 5
    faiss_k_substitutes: int = 10
    similarity_threshold: float = 0.8
    
    # LLM settings
    chef_model: str = "gpt-4-mini"
    trainee_model: str = "gpt-4-mini"
    use_structured_output: bool = True
    
    # Substitution system
    substitution_method: str = "hybrid"  # hybrid, llm_only, faiss_only, ontology_only
    validate_substitutions: bool = True
    
    # Conversation flow
    max_clarification_turns: int = 3
    auto_proceed_threshold: int = 2
    enable_allergy_warnings: bool = True
    
    # Experiment metadata
    experiment_id: str = ""
    description: str = ""
    baseline: bool = False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ExperimentConfig':
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    def get_variant_name(self) -> str:
        """Generate a descriptive name for this configuration variant"""
        components = []
        
        if not self.use_faiss_search:
            components.append("no_faiss")
        if not self.use_ontology_substitution:
            components.append("no_ontology")
        if not self.use_llm_validation:
            components.append("no_validation")
        if self.use_supervisor_pattern:
            components.append("supervisor")
        
        components.append(f"{self.trainee_experience_level}")
        components.append(f"{self.substitution_method}")
        
        return "_".join(components) if components else "baseline"

# Predefined experiment configurations
EXPERIMENT_CONFIGS = {
    "baseline": ExperimentConfig(
        experiment_id="baseline",
        description="Full system with all features enabled",
        baseline=True
    ),
    
    "no_faiss": ExperimentConfig(
        use_faiss_search=False,
        experiment_id="no_faiss",
        description="Disable FAISS search, use fuzzy matching only"
    ),
    
    "no_ontology": ExperimentConfig(
        use_ontology_substitution=False,
        experiment_id="no_ontology", 
        description="Disable ontology-based substitutions"
    ),
    
    "llm_only_substitution": ExperimentConfig(
        substitution_method="llm_only",
        experiment_id="llm_sub",
        description="Use only LLM for substitutions"
    ),
    
    "simple_trainee": ExperimentConfig(
        trainee_experience_level="advanced",
        max_clarification_turns=1,
        experiment_id="simple_trainee",
        description="Advanced trainee with minimal questions"
    ),
    
    "minimal_system": ExperimentConfig(
        use_faiss_search=False,
        use_ontology_substitution=False,
        use_llm_validation=False,
        chef_complexity_adaptation=False,
        experiment_id="minimal",
        description="Minimal system with basic features only"
    )
}