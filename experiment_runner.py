import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import pandas as pd
from pathlib import Path

from experiment_config import ExperimentConfig, EXPERIMENT_CONFIGS
from configurable_agents import ConfigurableChefAgent, ConfigurableTraineeAgent

class ExperimentMetrics:
    """Collect and track experiment metrics"""
    
    def __init__(self, experiment_id: str, config: ExperimentConfig):
        self.experiment_id = experiment_id
        self.config = config
        self.start_time = datetime.now()
        self.metrics = {
            "conversation_length": 0,
            "chef_turns": 0,
            "trainee_turns": 0,
            "clarification_requests": 0,
            "substitution_requests": 0,
            "recipe_selections": 0,
            "errors": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "completion_rate": 0.0,
            "user_satisfaction": None,
            "task_success": False
        }
        self.conversation_log = []
        self.timeline = []
    
    def log_turn(self, speaker: str, message: str, tokens: int = 0, cost: float = 0.0):
        """Log a conversation turn"""
        turn_data = {
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker,
            "message": message,
            "tokens": tokens,
            "cost": cost
        }
        self.conversation_log.append(turn_data)
        
        # Update metrics
        self.metrics["conversation_length"] += 1
        if speaker == "chef":
            self.metrics["chef_turns"] += 1
        elif speaker == "trainee":
            self.metrics["trainee_turns"] += 1
        
        self.metrics["total_tokens"] += tokens
        self.metrics["total_cost"] += cost
        
        # Detect special events
        if "clarif" in message.lower() or "question" in message.lower():
            self.metrics["clarification_requests"] += 1
        if "substitute" in message.lower() or "replace" in message.lower():
            self.metrics["substitution_requests"] += 1
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a timeline event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.timeline.append(event)
        
        # Update relevant metrics
        if event_type == "recipe_selected":
            self.metrics["recipe_selections"] += 1
        elif event_type == "error":
            self.metrics["errors"] += 1
        elif event_type == "task_completed":
            self.metrics["task_success"] = True
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize metrics and return summary"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate completion rate
        if self.metrics["conversation_length"] > 0:
            self.metrics["completion_rate"] = min(1.0, self.metrics["conversation_length"] / 10)  # Assume 10 turns = complete
        
        return {
            "experiment_id": self.experiment_id,
            "config": asdict(self.config),
            "duration_seconds": duration,
            "metrics": self.metrics,
            "conversation_log": self.conversation_log,
            "timeline": self.timeline
        }

class ExperimentRunner:
    """Run ablation experiments systematically"""
    
    def __init__(self, recipes_df: pd.DataFrame, output_dir: str = "experiment_results"):
        self.recipes_df = recipes_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def run_single_experiment(self, config: ExperimentConfig, 
                            test_scenarios: List[Dict[str, Any]], 
                            iterations: int = 1) -> List[Dict[str, Any]]:
        """Run a single experiment configuration"""
        experiment_results = []
        
        for iteration in range(iterations):
            for scenario_idx, scenario in enumerate(test_scenarios):
                # Create unique job ID
                job_id = f"{config.experiment_id}_iter{iteration}_scenario{scenario_idx}_{uuid.uuid4().hex[:8]}"
                
                print(f"Running: {config.experiment_id} - Iteration {iteration+1}/{iterations} - Scenario {scenario_idx+1}/{len(test_scenarios)}")
                
                # Initialize metrics tracking
                metrics = ExperimentMetrics(job_id, config)
                
                try:
                    # Run the experiment
                    result = self._run_cooking_session(config, scenario, metrics)
                    experiment_results.append(result)
                    
                except Exception as e:
                    print(f"Error in experiment {job_id}: {e}")
                    metrics.log_event("error", {"error": str(e)})
                    experiment_results.append(metrics.finalize())
        
        return experiment_results
    
    def _run_cooking_session(self, config: ExperimentConfig, 
                           scenario: Dict[str, Any], 
                           metrics: ExperimentMetrics) -> Dict[str, Any]:
        """Run a single cooking session"""
        
        # Initialize agents
        chef = ConfigurableChefAgent(
            job_id=metrics.experiment_id,
            config=config,
            recipes_df=self.recipes_df,
            trainee_profile=scenario.get("trainee_profile", {})
        )
        
        trainee = ConfigurableTraineeAgent(
            job_id=metrics.experiment_id,
            config=config,
            trainee_profile=scenario.get("trainee_profile", {})
        )
        
        # Start conversation
        user_query = scenario["user_query"]
        metrics.log_event("session_start", {"query": user_query, "scenario": scenario})
        
        # Recipe selection
        recipe_result = chef.select_recipe(user_query)
        metrics.log_turn("chef", f"Recipe search: {recipe_result}", tokens=50, cost=0.001)
        
        if "error" in recipe_result:
            metrics.log_event("error", recipe_result)
            return metrics.finalize()
        
        metrics.log_event("recipe_selected", recipe_result)
        
        # Simulate conversation turns
        max_turns = scenario.get("max_turns", 10)
        for turn in range(max_turns):
            # Chef provides step
            chef_message = f"Step {turn+1}: {scenario.get('steps', ['Generic step'])[turn % len(scenario.get('steps', ['Generic step']))]}"
            metrics.log_turn("chef", chef_message, tokens=75, cost=0.0015)
            
            # Trainee responds
            trainee_response = trainee.generate_response(chef_message, turn+1)
            metrics.log_turn("trainee", trainee_response, tokens=30, cost=0.0006)
            
            # Check for completion
            if trainee_response.strip().lower() == "next" and turn >= 3:
                metrics.log_event("task_completed", {"turns": turn+1})
                break
            
            # Add small delay to simulate real conversation
            time.sleep(0.1)
        
        return metrics.finalize()
    
    def run_ablation_study(self, test_scenarios: List[Dict[str, Any]], 
                          experiment_names: List[str] = None,
                          iterations: int = 3) -> pd.DataFrame:
        """Run complete ablation study"""
        
        if experiment_names is None:
            experiment_names = list(EXPERIMENT_CONFIGS.keys())
        
        all_results = []
        
        for exp_name in experiment_names:
            if exp_name not in EXPERIMENT_CONFIGS:
                print(f"Warning: Unknown experiment '{exp_name}', skipping")
                continue
                
            config = EXPERIMENT_CONFIGS[exp_name]
            print(f"\n{'='*60}")
            print(f"Running Ablation: {exp_name}")
            print(f"Description: {config.description}")
            print(f"{'='*60}")
            
            # Run experiments for this configuration
            results = self.run_single_experiment(config, test_scenarios, iterations)
            all_results.extend(results)
            
            # Save intermediate results
            self._save_results(results, f"{exp_name}_results.json")
        
        # Convert to DataFrame for analysis
        df = self._results_to_dataframe(all_results)
        
        # Save final results
        df.to_csv(self.output_dir / "ablation_study_results.csv", index=False)
        self._save_results(all_results, "complete_results.json")
        
        return df
    
    def _save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save results to JSON file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def _results_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        rows = []
        
        for result in results:
            row = {
                "experiment_id": result["experiment_id"],
                "config_name": result["config"]["experiment_id"],
                "duration_seconds": result["duration_seconds"],
                **result["metrics"]  # Flatten metrics
            }
            
            # Add key config parameters
            config = result["config"]
            row.update({
                "use_faiss_search": config["use_faiss_search"],
                "use_ontology_substitution": config["use_ontology_substitution"],
                "use_llm_validation": config["use_llm_validation"],
                "trainee_experience_level": config["trainee_experience_level"],
                "substitution_method": config["substitution_method"],
            })
            
            rows.append(row)
        
        return pd.DataFrame(rows)

# Predefined test scenarios
DEFAULT_TEST_SCENARIOS = [
    {
        "user_query": "I want to make chicken parmesan",
        "trainee_profile": {"experience_level": "beginner", "allergies": []},
        "max_turns": 8,
        "steps": ["Pound chicken", "Bread chicken", "Fry chicken", "Add sauce", "Bake with cheese"]
    },
    {
        "user_query": "Can I substitute beef for chicken in this recipe?",
        "trainee_profile": {"experience_level": "intermediate", "allergies": ["chicken"]},
        "max_turns": 6,
        "steps": ["Prepare substitute", "Adjust cooking time", "Season appropriately"]
    },
    {
        "user_query": "Show me pasta recipes",
        "trainee_profile": {"experience_level": "advanced", "allergies": []},
        "max_turns": 5,
        "steps": ["Boil water", "Add pasta", "Make sauce", "Combine", "Serve"]
    }
]

if __name__ == "__main__":
    # Example usage
    recipes_df = pd.read_csv("13k-recipes.csv")  # Your recipe dataset
    
    runner = ExperimentRunner(recipes_df)
    
    # Run ablation study
    results_df = runner.run_ablation_study(
        test_scenarios=DEFAULT_TEST_SCENARIOS,
        experiment_names=["baseline", "no_faiss", "llm_only_substitution", "minimal_system"],
        iterations=2
    )
    
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)
    print(results_df.groupby("config_name")[["task_success", "total_tokens", "total_cost"]].mean())