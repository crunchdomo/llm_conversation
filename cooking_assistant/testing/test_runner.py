"""Test runner with CLI interface for cooking conversation experiments."""

import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

from .test_scenarios import TestScenario, TestScenarioBuilder, ConversationType
from .llm_providers import LLMProvider, LLMProviderFactory, TestableChefAgent, TestableTraineeAgent
from ..core.graph.workflow import build_cooking_graph
from ..core.models import State
from langchain_core.messages import AIMessage, HumanMessage

import pandas as pd


class TestResult:
    """Container for test execution results."""
    
    def __init__(self, scenario: TestScenario, llm_provider: LLMProvider):
        self.scenario = scenario
        self.llm_provider = llm_provider
        self.test_id = str(uuid.uuid4())
        self.start_time = None
        self.end_time = None
        self.success = False
        self.error_message = None
        self.conversation_log = []
        self.metrics = {}
        self.final_state = None
        self.outcomes_achieved = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "test_id": self.test_id,
            "scenario_name": self.scenario.name,
            "conversation_type": self.scenario.conversation_type.value,
            "llm_provider": self.llm_provider.value,
            "user_query": self.scenario.user_query,
            "user_profile": asdict(self.scenario.user_profile),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            "success": self.success,
            "error_message": self.error_message,
            "conversation_log": self.conversation_log,
            "metrics": self.metrics,
            "outcomes_achieved": self.outcomes_achieved,
            "expected_outcomes": self.scenario.expected_outcomes,
            "metadata": self.scenario.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], scenario: TestScenario = None, llm_provider: LLMProvider = None):
        """Create TestResult from dictionary."""
        if scenario is None or llm_provider is None:
            # If we don't have the original objects, create minimal ones
            from .test_scenarios import UserProfile
            user_profile = UserProfile(**data.get("user_profile", {}))
            
            scenario = TestScenario(
                name=data["scenario_name"],
                conversation_type=ConversationType(data["conversation_type"]),
                user_query=data["user_query"],
                user_profile=user_profile,
                expected_outcomes=data.get("expected_outcomes", []),
                llm_providers=[LLMProvider(data["llm_provider"])],
                metadata=data.get("metadata", {})
            )
            llm_provider = LLMProvider(data["llm_provider"])
        
        result = cls(scenario, llm_provider)
        result.test_id = data["test_id"]
        result.start_time = datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None
        result.end_time = datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None
        result.success = data["success"]
        result.error_message = data.get("error_message")
        result.conversation_log = data.get("conversation_log", [])
        result.metrics = data.get("metrics", {})
        result.outcomes_achieved = data.get("outcomes_achieved", [])
        result.final_state = data.get("final_state")
        
        return result


class ConversationTestRunner:
    """Main test runner for cooking conversation experiments."""
    
    def __init__(self, recipes_csv_path: str = "13k-recipes.csv", sample_size: int = 100):
        self.recipes_csv_path = recipes_csv_path
        self.sample_size = sample_size
        self.recipes_df = None
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize mock components for testing
        self._load_recipes()
    
    def _load_recipes(self):
        """Load recipes for testing."""
        try:
            self.recipes_df = pd.read_csv(self.recipes_csv_path)
            if self.sample_size and len(self.recipes_df) > self.sample_size:
                self.recipes_df = self.recipes_df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
            print(f"Loaded {len(self.recipes_df)} recipes for testing")
        except FileNotFoundError:
            print(f"Warning: Could not load recipes from {self.recipes_csv_path}")
            # Create minimal mock dataframe
            self.recipes_df = pd.DataFrame([
                {
                    "Title": "Test Recipe",
                    "Cleaned_Ingredients": "['chicken', 'rice', 'vegetables']",
                    "Instructions": "Cook chicken. Add rice and vegetables. Serve hot."
                }
            ])
    
    def run_single_test(self, scenario: TestScenario, llm_provider: LLMProvider, timeout: int = 300) -> TestResult:
        """Run a single test scenario with a specific LLM provider."""
        result = TestResult(scenario, llm_provider)
        result.start_time = datetime.now()
        
        try:
            # Validate provider access
            if not LLMProviderFactory.validate_provider_access(llm_provider):
                raise Exception(f"No API key found for provider: {llm_provider.value}")
            
            # Create LLM wrapper
            llm_wrapper = LLMProviderFactory.create_provider(llm_provider)
            
            # Create agents with the LLM wrapper
            chef = TestableChefAgent(
                llm_wrapper=llm_wrapper,
                job_id=result.test_id,
                recipes_df=self.recipes_df,
                faiss_index=None,  # Mock for testing
                recipe_searcher=None,  # Mock for testing
                ingredient_substituter=None,  # Mock for testing
                trainee_experience_log=asdict(scenario.user_profile)
            )
            
            trainee = TestableTraineeAgent(
                llm_wrapper=llm_wrapper,
                job_id=result.test_id,
                trainee_experience_log=asdict(scenario.user_profile)
            )
            
            # Run conversation
            conversation_result = self._run_conversation(scenario, chef, trainee, timeout)
            
            # Process results
            result.success = conversation_result["success"]
            result.conversation_log = conversation_result["conversation_log"]
            result.final_state = conversation_result["final_state"]
            result.metrics = self._calculate_metrics(chef, trainee, conversation_result)
            result.outcomes_achieved = self._analyze_outcomes(scenario, conversation_result)
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            print(f"Test failed for {scenario.name} with {llm_provider.value}: {e}")
        
        result.end_time = datetime.now()
        return result
    
    def _run_conversation(self, scenario: TestScenario, chef: TestableChefAgent, trainee: TestableTraineeAgent, timeout: int) -> Dict[str, Any]:
        """Run the actual conversation flow."""
        
        # For testing purposes, we'll simulate a simplified conversation
        # In a full implementation, this would use your LangGraph workflow
        
        conversation_log = []
        messages = [chef.system_message]
        
        try:
            # Initial chef introduction
            intro_response = chef.respond([], "Introduce yourself and ask what the user wants to cook.")
            conversation_log.append({
                "speaker": "chef",
                "message": intro_response,
                "timestamp": datetime.now().isoformat()
            })
            messages.append(AIMessage(content=intro_response))
            
            # User query
            conversation_log.append({
                "speaker": "user",
                "message": scenario.user_query,
                "timestamp": datetime.now().isoformat()
            })
            messages.append(HumanMessage(content=scenario.user_query))
            
            # Chef response to user query
            chef_response = chef.respond(messages[1:], scenario.user_query)
            conversation_log.append({
                "speaker": "chef", 
                "message": chef_response,
                "timestamp": datetime.now().isoformat()
            })
            messages.append(AIMessage(content=chef_response))
            
            # Simplified conversation flow - a few back and forth exchanges
            for turn in range(min(scenario.max_turns, 5)):  # Limit turns for testing
                # Trainee response
                trainee_response = trainee.generate_response(chef_response, 5)
                conversation_log.append({
                    "speaker": "trainee",
                    "message": trainee_response,
                    "timestamp": datetime.now().isoformat()
                })
                messages.append(HumanMessage(content=trainee_response))
                
                # Chef response
                chef_response = chef.respond(messages[-3:], f"Respond to: {trainee_response}")
                conversation_log.append({
                    "speaker": "chef",
                    "message": chef_response,
                    "timestamp": datetime.now().isoformat()
                })
                messages.append(AIMessage(content=chef_response))
                
                # Check for natural ending
                if "next" in trainee_response and turn >= 2:
                    break
            
            return {
                "success": True,
                "conversation_log": conversation_log,
                "final_state": {"messages": messages, "turn_count": len(conversation_log)},
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "conversation_log": conversation_log,
                "final_state": None,
                "error": str(e)
            }
    
    def _calculate_metrics(self, chef: TestableChefAgent, trainee: TestableTraineeAgent, conversation_result: Dict) -> Dict[str, Any]:
        """Calculate metrics for the conversation."""
        metrics = {
            "total_turns": len(conversation_result.get("conversation_log", [])),
            "chef_turns": len([log for log in conversation_result.get("conversation_log", []) if log["speaker"] == "chef"]),
            "trainee_turns": len([log for log in conversation_result.get("conversation_log", []) if log["speaker"] == "trainee"]),
            "chef_responses": len(chef.token_cost_log),
            "trainee_responses": len(trainee.token_cost_log),
            "conversation_success": conversation_result.get("success", False)
        }
        
        # Add LLM-specific metrics if available
        if chef.token_cost_log:
            metrics["chef_provider"] = chef.token_cost_log[0].get("provider")
            metrics["chef_model"] = chef.token_cost_log[0].get("model")
        
        return metrics
    
    def _analyze_outcomes(self, scenario: TestScenario, conversation_result: Dict) -> List[str]:
        """Analyze which expected outcomes were achieved."""
        achieved_outcomes = []
        conversation_log = conversation_result.get("conversation_log", [])
        
        # Simple keyword-based outcome detection
        full_conversation = " ".join([log["message"] for log in conversation_log]).lower()
        
        for expected_outcome in scenario.expected_outcomes:
            if self._check_outcome_achieved(expected_outcome, full_conversation, conversation_log):
                achieved_outcomes.append(expected_outcome)
        
        return achieved_outcomes
    
    def _check_outcome_achieved(self, outcome: str, full_conversation: str, conversation_log: List) -> bool:
        """Check if a specific outcome was achieved."""
        outcome_lower = outcome.lower()
        
        # Basic keyword matching - you could make this more sophisticated
        outcome_indicators = {
            "recipe_found": ["recipe", "ingredients", "instructions"],
            "ingredients_listed": ["ingredients", "need", "require"],
            "step_by_step_guidance": ["step", "next", "first", "then"],
            "completion_confirmation": ["complete", "done", "finished", "congratulations"],
            "substitution_confirmed": ["substitute", "instead", "replace", "use"],
            "allergen_free": ["safe", "no allergens", "allergy-free"],
            "dietary_compliant": ["vegetarian", "vegan", "keto", "compliant"]
        }
        
        indicators = outcome_indicators.get(outcome_lower, [outcome_lower])
        return any(indicator in full_conversation for indicator in indicators)
    
    def run_batch_tests(self, scenarios: List[TestScenario], llm_providers: List[LLMProvider], max_workers: int = 3) -> List[TestResult]:
        """Run multiple test scenarios in parallel."""
        test_tasks = []
        
        # Create all test combinations
        for scenario in scenarios:
            for provider in llm_providers:
                if provider in scenario.llm_providers:  # Only run if scenario supports this provider
                    test_tasks.append((scenario, provider))
        
        print(f"Running {len(test_tasks)} test combinations with {max_workers} workers...")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.run_single_test, scenario, provider): (scenario, provider)
                for scenario, provider in test_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                scenario, provider = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "✅ PASS" if result.success else "❌ FAIL"
                    print(f"{status} {scenario.name} ({provider.value})")
                except Exception as e:
                    print(f"❌ ERROR {scenario.name} ({provider.value}): {e}")
        
        return results
    
    def save_results(self, results: List[TestResult], batch_name: str = None) -> str:
        """Save test results to JSON file with informative naming."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not batch_name:
            batch_name = f"test_batch_{timestamp}"
        
        # Generate informative filename based on results content
        filename = self._generate_results_filename(results, batch_name, timestamp)
        results_file = self.results_dir / filename
        
        # Convert results to serializable format
        results_data = {
            "batch_name": batch_name,
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "successful_tests": len([r for r in results if r.success]),
            "failed_tests": len([r for r in results if not r.success]),
            "results": [result.to_dict() for result in results]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        return str(results_file)
    
    def _generate_results_filename(self, results: List[TestResult], batch_name: str, timestamp: str) -> str:
        """Generate informative filename based on test results content."""
        if not results:
            return f"{timestamp}_{batch_name}_empty.json"
        
        # Extract unique LLM providers
        providers = list(set([r.llm_provider.value for r in results]))
        providers.sort()
        
        # Extract unique conversation types  
        conv_types = list(set([r.scenario.conversation_type.value for r in results]))
        conv_types.sort()
        
        # Create provider string (max 3 providers shown)
        if len(providers) == 1:
            provider_str = providers[0].replace('_', '')
        elif len(providers) <= 3:
            provider_str = '+'.join([p.replace('_', '') for p in providers])
        else:
            provider_str = f"{len(providers)}providers"
        
        # Create conversation type string (max 3 types shown)
        if len(conv_types) == 1:
            type_str = conv_types[0].replace('_', '-')
        elif len(conv_types) <= 3:
            type_str = '+'.join([ct.replace('_', '-') for ct in conv_types])
        else:
            type_str = "all-types"
        
        # Add summary stats
        total_tests = len(results)
        success_tests = len([r for r in results if r.success])
        success_rate = int((success_tests / total_tests) * 100) if total_tests > 0 else 0
        
        # Clean batch name for filename
        clean_batch_name = batch_name.replace(' ', '-').replace('_', '-')
        
        # Generate job ID (first 8 chars of a UUID)
        job_id = str(uuid.uuid4())[:8]
        
        return f"{timestamp}_{provider_str}_{type_str}_{clean_batch_name}_{success_rate}pct_{job_id}.json"
    
    def generate_summary_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate a summary report of test results."""
        total_tests = len(results)
        successful_tests = [r for r in results if r.success]
        failed_tests = [r for r in results if not r.success]
        
        # Group by conversation type
        by_conversation_type = {}
        for result in results:
            conv_type = result.scenario.conversation_type.value
            if conv_type not in by_conversation_type:
                by_conversation_type[conv_type] = {"total": 0, "success": 0, "fail": 0}
            by_conversation_type[conv_type]["total"] += 1
            if result.success:
                by_conversation_type[conv_type]["success"] += 1
            else:
                by_conversation_type[conv_type]["fail"] += 1
        
        # Group by LLM provider
        by_llm_provider = {}
        for result in results:
            provider = result.llm_provider.value
            if provider not in by_llm_provider:
                by_llm_provider[provider] = {"total": 0, "success": 0, "fail": 0}
            by_llm_provider[provider]["total"] += 1
            if result.success:
                by_llm_provider[provider]["success"] += 1
            else:
                by_llm_provider[provider]["fail"] += 1
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / total_tests if total_tests > 0 else 0
            },
            "by_conversation_type": by_conversation_type,
            "by_llm_provider": by_llm_provider,
            "common_failures": self._analyze_common_failures(failed_tests)
        }
    
    def _analyze_common_failures(self, failed_tests: List[TestResult]) -> Dict[str, int]:
        """Analyze common failure patterns."""
        failure_patterns = {}
        for result in failed_tests:
            if result.error_message:
                # Extract common error patterns
                error = result.error_message.lower()
                if "api key" in error:
                    failure_patterns["api_key_missing"] = failure_patterns.get("api_key_missing", 0) + 1
                elif "timeout" in error:
                    failure_patterns["timeout"] = failure_patterns.get("timeout", 0) + 1
                elif "rate limit" in error:
                    failure_patterns["rate_limit"] = failure_patterns.get("rate_limit", 0) + 1
                else:
                    failure_patterns["other"] = failure_patterns.get("other", 0) + 1
        return failure_patterns


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run cooking conversation tests with different LLMs")
    
    # Test selection
    parser.add_argument("--scenarios", nargs="+", help="Specific scenario names to run")
    parser.add_argument("--conversation-types", nargs="+", 
                      choices=[t.value for t in ConversationType],
                      help="Filter by conversation types")
    parser.add_argument("--llm-providers", nargs="+",
                      choices=[p.value for p in LLMProvider], 
                      help="LLM providers to test")
    
    # Test configuration
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum parallel workers")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per test in seconds")
    parser.add_argument("--sample-size", type=int, default=100, help="Recipe sample size")
    
    # Output configuration
    parser.add_argument("--batch-name", help="Name for this test batch")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for results")
    parser.add_argument("--recipes-csv", default="13k-recipes.csv", help="Path to recipes CSV")
    
    # Options
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("--list-providers", action="store_true", help="List available LLM providers")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # List scenarios if requested
    if args.list_scenarios:
        scenarios = TestScenarioBuilder.get_all_scenarios()
        print("Available test scenarios:")
        for scenario in scenarios:
            print(f"  {scenario.name} ({scenario.conversation_type.value})")
        return
    
    # List providers if requested
    if args.list_providers:
        available_providers = LLMProviderFactory.get_available_providers()
        all_providers = list(LLMProvider)
        
        print("LLM Providers:")
        for provider in all_providers:
            status = "✅ Available" if provider in available_providers else "❌ No API key"
            print(f"  {provider.value}: {status}")
        return
    
    # Initialize test runner
    runner = ConversationTestRunner(
        recipes_csv_path=args.recipes_csv,
        sample_size=args.sample_size
    )
    
    # Select scenarios
    all_scenarios = TestScenarioBuilder.get_all_scenarios()
    if args.scenarios:
        scenarios = [s for s in all_scenarios if s.name in args.scenarios]
    elif args.conversation_types:
        conv_types = [ConversationType(t) for t in args.conversation_types]
        scenarios = [s for s in all_scenarios if s.conversation_type in conv_types]
    else:
        scenarios = all_scenarios
    
    # Select LLM providers
    available_providers = LLMProviderFactory.get_available_providers()
    if args.llm_providers:
        requested_providers = [LLMProvider(p) for p in args.llm_providers]
        llm_providers = [p for p in requested_providers if p in available_providers]
        
        missing_providers = set(requested_providers) - set(llm_providers)
        if missing_providers:
            print(f"Warning: API keys missing for: {[p.value for p in missing_providers]}")
    else:
        llm_providers = available_providers
    
    if not scenarios:
        print("No scenarios selected!")
        return
    
    if not llm_providers:
        print("No LLM providers available! Please set API keys.")
        return
    
    # Show what will be run
    test_count = sum(1 for s in scenarios for p in llm_providers if p in s.llm_providers)
    print(f"Will run {test_count} tests:")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Providers: {[p.value for p in llm_providers]}")
    print(f"  Max workers: {args.max_workers}")
    
    if args.dry_run:
        print("Dry run complete.")
        return
    
    # Run tests
    print("\nStarting tests...")
    start_time = time.time()
    
    results = runner.run_batch_tests(
        scenarios=scenarios,
        llm_providers=llm_providers,
        max_workers=args.max_workers
    )
    
    end_time = time.time()
    print(f"\nTests completed in {end_time - start_time:.1f} seconds")
    
    # Save results
    results_file = runner.save_results(results, args.batch_name)
    
    # Generate summary
    summary = runner.generate_summary_report(results)
    print(f"\n📊 TEST SUMMARY:")
    print(f"  Total tests: {summary['summary']['total_tests']}")
    print(f"  Successful: {summary['summary']['successful_tests']}")
    print(f"  Failed: {summary['summary']['failed_tests']}")
    print(f"  Success rate: {summary['summary']['success_rate']:.1%}")
    
    print(f"\n🤖 BY LLM PROVIDER:")
    for provider, stats in summary['by_llm_provider'].items():
        success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {provider}: {stats['success']}/{stats['total']} ({success_rate:.1%})")
    
    print(f"\n💬 BY CONVERSATION TYPE:")
    for conv_type, stats in summary['by_conversation_type'].items():
        success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {conv_type}: {stats['success']}/{stats['total']} ({success_rate:.1%})")


if __name__ == "__main__":
    main()