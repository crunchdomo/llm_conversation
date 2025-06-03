#!/usr/bin/env python3
"""
Sample script demonstrating how to use the testing framework.

This script shows different ways to run cooking conversation tests
with multiple LLM providers and analyze the results.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cooking_assistant.testing import (
    TestScenarioBuilder, 
    LLMProvider, 
    LLMProviderFactory,
    ConversationTestRunner
)
from cooking_assistant.testing.batch_processor import run_batch_experiments
from cooking_assistant.testing.result_analyzer import ResultAnalyzer


def run_quick_validation():
    """Run a quick validation test with a few scenarios."""
    print("🧪 QUICK VALIDATION TEST")
    print("=" * 50)
    
    # Check available providers
    available_providers = LLMProviderFactory.get_available_providers()
    if not available_providers:
        print("❌ No LLM providers available. Please set API keys:")
        print("   export OPENAI_API_KEY='your_key'")
        print("   export ANTHROPIC_API_KEY='your_key'")
        return
    
    print(f"✅ Available providers: {[p.value for p in available_providers]}")
    
    # Select a few test scenarios
    scenarios = [
        TestScenarioBuilder.create_specific_recipe_scenarios()[0],  # chicken teriyaki
        TestScenarioBuilder.create_ingredient_based_scenarios()[0],  # chicken and onions
        TestScenarioBuilder.create_substitution_scenarios()[0]       # turkey for chicken
    ]
    
    # Use first 2 available providers
    providers = available_providers[:2]
    
    print(f"🎯 Running {len(scenarios)} scenarios with {len(providers)} providers...")
    
    # Run tests
    runner = ConversationTestRunner(sample_size=10)  # Small sample for speed
    results = runner.run_batch_tests(scenarios, providers, max_workers=2)
    
    # Save and analyze results
    results_file = runner.save_results(results, "quick_validation")
    summary = runner.generate_summary_report(results)
    
    print(f"\n📊 QUICK RESULTS:")
    print(f"   Success rate: {summary['summary']['success_rate']:.1%}")
    print(f"   Tests completed: {summary['summary']['total_tests']}")
    print(f"   Results saved: {results_file}")
    
    return results_file


def run_provider_comparison():
    """Compare performance between different LLM providers."""
    print("\n🤖 LLM PROVIDER COMPARISON")
    print("=" * 50)
    
    # Get available providers
    available_providers = LLMProviderFactory.get_available_providers()
    
    if len(available_providers) < 2:
        print("❌ Need at least 2 LLM providers for comparison")
        return
    
    # Use specific recipe scenarios for comparison (most standardized)
    scenarios = TestScenarioBuilder.create_specific_recipe_scenarios()
    
    print(f"🎯 Comparing {len(available_providers)} providers on {len(scenarios)} scenarios...")
    
    # Run batch experiment with recovery
    results, results_file = run_batch_experiments(
        scenarios=scenarios,
        llm_providers=available_providers,
        batch_name="provider_comparison",
        max_workers=2,
        retry_failed=True,
        sample_size=20  # Medium sample size
    )
    
    # Analyze results
    analyzer = ResultAnalyzer(results_file)
    comparison = analyzer.compare_providers()
    
    print(f"\n📊 PROVIDER COMPARISON:")
    print(comparison)
    
    # Create visualizations
    analyzer.create_visualizations("provider_comparison_plots")
    
    return results_file


def run_conversation_type_analysis():
    """Analyze performance across different conversation types."""
    print("\n💬 CONVERSATION TYPE ANALYSIS")
    print("=" * 50)
    
    available_providers = LLMProviderFactory.get_available_providers()
    if not available_providers:
        print("❌ No LLM providers available")
        return
    
    # Get one scenario from each conversation type
    all_scenarios = TestScenarioBuilder.get_all_scenarios()
    
    # Sample one scenario per conversation type
    conversation_types = set(s.conversation_type for s in all_scenarios)
    selected_scenarios = []
    for conv_type in conversation_types:
        scenarios_of_type = [s for s in all_scenarios if s.conversation_type == conv_type]
        if scenarios_of_type:
            selected_scenarios.append(scenarios_of_type[0])  # Take first one
    
    print(f"🎯 Testing {len(selected_scenarios)} conversation types...")
    
    # Use first available provider for consistency
    provider = available_providers[0]
    
    # Run tests
    runner = ConversationTestRunner(sample_size=15)
    results = runner.run_batch_tests(selected_scenarios, [provider], max_workers=1)
    
    # Save and analyze
    results_file = runner.save_results(results, "conversation_types")
    analyzer = ResultAnalyzer(results_file)
    summary = analyzer.generate_summary_report()
    
    print(f"\n📊 BY CONVERSATION TYPE:")
    for conv_type, stats in summary['by_conversation_type'].items():
        success_rate = stats.get('success_mean', 0)
        count = stats.get('success_count', 0)
        avg_duration = stats.get('duration_seconds_mean', 0)
        print(f"   {conv_type}: {success_rate:.1%} success, {avg_duration:.1f}s avg ({count} tests)")
    
    return results_file


def analyze_existing_results(results_file: str):
    """Demonstrate result analysis on existing test results."""
    print(f"\n📈 ANALYZING RESULTS: {results_file}")
    print("=" * 50)
    
    try:
        analyzer = ResultAnalyzer(results_file)
        
        # Generate comprehensive summary
        summary = analyzer.generate_summary_report()
        
        print("📊 OVERVIEW:")
        overview = summary['overview']
        print(f"   Total tests: {overview['total_tests']}")
        print(f"   Success rate: {overview['overall_success_rate']:.1%}")
        print(f"   Avg duration: {overview['average_duration']:.1f}s")
        print(f"   Avg turns: {overview['average_turns']:.1f}")
        
        # Find best performers
        best_scenarios = analyzer.find_best_performers('success_rate', top_n=3)
        print(f"\n🏆 TOP PERFORMERS:")
        for _, row in best_scenarios.iterrows():
            print(f"   {row['scenario_name']} + {row['llm_provider']}: {row['success']:.1%}")
        
        # Create detailed analysis
        output_dir = f"analysis_{Path(results_file).stem}"
        analyzer.create_visualizations(output_dir)
        analyzer.export_detailed_csv(f"{output_dir}/detailed_results.csv")
        
        print(f"📁 Detailed analysis saved to: {output_dir}/")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")


def main():
    """Run sample tests and demonstrations."""
    print("🧑‍🍳 COOKING CONVERSATION TESTING DEMO")
    print("=" * 60)
    
    # 1. Quick validation
    results_file_1 = run_quick_validation()
    
    # 2. Provider comparison (if multiple providers available)
    results_file_2 = run_provider_comparison()
    
    # 3. Conversation type analysis
    results_file_3 = run_conversation_type_analysis()
    
    # 4. Analyze one of the result files
    if results_file_1:
        analyze_existing_results(results_file_1)
    
    print(f"\n🎉 DEMO COMPLETE!")
    print(f"Check the test_results/ directory for all output files.")
    print(f"Run analysis on any results file with:")
    print(f"   python cooking_assistant/testing/result_analyzer.py <results_file.json>")


if __name__ == "__main__":
    main()