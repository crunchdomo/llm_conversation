#!/usr/bin/env python3
"""
Example script to run ablation studies on the cooking assistant system.
This demonstrates how to systematically test different configuration combinations.
"""

import pandas as pd
from experiment_config import ExperimentConfig, EXPERIMENT_CONFIGS
from experiment_runner import ExperimentRunner, DEFAULT_TEST_SCENARIOS
from analysis_tools import AblationAnalyzer, statistical_significance_test

def main():
    """Run a complete ablation study"""
    
    print("🧪 Starting Ablation Study for Cooking Assistant")
    print("=" * 60)
    
    # Load your recipe dataset
    try:
        recipes_df = pd.read_csv("13k-recipes.csv")
        print(f"✅ Loaded {len(recipes_df)} recipes")
    except FileNotFoundError:
        print("❌ Recipe dataset not found. Creating dummy data for demo.")
        recipes_df = pd.DataFrame({
            'Title': ['Chicken Parmesan', 'Beef Stew', 'Pasta Carbonara'],
            'Cleaned_Ingredients': [
                "['chicken breast', 'parmesan cheese', 'breadcrumbs']",
                "['beef chunks', 'carrots', 'potatoes']", 
                "['pasta', 'eggs', 'bacon', 'parmesan']"
            ],
            'Instructions': [
                'Step 1: Pound chicken. Step 2: Bread chicken. Step 3: Fry.',
                'Step 1: Brown beef. Step 2: Add vegetables. Step 3: Simmer.',
                'Step 1: Boil pasta. Step 2: Mix eggs. Step 3: Combine.'
            ]
        })
    
    # Custom test scenarios for more thorough testing
    extended_scenarios = DEFAULT_TEST_SCENARIOS + [
        {
            "user_query": "I need dairy-free pasta recipes",
            "trainee_profile": {"experience_level": "intermediate", "allergies": ["milk", "cheese"]},
            "max_turns": 7,
            "steps": ["Choose recipe", "Check ingredients", "Prepare substitutes", "Cook pasta", "Make sauce"]
        },
        {
            "user_query": "Quick 15-minute meals",
            "trainee_profile": {"experience_level": "advanced", "allergies": []},
            "max_turns": 4,
            "steps": ["Select quick recipe", "Prep ingredients", "Cook quickly", "Serve"]
        }
    ]
    
    # Initialize experiment runner
    runner = ExperimentRunner(recipes_df, output_dir="ablation_results")
    
    # Define which experiments to run
    experiments_to_run = [
        "baseline",           # Full system
        "no_faiss",          # Disable FAISS search
        "no_ontology",       # Disable ontology substitutions
        "llm_only_substitution",  # LLM-only substitutions
        "simple_trainee",    # Advanced trainee with minimal questions
        "minimal_system"     # Minimal features
    ]
    
    print(f"🔬 Running {len(experiments_to_run)} experiment configurations")
    print(f"📊 Testing {len(extended_scenarios)} scenarios each")
    print(f"🔄 {2} iterations per configuration")
    
    # Run the ablation study
    results_df = runner.run_ablation_study(
        test_scenarios=extended_scenarios,
        experiment_names=experiments_to_run,
        iterations=2  # Increase for more robust results
    )
    
    print("\n" + "=" * 60)
    print("📈 ANALYZING RESULTS")
    print("=" * 60)
    
    # Analyze results
    analyzer = AblationAnalyzer(results_df)
    
    # Generate comprehensive report
    report_path = analyzer.generate_report("ablation_analysis")
    
    # Quick summary
    print("\n🏆 QUICK SUMMARY:")
    print("-" * 40)
    
    # Best performing configuration
    avg_success = results_df.groupby("config_name")["task_success"].mean()
    best_config = avg_success.idxmax()
    print(f"Best performance: {best_config} ({avg_success[best_config]:.2%} success rate)")
    
    # Most cost-efficient
    efficiency = analyzer.cost_efficiency_analysis()
    most_efficient = efficiency.iloc[0]
    print(f"Most cost-efficient: {most_efficient['config_name']} ({most_efficient['success_per_dollar']:.1f} success/$)")
    
    # Feature impact summary
    feature_impacts = analyzer.feature_impact_analysis()
    most_impactful = max(feature_impacts.items(), key=lambda x: abs(x[1]))
    print(f"Most impactful feature: {most_impactful[0]} ({most_impactful[1]:+.3f} impact)")
    
    # Statistical significance tests
    print(f"\n🔍 STATISTICAL SIGNIFICANCE:")
    print("-" * 40)
    
    configs = list(avg_success.index)
    if len(configs) >= 2:
        # Test baseline vs others
        baseline_config = "baseline" if "baseline" in configs else configs[0]
        for config in configs[:3]:  # Test first 3 configs
            if config != baseline_config:
                sig_test = statistical_significance_test(results_df, baseline_config, config, "task_success")
                significance = "✅ Significant" if sig_test["significant"] else "❌ Not significant"
                print(f"{baseline_config} vs {config}: {significance} (p={sig_test['p_value']:.4f})")
    
    print(f"\n📋 Full analysis report: {report_path}")
    print("🎉 Ablation study complete!")

def create_custom_config_example():
    """Example of creating custom experiment configurations"""
    
    # Create a custom configuration for testing
    custom_config = ExperimentConfig(
        # Disable expensive features
        use_faiss_search=False,
        use_ontology_substitution=False,
        
        # Enable only LLM-based features
        use_llm_validation=True,
        use_structured_output=True,
        
        # Use faster models
        chef_model="gpt-3.5-turbo",
        trainee_model="gpt-3.5-turbo",
        
        # Aggressive settings for speed
        trainee_experience_level="advanced",
        max_clarification_turns=1,
        
        # Metadata
        experiment_id="fast_config",
        description="Optimized for speed over accuracy"
    )
    
    # Save custom config
    custom_config.save_to_file("custom_experiment_config.json")
    print("Custom configuration saved to custom_experiment_config.json")
    
    return custom_config

if __name__ == "__main__":
    # Run the main ablation study
    main()
    
    # Optionally create and test custom configurations
    print(f"\n{'='*60}")
    print("CREATING CUSTOM CONFIGURATION EXAMPLE")
    print("="*60)
    create_custom_config_example()