# Cooking Conversation Testing Framework

A comprehensive testing system for evaluating different conversation scenarios and LLM providers in your cooking assistant.

## Overview

This testing framework allows you to:

- **Test Multiple Conversation Types**: Specific recipes, ingredient-based queries, substitutions, dietary restrictions, allergen-safe cooking
- **Compare LLM Providers**: OpenAI (GPT-4, GPT-4o-mini), Anthropic (Claude), XAI (Grok)
- **Run Batch Experiments**: Execute hundreds of tests in parallel with progress tracking
- **Analyze Results**: Comprehensive analysis with visualizations and metrics

## Quick Start

### 1. Setup Environment Variables

```bash
# Set API keys for the LLM providers you want to test
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export XAI_API_KEY="your_xai_key"  # Optional
```

### 2. List Available Options

```bash
# See all available test scenarios
python cooking_assistant/test_cli.py --list-scenarios

# See which LLM providers are available (have API keys)
python cooking_assistant/test_cli.py --list-providers
```

### 3. Run Simple Tests

```bash
# Test specific recipe scenarios with available providers
python cooking_assistant/test_cli.py --conversation-types specific_recipe

# Test with specific LLM providers
python cooking_assistant/test_cli.py --llm-providers openai_gpt4_mini anthropic_claude_35_sonnet

# Run specific scenarios
python cooking_assistant/test_cli.py --scenarios specific_recipe_chicken_teriyaki ingredient_based_chicken_onions
```

### 4. Run Batch Experiments

```bash
# Run all available tests with 3 parallel workers
python cooking_assistant/test_cli.py --max-workers 3 --batch-name "full_experiment"

# Test substitution scenarios only
python cooking_assistant/test_cli.py --conversation-types substitution dietary_restriction --batch-name "substitution_test"

# Quick test with timeout
python cooking_assistant/test_cli.py --conversation-types specific_recipe --timeout 120 --batch-name "quick_test"
```

## Available Test Scenarios

### 1. Specific Recipe Scenarios
```python
# User asks for a particular recipe by name
"I want to make chicken teriyaki"
"Show me how to make pasta carbonara"
"I want to make beef stir fry"
```

### 2. Ingredient-Based Scenarios  
```python
# User has ingredients and wants recipe suggestions
"I have chicken and onions, what can I make?"
"I have carrots, bell peppers, and mushrooms"
"I have rice, eggs, and soy sauce"
```

### 3. General Recipe Scenarios
```python
# User asks how to make a general type of dish
"How do I make spaghetti bolognese?"
"Teach me to make chocolate chip cookies"
"How do you make chicken curry?"
```

### 4. Substitution Scenarios
```python
# User wants to substitute ingredients
"Can I use turkey instead of chicken in this stir fry?"
"I don't have heavy cream, can I use coconut milk?"
"I need to replace both butter and eggs in this cookie recipe"
```

### 5. Dietary Restriction Scenarios
```python
# User has specific dietary needs
"I want to make a vegan pasta dish"
"I need a keto-friendly dinner recipe"
"How can I make gluten-free bread?"
```

### 6. Allergen-Safe Scenarios
```python
# User has allergies to consider
"I need a nut-free dessert for my kid's school party"
"I'm allergic to dairy and eggs, can you help me make a birthday cake?"
"I love seafood but I'm allergic to shellfish, what can I cook?"
```

## LLM Providers Supported

| Provider | Model | API Key Env Var |
|----------|-------|-----------------|
| **OpenAI GPT-4** | `gpt-4` | `OPENAI_API_KEY` |
| **OpenAI GPT-4o-mini** | `gpt-4o-mini` | `OPENAI_API_KEY` |
| **OpenAI GPT-3.5** | `gpt-3.5-turbo` | `OPENAI_API_KEY` |
| **Claude 3.5 Sonnet** | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| **Claude 3.5 Haiku** | `claude-3-5-haiku-20241022` | `ANTHROPIC_API_KEY` |
| **Claude 3 Opus** | `claude-3-opus-20240229` | `ANTHROPIC_API_KEY` |
| **XAI Grok** | `grok-beta` | `XAI_API_KEY` |

## Advanced Usage

### Custom Test Configuration

```python
from cooking_assistant.testing import TestScenarioBuilder, LLMProvider, ConversationTestRunner

# Create custom scenarios
scenarios = TestScenarioBuilder.create_specific_recipe_scenarios()

# Select providers
providers = [LLMProvider.OPENAI_GPT4_MINI, LLMProvider.ANTHROPIC_CLAUDE_35_SONNET]

# Run tests
runner = ConversationTestRunner(sample_size=50)  # Use smaller recipe sample
results = runner.run_batch_tests(scenarios, providers, max_workers=2)

# Save results
runner.save_results(results, "custom_experiment")
```

### Batch Processing with Recovery

```python
from cooking_assistant.testing.batch_processor import run_batch_experiments

# Run with automatic retry and progress tracking
results, output_file = run_batch_experiments(
    scenarios=scenarios,
    llm_providers=providers,
    batch_name="experiment_with_recovery",
    max_workers=3,
    retry_failed=True,
    save_intermediate=True  # Save progress every 10 tests
)
```

### Result Analysis

```bash
# Analyze results from a test batch
python cooking_assistant/testing/result_analyzer.py test_results/experiment_1.json --output-dir analysis --create-plots --export-csv
```

```python
from cooking_assistant.testing.result_analyzer import ResultAnalyzer

# Load and analyze results
analyzer = ResultAnalyzer(results_file="test_results/experiment_1.json")

# Generate comprehensive report
summary = analyzer.generate_summary_report()

# Create visualizations
analyzer.create_visualizations("plots/")

# Compare LLM providers
comparison = analyzer.compare_providers()
print(comparison)

# Find best performers
best = analyzer.find_best_performers(metric='success_rate', top_n=10)
print(best)
```

## Output Structure

### Test Results File Structure
```json
{
  "batch_name": "experiment_1",
  "timestamp": "2024-01-15T10:30:00",
  "total_tests": 45,
  "successful_tests": 38,
  "failed_tests": 7,
  "results": [
    {
      "test_id": "uuid-1234",
      "scenario_name": "specific_recipe_chicken_teriyaki",
      "conversation_type": "specific_recipe",
      "llm_provider": "openai_gpt4_mini",
      "user_query": "I want to make chicken teriyaki",
      "success": true,
      "duration_seconds": 45.2,
      "conversation_log": [...],
      "metrics": {
        "total_turns": 8,
        "chef_turns": 4,
        "trainee_turns": 4
      },
      "outcomes_achieved": ["recipe_found", "ingredients_listed"],
      "expected_outcomes": ["recipe_found", "ingredients_listed", "step_by_step_guidance"]
    }
  ]
}
```

### Analysis Report Structure
```json
{
  "overview": {
    "total_tests": 45,
    "successful_tests": 38,
    "overall_success_rate": 0.844,
    "average_duration": 52.3,
    "average_turns": 7.2
  },
  "by_conversation_type": {
    "specific_recipe": {"success_rate": 0.92, "avg_duration": 48.1},
    "substitution": {"success_rate": 0.78, "avg_duration": 61.5}
  },
  "by_llm_provider": {
    "openai_gpt4_mini": {"success_rate": 0.89, "test_count": 22},
    "anthropic_claude_35_sonnet": {"success_rate": 0.83, "test_count": 23}
  }
}
```

## Common CLI Commands

```bash
# Quick test of all scenario types
python cooking_assistant/test_cli.py --conversation-types specific_recipe ingredient_based general_recipe --max-workers 2

# Test substitution capabilities across all providers
python cooking_assistant/test_cli.py --conversation-types substitution dietary_restriction allergen_safe --batch-name "substitution_study"

# Performance comparison between GPT-4 and Claude
python cooking_assistant/test_cli.py --llm-providers openai_gpt4_mini anthropic_claude_35_sonnet --batch-name "gpt_vs_claude"

# Test with beginner vs advanced user profiles
python cooking_assistant/test_cli.py --scenarios specific_recipe_chicken_teriyaki specific_recipe_beef_stir_fry --batch-name "experience_levels"

# Quick validation test (dry run)
python cooking_assistant/test_cli.py --conversation-types specific_recipe --dry-run

# Test with custom timeout and smaller sample
python cooking_assistant/test_cli.py --timeout 120 --sample-size 50 --max-workers 1 --batch-name "quick_validation"
```

## Metrics and Analysis

The framework tracks:

- **Success Rate**: Percentage of tests that completed without errors
- **Duration**: Time taken for each conversation
- **Turn Count**: Number of back-and-forth exchanges
- **Outcome Completion**: How many expected outcomes were achieved
- **Error Patterns**: Categorization of common failure types

### Key Performance Indicators

1. **Overall Success Rate**: Target >90% for production readiness
2. **Average Duration**: Target <60 seconds per conversation
3. **Outcome Completion Rate**: Target >80% for quality conversations
4. **Provider Consistency**: <10% variance between providers
5. **Error Rate by Type**: Track API, timeout, and logic errors

## Troubleshooting

### Common Issues

1. **"No API key found"**
   ```bash
   export OPENAI_API_KEY="your_key_here"
   # Check with: python cooking_assistant/test_cli.py --list-providers
   ```

2. **"No scenarios selected"**
   ```bash
   # Use --list-scenarios to see available options
   python cooking_assistant/test_cli.py --list-scenarios
   ```

3. **High failure rate**
   - Check API key validity
   - Reduce --max-workers to avoid rate limits
   - Increase --timeout for slow responses

4. **Memory issues with large batches**
   - Reduce --sample-size (recipe count)
   - Use --max-workers 1 for sequential processing
   - Enable --save-intermediate for recovery

### Performance Tuning

- **For speed**: Use `--max-workers 5` with fast models like GPT-4o-mini
- **For stability**: Use `--max-workers 1` with rate-limited APIs
- **For development**: Use `--sample-size 10` for quick iterations
- **For production**: Use `--timeout 300` and `--retry-failed`

## Integration with Your System

The testing framework is designed to work with your existing cooking assistant:

1. **Scenario Integration**: Add your own scenarios to `test_scenarios.py`
2. **LLM Integration**: The testable agents use your existing chef/trainee logic
3. **Recipe Integration**: Uses your 13k-recipes.csv dataset
4. **Metrics Integration**: Tracks tokens, costs, and conversation quality

This gives you a comprehensive way to validate improvements, compare approaches, and ensure consistent performance across different LLM providers and conversation types.