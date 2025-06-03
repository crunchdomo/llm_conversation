# Cooking Assistant

A modular multi-agent cooking conversation system using LangGraph, FAISS indexing, and LLM-based agents.

## Overview

This system simulates cooking conversations between a Chef AI and Trainee AI with varying experience levels. It includes:

- **Recipe Search**: FAISS-based semantic search across 13k+ recipes
- **Ingredient Substitution**: LLM + ontology-based ingredient substitution
- **Multi-Agent Conversations**: Chef and Trainee agents with different experience levels
- **LangGraph Workflows**: Structured conversation flows with state management
- **Scenario Testing**: Predefined scenarios for testing different user profiles

## Architecture

```
cooking_assistant/
├── core/
│   ├── models.py              # Data models and type definitions
│   ├── utils.py               # Utility functions
│   ├── agents/
│   │   ├── chef_agent.py      # Chef AI implementation
│   │   └── trainee_agent.py   # Trainee AI implementation
│   ├── search/
│   │   ├── faiss_index.py     # FAISS indexing for recipes/ingredients
│   │   ├── recipe_search.py   # Recipe search utilities
│   │   └── ingredient_substitution.py  # Substitution logic
│   └── graph/
│       └── workflow.py        # LangGraph workflow definition
├── scenarios.py              # Test scenarios for different user types
├── main.py                   # Main orchestration script
└── __init__.py
```

## Key Features

### 1. Multi-Experience Level Support
- **Beginner**: Asks many questions, needs detailed explanations
- **Intermediate**: Moderate questions, assumes basic knowledge
- **Advanced**: Minimal questions, focuses on complex techniques only

### 2. FAISS-Based Search
- Semantic recipe search by ingredients
- Ingredient similarity for substitutions
- Fast indexing of 13k+ recipes and 75k+ ingredients

### 3. Intelligent Substitution
- LLM-generated substitution candidates
- FAISS validation for ingredient formats
- Hybrid ranking (LLM + semantic similarity)

### 4. LangGraph Workflow
- Structured conversation phases (introduction → recipe selection → cooking)
- State management with retries and error handling
- Supervisor-based routing between agents

## Usage

### Command Line Interface

```bash
# List available test scenarios (uses 100 recipes by default for performance)
python -m cooking_assistant.main --list-scenarios

# Run specific scenario with sample size for testing
python -m cooking_assistant.main --scenario beginner_pasta --sample-size 50

# Interactive mode
python -m cooking_assistant.main --interactive --sample-size 50

# Custom recipes CSV
python -m cooking_assistant.main --recipes-csv path/to/recipes.csv --sample-size 100

# For full dataset (may be slow on first run)
python -m cooking_assistant.main --scenario beginner_pasta --sample-size 1000
```

### Programmatic Usage

```python
from cooking_assistant import CookingAssistant, UserProfile

# Initialize
assistant = CookingAssistant("13k-recipes.csv")

# Define user profile
profile = UserProfile(
    experience_level="beginner",
    allergies=["nuts"],
    preferred_cuisine="Italian"
)

# Run conversation
job_id = assistant.run_conversation(
    user_query="I want to make pasta", 
    user_profile=profile
)
```

### Scenario Testing

```python
from cooking_assistant import get_scenario_by_name

# Get predefined scenario
scenario = get_scenario_by_name("beginner_pasta")

# Run scenario
assistant = CookingAssistant()
job_id = assistant.run_scenario("beginner_pasta", visualize=True)
```

## Available Scenarios

### Beginner Scenarios
- `beginner_pasta`: Simple pasta making for complete beginners
- `beginner_with_allergies`: Baking with nut allergies

### Intermediate Scenarios  
- `intermediate_substitution`: Ingredient substitution in stir-fry
- `intermediate_technique_focus`: Learning risotto technique

### Advanced Scenarios
- `advanced_complex_dish`: Beef wellington for experienced cooks
- `advanced_dietary_restrictions`: Complex dietary modifications

### Substitution Scenarios
- `protein_substitution`: Replacing main protein with allergies
- `ingredient_unavailable`: Finding alternatives for missing ingredients

## Configuration

The system requires:

1. **Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   ```

2. **Recipe Dataset**: Place `13k-recipes.csv` in the project root

3. **Optional Dependencies**:
   - LangSmith for tracing (set `LANGCHAIN_API_KEY`)
   - Food ontology file `foodon-base.owl` for enhanced substitutions

## Output

Each conversation generates:
- **Conversation Log**: Complete message history with timestamps
- **Token Metrics**: Token usage and costs per agent
- **Session Data**: Saved as `cooking_session_{job_id}_full_log.json`

## Extension Points

### Adding New Scenarios
```python
from cooking_assistant.scenarios import CookingScenario
from cooking_assistant.core.models import UserProfile

new_scenario = CookingScenario(
    name="custom_scenario",
    description="Your scenario description",
    user_profile=UserProfile(experience_level="intermediate"),
    user_query="What should I cook?",
    expected_outcomes=["Expected behavior 1", "Expected behavior 2"]
)
```

### Custom Agents
Extend `ChefAgent` or `TraineeAgent` classes to modify behavior:

```python
from cooking_assistant.core.agents import ChefAgent

class CustomChefAgent(ChefAgent):
    def respond(self, conversation, prompt):
        # Custom logic here
        return super().respond(conversation, prompt)
```

### New Search Methods
Implement additional search backends by extending the search utilities:

```python
from cooking_assistant.core.search import RecipeSearcher

class CustomSearcher(RecipeSearcher):
    def search_by_custom_criteria(self, criteria):
        # Implementation here
        pass
```

## Dependencies

- `langchain-openai`: LLM interactions
- `langgraph`: Workflow management  
- `sentence-transformers`: Embedding generation
- `faiss-cpu`: Vector similarity search
- `pandas`: Data manipulation
- `thefuzz`: Fuzzy string matching
- `langsmith`: Optional tracing