# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains two distinct projects:

1. **LLM Conversation Tool** - A Python CLI application for multi-agent conversations using Ollama API (`llm-conversation` package)
2. **Cooking Conversation System** - An experimental system for chef-trainee conversations with recipe selection and ingredient substitution

## Development Commands

### Setup and Dependencies

This project uses Poetry for dependency management. Install dependencies with:
```bash
poetry install
```

To include development dependencies:
```bash
poetry install --with dev
```

### Running the Main LLM Conversation Tool

```bash
# Run directly
poetry run llm-conversation

# With configuration file
poetry run llm-conversation -c config.json

# Save conversation output
poetry run llm-conversation -o conversation.txt
```

### Code Quality Tools

```bash
# Linting (configured in pyproject.toml)
poetry run ruff check .
poetry run ruff format .

# Type checking
poetry run mypy src/ scripts/
```

### Testing

No test framework is currently set up. To add tests, you would need to configure pytest or similar.

## Architecture

### LLM Conversation Tool (`src/llm_conversation/`)

The main package has a modular architecture:

- **`__init__.py`** - Main entry point, CLI argument parsing, and conversation orchestration
- **`ai_agent.py`** - AIAgent class wrapping Ollama API interactions
- **`config.py`** - Configuration management using Pydantic models for validation
- **`conversation_manager.py`** - ConversationManager handles turn order, message routing, and conversation flow

Key architectural patterns:
- Uses Pydantic for configuration validation and JSON schema generation
- Supports multiple turn order strategies (round_robin, random, chain, moderator, vote)
- Streaming responses with Rich library for terminal UI
- Configurable agent parameters (model, temperature, context size, system prompt)

### Cooking Conversation System

This appears to be an experimental system with:

- **Recipe Management** - Uses pandas DataFrames for recipe storage and FAISS for semantic search
- **Configurable Agents** - `configurable_agents.py` defines chef/trainee agents with experiment configurations
- **Experiment Framework** - `experiment_config.py` and `experiment_runner.py` for ablation studies
- **Multiple Substitution Methods** - Supports FAISS embeddings, ontology-based, and LLM-based ingredient substitution

### External Dependencies

The cooking system includes a submodule `Exploiting-Food-Embeddings-for-Ingredient-Substitution/` which provides:
- Food2Vec embeddings
- FoodBERT models
- Ingredient normalization utilities
- Relation extraction for substitutions

## Key Configuration Files

- **`pyproject.toml`** - Poetry configuration, dependencies, and tool settings (ruff, mypy)
- **`schema.json`** - JSON schema for conversation configuration validation
- **Agent config example**:
  ```json
  {
    "agents": [
      {
        "name": "Agent Name",
        "model": "llama3.1:8b",
        "system_prompt": "You are...",
        "temperature": 0.8,
        "ctx_size": 2048
      }
    ],
    "settings": {
      "allow_termination": false,
      "use_markdown": true,
      "turn_order": "round_robin"
    }
  }
  ```

## Development Notes

- The project requires Python 3.13+ as specified in pyproject.toml
- Ollama must be installed and running for the LLM conversation tool
- Poetry is used for dependency management and virtual environment handling
- Rich library is used for terminal formatting and markdown rendering
- Type hints are used throughout with strict mypy configuration