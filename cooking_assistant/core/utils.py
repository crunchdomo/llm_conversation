"""Utility functions for cooking assistant."""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage

from .models import State


def save_conversation_to_json(job_id: str, state: State, chef_agent, trainee_agent):
    """Save conversation and metrics to JSON file."""
    conversation_log = []
    messages = state.get("messages", [])
    
    for i, message in enumerate(messages):
        entry = {
            "role": "trainee" if isinstance(message, HumanMessage) else "chef",
            "content": message.content,
            "timestamp": datetime.now().isoformat()
        }
        conversation_log.append(entry)

    # Calculate totals
    chef_total_tokens = sum(entry["total_tokens"] for entry in chef_agent.token_cost_log)
    chef_total_cost = sum(entry["cost"] for entry in chef_agent.token_cost_log)
    trainee_total_tokens = sum(entry["total_tokens"] for entry in trainee_agent.token_cost_log)
    trainee_total_cost = sum(entry["cost"] for entry in trainee_agent.token_cost_log)
    
    summary = {
        "chef_total_tokens": chef_total_tokens,
        "chef_total_cost": chef_total_cost,
        "trainee_total_tokens": trainee_total_tokens,
        "trainee_total_cost": trainee_total_cost,
        "overall_total_tokens": chef_total_tokens + trainee_total_tokens,
        "overall_total_cost": chef_total_cost + trainee_total_cost
    }

    # Save to file
    filename = f"cooking_session_{job_id}_full_log.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "job_id": job_id,
            "conversation": conversation_log,
            "metrics": summary,
            "chef_turns": chef_agent.token_cost_log,
            "trainee_turns": trainee_agent.token_cost_log
        }, f, indent=2, ensure_ascii=False)
    
    return filename


def append_with_accrual(conversation: List, message, chef_agent, trainee_agent, combined_accrual: List):
    """Append message with token cost tracking."""
    conversation.append(message)
    
    chef_cum = chef_agent.token_cost_log[-1] if chef_agent.token_cost_log else {
        "cumulative_tokens": 0, "cumulative_cost": 0
    }
    trainee_cum = trainee_agent.token_cost_log[-1] if trainee_agent.token_cost_log else {
        "cumulative_tokens": 0, "cumulative_cost": 0
    }
    
    combined_accrual.append({
        "chef_cumulative_tokens": chef_cum["cumulative_tokens"],
        "chef_cumulative_cost": chef_cum["cumulative_cost"],
        "trainee_cumulative_tokens": trainee_cum["cumulative_tokens"],
        "trainee_cumulative_cost": trainee_cum["cumulative_cost"],
        "overall_cumulative_tokens": chef_cum["cumulative_tokens"] + trainee_cum["cumulative_tokens"],
        "overall_cumulative_cost": chef_cum["cumulative_cost"] + trainee_cum["cumulative_cost"]
    })


def parse_llm_recipe(recipe_text: str) -> Dict[str, Any]:
    """Parse LLM-generated recipe text into structured format."""
    ingredients = []
    instructions = []
    in_ingredients = False
    in_instructions = False
    
    for line in recipe_text.splitlines():
        line_lower = line.lower()
        
        if "ingredient" in line_lower:
            in_ingredients = True
            in_instructions = False
            continue
        if "instruction" in line_lower or "directions" in line_lower:
            in_ingredients = False
            in_instructions = True
            continue
            
        if in_ingredients and line.strip().startswith("-"):
            ingredients.append(line.strip().lstrip("-").strip())
        elif in_instructions and line.strip() and not line_lower.startswith("note"):
            instructions.append(line.strip())
    
    return {
        "Title": "Adapted Recipe",
        "Cleaned_Ingredients": str(ingredients),
        "Instructions": "\n".join(instructions)
    }


def generate_job_id() -> str:
    """Generate unique job ID for session tracking."""
    return str(uuid.uuid4())


def print_token_summary(chef_agent, trainee_agent):
    """Print token usage summary."""
    chef_total_tokens = sum(entry["total_tokens"] for entry in chef_agent.token_cost_log)
    chef_total_cost = sum(entry["cost"] for entry in chef_agent.token_cost_log)
    trainee_total_tokens = sum(entry["total_tokens"] for entry in trainee_agent.token_cost_log)
    trainee_total_cost = sum(entry["cost"] for entry in trainee_agent.token_cost_log)
    
    print("\n=== Token and Cost Summary ===")
    print(f"ChefAgent:    {chef_total_tokens} tokens, ${chef_total_cost:.6f}")
    print(f"TraineeAgent: {trainee_total_tokens} tokens, ${trainee_total_cost:.6f}")
    print(f"TOTAL:        {chef_total_tokens + trainee_total_tokens} tokens, ${chef_total_cost + trainee_total_cost:.6f}")


def visualize_graph(graph):
    """Visualize the LangGraph structure."""
    try:
        from IPython.display import Image, display
        display(Image(graph.get_graph().draw_mermaid_png()))
        print(graph.get_graph().draw_mermaid())
        return graph.get_graph().draw_mermaid()
    except ImportError:
        print("IPython not available - cannot display graph visualization")
        return None