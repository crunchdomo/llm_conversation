"""LangGraph workflow for cooking conversation management."""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from ..models import State
from ..search.recipe_search import RecipeSearcher


def supervisor_router(state: State) -> str:
    """Route conversation flow based on current state."""
    if state.get("max_retries", 0) >= 3:
        return END
    
    phase = state["phase"]
    last_agent = state.get("current_agent")
    
    if phase == "chef_turn" and last_agent == "chef":
        state["max_retries"] += 1
    elif phase == "trainee_turn" and last_agent == "trainee":
        state["max_retries"] += 1
    else:
        state["max_retries"] = 0

    return {
        "introduction": "chef",
        "recipe_selection": "chef", 
        "ingredient_check": "chef",
        "chef_turn": "chef",      # Chef turn should route to chef node
        "trainee_turn": "trainee", # Trainee turn should route to trainee node
        "done": END
    }.get(phase, END)


def supervisor_node(state: State):
    """Supervisor node for managing conversation phases."""
    phase = state["phase"]
    
    if state.get("retries", 0) >= 3:
        state["phase"] = "done"
        return state

    if phase == "introduction":
        state["phase"] = "recipe_selection"
    elif state.get("detected_intent") == "substitution":
        state["phase"] = "substitution"
    elif phase == "recipe_selection":
        if state.get("selected_recipe"):
            state["phase"] = "ingredient_check"
        else:
            state["phase"] = "recipe_selection"
    elif phase == "ingredient_check":
        # After ingredients are checked, start the cooking conversation
        state["phase"] = "chef_turn" 
        state["step_idx"] = 0  # Start with first cooking step
    elif phase == "chef_turn":
        state["phase"] = "trainee_turn" if state.get("selected_recipe") else "recipe_selection"
    elif phase == "trainee_turn":
        state["phase"] = "chef_turn" if state.get("selected_recipe") else "recipe_selection"

    return state


def steps_from_recipe(state: State) -> list[str]:
    """Extract cooking steps from recipe."""
    if not state.get("selected_recipe"):
        return []
    
    instructions = state["selected_recipe"].get("Instructions", "")
    if not instructions:
        return []
    
    # Simple step parsing instead of using RecipeSearcher
    steps = []
    # Split by sentences and filter for actual steps
    sentences = instructions.replace('. ', '.\n').split('\n')
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20 and ('.' in sentence or len(sentences) == 1):
            steps.append(sentence)
    
    return steps[:8]  # Limit to first 8 steps for manageable conversation


def chef_node(state: State):
    """Chef agent node for providing cooking guidance."""
    phase = state["phase"]
    
    try:
        if phase == "introduction":
            response = state["chef_agent"].respond(
                state["messages"],
                "Introduce yourself as ChefAI, your friendly cooking assistant. Ask the user what they'd like to do: find a specific recipe, get similar recipes, or find recipes by ingredients."
            )
            state["messages"].append(AIMessage(content=response))
            state["phase"] = "recipe_selection"
            state["current_agent"] = "chef"
            return state
            
        if phase == "substitution":
            intent = state.get("last_intent")
            recipe = state.get("selected_recipe")
            
            # Handle substitution request
            faiss_subs = [sub for sub, _ in state["chef_agent"].faiss_index.find_ingredient_substitutes(
                intent.substitute_for, k=10
            )]
            
            validated_subs = state["chef_agent"].ingredient_substituter.validate_with_llm(
                state["chef_agent"], intent.substitute_for, faiss_subs, recipe
            )
            
            state["validated_substitutes"] = validated_subs
            msg = (
                f"ChefAI: For {intent.substitute_for} in {recipe['Title']}, the best substitutes are:\n"
                + "\n".join(f"- {sub}" for sub in validated_subs)
                + "\nWhich would you like to use?"
            )
            state["messages"].append(AIMessage(content=msg))
            state["phase"] = "await_substitution_choice"
            state["current_agent"] = "chef"
            return state

        if phase == "recipe_selection":
            try:
                response = state["chef_agent"].select_recipe(state["user_query"], state)
                state["messages"].append(AIMessage(content=response))
                
                if state.get("selected_recipe"):
                    state["phase"] = "ingredient_check"
                elif "Matching recipes:" in response or "Similar recipes:" in response:
                    state["phase"] = "user_select_recipe"
                else:
                    state["phase"] = "recipe_selection"
                    
                if "Error" in response:
                    state["retries"] = state.get("retries", 0) + 1
                    
                state["current_agent"] = "chef"
                return state
                
            except Exception as e:
                state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
                state["phase"] = "recipe_selection"
                state["retries"] = state.get("retries", 0) + 1
                state["current_agent"] = "chef"
                return state

        if phase == "ingredient_check":
            recipe = state.get("selected_recipe")
            if recipe:
                ingredients = recipe.get("Cleaned_Ingredients", [])
                if isinstance(ingredients, str):
                    # Parse if it's a string representation of a list
                    import ast
                    try:
                        ingredients = ast.literal_eval(ingredients)
                    except:
                        ingredients = [ingredients]
                
                ingredients_text = "\n".join(f"- {ing}" for ing in ingredients[:10])  # Show first 10
                
                try:
                    prompt = f"Great! I found a perfect recipe: {recipe['Title']}.\n\n" \
                            f"Here are the ingredients you'll need:\n{ingredients_text}\n\n" \
                            f"Do you have all these ingredients? If not, I can suggest substitutions!"
                    print(f"🔥 Chef responding to ingredient_check with prompt: {prompt[:100]}...")
                    response = state["chef_agent"].respond(state["messages"], prompt)
                    print(f"🔥 Chef response: {response[:100]}...")
                except Exception as e:
                    print(f"🔥 Chef respond error: {e}")
                    response = f"Error in ingredient check: {e}"
                state["messages"].append(AIMessage(content=response))
                state["current_agent"] = "chef"
                return state

        # General validation
        if phase in ["chef_turn", "ingredient_check"] and not state.get("selected_recipe"):
            response = "No recipe selected yet. Please select a recipe to continue."
            state["messages"].append(AIMessage(content=response))
            state["phase"] = "recipe_selection"
            state["current_agent"] = "chef"
            return state

        # Step-by-step cooking guidance
        if phase == "chef_turn":
            steps = steps_from_recipe(state)
            current_step = state.get("step_idx", 0)
            
            if current_step < len(steps):
                step_text = steps[current_step]
                try:
                    prompt = f"Step {current_step + 1}: {step_text}\nExplain this step clearly and wait for the user to respond."
                    print(f"🔥 Chef cooking step {current_step + 1}: {step_text[:50]}...")
                    response = state["chef_agent"].respond(state["messages"], prompt)
                    print(f"🔥 Chef cooking response: {response[:100]}...")
                except Exception as e:
                    print(f"🔥 Chef cooking error: {e}")
                    response = f"Error in cooking step: {e}"
                state["messages"].append(AIMessage(content=response))
                state["step_idx"] = current_step + 1  # Move to next step
            else:
                response = state["chef_agent"].respond(
                    state["messages"],
                    "Congratulations! You've completed the recipe. How did it turn out?"
                )
                state["messages"].append(AIMessage(content=response))
                state["phase"] = "done"
            
            state["current_agent"] = "chef"
            return state

    except Exception as e:
        state["messages"].append(AIMessage(content=f"Chef encountered an error: {e}"))
        state["phase"] = "recipe_selection"
        state["current_agent"] = "chef"
        return state

    return state


def trainee_node(state: State):
    """Trainee agent node for simulating student responses."""
    phase = state["phase"]
    
    if phase == "trainee_turn" and not state.get("selected_recipe"):
        state["messages"].append(HumanMessage(content="No recipe selected."))
        state["phase"] = "recipe_selection"
        state["current_agent"] = "trainee"
        return state

    if phase in ("trainee_choose_recipe", "user_select_recipe"):
        # Extract recipes from chef's message
        searcher = RecipeSearcher(state["chef_agent"].recipes_df)
        recipe_list = searcher.extract_recipes_from_message(state["messages"][-1].content)
        
        chosen_recipe_name = state["trainee_agent"].choose_recipe(recipe_list, state)
        recipe = searcher.get_recipe_by_name(chosen_recipe_name)
        
        if not recipe and recipe_list:
            recipe = searcher.get_recipe_by_name(recipe_list[0])
            
        if recipe:
            state["selected_recipe"] = recipe
            allergies = state["user_profile"].get("allergies", [])
            if searcher.contains_allergen(recipe.get('Cleaned_Ingredients', ''), allergies):
                state["phase"] = "allergy_warning"
            else:
                state["phase"] = "ingredient_check"
        else:
            state["phase"] = "recipe_selection"
            
        state["current_agent"] = "trainee"
        return state

    elif phase == "trainee_confirm_ingredients":
        response = state["trainee_agent"].confirm_ingredients(state)
        state["messages"].append(HumanMessage(content=response))
        
        if "yes" in response.lower():
            state["phase"] = "chef_turn"
        else:
            state["phase"] = "recipe_selection"
            
        state["current_agent"] = "trainee"
        return state

    elif phase == "trainee_turn":
        # Get the last chef message
        chef_msgs = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        last_chef_msg = chef_msgs[-1].content if chef_msgs else ""
        
        steps = steps_from_recipe(state)
        try:
            print(f"👤 Trainee responding to: {last_chef_msg[:50]}...")
            response = state["trainee_agent"].generate_response(last_chef_msg, len(steps))
            print(f"👤 Trainee response: {response[:100]}...")
        except Exception as e:
            print(f"👤 Trainee error: {e}")
            response = f"I'm having trouble understanding. Could you help me with that step?"
        state["messages"].append(HumanMessage(content=response))
        
        # Progress tracking
        if "next" in response.lower():
            state["step_idx"] = state.get("step_idx", 0) + 1
            state["same_step_turns"] = 0
        else:
            state["same_step_turns"] = state.get("same_step_turns", 0) + 1
            # Force progression after too many questions
            if state["same_step_turns"] > 2:
                state["step_idx"] = state.get("step_idx", 0) + 1
                state["same_step_turns"] = 0
                
        state["phase"] = "chef_turn"
        state["current_agent"] = "trainee"
        return state

    return state


def build_cooking_graph():
    """Build and compile the cooking conversation graph."""
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("chef", chef_node)
    graph_builder.add_node("trainee", trainee_node)
    
    # Start with supervisor
    graph_builder.add_edge(START, "supervisor")
    
    # Conditional routing
    graph_builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "chef": "chef",
            "trainee": "trainee",
            END: END
        }
    )
    
    # Agents return to supervisor
    graph_builder.add_edge("chef", "supervisor")
    graph_builder.add_edge("trainee", "supervisor")
    
    return graph_builder.compile()