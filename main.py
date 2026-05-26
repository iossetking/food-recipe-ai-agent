# %%
# Libraries import
import os
from typing import TypedDict, Annotated, Sequence
import requests

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_tavily import TavilySearch

# %%
INITIAL_MESSAGE = "Give me a recipe with pasta, tomatoes, and shrimp."

# %%
# Environment variables loading
load_dotenv()

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
SPOONACULAR_API_KEY = os.getenv('SPOONACULAR_API_KEY')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

# %%
# Instantiating tools and LLM
tavily_tool = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=5,
    include_answer=True,
    include_raw_content=False,
)

llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=NVIDIA_API_KEY,
    temperature=0.7,
    top_p=0.9,
    max_completion_tokens=2048,
)

# %%
# Agent state definition
class AgentState(TypedDict):
    message: Annotated[Sequence[BaseMessage], add_messages]
    search_results: dict
    memory: dict
    iteration_count: int

# %%
# Tool functions
def tavily_search(query: str) -> dict:
    return tavily_tool.invoke(query)

def get_recipe_by_ingredients(ingredients: str, number: int = 5) -> list:
    params = {
        "apiKey": SPOONACULAR_API_KEY,
        "ingredients": ingredients,
        "number": number,
    }
    return requests.get(
        "https://api.spoonacular.com/recipes/findByIngredients",
        params=params,
    ).json()

# %%
# Agent nodes
def search_node(state: AgentState) -> dict:
    query = str(state["message"][-1].content)

    tavily_results = tavily_search(query)
    spoonacular_results = get_recipe_by_ingredients(query)

    return {
        "search_results": {
            "tavily": tavily_results,
            "spoonacular": spoonacular_results,
        },
        "iteration_count": state["iteration_count"] + 1,
    }

def recipe_node(state: AgentState) -> dict:
    results = state["search_results"]
    tavily = results["tavily"]
    spoonacular = results["spoonacular"]

    tavily_answer = tavily.get("answer") or (
        tavily["results"][0]["content"] if tavily.get("results") else "No web results found."
    )

    recipe_names = [r["title"] for r in spoonacular] if isinstance(spoonacular, list) else []

    prompt = f"""
    You are a food recipe assistant.


    User request: {state["message"][-1].content}

    Web search context:
    {tavily_answer}

    Matching recipes from database:
    {recipe_names if recipe_names else "No matching recipes found."}

    Based on the above, recommend the best recipe. Include:
    - Recipe name
    - Key ingredients
    - Brief preparation steps
    """
    response = llm.invoke(prompt)
    return {"message": [AIMessage(content=response.content)]}

# Continuation condition
def should_continue(state: AgentState) -> str:
    if state["iteration_count"] >= 3:
        return END
    return END

# %%
# Graph construction
graph = StateGraph(AgentState)
graph.add_node("search_node", search_node)
graph.add_node("recipe_node", recipe_node)

graph.set_entry_point("search_node")
graph.add_edge("search_node", "recipe_node")
graph.add_conditional_edges("recipe_node", should_continue)

app = graph.compile()

# %%
# Initial state and execution
initial_state: AgentState = {
    "message": [HumanMessage(content=INITIAL_MESSAGE)],
    "search_results": {},
    "memory": {},
    "iteration_count": 0,
}

result = app.invoke(initial_state)
print(result["message"][-1].content)
