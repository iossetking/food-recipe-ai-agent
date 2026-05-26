import os
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
import requests
import json

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_tavily import TavilySearch
from langchain.tools import tool

load_dotenv()

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
SPOONACULAR_API_KEY = os.getenv('SPOONACULAR_API_KEY')

tavily_tool = TavilySearch(
  api_key=TAVILY_API_KEY,
  max_results=5,
  include_answer=True,
  include_raw_content=True
)

llm = ChatNVIDIA(
  model="gemini-2.0-flash",
  # api_key=,
  temperature=0.7,
  top_p=50,
  max_tokens=2048,
  reasoning_budget=1000,
  chat_template_kwargs={'enable_thinking': True},
)

class AgentState(TypedDict):
  message: Annotated[Sequence[BaseMessage], add_messages]
  search_results: dict
  memory: dict
  iteration_count: int

# @tool
def tavily_search(query):
  # response = tavily_tool.search(query)
  # return response
  pass

# @tool
def get_recipe_by_ingredients(ingredients):
  data = requests.get(f'https://api.spoonacular.com/recipes/findByIngredients?apiKey={SPOONACULAR_API_KEY}&{ingredients}').json()

  return data

# print(get_recipe_by_ingredients('apples,+flour,+sugar&number=2'))
# data = json.loads(tavily_search('Give me a dish with bread, jam, and cheese'))