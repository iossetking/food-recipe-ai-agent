# Food Recipe AI Agent

A LangGraph-powered AI agent that recommends food recipes by combining real-time web search (Tavily) with a recipe database (Spoonacular), then synthesizing the results through an LLM (NVIDIA NIM).

## What it does

Given a natural language request like _"Give me a recipe with eggs, bread, and mayonnaise"_, the agent:

1. Searches the web for relevant dishes and context
2. Queries the Spoonacular database for matching recipes by ingredients
3. Feeds both results into an LLM to generate a structured recipe recommendation

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Clone the repo

```bash
git clone https://github.com/your-username/food-recipe-ai-agent.git
cd food-recipe-ai-agent
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=nvapi-...
TAVILY_API_KEY=tvly-...
SPOONACULAR_API_KEY=...
```

| Variable | Where to get it |
|----------|----------------|
| `NVIDIA_API_KEY` | [build.nvidia.com](https://build.nvidia.com) |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) |
| `SPOONACULAR_API_KEY` | [spoonacular.com/food-api](https://spoonacular.com/food-api) |

### 4. Run

```bash
uv run python main.py
```

## Structure

```
food-recipe-ai-agent/
├── main.py          # Agent definition and entry point
├── pyproject.toml   # Dependencies
├── .env             # API keys (not committed)
└── README.md
```

### Agent graph

```
User Input (HumanMessage)
        │
        ▼
┌───────────────┐
│  search_node  │  → Tavily web search
│               │  → Spoonacular recipe DB lookup
│               │  → stores results in state
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  recipe_node  │  → builds prompt from search results
│               │  → calls LLM (Llama 3.1 8B via NVIDIA NIM)
│               │  → appends recommendation to message history
└───────┬───────┘
        │
        ▼
      END
```

### Agent state

| Field | Type | Description |
|-------|------|-------------|
| `message` | `Sequence[BaseMessage]` | Full conversation history (auto-appended) |
| `search_results` | `dict` | Raw Tavily + Spoonacular API responses |
| `memory` | `dict` | Reserved for future user preferences/history |
| `iteration_count` | `int` | Loop counter for controlling agent cycles |

## Tech stack

| Tool | Purpose |
|------|---------|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent graph orchestration |
| [LangChain NVIDIA AI Endpoints](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/) | LLM inference via NVIDIA NIM |
| [LangChain Tavily](https://python.langchain.com/docs/integrations/tools/tavily_search/) | Real-time web search |
| [Spoonacular API](https://spoonacular.com/food-api) | Recipe database search by ingredients |
