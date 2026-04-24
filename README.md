# AI Travel Planner

A multi-agent travel planning app built with LangChain, LangGraph, and a lightweight local web frontend.

The planner takes a destination, trip length, travel style, and interests, then coordinates four agents to produce a complete travel plan:

- Research Agent: destination context, attractions, climate, visa notes, safety, and transport
- Itinerary Agent: day-by-day schedule
- Budget Agent: estimated costs by category
- Report Agent: polished final travel plan

## Project Structure

```text
.
├── multi_agent_system.py   # LangGraph multi-agent workflow and CLI entry point
├── web_app.py              # Local web server and /api/plan endpoint
├── frontend/
│   ├── index.html          # Frontend layout
│   ├── styles.css          # Visual design
│   └── app.js              # Form handling, progress UI, and report rendering
└── .env                    # Local environment variables
```

## Requirements

- Python 3.10+
- An OpenAI API key
- Python packages:
  - `langchain`
  - `langchain-openai`
  - `langgraph`

If you already have the included `.venv`, use that environment.

## Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY="your-api-key-here"
```

If you need to install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install langchain langchain-openai langgraph
```

## Run The Web App

Start the local server:

```powershell
.\.venv\Scripts\python.exe web_app.py
```

Open the frontend:

```text
http://localhost:8000
```

Fill in the trip details, generate the plan, and copy the final report from the browser.

## Run The CLI

You can also run the original interactive terminal version:

```powershell
.\.venv\Scripts\python.exe multi_agent_system.py
```

## Notes

- The web server uses Python's standard library, so no additional web framework is required.
- Trip generation may take a little while because the agents run sequentially.
- Generated plans are AI-assisted and should be checked against current official travel, visa, health, and safety guidance before booking.
