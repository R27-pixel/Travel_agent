from __future__ import annotations

import os
import sys
from typing import TypedDict, Annotated
import operator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_local_env(path: str = ".env") -> None:
    """
    Load environment variables from a local .env file.

    The app needs an API key to call the LLM. This helper reads simple
    KEY=VALUE lines from `.env` and adds them to `os.environ` only when the
    variable has not already been set in the terminal. That lets local
    development use `.env` while still allowing deployed environments or shell
    variables to override it.

    Args:
        path: Path to the .env file. Defaults to the project root `.env`.

    Returns:
        None. Environment variables are updated in place.
    """
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Shared State
# TravelState is the data contract for the whole LangGraph workflow. Each
# agent receives the current state, reads the fields it needs, and returns only
# the fields it wants to update. LangGraph merges those updates into the state
# before passing it to the next agent.
# ---------------------------------------------------------------------------

class TravelState(TypedDict):
    """
    Shared context passed between all travel-planning agents.

    The first group of fields contains user input. The second group stores
    agent-generated outputs. The `messages` field keeps a running log of agent
    responses, which is useful for debugging or showing intermediate work.
    """
    # ── User inputs ──────────────────────────────────────────────────────────
    destination: str
    duration_days: int
    travel_style: str          # e.g. "budget", "mid-range", "luxury"
    interests: str             # e.g. "history, food, nature"

    # ── Agent outputs ────────────────────────────────────────────────────────
    research_notes: str        # ResearchAgent → destination facts
    itinerary: str             # ItineraryAgent → day-by-day plan
    budget_breakdown: str      # BudgetAgent → cost estimates
    final_report: str          # ReportAgent → full travel plan

    # ── Accumulated messages for transparency ────────────────────────────────
    messages: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# LLM (shared, single instance)
# ---------------------------------------------------------------------------

def get_llm() -> ChatGroq:
    """
    Create and return the shared Groq chat model client.

    This function centralizes model configuration so every agent uses the same
    model, temperature, and API-key lookup. It loads `.env`, checks that
    `groq_API_KEY` is available, and exits with a helpful message if the key is
    missing.

    Returns:
        A configured ChatGroq instance ready for `.invoke(...)` calls.
    """
    load_local_env()
    api_key = os.getenv("groq_API_KEY")
    if not api_key:
        print("\n[ERROR] groq_API_KEY environment variable is not set.")
        print("Export it with:  export groq_API_KEY='sk-...'")
        sys.exit(1)
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)


# ---------------------------------------------------------------------------
# Agent 1 – ResearchAgent
# ---------------------------------------------------------------------------

def research_agent(state: TravelState) -> dict:
    """
    Research the destination and return practical travel intelligence.

    This is the first agent in the workflow. It uses the user's destination,
    trip length, travel style, and interests to ask the LLM for a structured
    research brief. Later agents use this brief as context for itinerary and
    budget planning.

    Args:
        state: Current TravelState containing the user's trip inputs.

    Returns:
        A partial state update with `research_notes` and a logged message.
    """
    print("\n🔍  [ResearchAgent] Researching destination…")

    llm = get_llm()

    system_prompt = (
        "You are an expert travel researcher with encyclopedic knowledge of "
        "world destinations. Provide concise, practical, and accurate "
        "destination intelligence for trip planning purposes."
    )

    user_prompt = (
        f"Research the travel destination: {state['destination']}\n\n"
        f"Trip duration: {state['duration_days']} days\n"
        f"Traveller interests: {state['interests']}\n"
        f"Travel style: {state['travel_style']}\n\n"
        "Provide a structured research brief covering:\n"
        "1. Destination Overview (geography, culture, language, currency)\n"
        "2. Best Time to Visit & Climate\n"
        "3. Visa & Entry Requirements (general guidance)\n"
        "4. Top Attractions & Hidden Gems (match the interests listed)\n"
        "5. Local Cuisine Highlights\n"
        "6. Safety & Health Tips\n"
        "7. Getting Around (local transport options)\n"
        "Keep it clear and well-organised."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    research_notes = response.content
    print("   ✅  Research complete.")

    return {
        "research_notes": research_notes,
        "messages": [HumanMessage(content=f"[ResearchAgent]\n{research_notes}")],
    }


# ---------------------------------------------------------------------------
# Agent 2 – ItineraryAgent
# ---------------------------------------------------------------------------

def itinerary_agent(state: TravelState) -> dict:
    """
    Build a day-by-day itinerary from the research notes.

    This second agent receives the destination research generated by
    `research_agent` and turns it into a realistic schedule. It keeps the plan
    aligned with the user's interests and travel style, while asking the model
    to include practical timing and activity details.

    Args:
        state: Current TravelState with user inputs and `research_notes`.

    Returns:
        A partial state update with `itinerary` and a logged message.
    """
    print("\n🗺️   [ItineraryAgent] Building itinerary…")

    llm = get_llm()

    system_prompt = (
        "You are a seasoned travel planner who crafts inspiring, practical, "
        "and realistic day-by-day itineraries. You balance must-see highlights "
        "with off-the-beaten-path experiences and always account for travel "
        "time between locations."
    )

    user_prompt = (
        f"Create a {state['duration_days']}-day itinerary for "
        f"{state['destination']}.\n\n"
        f"Travel style: {state['travel_style']}\n"
        f"Interests: {state['interests']}\n\n"
        "Use the following research brief to inform the plan:\n"
        f"{state['research_notes']}\n\n"
        "Format:\n"
        "• Day X – [Theme/Title]\n"
        "  Morning  | Mid-day  | Afternoon  | Evening\n"
        "  (include specific places, activities, and practical tips)\n\n"
        "End with a section: 'Practical Tips' (3–5 bullet points)."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    itinerary = response.content
    print("   ✅  Itinerary ready.")

    return {
        "itinerary": itinerary,
        "messages": [HumanMessage(content=f"[ItineraryAgent]\n{itinerary}")],
    }


# ---------------------------------------------------------------------------
# Agent 3 – BudgetAgent
# ---------------------------------------------------------------------------

def budget_agent(state: TravelState) -> dict:
    """
    Estimate the trip budget by major travel expense category.

    This third agent uses the destination, duration, and travel style to create
    low/high cost ranges for flights, hotels, food, transport, activities, and
    miscellaneous expenses. The final report agent later combines this budget
    with the research and itinerary.

    Args:
        state: Current TravelState with user inputs and prior agent outputs.

    Returns:
        A partial state update with `budget_breakdown` and a logged message.
    """
    print("\n💰  [BudgetAgent] Estimating budget…")

    llm = get_llm()

    system_prompt = (
        "You are a meticulous travel budget analyst. You provide realistic "
        "cost estimates in USD for all major travel expense categories, "
        "clearly labelled by travel style (budget / mid-range / luxury). "
        "Always include a total estimate range and money-saving tips."
    )

    user_prompt = (
        f"Estimate the total travel budget for:\n"
        f"• Destination: {state['destination']}\n"
        f"• Duration: {state['duration_days']} days\n"
        f"• Travel style: {state['travel_style']}\n\n"
        "Break down costs into:\n"
        "1. Flights (round-trip, economy from a major hub)\n"
        "2. Accommodation (per night × nights)\n"
        "3. Food & Dining (per day)\n"
        "4. Local Transport (per day)\n"
        "5. Activities & Entrance Fees\n"
        "6. Miscellaneous (souvenirs, tips, SIM card, etc.)\n\n"
        "Provide a low-end and high-end range for each category, "
        "then a TOTAL estimated range.\n"
        "Add 3 money-saving tips specific to this destination."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    budget_breakdown = response.content
    print("   ✅  Budget estimated.")

    return {
        "budget_breakdown": budget_breakdown,
        "messages": [HumanMessage(content=f"[BudgetAgent]\n{budget_breakdown}")],
    }


# ---------------------------------------------------------------------------
# Agent 4 – ReportAgent
# ---------------------------------------------------------------------------

def report_agent(state: TravelState) -> dict:
    """
    Combine all agent outputs into the final travel plan.

    This is the final agent in the graph. It receives the destination research,
    itinerary, and budget estimate, then asks the LLM to synthesize them into a
    polished report that a traveler can read and use directly.

    Args:
        state: Current TravelState containing research, itinerary, and budget.

    Returns:
        A partial state update with `final_report` and a logged message.
    """
    print("\n📋  [ReportAgent] Compiling final travel plan…")

    llm = get_llm()

    system_prompt = (
        "You are a professional travel writer and editor. Your job is to "
        "synthesise research, itinerary, and budget information into a "
        "beautifully written, comprehensive, and actionable travel plan. "
        "Use clear headings, engaging language, and a warm but professional tone."
    )

    user_prompt = (
        f"Compile a complete travel plan for a {state['duration_days']}-day "
        f"trip to {state['destination']} ({state['travel_style']} style).\n\n"
        "=== RESEARCH NOTES ===\n"
        f"{state['research_notes']}\n\n"
        "=== ITINERARY ===\n"
        f"{state['itinerary']}\n\n"
        "=== BUDGET BREAKDOWN ===\n"
        f"{state['budget_breakdown']}\n\n"
        "Structure the final report with these sections:\n"
        "1. 🌍  Trip Overview\n"
        "2. 📋  Destination Highlights\n"
        "3. 🗓️  Day-by-Day Itinerary\n"
        "4. 💰  Budget Summary\n"
        "5. ✅  Pre-Trip Checklist (packing essentials, documents, apps)\n"
        "6. 🌟  Final Tips & Recommendations\n\n"
        "Make it inspiring and ready to hand to the traveller."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    final_report = response.content
    print("   ✅  Final report compiled.")

    return {
        "final_report": final_report,
        "messages": [HumanMessage(content=f"[ReportAgent]\n{final_report}")],
    }


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph travel-planning workflow.

    Each node is one agent function. The edges define the order of execution,
    so the output from one agent becomes available to the next agent through
    the shared TravelState object.

    Flow:
        START → research_agent → itinerary_agent → budget_agent
              → report_agent → END

    Returns:
        A compiled LangGraph app that can be run with `.invoke(initial_state)`.
    """
    graph = StateGraph(TravelState)

    # Register nodes (each node = one agent function)
    graph.add_node("research",  research_agent)
    graph.add_node("itinerary", itinerary_agent)
    graph.add_node("budget",    budget_agent)
    graph.add_node("report",    report_agent)

    # Define sequential edges
    graph.add_edge(START,       "research")
    graph.add_edge("research",  "itinerary")
    graph.add_edge("itinerary", "budget")
    graph.add_edge("budget",    "report")
    graph.add_edge("report",    END)

    return graph.compile()


# ---------------------------------------------------------------------------
# User Input Collection
# ---------------------------------------------------------------------------

def collect_user_input() -> dict:
    """
    Collect trip details from the terminal for CLI usage.

    This function is only used when running `multi_agent_system.py` directly.
    It prompts for destination, duration, travel style, and interests, applies
    sensible defaults when the user leaves a field blank, and asks for final
    confirmation before the agent graph starts.

    Returns:
        A complete initial TravelState dictionary ready for graph execution.
    """

    print("=" * 60)
    print("  ✈️   AI Travel Planner — Multi-Agent System")
    print("=" * 60)
    print("Powered by: LangChain + LangGraph (4 collaborative agents)\n")

    destination = input("🌍  Where do you want to travel? (e.g. Kyoto, Japan): ").strip()
    if not destination:
        destination = "Kyoto, Japan"

    duration_input = input("📅  How many days? (e.g. 7): ").strip()
    try:
        duration_days = int(duration_input)
    except ValueError:
        duration_days = 7

    print("\n💼  Travel style options: budget | mid-range | luxury")
    travel_style = input("    Your choice (default: mid-range): ").strip().lower()
    if travel_style not in {"budget", "mid-range", "luxury"}:
        travel_style = "mid-range"

    interests = input(
        "\n🎯  Your interests (comma-separated, e.g. history, food, hiking): "
    ).strip()
    if not interests:
        interests = "culture, food, sightseeing"

    print("\n" + "=" * 60)
    print(f"  Destination : {destination}")
    print(f"  Duration    : {duration_days} days")
    print(f"  Style       : {travel_style}")
    print(f"  Interests   : {interests}")
    print("=" * 60)
    confirm = input("\nProceed? (y/n, default y): ").strip().lower()
    if confirm == "n":
        print("Exiting. Have a great trip anyway! 🌟")
        sys.exit(0)

    return {
        "destination":    destination,
        "duration_days":  duration_days,
        "travel_style":   travel_style,
        "interests":      interests,
        "research_notes": "",
        "itinerary":      "",
        "budget_breakdown": "",
        "final_report":   "",
        "messages":       [],
    }


def run_travel_plan(
    destination: str,
    duration_days: int,
    travel_style: str,
    interests: str,
) -> TravelState:
    """
    Run the complete multi-agent planner from code or the web API.

    The frontend calls this function through `web_app.py`. It normalizes the
    incoming values, fills in defaults, validates the travel style, builds the
    LangGraph workflow, and executes the full agent chain.

    Args:
        destination: Place the user wants to visit.
        duration_days: Number of days for the trip.
        travel_style: One of `budget`, `mid-range`, or `luxury`.
        interests: Comma-separated traveler interests.

    Returns:
        The completed TravelState containing all agent outputs.
    """
    initial_state = {
        "destination": destination.strip() or "Kyoto, Japan",
        "duration_days": max(1, int(duration_days or 7)),
        "travel_style": travel_style.strip().lower() or "mid-range",
        "interests": interests.strip() or "culture, food, sightseeing",
        "research_notes": "",
        "itinerary": "",
        "budget_breakdown": "",
        "final_report": "",
        "messages": [],
    }

    if initial_state["travel_style"] not in {"budget", "mid-range", "luxury"}:
        initial_state["travel_style"] = "mid-range"

    app = build_graph()
    return app.invoke(initial_state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Run the command-line version of the travel planner.

    The CLI flow collects user input, builds the LangGraph app, runs all agents
    sequentially, prints the final report, and optionally saves it to
    `travel_plan.txt`.

    Returns:
        None.
    """
    # 1. Collect user input
    initial_state = collect_user_input()

    # 2. Build the LangGraph multi-agent pipeline
    print("\n🚀  Launching multi-agent pipeline…\n")
    app = build_graph()

    # 3. Run the graph (agents execute sequentially, passing shared state)
    final_state = app.invoke(initial_state)

    # 4. Display the final report
    print("\n" + "=" * 60)
    print("  📄  YOUR PERSONALISED TRAVEL PLAN")
    print("=" * 60 + "\n")
    print(final_state["final_report"])

    # 5. Optionally save to file
    print("\n" + "-" * 60)
    save = input("💾  Save report to 'travel_plan.txt'? (y/n): ").strip().lower()
    if save == "y":
        output_path = "travel_plan.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"TRAVEL PLAN — {initial_state['destination']}\n")
            f.write(f"Generated by AI Travel Planner (Multi-Agent System)\n")
            f.write("=" * 60 + "\n\n")
            f.write(final_state["final_report"])
        print(f"   ✅  Saved to {output_path}")

    print("\n🌟  Safe travels! Bon voyage! ✈️\n")


if __name__ == "__main__":
    main()
