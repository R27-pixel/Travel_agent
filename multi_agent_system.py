"""
multi_agent_system.py
=====================
A multi-agent AI Travel Planner built with LangChain + LangGraph.

Agents
------
1. ResearchAgent      – gathers destination facts, weather, visa info
2. ItineraryAgent     – builds a day-by-day itinerary
3. BudgetAgent        – estimates costs (flights, hotels, food, activities)
4. ReportAgent        – compiles everything into a polished travel plan

Workflow (LangGraph)
--------------------
START → research → itinerary → budget → report → END

Usage
-----
    python multi_agent_system.py

Requirements
------------
    pip install langchain langchain-groq langgraph

Set your API key before running:
    export groq_API_KEY="sk-..."
"""

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
    Loads simple KEY=VALUE pairs from a local .env file when the variables are
    not already set by the shell.
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
# ---------------------------------------------------------------------------

class TravelState(TypedDict):
    """
    Shared context that flows through every agent node.
    Each agent reads what it needs and writes its own output key.
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
    Gathers essential destination intel: geography, climate, visa notes,
    cultural tips, top attractions, and best travel season.
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
    Creates a detailed day-by-day itinerary using the research notes,
    tailored to the traveller's style and interests.
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
    Estimates trip costs broken down by category, calibrated to the
    traveller's chosen travel style and destination.
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
    Synthesises all prior agent outputs into one polished, reader-friendly
    travel plan document.
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
    Defines the LangGraph workflow with nodes, edges, and execution order.

    Flow:
        START → research_agent → itinerary_agent → budget_agent
              → report_agent → END
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
    """Interactively collects trip parameters from the user."""

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
    """Runs the full multi-agent planner and returns the completed state."""
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
