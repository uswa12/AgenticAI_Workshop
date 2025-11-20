#"""Task definitions for the Agentic AI Workshop crew."""
from __future__ import annotations

from typing import List

from crewai import Task

from tools import create_calculator_tool, create_rag_tool, create_web_search_tool


from crewai import Task

def create_planning_task(agent) -> Task:
    """Task 1: Draft a personalized travel itinerary plan."""
    return Task(
        description=(
            "Analyze the user's travel preferences, budget, and trip duration. "
            "Create a milestone-based itinerary structured into morning, afternoon, and evening activities. "
            "Include transport modes, distances, estimated times, cultural experiences, hidden gems, local food, and free attractions. "
            "Account for weather, season, and user interests."
        ),
        expected_output=(
            "A structured, actionable travel plan with 3–5 milestones (e.g., Day 1, Day 2, Day 3), "
            "complete with timings, locations, travel durations, recommended meals, and notable experiences. "
            "Should be practical, budget-friendly, and optimized for maximum enjoyment."
        ),
        agent=agent,
        name="Travel Planning",  # Task name
    )



def create_research_task(agent, tools=None) -> Task:
    """Task 2: Light travel research for itinerary planning."""

    # Fallback tools (safe & minimal)
    tools = list(tools) if tools is not None else [
        create_web_search_tool(),  # RAG removed for speed
        create_calculator_tool(),
    ]

    return Task(
        description=(
            "Collect short and reliable travel information for the selected destination. "
            "Focus ONLY on: best season to visit, 2 key attractions, 1 local food suggestion, "
            "and typical cost range for budget travelers."
        ),
        expected_output=(
            "A short bullet list including:\n"
            "Best visiting season\n"
            "2 recommended attractions (with short notes)\n"
            "1 local food or cultural experience\n"
            "Approximate budget range (e.g. $200–$350)\n"
            "*Do not provide citations—only concise facts.*"
        ),
        agent=agent,
        tools=tools,
        name="Quick Travel Research",
    )


def create_writing_task(agent) -> Task:
    """Task: Author travel deliverables for the Travel Writer / Concierge Agent."""
    return Task(
        name="Travel Itinerary Authoring",
        description=(
            "Using the provided travel outlines and research, create detailed, user-friendly travel itineraries. "
            "Include day-by-day schedules, suggested activities, booking and packing checklists, and concise local tips. "
            "Ensure the output is clear, actionable, and easy to follow for travelers of all experience levels."
        ),
        expected_output=(
            "A polished, Markdown-formatted travel guide including:\n"
            "- Day-by-day itinerary with activities and timings\n"
            "- Booking checklist for accommodations, transport, and tours\n"
            "- Packing checklist tailored to the destination and planned activities\n"
            "- Concise, practical local tips (restaurants, attractions, cultural advice)"
        ),
        agent=agent,
    )


def create_review_task(agent) -> Task:
    """Task 4 placeholder: review compiled deliverables."""
    return Task(
        description=(
            "Add description for Task 4."
            # "Review the draft content for '{topic}' for accuracy, completeness, and pedagogy. Provide an executive summary of strengths, "
            # "list gaps or issues, and suggest concrete improvements."
        ),
        expected_output=(
            "Add expected output for Task 4."
            # "A review report with sections for Summary, Major Findings, Minor Suggestions, and Final Recommendation."
        ),
        agent=agent,
        name="Task 4",  # "Reviewing"
    )


def build_workshop_tasks(planner, researcher, writer, reviewer, research_tools=None) -> List[Task]:
    """Convenience helper to create the full task list order."""
    return [
        create_planning_task(planner),
        create_research_task(researcher, tools=research_tools),
        create_writing_task(writer),
        create_review_task(reviewer),
    ]
