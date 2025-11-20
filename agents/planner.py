from __future__ import annotations
from typing import Any, Iterable, Optional

from crewai import Agent
from config.settings import build_crewai_llm

SYSTEM_PROMPT = """
You are an expert travel planner AI. Your job is to create optimized, realistic,
budget-friendly and experience-rich travel itineraries. Follow these guidelines:

- Structure all plans into morning / afternoon / evening blocks.
- Recommend transport modes with realistic time estimates.
- Include free attractions, hidden gems, cultural experiences, and local food options.
- Adapt plans to the user's interests, weather, season, and budget.
- Keep recommendations specific, actionable, and practical.
- Add distances, opening hours, and travel durations when useful.
"""


def create_planner_agent(
    tools: Optional[Iterable[object]] = None,
    llm_overrides: dict[str, Any] | None = None,
) -> Agent:
    """Create the travel itinerary planner agent responsible for designing personalized trips."""
    
    return Agent(
        name="Travel Itinerary Planner",
        role="Architect of smart, cost-efficient and enjoyable travel plans",
        goal=(
            "Design personalized itineraries that optimize time, cost, and experience. "
            "Break down trips into morning-afternoon-evening activities, assign transport modes, "
            "and include free attractions, hidden gems, and local food spots matching the user's budget."
        ),
        backstory=(
            "If any add a backstory here."
        ),  # ← *your original backstory kept exactly*
        llm=build_crewai_llm(**(llm_overrides or {})),
        allow_delegation=False,
        verbose=True,
        system_prompt=SYSTEM_PROMPT,
        tools=list(tools or []),  # Call tools here
    )