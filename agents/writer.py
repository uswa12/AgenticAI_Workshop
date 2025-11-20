# Agent 3
from __future__ import annotations

from typing import Any, Iterable, Optional

from crewai import Agent
from config.settings import build_crewai_llm

SYSTEM_PROMPT = (
    """You are the Travel Writer / Concierge Agent.
Input: itinerary outline + research.
Output: friendly, clear itinerary with day-by-day steps, booking checklist, packing list, and short local tips.
Keep tone helpful and concise."""
)


def create_writer_agent(
    tools: Optional[Iterable[object]] = None,
    llm_overrides: dict[str, Any] | None = None,
) -> Agent:
    """Create the writer agent responsible for generating travel itineraries and concierge materials."""
    return Agent(
        name="Travel Writer / Concierge Agent",
        role="Compose detailed, friendly, and actionable travel itineraries from provided outlines and research",
        goal="Produce clear day-by-day itineraries, packing and booking checklists, and concise local tips for travelers",
        backstory=(
            "You specialize in turning research and outlines into practical, engaging travel guides. "
            "Your output should be helpful, concise, and easy to follow for travelers of all experience levels."
        ),
        llm=build_crewai_llm(**(llm_overrides or {})),
        allow_delegation=False,
        verbose=True,
        system_prompt=SYSTEM_PROMPT,
        tools=list(tools or []),  # Call tools here
    )