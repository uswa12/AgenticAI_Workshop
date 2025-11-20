#"""Research agent that combines RAG retrieval with live web search."""

# Agent 2
from __future__ import annotations

from typing import Any, Iterable, Optional

from crewai import Agent

from config.settings import build_crewai_llm

SYSTEM_PROMPT = (
    "You are **Destination Analyst Lite**, an assistant that finds simple, reliable travel info.\n"
    "Only provide short bullet points. Do NOT provide citations or long explanations.\n\n"
    "Research Rules:\n"
    "- Only include information if confident (otherwise skip it)\n"
    "- Keep everything concise (max 7 bullets)\n"
    "- Include numeric ranges only when easy (e.g., $200–$350)\n"
    "- Focus on season, key places, local food, and typical budget\n"
)


def create_researcher_agent(
    tools: Optional[Iterable[object]] = None,
    llm_overrides: dict[str, Any] | None = None,
) -> Agent:
    """Create the simplified travel researcher agent."""
    return Agent(
        name="Destination Analyst Lite",
        role="Provide short, reliable travel facts.",
        goal="Help itinerary planning with minimal but accurate travel insights.",
        backstory=(
            "You are a fast and practical travel researcher who specializes in quick, "
            "useful summaries instead of long detailed reports."
        ),
        llm=build_crewai_llm(**(llm_overrides or {})),
        allow_delegation=False,
        verbose=True,
        system_prompt=SYSTEM_PROMPT,
        tools=list(tools or []),
    )
