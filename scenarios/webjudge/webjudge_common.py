"""
Common models and utilities for WebJudge multi-agent system.
"""

from pydantic import BaseModel
from typing import List, Literal

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)


class TaskEvaluation(BaseModel):
    """Evaluation result for a web navigation task."""
    success: bool
    status: Literal["success", "failure"]
    key_points: str
    reasoning: str
    final_score: float


def webjudge_agent_card(agent_name: str, card_url: str) -> AgentCard:
    """Create agent card for WebJudge orchestrator."""
    skill = AgentSkill(
        id='evaluate_web_navigation',
        name='Evaluate web navigation task completion',
        description='Orchestrate web navigation task execution and evaluate results using WebJudge methodology.',
        tags=['web-navigation', 'evaluation', 'webjudge'],
        examples=["""
{
  "participants": {
    "web_agent": "https://web-agent.example.com:443"
  },
  "config": {
    "task_description": "Find the store location and hours of the closest Trader Joe's to zip code 90028",
    "start_url": "https://www.traderjoes.com/",
    "max_steps": 30
  }
}
"""]
    )
    agent_card = AgentCard(
        name=agent_name,
        description='Orchestrate web navigation tasks and evaluate completion using WebJudge methodology with key point extraction and screenshot judging.',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card
