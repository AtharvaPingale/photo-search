from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.agent.graph import AgentAnswer, run_agent

router = APIRouter(tags=["agent"])


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@router.post("/agent/ask", response_model=AgentAnswer)
def ask(body: AskIn) -> AgentAnswer:
    return run_agent(body.question)
