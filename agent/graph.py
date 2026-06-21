"""
LangGraph tool-calling retention agent.

Builds a ReAct-style agent (LLM + tools) that gathers customer context and
returns a structured outreach plan. Import-safe: if langgraph/langchain or an
API key are unavailable, callers fall back to the deterministic composer.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import SYSTEM_PROMPT, build_task_message
from agent.tools import build_langchain_tools
from settings import settings


def llm_available() -> bool:
    if not settings.openai_api_key:
        return False
    try:
        import langgraph  # noqa: F401
        import langchain_openai  # noqa: F401
        return True
    except Exception:
        return False


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of the model's final message."""
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    return json.loads(text)


def compose_with_agent(customer_id: int, segment: str, playbook: str, max_discount_pct: float) -> dict:
    """Run the LangGraph agent. Returns the parsed plan plus a tool trace."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    model = ChatOpenAI(
        model=settings.agent_llm_model,
        temperature=settings.agent_temperature,
        api_key=settings.openai_api_key,
    )
    tools = build_langchain_tools()
    agent = create_react_agent(model, tools)

    task = build_task_message(customer_id, segment, playbook, max_discount_pct)
    result = agent.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=task)]},
        config={"recursion_limit": 12},
    )

    messages = result["messages"]
    tools_used = [
        tc["name"]
        for m in messages
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    final = messages[-1].content if messages else ""
    plan = _extract_json(final)
    plan["_tools_used"] = tools_used
    return plan
