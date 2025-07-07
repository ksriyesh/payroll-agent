import pytest
from langsmith import unit

from react_agent import graph


@pytest.mark.asyncio
@unit
async def test_react_agent_simple_passthrough() -> None:
    res = await graph.ainvoke(
        {"messages": [("user", "Who is the founder of LangChain?")]},
        {"configurable": {
            "model": "groq/llama-3.1-8b-instant",
            "system_prompt": "You are a helpful AI assistant specialized in payroll and HR management."
        }},
    )

    assert "harrison" in str(res["messages"][-1].content).lower()
