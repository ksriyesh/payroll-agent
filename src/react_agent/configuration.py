"""Configuration for the payroll agent."""

import os
from typing import Annotated
from pydantic import BaseModel, Field
from react_agent import prompts

# Optional LangSmith tracing
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "payroll-agent"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


class Configuration(BaseModel):
    """Configuration for the payroll system."""

    vlm_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = "openai/gpt-4o"
    vlm_system_prompt: str = prompts.VLM_DOC_PROCESSOR_PROMPT
    
    react_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = "openai/gpt-4o"
    react_system_prompt: str = prompts.UPDATE_CHANGE_AGENT_PROMPT
    
    payroll_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = "openai/gpt-4o"
    payroll_system_prompt: str = prompts.PAYROLL_GENERATOR_PROMPT

    @classmethod
    def from_context(cls) -> "Configuration":
        """Create configuration from context."""
        return cls()
