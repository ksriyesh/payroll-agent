"""Configuration for the payroll agent."""

import os
from typing import Annotated
from pydantic import BaseModel, Field
from . import prompts

# Optional LangSmith tracing
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "payroll-agent"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


class Configuration(BaseModel):
    """Configuration for the payroll system."""

    # Model configurations
    vlm_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = "openai/gpt-4o"
    vlm_system_prompt: str = prompts.VLM_DOC_PROCESSOR_PROMPT
    
    react_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = "openai/gpt-4o"
    react_system_prompt: str = prompts.UPDATE_CHANGE_AGENT_PROMPT
    
    payroll_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = "openai/gpt-4o"
    payroll_system_prompt: str = prompts.PAYROLL_GENERATOR_PROMPT
    
    # API Mode toggle - set to True to enable Azure OpenAI + MAAS mode
    # When True, the system will look for Azure OpenAI or MAAS environment variables
    # When False, the system will only use standard OpenAI API key
    use_azure_or_maas: bool = False

    @classmethod
    def from_context(cls) -> "Configuration":
        """Create configuration from context."""
        return cls()
