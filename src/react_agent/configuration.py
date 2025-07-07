"""Configuration for the payroll parsing agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Configuration(BaseSettings):
    """Configuration for the payroll parsing agent."""
    
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
    
    # Text Model Configuration (Groq)
    model: str = Field(
        default="llama-3.1-8b-instant",  # Groq text model
        description="The chat model to use for text processing"
    )
    
    # Vision Model Configuration (OpenAI)
    vision_model: str = Field(
        default="gpt-4o-mini",  # OpenAI vision model
        description="Vision model for document processing (OpenAI)"
    )
    
    # Model parameters
    temperature: float = Field(
        default=0.1,
        description="Temperature for model generation"
    )
    
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for model generation"
    )
    
    # API Configuration
    groq_api_key: Optional[str] = Field(
        default=None,
        description="Groq API key for text model access"
    )
    
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for vision model access"
    )
    
    # Processing limits
    max_file_size: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        description="Maximum file size for upload in bytes"
    )
    
    max_pages: int = Field(
        default=50,
        description="Maximum pages to process in a document"
    )
    
    # System prompt
    system_prompt: str = Field(
        default="""You are a specialized payroll document parsing assistant. Your role is to:

1. **Acknowledge VLM Processing**: When you receive pre-processed document data from the VLM system, acknowledge receipt and analyze the extracted information.

2. **Make Definitive Decisions**: Based on the VLM analysis, you must either:
   - Return complete structured JSON payroll data if sufficient information is available
   - Ask 1-2 specific clarification questions if critical data is missing

3. **Response Format**: 
   - If data is complete → Provide JSON format with employee payroll information
   - If data is incomplete → Ask specific questions about missing information (employee names, pay rates, hours, etc.)

4. **End Interaction**: Your response should be definitive and end the conversation. Do not continue processing after providing JSON or asking clarification questions.

5. **Data Structure**: When returning JSON, use this format:
   ```json
   {
     "status": "success",
     "employees": [
       {
         "employee_id": "string or null",
         "name": "full name",
         "pay_rate": number or null,
         "hours_worked": number or null,
         "overtime_hours": number or null,
         "gross_pay": number or null,
         "deductions": number or null,
         "net_pay": number or null,
         "pay_period": "string or null",
         "position": "string or null"
       }
     ],
     "extraction_complete": true
   }
   ```

Be professional, accurate, and decisive in your responses.""",
        description="System prompt for the payroll parsing agent"
    )
    
    def model_post_init(self, __context: Any) -> None:
        """Log configuration after initialization."""
        logger.debug(f"⚙️ Configuration initialized: Text={self.model}, Vision={self.vision_model} (temp: {self.temperature})")
    
    def __str__(self) -> str:
        return f"Configuration(text_model={self.model}, vision_model={self.vision_model}, temperature={self.temperature})"
