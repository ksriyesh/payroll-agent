"""Define the state structures for the agent."""

from __future__ import annotations

from typing import Sequence, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmployeePayInfo(BaseModel):
    """Information about an employee's pay details extracted from documents."""
    
    employee_id: Optional[str] = Field(None, description="Employee ID")
    name: str = Field(..., description="Employee name")
    pay_rate: Optional[float] = Field(None, description="Pay rate (hourly/salary)")
    hours_worked: Optional[float] = Field(None, description="Regular hours worked")
    overtime_hours: Optional[float] = Field(None, description="Overtime hours worked")
    gross_pay: Optional[float] = Field(None, description="Gross pay amount")
    deductions: Optional[float] = Field(None, description="Total deductions")
    net_pay: Optional[float] = Field(None, description="Net pay amount")
    pay_period: Optional[str] = Field(None, description="Pay period")
    position: Optional[str] = Field(None, description="Job position/title")
    
    def model_post_init(self, __context: Any) -> None:
        """Log employee creation."""
        logger.debug(f"👤 Employee created: {self.name} (ID: {self.employee_id})")
    
    model_config = {
        "arbitrary_types_allowed": True
    }


class DocumentInfo(BaseModel):
    """Information about uploaded payroll document."""
    
    filename: str = Field(..., description="Name of the uploaded file")
    file_type: str = Field(..., description="Type of file (pdf, jpeg, etc.)")
    file_size: int = Field(..., description="Size of file in bytes")
    pages: int = Field(1, description="Number of pages in document")
    processed: bool = Field(False, description="Whether document has been processed by VLM")
    
    def model_post_init(self, __context: Any) -> None:
        """Log document info creation."""
        logger.debug(f"📄 Document info created: {self.filename} ({self.file_type}, {self.file_size} bytes, {self.pages} pages)")
    
    model_config = {
        "arbitrary_types_allowed": True
    }


class PayrollContext(BaseModel):
    """Additional context provided by user for better document processing."""
    
    company_name: Optional[str] = Field(None, description="Company name")
    pay_period_type: Optional[str] = Field(None, description="weekly, bi-weekly, monthly, etc.")
    expected_employees: Optional[List[str]] = Field(None, description="Expected employee names")
    document_type: Optional[str] = Field(None, description="payslip, timesheet, payroll_summary, etc.")
    currency: Optional[str] = Field(None, description="Currency used (USD, EUR, etc.)")
    additional_notes: Optional[str] = Field(None, description="Any additional context from user")
    
    def model_post_init(self, __context: Any) -> None:
        """Log context creation."""
        logger.debug(f"📋 Payroll context created: {self.company_name} ({self.pay_period_type})")
    
    model_config = {
        "arbitrary_types_allowed": True
    }


class InputState(BaseModel):
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="Messages tracking the primary execution state of the agent."
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """

    def model_post_init(self, __context: Any) -> None:
        """Log input state creation."""
        logger.debug(f"📥 Input state created: {len(self.messages)} messages")

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "forbid"
    }


class State(InputState):
    """Represents the complete state of the payroll parsing agent.

    This class stores document information, extracted employee data, and processing context.
    """

    is_last_step: IsLastStep = Field(
        default=False,
        description="Indicates whether the current step is the last one before the graph raises an error."
    )
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """
    
    # Document upload and processing state
    document_uploaded: bool = Field(
        default=False,
        description="Whether a document has been uploaded"
    )
    
    file_path: str = Field(
        default="",
        description="Path to the uploaded file"
    )
    
    context_query: str = Field(
        default="",
        description="User-provided context or query about the document"
    )
    
    # VLM processing state
    vlm_processing_complete: bool = Field(
        default=False,
        description="Whether VLM processing of document is complete"
    )
    
    # Document information
    document_info: Optional[DocumentInfo] = Field(
        None, 
        description="Information about uploaded payroll document"
    )
    
    # Extracted data
    text_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted text and positions from document"
    )
    
    extracted_text: str = Field(
        default="",
        description="Full extracted text from VLM analysis"
    )
    
    # Payroll data
    employees: List[EmployeePayInfo] = Field(
        default_factory=list,
        description="List of employees and their pay information extracted from document"
    )
    
    payroll_context: Optional[PayrollContext] = Field(
        None,
        description="Additional context provided by user for better processing"
    )
    
    # Processing status
    extraction_complete: bool = Field(
        default=False,
        description="Whether payroll data extraction is complete"
    )
    
    context_gathered: bool = Field(
        default=False,
        description="Whether additional context has been gathered from user"
    )
    
    # Processing metadata
    processing_errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during processing"
    )

    def model_post_init(self, __context: Any) -> None:
        """Log state initialization."""
        logger.debug(f"🔄 State initialized: {len(self.messages)} messages, document_uploaded={self.document_uploaded}")

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "forbid"
    }
