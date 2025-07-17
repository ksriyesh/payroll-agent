"""Define the state structures for the agent."""

from __future__ import annotations

from typing import List, Optional, Sequence, Dict, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated


class EmployeeData(BaseModel):
    """Employee data model with all required fields."""
    model_config = ConfigDict(extra='forbid')
    
    name: str = Field(..., description="Employee name")
    regular_hours: float = Field(default=0, description="Regular hours worked")
    overtime_hours: float = Field(default=0, description="Overtime hours worked") 
    payrate: float = Field(..., description="Hourly pay rate")


class PayrollEmployee(BaseModel):
    """Individual employee payroll calculation."""
    model_config = ConfigDict(extra='forbid')
    
    name: str = Field(..., description="Employee name")
    regular_hours: float = Field(..., description="Regular hours worked")
    overtime_hours: float = Field(..., description="Overtime hours worked")
    payrate: float = Field(..., description="Hourly pay rate")
    regular_pay: float = Field(..., description="Regular pay amount")
    overtime_pay: float = Field(..., description="Overtime pay amount")
    total_pay: float = Field(..., description="Total pay amount")


class PayrollEmployeeList(BaseModel):
    """List of payroll employees extracted from documents."""
    model_config = ConfigDict(extra='forbid')
    
    employees: List[EmployeeData] = Field(default_factory=list, description="List of extracted employee data")


class PayrollReport(BaseModel):
    """Payroll report output."""
    model_config = ConfigDict(extra='forbid')
    
    employees: List[PayrollEmployee] = Field(default_factory=list, description="Employee payroll data")
    total_payroll: float = Field(default=0, description="Total payroll amount")
    summary: str = Field(default="", description="Payroll summary")


class State(BaseModel):
    """Complete state of the agent."""
    
    messages: Annotated[Sequence[AnyMessage], add_messages] = Field(default_factory=list)
    is_last_step: IsLastStep = Field(default=False)
    
    existing_employees: List[EmployeeData] = Field(default_factory=list)
    updated_employees: List[EmployeeData] = Field(default_factory=list)
    updates_list: List[EmployeeData] = Field(default_factory=list)
    temp_merged_list: List[Dict[str, Any]] = Field(default_factory=list, description="Temporarily stored merged employee data for confirmation")
    
    document_content: Optional[str] = Field(default=None)
    document_uploaded: bool = Field(default=False)
    document_processing_done: bool = Field(default=False)
    user_approval: bool = Field(default=False)
    trigger_payroll: bool = Field(default=False)
    
    # File data fields for direct graph processing
    file_data: Optional[str] = Field(default=None, description="Base64 encoded file data")
    file_path: Optional[str] = Field(default=None, description="Original file name/path")
    file_type: Optional[str] = Field(default=None, description="MIME type of the file")
    
    current_pay_data: Optional[Dict[str, Any]] = Field(default=None)


class InputState(BaseModel):
    messages: Annotated[Sequence[AnyMessage], add_messages] = Field(default_factory=list)
    
class OutputState(BaseModel):
    messages: Annotated[Sequence[AnyMessage], add_messages] = Field(default_factory=list)