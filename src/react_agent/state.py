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
    current_employees: List[EmployeeData] = Field(default_factory=list)
    
    document_content: Optional[str] = Field(default=None)
    document_uploaded: bool = Field(default=False)
    user_approval: bool = Field(default=False)
    trigger_payroll: bool = Field(default=False)
    
    current_pay_data: Optional[Dict[str, Any]] = Field(default=None)
