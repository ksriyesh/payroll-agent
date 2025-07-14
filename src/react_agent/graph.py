"""Simple 3-agent payroll system."""

from typing import Dict, List, Literal, cast
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from react_agent.configuration import Configuration
from react_agent.state import State, EmployeeData, PayrollReport
from react_agent.utils import load_chat_model
from react_agent.tools import merge_employees


async def init_node(state: State) -> Dict:
    """Initialize with sample employees."""
    employees = [
        EmployeeData(name="John Doe", regular_hours=40.0, overtime_hours=5.0, payrate=60.0),
        EmployeeData(name="Jane Smith", regular_hours=38.5, overtime_hours=2.5, payrate=50.0),
        EmployeeData(name="Bob Johnson", regular_hours=42.0, overtime_hours=8.0, payrate=45.0),
        EmployeeData(name="Alice Williams", regular_hours=35.0, overtime_hours=3.0, payrate=55.0),
        EmployeeData(name="Charlie Brown", regular_hours=40.0, overtime_hours=0.0, payrate=48.0)
    ]
    return {"existing_employees": employees}


async def vlm_processor(state: State) -> Dict:
    """Process document to extract employee data."""
    config = Configuration.from_context()
    
    # Create proper Pydantic model for structured output
    class EmployeeList(BaseModel):
        employees: List[EmployeeData] = Field(..., description="List of extracted employees")
    
    model = load_chat_model(config.vlm_model).with_structured_output(EmployeeList)
    
    if not state.document_content:
        return {"updated_employees": []}
    
    response = await model.ainvoke([
        {"role": "system", "content": config.vlm_system_prompt},
        {"role": "user", "content": f"Extract data from: {state.document_content}"}
    ])
    
    return {"updated_employees": response.employees}


async def update_agent(state: State) -> Dict:
    """Main interactive agent."""
    config = Configuration.from_context()
    
    from react_agent.tools import update_state, merge_employees
    
    # Check if user is asking for structured output
    if state.messages and state.current_pay_data:
        last_message = state.messages[-1].content.lower()
        if "structured output" in last_message or "json" in last_message or "raw data" in last_message:
            import json
            structured_output = json.dumps(state.current_pay_data, indent=2)
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content=f"Here's the structured output in JSON format:\n\n```json\n{structured_output}\n```")]}
    
    # If user has already approved and we have current employees, just acknowledge
    if state.user_approval and state.current_employees and state.current_pay_data:
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content="Thank you for confirming! The payroll has been successfully generated and is ready for processing.")]}
    
    model = load_chat_model(config.react_model).bind_tools([update_state])
    
    # Check if we should automatically merge
    should_auto_merge = (
        len(state.existing_employees) > 0 and 
        len(state.updated_employees) > 0 and 
        len(state.current_employees) == 0 and
        not state.user_approval
    )
    
    if should_auto_merge:
        # Automatically merge employee lists
        merged_employees = merge_employees(state.existing_employees, state.updated_employees)
        
        # Convert to dictionaries for state update
        merged_dict = [
            {
                "name": emp.name,
                "regular_hours": emp.regular_hours,
                "overtime_hours": emp.overtime_hours,
                "payrate": emp.payrate
            } for emp in merged_employees
        ]
        
        # Create summary of changes
        updated_names = {emp.name for emp in state.updated_employees}
        existing_names = {emp.name for emp in state.existing_employees}
        
        added_names = [emp.name for emp in state.updated_employees if emp.name not in existing_names]
        updated_names_list = [emp.name for emp in state.updated_employees if emp.name in existing_names]
        preserved_names = [emp.name for emp in state.existing_employees if emp.name not in updated_names]
        
        summary_parts = [
            "✅ **Employee data merged successfully!**",
            f"📋 **Final employee list: {len(merged_employees)} employees**",
            ""
        ]
        
        if added_names:
            summary_parts.append(f"➕ **Added**: {', '.join(added_names)}")
        if updated_names_list:
            summary_parts.append(f"🔄 **Updated**: {', '.join(updated_names_list)}")
        if preserved_names:
            summary_parts.append(f"💾 **Preserved**: {', '.join(preserved_names)}")
        
        summary_parts.extend([
            "",
            "**Current employees:**"
        ])
        
        for emp in merged_employees:
            summary_parts.append(f"- {emp.name}: {emp.regular_hours}h regular, {emp.overtime_hours}h overtime @ ${emp.payrate}/hr")
        
        summary_parts.extend([
            "",
            "Please confirm this data is correct to proceed with payroll generation."
        ])
        
        # Use tool to update state and return summary
        from langchain_core.messages import AIMessage
        tool_response = AIMessage(
            content="\n".join(summary_parts),
            tool_calls=[{
                "name": "update_state",
                "args": {
                    "employees": merged_dict,
                    "target_list": "current_employees"
                },
                "id": "auto_merge_001",
                "type": "tool_call"
            }]
        )
        
        return {"messages": [tool_response]}
    
    # Regular interaction mode - provide detailed context
    context_parts = []
    
    # Add existing employees details
    if state.existing_employees:
        context_parts.append("EXISTING EMPLOYEES (Previous pay period):")
        for emp in state.existing_employees:
            context_parts.append(f"- {emp.name}: {emp.regular_hours}h regular, {emp.overtime_hours}h overtime @ ${emp.payrate}/hr")
    else:
        context_parts.append("EXISTING EMPLOYEES: None")
    
    # Add updated employees details
    if state.updated_employees:
        context_parts.append("\nUPDATED EMPLOYEES (From current document):")
        for emp in state.updated_employees:
            context_parts.append(f"- {emp.name}: {emp.regular_hours}h regular, {emp.overtime_hours}h overtime @ ${emp.payrate}/hr")
    else:
        context_parts.append("\nUPDATED EMPLOYEES: None")
    
    # Add current employees details if available
    if state.current_employees:
        context_parts.append("\nCURRENT EMPLOYEES (Merged list):")
        for emp in state.current_employees:
            context_parts.append(f"- {emp.name}: {emp.regular_hours}h regular, {emp.overtime_hours}h overtime @ ${emp.payrate}/hr")
    else:
        context_parts.append("\nCURRENT EMPLOYEES: None (needs merging)")
    
    # Add approval status
    context_parts.append(f"\nUSER APPROVAL: {state.user_approval}")
    
    context = "\n".join(context_parts)
    
    prompt = config.react_system_prompt.format(
        existing_count=len(state.existing_employees),
        updated_count=len(state.updated_employees),
        current_count=len(state.current_employees),
        user_approval=state.user_approval
    )
    
    response = await model.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": context},
        *state.messages
    ])
    
    return {"messages": [response]}


async def payroll_agent(state: State) -> Dict:
    """Generate payroll report with manual calculations."""
    from react_agent.state import PayrollReport, PayrollEmployee
    
    # Use current employees or merge if needed
    employees = state.current_employees or merge_employees(state.existing_employees, state.updated_employees)
    
    # Calculate payroll for each employee
    payroll_employees = []
    total_payroll = 0.0
    
    for emp in employees:
        # Calculate regular pay
        regular_pay = emp.regular_hours * emp.payrate
        
        # Calculate overtime pay (1.5x rate)
        overtime_pay = emp.overtime_hours * (emp.payrate * 1.5)
        
        # Calculate total pay
        total_pay = regular_pay + overtime_pay
        total_payroll += total_pay
        
        # Create PayrollEmployee object
        payroll_emp = PayrollEmployee(
            name=emp.name,
            regular_hours=emp.regular_hours,
            overtime_hours=emp.overtime_hours,
            payrate=emp.payrate,
            regular_pay=regular_pay,
            overtime_pay=overtime_pay,
            total_pay=total_pay
        )
        payroll_employees.append(payroll_emp)
    
    # Create summary
    summary = f"Payroll calculated for {len(payroll_employees)} employees. Total payroll: ${total_payroll:.2f}"
    
    # Create PayrollReport
    report = PayrollReport(
        employees=payroll_employees,
        total_payroll=total_payroll,
        summary=summary
    )
    
    # Convert to dictionary for state storage
    report_dict = {
        "employees": [emp.model_dump() for emp in payroll_employees],
        "total_payroll": total_payroll,
        "summary": summary
    }
    
    return {
        "current_pay_data": report_dict,
        "messages": [AIMessage(content=f"✅ Payroll calculated: {len(payroll_employees)} employees, Total: ${total_payroll:.2f}")]
    }


async def tool_executor(state: State) -> Dict:
    """Execute tools."""
    from langchain_core.messages import ToolMessage
    
    messages = []
    updates = {}
    
    for tool_call in state.messages[-1].tool_calls:
        if tool_call["name"] == "update_state":
            try:
                target = tool_call["args"]["target_list"]
                emp_data = tool_call["args"]["employees"]
                
                # Convert to EmployeeData objects
                employees = [EmployeeData(**emp) for emp in emp_data]
                
                # Update state
                if target == "current_employees":
                    updates["current_employees"] = employees
                    updates["user_approval"] = True
                    updates["trigger_payroll"] = True
                elif target == "updated_employees":
                    updates["updated_employees"] = employees
                elif target == "existing_employees":
                    updates["existing_employees"] = employees
                
                messages.append(ToolMessage(
                    content=f"Updated {target} with {len(employees)} employees",
                    tool_call_id=tool_call["id"]
                ))
                
            except KeyError as e:
                error_msg = f"Error in tool call: missing {e} in arguments"
                messages.append(ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call["id"]
                ))
            except Exception as e:
                error_msg = f"Error processing tool call: {str(e)}"
                messages.append(ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call["id"]
                ))
    
    return {"messages": messages, **updates}


# Simple routing
def route_init(state: State) -> Literal["vlm_processor", "update_agent"]:
    return "vlm_processor" if state.document_uploaded else "update_agent"

def route_vlm(state: State) -> Literal["update_agent"]:
    return "update_agent"

def route_update(state: State) -> Literal["tool_executor", "__end__"]:
    return "tool_executor" if state.messages and state.messages[-1].tool_calls else "__end__"

def route_tools(state: State) -> Literal["update_agent", "payroll_agent"]:
    return "payroll_agent" if state.trigger_payroll else "update_agent"

def route_payroll(state: State) -> Literal["update_agent"]:
    return "update_agent"


# Build graph
builder = StateGraph(State)
builder.add_node("init", init_node)
builder.add_node("vlm_processor", vlm_processor)
builder.add_node("update_agent", update_agent)
builder.add_node("payroll_agent", payroll_agent)
builder.add_node("tool_executor", tool_executor)

builder.add_edge("__start__", "init")
builder.add_conditional_edges("init", route_init)
builder.add_conditional_edges("vlm_processor", route_vlm)
builder.add_conditional_edges("update_agent", route_update)
builder.add_conditional_edges("tool_executor", route_tools)
builder.add_conditional_edges("payroll_agent", route_payroll)

graph = builder.compile()
