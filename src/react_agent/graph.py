"""Clean 3-node payroll workflow with automatic conflict resolution."""

import logging
import json
from typing import Dict, List, Literal
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel

from .configuration import Configuration
from .state import State, EmployeeData
from .utils import load_chat_model, load_previous_payperiod_data

logger = logging.getLogger(__name__)


async def init_node(state: State) -> Dict:
    """Initialize workflow: check for file uploads and load existing employees."""
    logger.info("=== ENTERING init_node ===")
    
    result = {}
    
    # Check if file is uploaded in the current request
    if state.file_data and state.file_path:
        logger.info(f"File detected: {state.file_path}")
        result['document_uploaded'] = True
    
    # Load existing employees from previous pay period (hardcoded for now)
    if not state.existing_employees:
        logger.info("Loading existing employees from previous pay period")
        result['existing_employees'] = await load_previous_payperiod_data()
    
    logger.info(f"Init complete - Document uploaded: {result.get('document_uploaded', False)}")
    return result
    

async def vlm_processor(state: State) -> Dict:
    """Process document with VLM and extract structured employee data."""
    logger.info("=== ENTERING vlm_processor ===")
    
    import os
    import base64
    import tempfile
    from .utils import process_document_with_files_api
    from .prompts import VLM_DOC_PROCESSOR_PROMPT
    
    config = Configuration.from_context()

    # Define Pydantic model for structured extraction
    class EmployeeList(BaseModel):
        employees: List[EmployeeData]

    temp_file_path = None
    try:
        # Create temporary file from uploaded data
        if state.file_data:
            file_extension = ".jpg"  # Default
            if state.file_type:
                if "pdf" in state.file_type.lower():
                    file_extension = ".pdf"
                elif "png" in state.file_type.lower():
                    file_extension = ".png"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                base64_data = state.file_data.split(",", 1)[-1]
                file_content = base64.b64decode(base64_data)
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
        # Process document using files API
        if temp_file_path:
            logger.info(f"Processing document: {temp_file_path}")
            result = await process_document_with_files_api(
                file_path=temp_file_path,
                prompt=VLM_DOC_PROCESSOR_PROMPT.strip(),
                model_name=config.vlm_model,
                vlm_system_prompt=config.vlm_system_prompt,
                output_schema=EmployeeList
            )

            if result and "structured_data" in result and "employees" in result["structured_data"]:
                employees_data = result["structured_data"]["employees"]
                logger.info(f"Successfully extracted {len(employees_data)} employees")
                
                return {
                    "updates_list": employees_data,
                    "document_uploaded": False,
                    "document_processing_done": True
                }
        
        logger.warning("Could not extract employee data from document")
        return {
            "updates_list": [],
            "document_uploaded": False,
            "document_processing_done": True
        }

    except Exception as e:
        logger.error(f"Error in vlm_processor: {str(e)}")
        return {
            "updates_list": [],
            "document_uploaded": False,
            "document_processing_done": True
        }
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}")


def merge_with_conflict_resolution(existing_employees: List[EmployeeData], updates_list: List[EmployeeData]) -> List[Dict]:
    """
    Merge existing employees with updates using automatic conflict resolution:
    - Missing employees: Add from existing_employees with same data
    - Conflicts: updates_list takes precedence 
    - New employees: Add from updates_list
    """
    logger.info(f"Merging {len(existing_employees)} existing + {len(updates_list)} updates")
    
    merged = {}
    
    # First, add all existing employees
    for emp in existing_employees:
        emp_dict = emp.model_dump() if hasattr(emp, 'model_dump') else emp
        merged[emp_dict["name"]] = emp_dict
    
    # Then, update/add from updates_list (takes precedence)
    for emp in updates_list:
        emp_dict = emp.model_dump() if hasattr(emp, 'model_dump') else emp
        
        if emp_dict["name"] in merged:
            # Employee exists - merge with updates_list taking precedence
            existing_emp = merged[emp_dict["name"]]
            merged_emp = {
                "name": emp_dict["name"],
                "regular_hours": emp_dict["regular_hours"],  # Updates take precedence
                "overtime_hours": emp_dict["overtime_hours"],  # Updates take precedence
                "payrate": emp_dict["payrate"] if emp_dict["payrate"] > 0 else existing_emp["payrate"]  # Keep existing payrate if updates has 0
            }
            merged[emp_dict["name"]] = merged_emp
        else:
            # New employee from updates_list
            merged[emp_dict["name"]] = emp_dict
    
    result = list(merged.values())
    logger.info(f"Merge complete: {len(result)} total employees")
    return result


async def update_agent(state: State) -> Dict:
    """React agent for handling user interactions and employee data management."""
    logger.info("=== ENTERING update_agent ===")
    
    config = Configuration.from_context()
    model = load_chat_model(fully_specified_name=config.react_model)
    
    # PRIORITY 1: Handle user confirmation if temp_merged_list exists
    if state.temp_merged_list:
        logger.info(f"temp_merged_list has {len(state.temp_merged_list)} employees - checking for confirmation")
        
        # Check for confirmation in natural language
        last_user_message = ""
        for msg in reversed(state.messages):
            if hasattr(msg, "type") and msg.type == "human":
                last_user_message = msg.content.lower().strip()
                break
            elif hasattr(msg, "content") and not hasattr(msg, "type"):
                # Handle dict messages
                last_user_message = msg.get("content", "").lower().strip()
                break
            elif isinstance(msg, dict) and "content" in msg:
                # Handle dict-style messages
                last_user_message = msg["content"].lower().strip()
                break
        
        logger.info(f"Last user message detected: '{last_user_message}'")
        
        # Natural language confirmation detection - more comprehensive
        confirmation_indicators = [
            "confirm", "yes", "approve", "looks good", "correct", "proceed", 
            "generate payroll", "ok", "okay", "accepted", "accept", "agreed", 
            "agree", "confirmed", "good", "right", "yep", "yeah", "y"
        ]
        is_confirmed = any(indicator in last_user_message for indicator in confirmation_indicators)
        
        # Also check if the message is exactly "yes" or similar short confirmations
        if last_user_message in ["yes", "y", "ok", "okay", "confirm", "agreed", "accept", "proceed"]:
            is_confirmed = True
        
        logger.info(f"Confirmation check - is_confirmed: {is_confirmed} (message: '{last_user_message}')")
        
        if is_confirmed:
            logger.info("✅ User confirmed merged data - generating payroll automatically")
            
            # Calculate payroll for confirmed employees
            payroll_employees = []
            total_payroll = 0.0
            
            for emp in state.temp_merged_list:
                # Calculate regular pay
                regular_pay = emp['regular_hours'] * emp['payrate']
                
                # Calculate overtime pay (1.5x rate)
                overtime_pay = emp['overtime_hours'] * (emp['payrate'] * 1.5)
                
                # Calculate total pay
                total_pay = regular_pay + overtime_pay
                total_payroll += total_pay
                
                # Create payroll entry
                payroll_entry = {
                    "name": emp['name'],
                    "regular_hours": emp['regular_hours'],
                    "overtime_hours": emp['overtime_hours'],
                    "payrate": emp['payrate'],
                    "regular_pay": regular_pay,
                    "overtime_pay": overtime_pay,
                    "total_pay": total_pay
                }
                payroll_employees.append(payroll_entry)
            
            logger.info(f"📊 Payroll calculated for {len(payroll_employees)} employees, total: ${total_payroll:.2f}")
            
            # Create payroll report
            payroll_report = {
                "employees": payroll_employees,
                "total_payroll": total_payroll,
                "summary": f"Payroll calculated for {len(payroll_employees)} employees. Total payroll: ${total_payroll:.2f}"
            }
            
            # Format payroll report for display
            payroll_display = []
            payroll_display.append("**📊 PAYROLL REPORT**")
            payroll_display.append("=" * 50)
            
            for emp in payroll_employees:
                payroll_display.append(f"\n**{emp['name']}**")
                payroll_display.append(f"• Regular: {emp['regular_hours']}h × ${emp['payrate']:.2f} = ${emp['regular_pay']:.2f}")
                if emp['overtime_hours'] > 0:
                    payroll_display.append(f"• Overtime: {emp['overtime_hours']}h × ${emp['payrate'] * 1.5:.2f} = ${emp['overtime_pay']:.2f}")
                payroll_display.append(f"• **Total Pay: ${emp['total_pay']:.2f}**")
            
            payroll_display.append("\n" + "=" * 50)
            payroll_display.append(f"**TOTAL PAYROLL: ${total_payroll:.2f}**")
            
            system_prompt = f"""Perfect! You've confirmed the employee data. I've automatically generated the complete payroll report:

{chr(10).join(payroll_display)}

The payroll has been calculated and is ready for processing. You can now:
- Export this data for your payroll system
- Make any final adjustments if needed
- Process payments based on these calculations

Is there anything else you'd like me to help you with?"""
            
            recent_messages = state.messages[-2:] if len(state.messages) > 2 else state.messages
            response = await model.ainvoke([
                {"role": "system", "content": system_prompt},
                *recent_messages
            ])
            
            if not isinstance(response, AIMessage):
                response = AIMessage(content=response.content)
            
            logger.info("✅ Payroll report generated and returning final state")
            
            return {
                "messages": [response],
                "updated_employees": state.temp_merged_list,
                "temp_merged_list": [],  # Clear temp data
                "user_approval": True,
                "current_pay_data": payroll_report,
                "document_processing_done": True
            }
        else:
            logger.info(f"❌ User message '{last_user_message}' not recognized as confirmation")
            # User hasn't confirmed yet, show the merged data and ask for confirmation
            merged_summary = []
            for emp in state.temp_merged_list:
                merged_summary.append(f"• {emp['name']}: {emp['regular_hours']}h regular, {emp['overtime_hours']}h overtime @ ${emp['payrate']}/hr")
            
            system_prompt = f"""I've successfully processed your document and merged the data with existing employee information:

**MERGED EMPLOYEE DATA ({len(state.temp_merged_list)} employees):**
{chr(10).join(merged_summary)}

**Conflict Resolution Applied:**
- Existing employees: Hours updated from document
- New employees: Added from document  
- Pay rates: Preserved existing rates when document showed $0

Please review this data carefully. If everything looks correct, just say "yes" or "confirm" to proceed with payroll generation. If you need any changes, let me know specifically what to modify."""
            
            recent_messages = state.messages[-2:] if len(state.messages) > 2 else state.messages
            response = await model.ainvoke([
                {"role": "system", "content": system_prompt},
                *recent_messages
            ])
            
            if not isinstance(response, AIMessage):
                response = AIMessage(content=response.content)
            
            return {
                "messages": [response],
                "temp_merged_list": state.temp_merged_list  # Keep temp data for confirmation
            }
    
    # PRIORITY 2: Handle interactions when updated_employees is populated
    if state.updated_employees:
        # Create current employee context
        employee_summary = []
        for emp in state.updated_employees:
            emp_dict = emp if isinstance(emp, dict) else emp.model_dump()
            employee_summary.append(f"• {emp_dict['name']}: {emp_dict['regular_hours']}h regular, {emp_dict['overtime_hours']}h overtime @ ${emp_dict['payrate']}/hr")
        
        system_prompt = f"""You are a payroll assistant. The user has confirmed employee data and can now make changes or proceed to payroll.

CURRENT CONFIRMED EMPLOYEES ({len(state.updated_employees)} total):
{chr(10).join(employee_summary)}

You can help the user:
1. Modify individual employee hours, overtime, or pay rates
2. Add new employees 
3. Remove employees
4. Answer questions about the data
5. Proceed to payroll generation when ready

Handle all requests naturally through conversation. When making changes, update the data and provide a JSON summary of the changes."""
        
        recent_messages = state.messages[-5:] if len(state.messages) > 5 else state.messages
        response = await model.ainvoke([
            {"role": "system", "content": system_prompt},
            *recent_messages
        ])
        
        if not isinstance(response, AIMessage):
            response = AIMessage(content=response.content)
        
        return {"messages": [response]}
    
    # PRIORITY 3: If updated_employees is empty and we have updates_list, perform initial merge
    if not state.updated_employees and state.updates_list:
        logger.info("Performing initial merge of updates_list with existing_employees")
        
        merged_list = merge_with_conflict_resolution(state.existing_employees, state.updates_list)
        
        # Create system prompt for showing merged data and asking for confirmation
        merged_summary = []
        for emp in merged_list:
            merged_summary.append(f"• {emp['name']}: {emp['regular_hours']}h regular, {emp['overtime_hours']}h overtime @ ${emp['payrate']}/hr")
        
        system_prompt = f"""You are a payroll assistant. I've just processed a document and merged it with existing employee data.

MERGED EMPLOYEE DATA ({len(merged_list)} employees):
{chr(10).join(merged_summary)}

CONFLICT RESOLUTION APPLIED:
- Existing employees: Preserved with updated hours from document
- New employees: Added from document  
- Hours: Document data takes precedence
- Pay rates: Existing rates preserved when document shows $0

Your task: Present this merged data clearly to the user and ask for their confirmation to proceed. Explain that they can make changes if needed before confirming."""
        
        recent_messages = state.messages[-5:] if len(state.messages) > 5 else state.messages
        
        response = await model.ainvoke([
            {"role": "system", "content": system_prompt},
            *recent_messages
        ])
        
        if not isinstance(response, AIMessage):
            response = AIMessage(content=response.content)
        
        return {
            "messages": [response],
            "temp_merged_list": merged_list  # Store temporarily for confirmation
        }
    
    # PRIORITY 4: Default response for other cases
    response = await model.ainvoke([
        {"role": "system", "content": "You are a payroll assistant. Help the user with their payroll processing needs."},
        *state.messages[-5:]
    ])
    
    if not isinstance(response, AIMessage):
        response = AIMessage(content=response.content)
    
    return {"messages": [response]}


# Routing functions
def route_init(state: State) -> Literal["vlm_processor", "update_agent"]:
    """Route from init: go to vlm_processor if document uploaded, otherwise update_agent."""
    if state.document_uploaded and not state.document_processing_done:
        route = "vlm_processor"
    else:
        route = "update_agent"
    
    logger.info(f"=== ROUTING: init_node -> {route} ===")
    return route


def route_vlm(state: State) -> Literal["update_agent"]:
    """Route from vlm_processor: always go to update_agent."""
    logger.info("=== ROUTING: vlm_processor -> update_agent ===")
    return "update_agent"


def route_update(state: State) -> Literal["__end__"]:
    """Route from update_agent: always end (user can continue conversation)."""
    logger.info("=== ROUTING: update_agent -> __end__ ===")
    return "__end__"
    

# Build the workflow graph
builder = StateGraph(State)
builder.add_node("init", init_node)
builder.add_node("vlm_processor", vlm_processor)
builder.add_node("update_agent", update_agent)

builder.add_edge("__start__", "init")
builder.add_conditional_edges("init", route_init)
builder.add_conditional_edges("vlm_processor", route_vlm)
builder.add_conditional_edges("update_agent", route_update)

graph = builder.compile()
