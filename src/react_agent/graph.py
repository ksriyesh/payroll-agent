"""Simple 3-agent payroll system."""

import logging
import json
import pprint
from typing import Dict, List, Literal, cast, Type
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from .configuration import Configuration
from .state import State, EmployeeData, PayrollReport
from .utils import load_chat_model
from .tools import merge_employees
from .utils import load_previous_payperiod_data
logger = logging.getLogger(__name__)


def log_state_beautifully(state: State, node_name: str):
    """Log the state in a beautiful and structured manner."""
    logger.info(f"\n{'='*20} STATE AFTER {node_name} {'='*20}")
    
    # Create a dictionary representation of the state for pretty printing
    state_dict = {
        'messages_count': len(state.messages),
        'document_uploaded': state.document_uploaded,
        'document_processing_done': state.document_processing_done,
        'user_approval': state.user_approval,
        'trigger_payroll': state.trigger_payroll,
        'existing_employees': state.existing_employees,
        'updated_employees': state.updated_employees,
        'updates_list': state.updates_list,
        'file_data_exists': bool(state.file_data),
        'file_path': state.file_path,
        'file_type': state.file_type,
    }
    
    # Add message details
    if state.messages:
        last_message = state.messages[-1]
        state_dict["messages"] = state.messages
        state_dict['last_message'] = {
            'type': type(last_message).__name__,
            'has_tool_calls': hasattr(last_message, 'tool_calls') and bool(last_message.tool_calls),
        }
        if hasattr(last_message, 'content'):
            preview = str(last_message.content) + '...' if len(str(last_message.content)) > 50 else str(last_message.content)
            state_dict['last_message']['content_preview'] = preview
    
    # Pretty print the state dictionary
    formatted_state = pprint.pformat(state_dict, indent=2)
    logger.info(f"\n{formatted_state}\n{'='*50}\n")



async def init_node(state: State) -> Dict:
    """Initialize the workflow and check for file inputs in messages."""
    logger.info(f"=== ENTERING init_node ===")
    logger.info(f"State object ID: {id(state)}")
    logger.info(f"State messages count: {len(state.messages)}")
    logger.info(f"Document uploaded: {state.document_uploaded}")
    logger.info(f"File data exists: {bool(state.file_data)}")
    
    result = {}
    
    # Check if there's a file in the messages
    logger.info(f"Checking for file in messages: {state.messages}")
    if state.messages and len(state.messages) > 0:
        last_message = state.messages[-1]
        
        # Check if the message has content and if it's a list (multimodal content)
        if hasattr(last_message, 'content') and isinstance(last_message.content, list):
            # Look for file or image type content
            for content_item in last_message.content:
                if isinstance(content_item, dict) and content_item.get('type') in ['file', 'image']:
                    # Found a file, set document_uploaded to True
                    state.document_uploaded = True
                    logger.info("File detected in messages, setting document_uploaded=True")
                    break
    
    
    # Check if existing_employees is empty, if so load previous pay period data
    if not state.existing_employees:
        logger.info("No existing employees found, loading previous pay period data")
        result['existing_employees'] = await load_previous_payperiod_data()
    
    # Log the state beautifully after processing
    log_state_beautifully(state, "init_node")
    
    return result

    

async def vlm_processor(state: State) -> Dict:
    """
    Process document content with VLM, streamlining file handling and data extraction.
    """
    logger.info("=== ENTERING vlm_processor ===")
    
    # Import necessary libraries and modules
    import os
    import base64
    import tempfile
    import json
    import re
    from .utils import process_document_with_files_api
    from .prompts import VLM_DOC_PROCESSOR_PROMPT
    from .state import EmployeeData
    from pydantic import BaseModel, Field
    from typing import List
    
    config = Configuration.from_context()

    # Define a Pydantic model for structured data extraction
    class EmployeeList(BaseModel):
        """Model for structured extraction of employee data from documents."""
        employees: List[EmployeeData] = Field(
            ..., 
            description="List of extracted employees with their payroll information"
        )

    # If updates_list is already populated, no need to re-process
    if state.updates_list:
        logger.info(f"Using pre-populated updates_list with {len(state.updates_list)} employees.")
        return {"updates_list": state.updates_list, "document_processing_done": True}

    temp_file_path = None
    try:
        # If file_data is available, it's the source of truth. Create a temp file to ensure a valid path.
        if state.file_data:
            file_extension = ".jpg"  # Default
            if state.file_type:
                if "pdf" in state.file_type.lower():
                    file_extension = ".pdf"
                elif "png" in state.file_type.lower():
                    file_extension = ".png"
            
            # Create a temporary file to store the uploaded data
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                base64_data = state.file_data.split(",", 1)[-1]
                file_content = base64.b64decode(base64_data)
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            state.file_path = temp_file_path
            logger.info(f"Created temporary file from file_data: {state.file_path}")

        # Process the document using the files API if a file path is available
        if state.file_path:
            logger.info(f"Processing document using files API: {state.file_path}")
            result = await process_document_with_files_api(
                file_path=state.file_path,
                prompt=VLM_DOC_PROCESSOR_PROMPT.strip(),
                model_name=config.vlm_model,
                vlm_system_prompt=config.vlm_system_prompt,
                output_schema=EmployeeList
            )
            logger.info(f"Files API processing result: {result}")

            # Check for structured data in the result
            if result and "structured_data" in result and "employees" in result["structured_data"]:
                employees_data = result["structured_data"]["employees"]
                logger.info(f"Successfully extracted {len(employees_data)} employees via files API.")
                
                # Convert Pydantic models to dicts for state compatibility
                employee_dicts = [emp.model_dump() if hasattr(emp, 'model_dump') else emp for emp in employees_data]
                return {"updates_list": employee_dicts, "document_processing_done": True}
            
            # Fallback to content if structured data is not found
            if result and "content" in result:
                state.document_content = result["content"]
                logger.info("Got text content from files API, will attempt to parse.")
        
        # If processing falls through or started with document_content, parse it
        if state.document_content:
            logger.info("Attempting to parse document_content for employee data.")
            
            # Try to find a JSON object within the text content
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', state.document_content, re.DOTALL)
            if json_match:
                try:
                    parsed_data = json.loads(json_match.group(1))
                    if "employees" in parsed_data:
                        logger.info(f"Successfully parsed JSON from document_content with {len(parsed_data['employees'])} employees.")
                        return {"updates_list": parsed_data["employees"], "document_processing_done": True}
                except json.JSONDecodeError:
                    logger.warning("Found a JSON-like block, but failed to parse it.")

            # If parsing fails, use the model for a final extraction attempt
            logger.info("Falling back to model-based extraction from text content.")
            model = load_chat_model(fully_specified_name=config.vlm_model)
            structured_model = model.with_structured_output(EmployeeList)
            response = await structured_model.ainvoke([
                {"role": "system", "content": config.vlm_system_prompt},
                {"role": "user", "content": f"Extract payroll data from the following text: {state.document_content}"}
            ])
            
            if response.employees:
                logger.info(f"Successfully extracted {len(response.employees)} employees via model fallback.")
                employee_dicts = [emp.model_dump() for emp in response.employees]
                return {"updates_list": employee_dicts, "document_processing_done": True , "document_uploaded":False}

        # If no data could be extracted, return an empty list
        logger.warning("Could not extract any employee data from the document.")
        return {"updates_list": [], "document_processing_done": True}

    except Exception as e:
        logger.error(f"An unexpected error occurred in vlm_processor: {str(e)}")
        return {"updates_list": [], "document_processing_done": True}
    
    finally:
        # Clean up the temporary file if one was created
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info(f"Removed temporary file: {temp_file_path}")



async def update_agent(state: State) -> Dict:
    """Update the agent with the latest state."""
    logger.info(f"=== ENTERING update_agent ===")
    logger.info(f"State object ID: {id(state)}")
    logger.info(f"State messages count: {len(state.messages)}")
    if state.messages:
        logger.info(f"Last message type: {type(state.messages[-1]).__name__}")
        if hasattr(state.messages[-1], 'content'):
            logger.info(f"Last message content preview: {str(state.messages[-1].content)[:50]}...")
    
    # Get configuration
    config = Configuration.from_context()
    
    # If user has already approved and we're generating payroll
    if state.user_approval and state.trigger_payroll:
        # Call payroll agent to generate the report
        payroll_result = await payroll_agent(state)
        return {
            "messages": [AIMessage(content="✅ Payroll has been generated successfully! You can view the complete report above.")],
            **payroll_result
        }
    
    # Check if we have updates_list from document processing and move to updated_employees
    if state.updates_list and not state.updated_employees:
        # Convert updates_list to updated_employees
        updated_employees = []
        for emp in state.updates_list:
            if isinstance(emp, dict):
                updated_employees.append(EmployeeData(**emp))
            else:
                updated_employees.append(emp)
        
        return {
            "updated_employees": updated_employees,
            "updates_list": [],  # Clear updates_list since we've processed it
            "messages": [AIMessage(content=f"📄 Document processed! Found {len(updated_employees)} employees. I'll now help you review and merge this data with your existing employees.")]
        }
    
    # Load the model
    model = load_chat_model(fully_specified_name=config.react_model)
    
    # Process the conversation with the user
    current_messages = state.messages
    recent_messages = current_messages[-10:] if len(current_messages) > 10 else current_messages
    
    # Create detailed context about all employee data
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
    
    # Add approval status
    context_parts.append(f"\nUSER APPROVAL: {state.user_approval}")
    
    # Create system prompt with detailed instructions
    system_prompt = (
        "You are a payroll assistant. Help users process employee payroll data. "
        "Your main tasks:\n"
        "1. Review existing and updated employee data\n"
        "2. Help merge/resolve conflicts automatically\n"
        "3. Present a summary and ask for confirmation\n"
        "4. Wait for explicit user confirmation before proceeding to payroll\n\n"
        "When user confirms (says 'yes', 'confirm', 'approve', 'proceed', etc.), "
        "use the update_state tool to update existing_employees with merged data.\n\n"
        "Current employee data:\n" + "\n".join(context_parts)
    )
    
    # Check if user is providing approval in their last message
    last_user_message = ""
    for msg in reversed(recent_messages):
        if hasattr(msg, "type") and msg.type == "human":
            last_user_message = msg.content.lower()
            break
    
    # Look for confirmation keywords
    confirmation_keywords = ["confirm", "approved", "looks good", "correct", "yes", "proceed", "generate payroll"]
    is_confirmed = any(keyword in last_user_message for keyword in confirmation_keywords)
    
    # If user confirmed and we have data to merge
    if is_confirmed and (state.updated_employees or state.existing_employees):
        from .tools import merge_employees
        
        # Merge employees
        if state.existing_employees and state.updated_employees:
            merged_employees = merge_employees(state.existing_employees, state.updated_employees)
        elif state.updated_employees:
            merged_employees = state.updated_employees
        else:
            merged_employees = state.existing_employees
        
        # Return with approval flags set and trigger payroll
        return {
            "user_approval": True,
            "trigger_payroll": True,
            "existing_employees": [emp.model_dump() if hasattr(emp, 'model_dump') else emp for emp in merged_employees],
            "messages": [AIMessage(content="✅ Thank you for confirming! Processing payroll now...")]
        }
    
    # For regular conversation, invoke the model
    response = await model.ainvoke([
        {"role": "system", "content": system_prompt},
        *recent_messages
    ])
    
    # Convert response to AIMessage if needed
    if not isinstance(response, AIMessage):
        response = AIMessage(content=response.content)
    
    # Log the state beautifully after processing
    log_state_beautifully(state, "update_agent")
    
    return {"messages": [response]}


async def payroll_agent(state: State) -> Dict:
    """Generate payroll report with manual calculations."""
    logger.info(f"=== ENTERING payroll_agent ===")
    logger.info(f"State object ID: {id(state)}")
    logger.info(f"State messages count: {len(state.messages)}")
    logger.info(f"Existing employees count: {len(state.existing_employees)}")
    logger.info(f"Updated employees count: {len(state.updated_employees)}")
    
    from .state import PayrollReport, PayrollEmployee
    
    # Merge existing and updated employees
    employees = merge_employees(state.existing_employees, state.updated_employees)
    
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
    
    # Create the payroll message
    payroll_message = AIMessage(content=f"✅ Payroll calculated: {len(payroll_employees)} employees, Total: ${total_payroll:.2f}")
    
    # Log the state beautifully after processing
    log_state_beautifully(state, "payroll_agent")
    
    return {
        "current_pay_data": report_dict,
        "messages": [payroll_message]
    }


async def tool_executor(state: State) -> Dict:
    """Execute tools."""
    logger.info(f"=== ENTERING tool_executor ===")
    logger.info(f"State object ID: {id(state)}")
    logger.info(f"State messages count: {len(state.messages)}")
    if state.messages:
        logger.info(f"Last message type: {type(state.messages[-1]).__name__}")
        if hasattr(state.messages[-1], 'tool_calls') and state.messages[-1].tool_calls:
            logger.info(f"Tool calls count: {len(state.messages[-1].tool_calls)}")
            for i, tool_call in enumerate(state.messages[-1].tool_calls):
                logger.info(f"Tool call {i}: {tool_call['name']}")
    
    from langchain_core.messages import ToolMessage
    
    tool_messages = []
    updates = {}
    
    for tool_call in state.messages[-1].tool_calls:
        if tool_call["name"] == "update_state":
            try:
                target = tool_call["args"]["target_list"]
                emp_data = tool_call["args"]["employees"]
                
                # Convert to EmployeeData objects
                employees = [EmployeeData(**emp) for emp in emp_data]
                
                # Update state - Remove automatic approval logic
                if target == "updated_employees":
                    updates["updated_employees"] = employees
                elif target == "existing_employees":
                    updates["existing_employees"] = employees
                
                tool_message = ToolMessage(
                    content=f"Updated {target} with {len(employees)} employees",
                    tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_message)
                
            except KeyError as e:
                error_msg = f"Error in tool call: missing {e} in arguments"
                tool_message = ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_message)
            except Exception as e:
                error_msg = f"Error processing tool call: {str(e)}"
                tool_message = ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_message)
    
    # Log the state beautifully after processing
    log_state_beautifully(state, "tool_executor")
    
    return {"messages": tool_messages, **updates}


# Simple routing
def route_init(state: State) -> Literal["vlm_processor", "update_agent"]:
    route = "vlm_processor" if state.document_uploaded else "update_agent"
    logger.info(f"=== ROUTING: init_node -> {route} ===")
    logger.info(f"State object ID: {id(state)}")
    logger.info(f"document_uploaded: {state.document_uploaded}")
    
    # Log the state beautifully after routing decision
    log_state_beautifully(state, "route_init")
    
    return route

def route_vlm(state: State) -> Literal["update_agent"]:
    logger.info(f"=== ROUTING: vlm_processor -> update_agent ===")
    logger.info(f"State object ID: {id(state)}")
    
    # Log the state beautifully after routing decision
    log_state_beautifully(state, "route_vlm")
    
    return "update_agent"

def route_update(state: State) -> Literal["tool_executor", "update_agent", "__end__"]:
    # If user has approved and payroll is triggered, end the workflow
    if state.user_approval and state.trigger_payroll:
        logger.info(f"=== ROUTING: update_agent -> __end__ (payroll completed) ===")
        return "__end__"
    
    # Check if the last message has tool calls
    has_tool_calls = False
    if state.messages and hasattr(state.messages[-1], 'tool_calls') and state.messages[-1].tool_calls:
        has_tool_calls = True
    
    # Check if we're in a state where we should be waiting for user input
    waiting_for_approval = (
        # We have existing or updated employees (data to process)
        (state.existing_employees or state.updated_employees) and
        # User hasn't approved yet
        not state.user_approval and
        # We're not triggering payroll
        not state.trigger_payroll and
        # We have messages (active conversation)
        state.messages and
        # The last message is from the assistant (asking for confirmation)
        hasattr(state.messages[-1], 'content') and 
        isinstance(state.messages[-1], AIMessage)
    )
    
    # Determine the route
    if has_tool_calls:
        route = "tool_executor"
    elif waiting_for_approval:
        # End the workflow to wait for user input - this prevents infinite loops
        route = "__end__"
    else:
        # End the workflow
        route = "__end__"
    
    logger.info(f"=== ROUTING: update_agent -> {route} ===")
    logger.info(f"State object ID: {id(state)}")
    logger.info(f"Has tool calls: {has_tool_calls}")
    logger.info(f"User approval: {state.user_approval}")
    logger.info(f"Trigger payroll: {state.trigger_payroll}")
    logger.info(f"Waiting for approval: {waiting_for_approval}")
    if has_tool_calls and state.messages:
        logger.info(f"Tool calls count: {len(state.messages[-1].tool_calls)}")
    
    # Log the state beautifully after routing decision
    log_state_beautifully(state, "route_update")
    
    return route

def route_tools(state: State) -> Literal["update_agent"]:
    # Always route back to update_agent since payroll_agent is removed
    logger.info(f"=== ROUTING: tool_executor -> update_agent ===")
    logger.info(f"State object ID: {id(state)}")
    
    # Log the state beautifully after routing decision
    log_state_beautifully(state, "route_tools")
    
    return "update_agent"

# Payroll agent routing function removed


# Build graph
builder = StateGraph(State)
builder.add_node("init", init_node)
builder.add_node("vlm_processor", vlm_processor)
builder.add_node("update_agent", update_agent)
builder.add_node("tool_executor", tool_executor)

builder.add_edge("__start__", "init")
builder.add_conditional_edges("init", route_init)
builder.add_conditional_edges("vlm_processor", route_vlm)
builder.add_conditional_edges("update_agent", route_update)
builder.add_conditional_edges("tool_executor", route_tools)

graph = builder.compile()
