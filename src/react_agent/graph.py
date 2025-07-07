"""Define the graph for the payroll document parsing agent."""

import os
import base64
import re
from datetime import datetime, timezone
from typing import Literal, Dict, Any
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages

from src.react_agent.configuration import Configuration
from src.react_agent.state import State, DocumentInfo
from src.react_agent.utils import load_chat_model

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_file_path(message_content: str) -> Dict[str, Any]:
    """Detect file path in message content.
    
    Returns:
        Dict with file_path, file_name, file_type, or None if no file detected
    """
    if not isinstance(message_content, str):
        return None
    
    # Look for file path pattern: process_document:/path/to/file.ext
    file_pattern = r'process_document:([^\s]+)'
    match = re.search(file_pattern, message_content)
    
    if match:
        file_path = match.group(1)
        file_name = os.path.basename(file_path)
        file_type = file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
        
        return {
            "file_path": file_path,
            "file_name": file_name,
            "file_type": file_type
        }
    
    return None


async def vlm_processing_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """VLM processing node - automatically processes uploaded files with vision analysis.
    
    This node directly processes documents and extracts text/positions using VLM.
    """
    logger.info("🔄 VLM Processing Node - Starting")
    logger.debug(f"State: document_uploaded={state.document_uploaded}, file_path={state.file_path}")
    
    try:
        # Check if there's a file to process
        if not state.file_path:
            logger.warning("❌ No file path provided for VLM processing")
            return {
                "messages": [AIMessage(content="No file provided for processing.")],
                "vlm_processing_complete": False
            }
        
        logger.info(f"📄 Processing file: {state.file_path}")
        
        # Import VLM processing function
        from src.react_agent.tools import process_document_with_vlm
        
        logger.debug("🧠 Calling VLM processing function")
        
        # DIRECTLY process with VLM in this node
        vlm_result = await process_document_with_vlm(
            file_path=state.file_path,
            context_query=state.context_query or "Extract all payroll information from this document."
        )
        
        logger.debug(f"VLM Result keys: {list(vlm_result.keys())}")
        logger.debug(f"VLM Success: {vlm_result.get('success', False)}")
        
        if not vlm_result.get("success", False):
            logger.error(f"❌ VLM processing failed: {vlm_result.get('error', 'Unknown error')}")
            return {
                "messages": [AIMessage(content=f"Error processing document: {vlm_result.get('error', 'Unknown error')}")]
            }
        
        logger.info("✅ VLM processing completed successfully")
        logger.debug(f"Extracted text length: {len(vlm_result.get('text_data', {}).get('full_text', ''))}")
        logger.debug(f"Employees found: {len(vlm_result.get('employees', []))}")
        
        # Update state with VLM results
        state_updates = {
            "vlm_processing_complete": True,
            "document_info": vlm_result.get("document_info"),
            "text_data": vlm_result.get("text_data", {}),
            "extracted_text": vlm_result.get("extracted_text", "")
        }
        
        if vlm_result.get("employees"):
            state_updates["employees"] = vlm_result["employees"]
            logger.info(f"👥 Found {len(vlm_result['employees'])} employees")
        
        # Create message to pass VLM results to react agent
        doc_info = vlm_result.get('document_info')
        doc_filename = doc_info.filename if doc_info else 'Unknown'
        text_data = vlm_result.get('text_data', {})
        full_text = text_data.get('full_text', '') if isinstance(text_data, dict) else ''
        
        vlm_data_message = HumanMessage(
            content=f"""
Document processed successfully! 📄

VLM Analysis Results:
- Document: {doc_filename}
- Text extracted: {len(full_text)} characters
- Pages processed: {vlm_result.get('total_pages', 1)}
- Employees found: {len(vlm_result.get('employees', []))}
- Needs clarification: {vlm_result.get('needs_clarification', False)}

Text Preview:
{full_text[:500]}...

VLM Analysis:
{vlm_result.get('extracted_text', '')[:1000]}...

Employees Data:
{vlm_result.get('employees', [])}
"""
        )
        
        state_updates["messages"] = [vlm_data_message]
        logger.info("📤 VLM results prepared for ReAct agent")
        
        return state_updates
        
    except Exception as e:
        logger.error(f"❌ Error in VLM processing node: {str(e)}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"Error in VLM processing: {str(e)}")]
        }


async def react_agent_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """React agent node - processes VLM results and allows interactive payroll editing.
    
    This agent:
    1. Initially processes VLM results and shows extracted data
    2. Allows user to interact and modify payroll details
    3. Handles requests to update employee information
    4. Provides final JSON when user is satisfied
    """
    logger.info("🤖 ReAct Agent Node - Starting")
    logger.debug(f"VLM processing complete: {state.vlm_processing_complete}")
    logger.debug(f"Employees in state: {len(state.employees) if state.employees else 0}")
    
    try:
        # Load the chat model
        model = load_chat_model(config)
        logger.debug(f"Model loaded: {model}")
        
        # Get the latest user message
        latest_message = state.messages[-1] if state.messages else None
        user_input = latest_message.content if latest_message else ""
        
        # Check if user is requesting to finalize/export data
        if any(keyword in user_input.lower() for keyword in ["export", "finalize", "final json", "done", "complete"]):
            logger.info("📊 User requesting final JSON export")
            return await export_final_payroll_data(state)
        
        # Check if user is making payroll modifications
        if state.vlm_processing_complete and any(keyword in user_input.lower() for keyword in 
                                                ["update", "change", "modify", "edit", "fix", "correct", "set"]):
            logger.info("✏️ User requesting payroll modifications")
            return await handle_payroll_modifications(state, model, user_input)
        
        # Initial VLM processing complete - show extracted data
        if state.vlm_processing_complete and state.employees:
            logger.info("✅ Initial VLM processing complete - showing extracted data")
            
            # Create summary of extracted data
            employee_summary = []
            for i, emp in enumerate(state.employees, 1):
                summary = f"**Employee {i}: {emp.name}**\n"
                if emp.employee_id:
                    summary += f"  - ID: {emp.employee_id}\n"
                if emp.position:
                    summary += f"  - Position: {emp.position}\n"
                if emp.pay_rate:
                    summary += f"  - Pay Rate: ${emp.pay_rate:.2f}\n"
                if emp.hours_worked:
                    summary += f"  - Hours: {emp.hours_worked}\n"
                if emp.overtime_hours:
                    summary += f"  - Overtime: {emp.overtime_hours}\n"
                if emp.gross_pay:
                    summary += f"  - Gross Pay: ${emp.gross_pay:.2f}\n"
                if emp.deductions:
                    summary += f"  - Deductions: ${emp.deductions:.2f}\n"
                if emp.net_pay:
                    summary += f"  - Net Pay: ${emp.net_pay:.2f}\n"
                if emp.pay_period:
                    summary += f"  - Pay Period: {emp.pay_period}\n"
                
                employee_summary.append(summary)
            
            response_message = f"""
✅ **Payroll Data Successfully Extracted!**

I've processed your payroll document and extracted the following information:

**Document Details:**
- File: {state.document_info.filename if state.document_info else 'Unknown'}
- Pages: {state.document_info.pages if state.document_info else 'Unknown'}
- Processing: Complete ✅

**Extracted Employee Data:**
{chr(10).join(employee_summary)}

🔧 **You can now interact with me to:**
- **Modify any employee details**: "Update Alice's pay rate to $25/hour"
- **Add missing information**: "Set Bob's employee ID to EMP001"
- **Fix incorrect data**: "Change Clara's deductions to $150"
- **Add new employees**: "Add employee David with $30/hour rate"
- **Remove employees**: "Remove employee Alice"
- **Calculate missing values**: "Calculate net pay for all employees"

💬 **Just tell me what you'd like to change, and I'll update the payroll data accordingly!**

📊 When you're satisfied with the data, say "export" or "finalize" to get the final JSON.
"""
            
            logger.info(f"📊 Showing initial data for {len(state.employees)} employees")
            return {
                "messages": [AIMessage(content=response_message)],
                "extraction_complete": False  # Keep interaction open
            }
            
        elif state.vlm_processing_complete and not state.employees:
            logger.info("❓ VLM processed but no employee data - asking clarification")
            
            clarification_message = f"""
I've processed your payroll document, but I need some help extracting the employee information.

**What I found:**
- Document text: {len(state.text_data.get('full_text', '')) if state.text_data else 0} characters
- Analysis: {state.extracted_text[:200] if state.extracted_text else 'No analysis available'}...

**Please help me by:**
1. **Telling me about employees**: "There are 3 employees: Alice, Bob, and Clara"
2. **Providing pay details**: "Alice earns $25/hour, worked 40 hours"
3. **Sharing specific information**: "This is a bi-weekly payroll for March 2024"

💬 **You can also manually add employees**: "Add employee Alice with $25/hour rate, worked 40 hours"

I'll use your input to build the payroll data, and you can modify it as needed!
"""
            
            logger.info("❓ Requesting clarification from user")
            return {
                "messages": [AIMessage(content=clarification_message)],
                "extraction_complete": False
            }
        
        # Handle general conversation and payroll data building
        elif user_input and not state.vlm_processing_complete:
            logger.info("💬 Handling general conversation")
            
            # Create a prompt for general payroll assistance
            system_prompt = """You are a helpful payroll processing assistant. Help users with:
1. Uploading payroll documents for processing
2. Understanding payroll data extraction
3. Answering payroll-related questions
4. Guiding them through the process

Be friendly and informative."""
            
            messages = [
                HumanMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]
            
            response = await model.ainvoke(messages)
            
            return {
                "messages": [AIMessage(content=response.content)],
                "extraction_complete": False
            }
        
        else:
            logger.info("🔄 Default state - prompting for document upload")
            return {
                "messages": [AIMessage(content="Please upload a payroll document to begin processing, or ask me any questions about payroll data extraction!")],
                "extraction_complete": False
            }
    
    except Exception as e:
        logger.error(f"❌ Error in ReAct agent node: {str(e)}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"Error in agent processing: {str(e)}")],
            "extraction_complete": False
        }


async def handle_payroll_modifications(state: State, model, user_input: str) -> Dict[str, Any]:
    """Handle user requests to modify payroll data."""
    logger.info("✏️ Processing payroll modification request")
    
    # Create a prompt for the LLM to understand the modification request
    current_employees = [emp.model_dump() for emp in state.employees]
    
    modification_prompt = f"""You are helping a user modify payroll data. 

CURRENT PAYROLL DATA:
{current_employees}

USER REQUEST: {user_input}

TASK: Understand what the user wants to modify and provide the updated payroll data.

RESPOND WITH:
1. A brief acknowledgment of what you're changing
2. The updated employee data in the same JSON format

EXAMPLES:
- "Update Alice's pay rate to $25" → Change Alice's pay_rate to 25.0
- "Set Bob's employee ID to EMP001" → Change Bob's employee_id to "EMP001"
- "Add employee David with $30/hour" → Add new employee David with pay_rate 30.0
- "Remove employee Alice" → Remove Alice from the list
- "Calculate net pay for all" → Calculate net_pay = gross_pay - deductions for all employees

Return the response in this format:
ACKNOWLEDGMENT: [what you're doing]
UPDATED_DATA: [complete updated JSON array of all employees]
"""
    
    try:
        response = await model.ainvoke([HumanMessage(content=modification_prompt)])
        response_text = response.content
        
        # Parse the response to extract acknowledgment and updated data
        if "ACKNOWLEDGMENT:" in response_text and "UPDATED_DATA:" in response_text:
            parts = response_text.split("UPDATED_DATA:")
            acknowledgment = parts[0].replace("ACKNOWLEDGMENT:", "").strip()
            updated_data_text = parts[1].strip()
            
            # Clean JSON
            if updated_data_text.startswith('```json'):
                updated_data_text = updated_data_text[7:]
            if updated_data_text.endswith('```'):
                updated_data_text = updated_data_text[:-3]
            updated_data_text = updated_data_text.strip()
            
            # Parse updated employee data
            import json
            from src.react_agent.state import EmployeePayInfo
            
            try:
                updated_employees_data = json.loads(updated_data_text)
                
                # Convert to EmployeePayInfo objects
                updated_employees = []
                for emp_data in updated_employees_data:
                    # Calculate net pay if missing
                    if emp_data.get('net_pay') is None and emp_data.get('gross_pay') and emp_data.get('deductions'):
                        emp_data['net_pay'] = emp_data['gross_pay'] - emp_data['deductions']
                    
                    employee = EmployeePayInfo(
                        employee_id=emp_data.get('employee_id'),
                        name=emp_data.get('name', ''),
                        pay_rate=emp_data.get('pay_rate'),
                        hours_worked=emp_data.get('hours_worked'),
                        overtime_hours=emp_data.get('overtime_hours'),
                        gross_pay=emp_data.get('gross_pay'),
                        deductions=emp_data.get('deductions'),
                        net_pay=emp_data.get('net_pay'),
                        pay_period=emp_data.get('pay_period'),
                        position=emp_data.get('position')
                    )
                    updated_employees.append(employee)
                
                # Create response message
                response_message = f"""
✅ **{acknowledgment}**

**Updated Employee Data:**
"""
                
                for i, emp in enumerate(updated_employees, 1):
                    response_message += f"\n**Employee {i}: {emp.name}**"
                    if emp.employee_id:
                        response_message += f"\n  - ID: {emp.employee_id}"
                    if emp.position:
                        response_message += f"\n  - Position: {emp.position}"
                    if emp.pay_rate:
                        response_message += f"\n  - Pay Rate: ${emp.pay_rate:.2f}"
                    if emp.hours_worked:
                        response_message += f"\n  - Hours: {emp.hours_worked}"
                    if emp.overtime_hours:
                        response_message += f"\n  - Overtime: {emp.overtime_hours}"
                    if emp.gross_pay:
                        response_message += f"\n  - Gross Pay: ${emp.gross_pay:.2f}"
                    if emp.deductions:
                        response_message += f"\n  - Deductions: ${emp.deductions:.2f}"
                    if emp.net_pay:
                        response_message += f"\n  - Net Pay: ${emp.net_pay:.2f}"
                    if emp.pay_period:
                        response_message += f"\n  - Pay Period: {emp.pay_period}"
                    response_message += "\n"
                
                response_message += "\n💬 **What else would you like to modify?** You can continue making changes or say 'export' to get the final JSON."
                
                logger.info(f"✅ Successfully updated employee data: {len(updated_employees)} employees")
                
                return {
                    "messages": [AIMessage(content=response_message)],
                    "employees": updated_employees,
                    "extraction_complete": False
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parsing error: {e}")
                return {
                    "messages": [AIMessage(content=f"Sorry, I had trouble parsing the updated data. Please try rephrasing your request.")],
                    "extraction_complete": False
                }
        
        else:
            # Fallback response
            return {
                "messages": [AIMessage(content=f"I understand you want to modify the payroll data, but I need more specific details. Please try something like 'Update Alice's pay rate to $25' or 'Add employee David with $30/hour'.")],
                "extraction_complete": False
            }
    
    except Exception as e:
        logger.error(f"❌ Error handling payroll modifications: {str(e)}")
        return {
            "messages": [AIMessage(content=f"Sorry, I encountered an error while processing your request: {str(e)}")],
            "extraction_complete": False
        }


async def export_final_payroll_data(state: State) -> Dict[str, Any]:
    """Export final payroll data as JSON."""
    logger.info("📊 Exporting final payroll data")
    
    if not state.employees:
        return {
            "messages": [AIMessage(content="No employee data to export. Please process a document or add employees first.")],
            "extraction_complete": False
        }
    
    # Create final JSON response
    employee_data = [emp.model_dump() for emp in state.employees]
    
    json_response = {
        "status": "success",
        "message": "Payroll data finalized and exported",
        "document_info": state.document_info.model_dump() if state.document_info else {},
        "employees": employee_data,
        "extraction_complete": True,
        "export_timestamp": datetime.now().isoformat()
    }
    
    import json
    json_str = json.dumps(json_response, indent=2)
    
    response_message = f"""
✅ **Final Payroll Data Exported**

Your payroll data has been finalized and is ready for use:

**Summary:**
- **Employees**: {len(state.employees)}
- **Document**: {state.document_info.filename if state.document_info else 'Manual Entry'}
- **Export Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Final JSON:**
```json
{json_str}
```

🎉 **Your payroll data is now ready for integration or further processing!**
"""
    
    logger.info(f"📊 Exported final data for {len(state.employees)} employees")
    
    return {
        "messages": [AIMessage(content=response_message)],
        "extraction_complete": True
    }


def route_agent_output(state: State) -> Literal[END, "react_agent"]:
    """Determine the next step based on the agent's output.
    
    Continue conversation unless extraction is explicitly marked complete.
    """
    logger.info("🔀 Routing after agent output")
    logger.debug(f"Extraction complete: {state.extraction_complete}")
    
    if state.extraction_complete:
        logger.info("✅ Extraction complete - ending conversation")
        return END
    else:
        logger.info("🔄 Continuing conversation - routing back to agent")
        return "react_agent"


def route_vlm_or_agent(state: State) -> Literal["vlm_processing", "react_agent"]:
    """Route to VLM processing if file upload detected, otherwise to react agent."""
    logger.info("🔀 Routing decision")
    logger.debug(f"Document uploaded: {state.document_uploaded}")
    logger.debug(f"VLM processing complete: {state.vlm_processing_complete}")
    logger.debug(f"File path: {state.file_path}")
    
    if state.document_uploaded and not state.vlm_processing_complete:
        logger.info("📄 Routing to VLM processing (new document)")
        return "vlm_processing"
    else:
        logger.info("🤖 Routing to ReAct agent")
        return "react_agent"


# Create the main workflow graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("vlm_processing", vlm_processing_node)
workflow.add_node("react_agent", react_agent_node)

# Define the workflow edges
workflow.add_conditional_edges(START, route_vlm_or_agent)
workflow.add_conditional_edges("vlm_processing", lambda state: "react_agent")  # Always go to agent after VLM
workflow.add_conditional_edges("react_agent", route_agent_output)  # Either continue conversation or end

# Compile the graph
graph = workflow.compile()

# Make the graph available for import
__all__ = ["graph"]
