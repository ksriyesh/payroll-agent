"""FastAPI backend server for the payroll agent."""

import os
import base64
import io
import math
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the payroll agent components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.react_agent.graph import graph
from src.react_agent.configuration import Configuration
from src.react_agent.state import EmployeeData, PayrollReport, PayrollEmployee
from src.react_agent.tools import merge_employees
from src.react_agent.graph import graph as agent_graph

def get_agent_graph():
    """Get the agent graph from the react_agent module."""
    return agent_graph

# Create FastAPI app
app = FastAPI(title="Payroll Agent API", 
              description="Backend API for the Payroll Agent application",
              version="1.0.0")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class EmployeeDataRequest(BaseModel):
    name: str
    regular_hours: float
    overtime_hours: float
    payrate: float

class EmployeeListRequest(BaseModel):
    employees: List[EmployeeDataRequest]

class ChatRequest(BaseModel):
    content: str
    existing_employees: List[EmployeeDataRequest] = []
    updated_employees: List[EmployeeDataRequest] = []
    current_employees: List[EmployeeDataRequest] = []
    user_approval: bool = False
    trigger_payroll: bool = False
    current_pay_data: Optional[Dict[str, Any]] = None
    file_data: Optional[str] = None  # Base64 encoded file data
    file_path: Optional[str] = None  # Original file name/path
    file_type: Optional[str] = None  # MIME type of the file

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    workflow_state: Dict[str, Any] = {}

def get_agent_config():
    """Get the standard configuration for the agent."""
    from src.react_agent.configuration import Configuration
    config = Configuration.from_context()
    
    return {
        "configurable": {
            "vlm_model": "openai/gpt-4o",
            "react_model": "openai/gpt-4o",
            "max_employees": 1000,
            "currency_symbol": "$",
            "default_overtime_multiplier": 1.5,
            "use_azure_or_maas": config.use_azure_or_maas
        }
    }

async def process_image(image_data, max_size_mb=200):
    """Process image data and convert to base64 for vision model.
    
    Args:
        image_data: Raw image bytes
        max_size_mb: Maximum file size in MB to process
    
    Returns:
        Processed image data as base64 string or list of image chunks
    """
    try:
        # Check if file is too large (convert MB to bytes)
        max_bytes = max_size_mb * 1024 * 1024
        if len(image_data) > max_bytes:
            logger.warning(f"Image size {len(image_data)} bytes exceeds maximum allowed size of {max_bytes} bytes")
            raise ValueError(f"Image size {len(image_data)/1024/1024:.2f} MB exceeds maximum allowed size of {max_size_mb} MB")
            
        # Convert image to base64 for vision model
        logger.info(f"Image data size: {len(image_data)} bytes")
        image = Image.open(io.BytesIO(image_data))
        
        # Calculate target size to keep base64 output under reasonable size for API
        # OpenAI's API has a token limit, and large base64 strings consume many tokens
        TARGET_TOKEN_COUNT = 50000  # Target token count (conservative estimate)
        MAX_BASE64_SIZE = TARGET_TOKEN_COUNT * 3 / 4  # Approx 3/4 bytes per token for base64
        
        # Resize large images to reduce token count
        original_width, original_height = image.size
        logger.info(f"Original image dimensions: {original_width}x{original_height}")
        
        # For very large images, we need resizing but maintain quality
        if len(image_data) > 1024 * 1024:  # If over 1MB
            # Calculate scaling factor based on original size
            # Less aggressive scaling to maintain readability
            scale_factor = min(1.0, math.sqrt(MAX_BASE64_SIZE / len(image_data)) * 0.7)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Cap maximum dimensions to reasonable values
            max_dimension = 1500
            if new_width > max_dimension or new_height > max_dimension:
                if new_width > new_height:
                    new_height = int(new_height * (max_dimension / new_width))
                    new_width = max_dimension
                else:
                    new_width = int(new_width * (max_dimension / new_height))
                    new_height = max_dimension
            
            # Ensure minimum dimensions for readability
            new_width = max(new_width, 800)
            new_height = max(new_height, 600)
            
            # Resize image
            logger.info(f"Resizing image to {new_width}x{new_height}")
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to RGB if needed (to ensure compatibility)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPEG with appropriate compression
        # Maintain higher quality for better text recognition
        quality = 90
        if original_width * original_height > 4000000:  # > 4 megapixels
            quality = 85
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality, optimize=True)
        img_bytes = buffered.getvalue()
        img_str = base64.b64encode(img_bytes).decode()
        logger.info(f"Image processed: {len(img_str)} bytes, quality={quality}")
        
        # Check if the processed image is still too large for the API
        if len(img_str) > MAX_BASE64_SIZE * 1.5:  # If still too large
            logger.warning(f"Processed image still too large ({len(img_str)} bytes). Using better compression approach.")
            # Try again with better compression approach - maintain higher quality
            # First try with a moderate quality reduction
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=75, optimize=True)
            img_bytes = buffered.getvalue()
            img_str = base64.b64encode(img_bytes).decode()
            
            # If still too large, try resizing the image instead of further quality reduction
            if len(img_str) > MAX_BASE64_SIZE * 1.2:
                # Calculate new dimensions - reduce by 25%
                current_width, current_height = image.size
                new_width = int(current_width * 0.75)
                new_height = int(current_height * 0.75)
                
                # Resize and compress
                resized_image = image.resize((new_width, new_height), Image.LANCZOS)
                buffered = io.BytesIO()
                resized_image.save(buffered, format="JPEG", quality=75, optimize=True)
                img_bytes = buffered.getvalue()
                img_str = base64.b64encode(img_bytes).decode()
            
            logger.info(f"Image reprocessed with better approach: {len(img_str)} bytes")
        
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Payroll Agent API is running"}


class ApiModeToggle(BaseModel):
    """Request model for toggling API mode."""
    use_azure_or_maas: bool


@app.post("/toggle-api-mode")
async def toggle_api_mode(request: ApiModeToggle):
    """Toggle between Azure/MAAS mode and standard OpenAI mode."""
    try:
        from src.react_agent.configuration import Configuration
        config = Configuration.from_context()
        
        # Update the configuration
        config.use_azure_or_maas = request.use_azure_or_maas
        
        # Save the updated configuration (in memory for now)
        # In a production environment, you might want to persist this to a file or database
        
        mode = "Azure/MAAS" if request.use_azure_or_maas else "Standard OpenAI"
        logger.info(f"API mode toggled to: {mode}")
        
        return {
            "success": True,
            "message": f"API mode set to {mode}",
            "data": {"use_azure_or_maas": config.use_azure_or_maas}
        }
    except Exception as e:
        logger.error(f"Error toggling API mode: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Error toggling API mode: {str(e)}",
            "data": None
        }

@app.post("/process-document", response_model=ApiResponse)
async def process_document(file: UploadFile = File(...)):
    """Process uploaded document to extract employee data using Files API."""
    try:
        # Get file size first without reading entire content
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()  # Get position (file size)
        file.file.seek(0)  # Reset to beginning
        
        # Check file size limit (200MB)
        MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB in bytes
        if file_size > MAX_FILE_SIZE:
            logger.error(f"File too large: {file_size/1024/1024:.2f} MB exceeds limit of 200 MB")
            return {
                "success": False,
                "message": f"File too large: {file_size/1024/1024:.2f} MB exceeds maximum allowed size of 200 MB",
                "data": None
            }
        
        # Create a temporary file to save the uploaded content
        import tempfile
        import os
        
        # Create temp file with appropriate extension
        file_extension = os.path.splitext(file.filename)[1] if os.path.splitext(file.filename)[1] else ".tmp"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        
        try:
            # Write the file content to the temp file
            file_content = await file.read()
            temp_file.write(file_content)
            temp_file.close()  # Close the file so it can be read by the Files API
            
            logger.info(f"\nFile saved to temporary location: {temp_file.name}")
            logger.info(f"File content size: {len(file_content)} bytes ({len(file_content)/1024/1024:.2f} MB)")
            logger.info(f"File content type: {file.content_type}")
            logger.info(f"File filename: {file.filename}")
            
            # Import the process_document_with_files_api function
            from src.react_agent.utils import process_document_with_files_api
            
            # Get the model name from config
            config = get_agent_config()
            model_name = config["configurable"]["vlm_model"]
            
            # Process the document using Files API
            prompt = "Extract employee data from this document. For each employee, extract their name, regular hours worked, overtime hours worked . Format the data as structured JSON."
            
            # Get the VLM system prompt from config
            config_obj = Configuration.from_context()
            vlm_system_prompt = config_obj.vlm_system_prompt
            
            # Process the document using Files API
            response = await process_document_with_files_api(
                file_path=temp_file.name,
                prompt=prompt,
                model_name=model_name,
                vlm_system_prompt=vlm_system_prompt
            )
            
            # Extract document content and employee data from response
            document_content = response["content"]
            employees_data = response.get("employees_data", [])
            logger.info(f"Document processed with Files API. Response preview: {document_content[:100]}...")
            logger.info(f"Extracted {len(employees_data)} employees from document")
            
            # Set up initial state for the graph
            initial_state = {
                "messages": [],
                "existing_employees": [],
                "updated_employees": [],
                "updates_list": employees_data,
                "document_uploaded": True,
                "document_content": document_content,
                "user_approval": False,
                "trigger_payroll": False,
                "current_pay_data": None
            }
            
            # Run the workflow
            try:
                result = await graph.ainvoke(initial_state, config=get_agent_config())
                
                # Log the result structure for debugging
                logger.info(f"Graph result type: {type(result)}")
                if isinstance(result, dict):
                    logger.info(f"Graph result keys: {result.keys()}")
                
                # Extract updated employees
                updated_employees = []
                
                # Check if updated_employees is in the result
                if "updated_employees" in result:
                    updated_employees = result["updated_employees"]
                    logger.info(f"Found {len(updated_employees)} employees in result")
                elif "state" in result and "updated_employees" in result["state"]:
                    updated_employees = result["state"]["updated_employees"]
                    logger.info(f"Found {len(updated_employees)} employees in state")
            except Exception as e:
                logger.error(f"Error running graph workflow: {str(e)}")
                # Continue with empty employees list
                updated_employees = []
            # Convert Pydantic models to dictionaries if needed
            if updated_employees and hasattr(updated_employees[0], "model_dump"):
                updated_employees = [emp.model_dump() for emp in updated_employees]
            
            return {
                "success": True,
                "message": "Document processed successfully using Files API",
                "data": {
                    "updated_employees": updated_employees
                }
            }
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file.name)
                logger.info(f"Temporary file removed: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Error processing document: {str(e)}",
            "data": None
        }
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Error processing document: {str(e)}",
            "data": None
        }

@app.post("/merge-employees", response_model=ApiResponse)
async def merge_employee_lists(request: EmployeeListRequest = Body(..., embed=True), 
                              updated: EmployeeListRequest = Body(..., embed=True)):
    """Merge existing and updated employee lists."""
    try:
        # Convert to EmployeeData objects
        existing_employees = [EmployeeData(**emp.dict()) for emp in request.employees]
        updated_employees = [EmployeeData(**emp.dict()) for emp in updated.employees]
        
        # Merge employees
        merged_employees = merge_employees(existing_employees, updated_employees)
        
        # Convert back to dictionaries
        merged_dict = [emp.model_dump() for emp in merged_employees]
        
        return {
            "success": True,
            "message": f"Successfully merged {len(merged_employees)} employees",
            "data": {
                "merged_employees": merged_dict
            }
        }
    
    except Exception as e:
        logger.error(f"Error merging employees: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Error merging employees: {str(e)}",
            "data": None
        }

@app.post("/generate-payroll", response_model=ApiResponse)
async def generate_payroll(request: EmployeeListRequest):
    """Generate payroll report for the given employees."""
    try:
        # Convert to EmployeeData objects
        employees = [EmployeeData(**emp.dict()) for emp in request.employees]
        
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
        
        # Create report dictionary
        report_dict = {
            "employees": [emp.model_dump() for emp in payroll_employees],
            "total_payroll": total_payroll,
            "summary": summary
        }
        
        return {
            "success": True,
            "message": f"Payroll calculated for {len(payroll_employees)} employees",
            "data": {
                "payroll_report": report_dict
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating payroll: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Error generating payroll: {str(e)}",
            "data": None
        }


@app.post("/chat")
async def chat(request: dict):
    """Process a chat message and return the agent's response.
    
    This unified endpoint can handle both text messages and document uploads.
    Uses LangChain's Multimodal Inputs API for processing documents.
    """
    try:
        import traceback
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        from src.react_agent.utils import load_chat_model
        # Extract request data
        content = request.get("content", "")
        file_data = request.get("file_data")
        file_path = request.get("file_path")
        file_type = request.get("file_type")
        
        # Initialize state using the State model from state.py
        from src.react_agent.state import State, EmployeeData
        
        # Convert dict employees to EmployeeData objects if needed
        existing_employees = []
        for emp in request.get("existing_employees", []):
            if isinstance(emp, dict):
                existing_employees.append(EmployeeData(**emp))
            else:
                existing_employees.append(emp)
                
        updated_employees = []
        for emp in request.get("updated_employees", []):
            if isinstance(emp, dict):
                updated_employees.append(EmployeeData(**emp))
            else:
                updated_employees.append(emp)
                
        # Removed current_employees handling as it's no longer needed
        
        # Create state object
        state = State(
            existing_employees=existing_employees,
            updated_employees=updated_employees,
            user_approval=request.get("user_approval", False),
            trigger_payroll=request.get("trigger_payroll", False),
            current_pay_data=request.get("current_pay_data"),
        )
        
        # Check if a file was uploaded
        if file_data:
            # Set document_uploaded flag
            state.document_uploaded = True
            
            # Log file upload
            logging.info(f"File upload detected in chat: {file_path}")
            
            # Create a multimodal message with the file data
            # This will be processed by the graph workflow, not here
            from langchain_core.messages import HumanMessage
            
            # Prepare prompt text
            prompt_text = content if content else "Please process this document and extract employee data."
            
            # Create a human message with the file content
            # We're not processing the image here anymore, just adding it to the message
            try:
                # Store the file data in the state for the graph to process
                state.file_data = file_data
                state.file_path = file_path
                state.file_type = file_type
                
                logging.info("Added file data to state for graph processing")
            except Exception as e:
                logging.error(f"Error adding file data to state: {str(e)}")
        
        # Get the agent graph
        graph = get_agent_graph()
        
        # Add user message to state
        state.messages.append(HumanMessage(content=content))
        
        # Convert any EmployeeData objects to dictionaries before invoking the graph
        # This prevents Pydantic validation errors when the graph processes the state
        if hasattr(state, 'updated_employees') and state.updated_employees:
            state.updated_employees = [emp.model_dump() if hasattr(emp, 'model_dump') else emp for emp in state.updated_employees]
        
        if hasattr(state, 'existing_employees') and state.existing_employees:
            state.existing_employees = [emp.model_dump() if hasattr(emp, 'model_dump') else emp for emp in state.existing_employees]
            
        # Removed current_employees conversion as it's no longer needed
            
        # Invoke the graph with the state - use ainvoke for async functions
        try:
            # Create a fresh state object for each invocation to prevent recursion
            from react_agent.state import State
            from copy import deepcopy
            
            # Create a new state with only the essential data
            fresh_state = State(
                messages=state.messages,  # Keep the messages
                document_uploaded=state.document_uploaded,
                document_content=state.document_content,
                document_processing_done=state.document_processing_done,
                existing_employees=state.existing_employees,
                updated_employees=state.updated_employees,
                user_approval=state.user_approval,
                trigger_payroll=state.trigger_payroll,
                file_data=state.file_data,
                file_path=state.file_path,
                file_type=state.file_type
            )
            
            # Use the fresh state to prevent recursion
            result = await graph.ainvoke(fresh_state)
        except Exception as e:
            logging.error(f"Error invoking graph: {str(e)}")
            # Return error response
            return {
                "success": False,
                "data": {
                    "response": f"I encountered an error processing your request: {str(e)}",
                    "workflow_state": {}
                }
            }
        
        # Extract the last AI message as the response
        response_content = "I'm not sure how to respond to that."
        for message in reversed(result.get('messages', [])):
            if isinstance(message, AIMessage):
                response_content = message.content
                break
            elif isinstance(message, dict) and 'content' in message:
                response_content = message['content']
        
        # Prepare response data - keep it simple and agile
        response_data = {
            "success": True,
            "data": {
                "response": response_content,
                "workflow_state": {
                    "document_uploaded": result.get('document_uploaded', False),
                    "document_processing_done": result.get('document_processing_done', False),
                    "updated_employees": [emp.dict() if hasattr(emp, 'dict') else emp for emp in result.get('updated_employees', [])],
                    "existing_employees": [emp.dict() if hasattr(emp, 'dict') else emp for emp in result.get('existing_employees', [])],
                    "user_approval": result.get('user_approval', False),
                    "trigger_payroll": result.get('trigger_payroll', False),
                    "current_pay_data": result.get('current_pay_data')
                }
            }
        }
        
        return response_data
    
    except Exception as e:
        logging.error(f"Error processing chat: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "data": None
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
