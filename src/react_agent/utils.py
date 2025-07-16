"""Utility & helper functions."""

import os
import logging
from typing import Optional, Dict, Any, List, Union, Type
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from .state import EmployeeData
from .configuration import Configuration

logger = logging.getLogger(__name__)

def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
        
    Returns:
        BaseChatModel: The initialized chat model.
    """
    # Get configuration to check if Azure/MAAS mode is enabled
    config = Configuration.from_context()
    
    # Check for Azure OpenAI configuration
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    maas_base_url = os.getenv("MAAS_BASE_URL")
    maas_api_key = os.getenv("MAAS_API_KEY")
    
    # Parse the model name
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = "openai"
        model = fully_specified_name
    
    # Only use Azure or MAAS if the toggle is enabled
    if config.use_azure_or_maas:
        # If MAAS configuration is available, use it
        if maas_base_url and maas_api_key:
            logger.info(f"Using MAAS with base URL: {maas_base_url}")
            return ChatOpenAI(
                model=model,
                openai_api_base=maas_base_url,
                openai_api_key=maas_api_key,
                temperature=0
            )
        # If Azure OpenAI configuration is available, use it
        elif azure_api_key and azure_endpoint:
            logger.info(f"Using Azure OpenAI with endpoint: {azure_endpoint}")
            # For Azure, we need deployment name which is typically the model name
            deployment_name = model.replace(".", "")
            return AzureChatOpenAI(
                deployment_name=deployment_name,
                openai_api_version="2023-05-15",
                azure_endpoint=azure_endpoint,
                azure_api_key=azure_api_key,
                temperature=0
            )
    
    # Use standard OpenAI if toggle is disabled or no Azure/MAAS configs available
    logger.info(f"Using standard OpenAI API with model: {model}")
    return ChatOpenAI(model=model, temperature=0)


async def process_document_with_files_api(file_path: str, prompt: str, model_name: str, vlm_system_prompt: str = None, output_schema: Type = None) -> Dict[str, Any]:
    """Process a document using LangChain's multimodal inputs approach with structured output support.
    
    Args:
        file_path: Path to the file to process
        prompt: Prompt to send to the model
        model_name: Name of the model to use
        vlm_system_prompt: Optional system prompt for the VLM model
        output_schema: Optional Pydantic model to use for structured output
        
    Returns:
        Dict containing the model's response and structured data if output_schema is provided
    """
    import os
    import mimetypes
    import base64
    from pathlib import Path
    from typing import Type, Dict, Any, Optional, List, Union
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel
    
    # Get configuration to check if Azure/MAAS mode is enabled
    config = Configuration.from_context()
    
    # Load the appropriate chat model
    model = load_chat_model(model_name)
    
    # Determine the file's MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        # Default to jpeg if we can't determine the type
        mime_type = "image/jpeg"
    
    # Read file as binary and encode to base64
    with open(file_path, "rb") as file:
        file_content = file.read()
        base64_content = base64.b64encode(file_content).decode("utf-8")
    
    # Get the filename from the path
    filename = Path(file_path).name
    
    try:
        # Create system message for instructions
        system_message = SystemMessage(content=vlm_system_prompt if vlm_system_prompt else "Extract employee data from documents with high accuracy.")
        
        # Create a multimodal message using the latest format
        # For images, use the new image format with source_type
        if mime_type and mime_type.startswith('image/'):
            human_message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": base64_content,
                        "mime_type": mime_type
                    }
                ]
            )
        else:
            # For PDFs and other document types
            # Note: OpenAI's latest API prefers using the 'file' type for documents
            human_message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "file",
                        "source_type": "base64",
                        "data": base64_content,
                        "mime_type": mime_type,
                        "name": filename  # Using 'name' instead of 'filename' for better compatibility
                    }
                ]
            )
        
        # Check if we should use structured output
        if output_schema and issubclass(output_schema, BaseModel):
            # Use structured output with the provided Pydantic model
            logger.info(f"Using structured output with schema: {output_schema.__name__}")
            structured_model = model.with_structured_output(output_schema)
            
            # Invoke the model with structured output
            try:
                structured_response = await structured_model.ainvoke([system_message, human_message])
                logger.info(f"Successfully received structured response of type: {type(structured_response).__name__}")
                
                # Convert the structured response to a dictionary
                if hasattr(structured_response, 'model_dump'):
                    # For Pydantic v2
                    structured_data = structured_response.model_dump()
                elif hasattr(structured_response, 'dict'):
                    # For Pydantic v1
                    structured_data = structured_response.dict()
                else:
                    # Fallback if it's not a standard Pydantic model
                    structured_data = structured_response
                
                # Return both the raw structured response and the dictionary version
                return {
                    "structured_response": structured_response,
                    "structured_data": structured_data,
                    "content": str(structured_response),  # For compatibility with existing code
                    "employees_data": structured_data.get("employees", []) if isinstance(structured_data, dict) else []
                }
            except Exception as e:
                logger.error(f"Error in structured output processing: {str(e)}")
                # Fall back to standard processing if structured output fails
                logger.info("Falling back to standard processing")
        
        # Standard processing without structured output
        logger.info(f"Invoking {model_name} with multimodal input (standard processing)")
        response = await model.ainvoke([system_message, human_message])
        
        # Parse the model's response to extract structured employee data
        import json
        import re
        from .state import EmployeeData
        
        content = response.content
        employees_data = []
        
        try:
            # Try to parse as JSON or extract JSON from the response
            json_patterns = [
                # Full content as JSON
                lambda c: json.loads(c),
                # JSON in code block
                lambda c: json.loads(re.search(r'```(?:json)?\s*({.*?})\s*```', c, re.DOTALL).group(1)),
                # JSON with employees key
                lambda c: json.loads(re.search(r'{\s*"employees"\s*:\s*\[.*?\]\s*}', c, re.DOTALL).group(0))
            ]
            
            parsed_data = None
            for pattern_func in json_patterns:
                try:
                    parsed_data = pattern_func(content)
                    break
                except (json.JSONDecodeError, AttributeError, TypeError):
                    continue
            
            # If we successfully parsed JSON with employees, extract them
            if parsed_data and isinstance(parsed_data, dict) and "employees" in parsed_data:
                logger.info(f"Found structured output with {len(parsed_data['employees'])} employees")
                
                # Validate and convert to EmployeeData objects
                for emp_data in parsed_data["employees"]:
                    try:
                        employee = {
                            "name": emp_data["name"],
                            "regular_hours": float(emp_data["regular_hours"]),
                            "overtime_hours": float(emp_data["overtime_hours"]),
                            "payrate": float(emp_data["payrate"])
                        }
                        employees_data.append(employee)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"Error parsing employee data: {str(e)}")
        except Exception as e:
            logger.warning(f"Could not parse structured output: {str(e)}")
        
        # Return both the raw content and structured employee data
        return {
            "content": response.content,
            "employees_data": employees_data
        }
    except Exception as e:
        logger.error(f"Error processing document with LangChain: {str(e)}")
        raise e




async def load_previous_payperiod_data() -> List[EmployeeData]:
    """Load hardcoded previous pay period employee data."""
    employees = [
        EmployeeData(name="John Doe", regular_hours=40.0, overtime_hours=5.0, payrate=60.0),
        EmployeeData(name="Jane Smith", regular_hours=38.5, overtime_hours=2.5, payrate=50.0),
        EmployeeData(name="Bob Johnson", regular_hours=42.0, overtime_hours=8.0, payrate=45.0),
        EmployeeData(name="Alice Williams", regular_hours=35.0, overtime_hours=3.0, payrate=55.0),
        EmployeeData(name="Charlie Brown", regular_hours=40.0, overtime_hours=0.0, payrate=48.0)
    ]
    return employees