"""This module provides tools for payroll document processing and analysis.

All operations are async to prevent blocking the event loop.
"""

import os
import base64
import io
import asyncio
import json
import re
from typing import Any, List, Optional, Dict
from decimal import Decimal
from datetime import datetime
import logging

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from src.react_agent.configuration import Configuration
from src.react_agent.state import DocumentInfo, PayrollContext, EmployeePayInfo

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def encode_image_to_base64(image_data: bytes, image_format: str = "PNG") -> str:
    """Encode image data to base64 string (async to avoid blocking)."""
    def _encode():
        return base64.b64encode(image_data).decode('utf-8')
    
    return await asyncio.to_thread(_encode)


async def convert_pdf_to_images(file_bytes: bytes) -> List[bytes]:
    """Convert PDF to images asynchronously."""
    def _convert_pdf():
        import fitz  # Import inside function to avoid blocking
        
        images = []
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Convert page to image (PNG format, high quality)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x scaling
            img_data = pix.tobytes("png")
            images.append(img_data)
        
        doc.close()
        return images
    
    return await asyncio.to_thread(_convert_pdf)


async def convert_image_to_png(file_bytes: bytes) -> bytes:
    """Convert image to PNG format asynchronously."""
    def _convert_image():
        from PIL import Image  # Import inside function to avoid blocking
        
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to RGB if necessary (for RGBA, CMYK, etc.)
        if image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')
        
        # Save as PNG
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG', optimize=True)
        return img_buffer.getvalue()
    
    return await asyncio.to_thread(_convert_image)


async def convert_document_to_images(file_path: str) -> List[bytes]:
    """Convert any document type to a list of images for VLM processing.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of image bytes (PNG format) ready for VLM processing
    """
    logger.info(f"🔄 Converting document to images: {file_path}")
    
    # Import inside function to avoid blocking during module load
    def _import_dependencies():
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            return fitz, Image
        except ImportError as e:
            logger.error(f"❌ Missing dependencies: {e}")
            raise ImportError(f"Missing dependencies for document processing: {e}")
    
    try:
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        logger.debug(f"File extension: {file_ext}")
        
        if file_ext == '.pdf':
            logger.info("📄 Processing PDF document")
            # Process PDF in thread to avoid blocking
            def process_pdf():
                fitz, Image = _import_dependencies()
                doc = fitz.open(file_path)
                images = []
                
                logger.debug(f"PDF pages: {len(doc)}")
                for page_num in range(len(doc)):
                    logger.debug(f"Processing page {page_num + 1}")
                    page = doc.load_page(page_num)
                    
                    # Render page to image with 2x scaling for better quality
                    mat = fitz.Matrix(2.0, 2.0)  # 2x scaling
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PNG bytes
                    png_bytes = pix.tobytes("png")
                    images.append(png_bytes)
                    logger.debug(f"Page {page_num + 1} converted, size: {len(png_bytes)} bytes")
                
                doc.close()
                return images
            
            images = await asyncio.to_thread(process_pdf)
            logger.info(f"✅ PDF converted to {len(images)} images")
            return images
        
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
            logger.info(f"🖼️ Processing image file: {file_ext}")
            # Process image in thread to avoid blocking
            def process_image():
                fitz, Image = _import_dependencies()
                image = Image.open(file_path)
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    logger.debug(f"Converting from {image.mode} to RGB")
                    image = image.convert('RGB')
                
                # Save as PNG bytes
                import io
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                png_bytes = img_buffer.getvalue()
                logger.debug(f"Image converted, size: {len(png_bytes)} bytes")
                return [png_bytes]
            
            images = await asyncio.to_thread(process_image)
            logger.info("✅ Image converted to PNG")
            return images
        
        else:
            logger.error(f"❌ Unsupported file type: {file_ext}")
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    except Exception as e:
        logger.error(f"❌ Error converting document to images: {str(e)}", exc_info=True)
        raise


async def extract_text_from_document(file_path: str) -> Dict[str, Any]:
    """Extract text and positions from document using OCR/text extraction.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary with extracted text and position information
    """
    def _extract_text():
        import fitz  # PyMuPDF
        
        text_data = {
            "full_text": "",
            "pages": [],
            "text_blocks": []
        }
        
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get text with position information
                text_dict = page.get_text("dict")
                page_text = page.get_text()
                
                text_data["pages"].append({
                    "page_num": page_num + 1,
                    "text": page_text,
                    "blocks": text_dict.get("blocks", [])
                })
                
                text_data["full_text"] += page_text + "\n"
            
            doc.close()
        else:
            # For images, use OCR
            try:
                import pytesseract
                from PIL import Image
                
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)
                
                text_data["full_text"] = text
                text_data["pages"] = [{
                    "page_num": 1,
                    "text": text,
                    "blocks": []
                }]
                
            except Exception as e:
                text_data["error"] = f"OCR failed: {str(e)}"
        
        return text_data
    
    return await asyncio.to_thread(_extract_text)


async def process_document_with_vlm(
    file_path: str,
    context_query: str = "Extract all payroll information from this document."
) -> Dict[str, Any]:
    """Process payroll document following intended workflow:
    1. Convert document to VLM-friendly format
    2. VLM extracts text and positions 
    3. Feed structured data to React agent
    """
    logger.info(f"🧠 Starting VLM workflow for: {file_path}")
    logger.debug(f"Context query: {context_query}")
    
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }
        
        # Get file info
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        file_size = os.path.getsize(file_path)
        
        logger.info(f"📄 File info: {file_name} ({file_ext}, {file_size} bytes)")
        
        # STEP 1: Convert document to VLM-friendly format
        logger.info("🔄 Step 1: Converting document to VLM-friendly format")
        images = await convert_document_to_images(file_path)
        logger.info(f"✅ Document converted to {len(images)} VLM-ready images")
        
        # STEP 2: VLM extracts text and positions
        logger.info("🧠 Step 2: VLM processing for text and position extraction")
        
        # Try OpenAI VLM first, fallback to text-based VLM simulation
        vlm_analysis = await extract_with_vlm(images, context_query, file_path)
        
        # STEP 3: Structure data for React agent
        logger.info("📊 Step 3: Structuring VLM data for React agent")
        employees = await parse_vlm_structured_data(vlm_analysis)
        
        logger.info(f"✅ VLM workflow complete: Found {len(employees)} employees")
        
        # Create document info
        doc_info = DocumentInfo(
            filename=file_name,
            file_type=file_ext,
            file_size=file_size,
            pages=len(images),
            processed=True
        )
        
        result = {
            "success": True,
            "document_info": doc_info,
            "text_data": vlm_analysis.get("text_data", {}),
            "extracted_text": vlm_analysis.get("extracted_text", ""),
            "vlm_analysis": vlm_analysis.get("vlm_raw_analysis", []),
            "employees": employees,
            "total_pages": len(images),
            "needs_clarification": len(employees) == 0,
            "processing_time": datetime.now().isoformat()
        }
        
        logger.info("✅ VLM workflow completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ VLM workflow error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "processing_time": datetime.now().isoformat()
        }


async def extract_with_vlm(images: List[bytes], context_query: str, file_path: str) -> Dict[str, Any]:
    """VLM text and position extraction with fallback strategies."""
    logger.info("🔍 Starting VLM text and position extraction")
    
    # Strategy 1: Try OpenAI VLM
    openai_result = await try_openai_vlm(images, context_query)
    if openai_result["success"]:
        logger.info("✅ OpenAI VLM extraction successful")
        return openai_result
    
    # Strategy 2: Fallback to enhanced text-based VLM simulation
    logger.info("🔄 Falling back to enhanced text-based VLM simulation")
    return await simulate_vlm_extraction(file_path, context_query)


async def try_openai_vlm(images: List[bytes], context_query: str) -> Dict[str, Any]:
    """Try OpenAI VLM processing."""
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("❌ OpenAI API key not found")
            return {"success": False, "error": "No OpenAI API key"}
        
        # Initialize OpenAI vision model
        config = Configuration()
        vlm_model = ChatOpenAI(
            model=config.vision_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=openai_api_key,
            timeout=30,
            max_retries=1  # Reduced retries for faster fallback
        )
        
        logger.debug(f"OpenAI VLM initialized: {config.vision_model}")
        
        # VLM prompt for text and position extraction
        vlm_prompt = f"""Extract text and positions from this payroll document.

TASK: Extract ALL text content and identify the spatial positions/structure of payroll data.

FOCUS ON:
- Employee names and IDs (with their document positions)
- Pay rates and salary information (with positions)
- Hours worked (regular, overtime) (with positions)
- Pay periods and dates (with positions)
- Deductions and taxes (with positions)
- Net/gross pay amounts (with positions)

CONTEXT: {context_query}

PROVIDE:
1. Complete text extraction
2. Structured payroll data with positions
3. Spatial relationships between data elements

Be thorough and extract every piece of payroll information with its position context."""
        
        vlm_results = []
        
        for i, image_bytes in enumerate(images):
            logger.debug(f"Processing image {i+1}/{len(images)} with OpenAI VLM")
            
            # Encode image
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create message
            message = HumanMessage(
                content=[
                    {"type": "text", "text": vlm_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high"
                        }
                    }
                ]
            )
            
            # VLM processing
            vlm_response = await vlm_model.ainvoke([message])
            vlm_analysis = vlm_response.content
            
            vlm_results.append({
                "page": i + 1,
                "analysis": vlm_analysis
            })
            
            logger.debug(f"OpenAI VLM page {i+1}: {len(vlm_analysis)} characters")
        
        # Combine analysis
        combined_analysis = "\n\n".join([f"Page {r['page']}:\n{r['analysis']}" for r in vlm_results])
        
        return {
            "success": True,
            "extracted_text": combined_analysis,
            "vlm_raw_analysis": vlm_results,
            "text_data": {"full_text": combined_analysis, "source": "openai_vlm"}
        }
        
    except Exception as e:
        logger.warning(f"⚠️ OpenAI VLM failed: {str(e)}")
        return {"success": False, "error": str(e)}


async def simulate_vlm_extraction(file_path: str, context_query: str) -> Dict[str, Any]:
    """Enhanced text-based VLM simulation with position awareness."""
    logger.info("🔄 Simulating VLM extraction with enhanced text processing")
    
    try:
        # Extract text with position information using PyMuPDF
        text_data = await extract_text_from_document(file_path)
        
        # Create VLM-style analysis using Groq
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=4096,
            timeout=30
        )
        
        # VLM simulation prompt
        vlm_simulation_prompt = f"""You are a Vision Language Model analyzing a payroll document. 

EXTRACTED TEXT:
{text_data.get('full_text', '')}

CONTEXT: {context_query}

TASK: Analyze this payroll document text as if you were a VLM that can see the document layout and positions.

PROVIDE:
1. Detailed analysis of payroll structure and layout
2. Identification of employee records and their data positions
3. Extraction of key payroll information with spatial context
4. Recognition of tables, forms, and data relationships

Format your response as a comprehensive VLM analysis that includes:
- Document structure analysis
- Employee data identification with positions
- Payroll calculations and relationships
- Data validation and completeness assessment

Analyze thoroughly as if you can see the visual layout of the document."""
        
        # Get VLM-style analysis
        response = await llm.ainvoke([HumanMessage(content=vlm_simulation_prompt)])
        vlm_style_analysis = response.content
        
        logger.info(f"✅ VLM simulation complete: {len(vlm_style_analysis)} characters")
        
        return {
            "success": True,
            "extracted_text": vlm_style_analysis,
            "vlm_raw_analysis": [{"page": 1, "analysis": vlm_style_analysis}],
            "text_data": text_data
        }
        
    except Exception as e:
        logger.error(f"❌ VLM simulation failed: {str(e)}")
        raise


async def parse_vlm_structured_data(vlm_analysis: Dict[str, Any]) -> List[EmployeePayInfo]:
    """Parse VLM analysis into structured employee data for React agent."""
    logger.info("📊 Parsing VLM analysis into structured data")
    
    extracted_text = vlm_analysis.get("extracted_text", "")
    
    if not extracted_text:
        logger.warning("❌ No VLM analysis text to parse")
        return []
    
    try:
        # Use Groq to parse VLM analysis into structured JSON
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=4096,
            timeout=30
        )
        
        parsing_prompt = f"""Parse the following VLM analysis of a payroll document into structured JSON.

VLM ANALYSIS:
{extracted_text}

Extract ALL employee payroll information and return as a JSON array. Each employee should have:

{{
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
}}

INSTRUCTIONS:
- Extract ALL employees found in the VLM analysis
- Calculate net_pay = gross_pay - deductions if not provided
- Use null for missing values
- Return ONLY the JSON array, no additional text
- Be accurate with numbers from the VLM analysis

JSON Array:"""
        
        response = await llm.ainvoke([HumanMessage(content=parsing_prompt)])
        response_text = response.content.strip()
        
        # Clean JSON response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        employees_data = json.loads(response_text)
        
        # Convert to EmployeePayInfo objects
        employees = []
        for emp_data in employees_data:
            try:
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
                employees.append(employee)
                logger.debug(f"✅ Parsed employee: {employee.name}")
            except Exception as e:
                logger.error(f"❌ Error creating employee object: {e}")
                continue
        
        logger.info(f"✅ Parsed {len(employees)} employees from VLM analysis")
        return employees
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parsing error: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ Error parsing VLM data: {str(e)}")
        return []


async def gather_payroll_context(
    company_name: Optional[str] = None,
    pay_period_type: Optional[str] = None,
    expected_employees: Optional[List[str]] = None,
    document_type: Optional[str] = None,
    currency: Optional[str] = None,
    additional_notes: Optional[str] = None
) -> PayrollContext:
    """Gather additional context from user to improve payroll document processing.
    
    Args:
        company_name: Name of the company
        pay_period_type: Type of pay period (weekly, bi-weekly, monthly, etc.)
        expected_employees: List of expected employee names
        document_type: Type of document (payslip, timesheet, payroll_summary, etc.)
        currency: Currency used (USD, EUR, etc.)
        additional_notes: Any additional context notes
    
    Returns:
        PayrollContext object with gathered information
    """
    context = PayrollContext(
        company_name=company_name,
        pay_period_type=pay_period_type,
        expected_employees=expected_employees,
        document_type=document_type,
        currency=currency,
        additional_notes=additional_notes
    )
    
    return context


async def parse_employee_data_json(extracted_text: str) -> List[EmployeePayInfo]:
    """Extract structured employee pay information from VLM analysis using Groq LLM.
    
    Args:
        extracted_text: VLM analysis text from payroll document
        
    Returns:
        List of EmployeePayInfo objects
    """
    logger.info("🧮 Parsing employee data from VLM analysis")
    logger.debug(f"Input text length: {len(extracted_text)}")
    
    if not extracted_text or not extracted_text.strip():
        logger.warning("❌ No text provided for employee data parsing")
        return []
    
    try:
        # Create LLM for structured data extraction
        llm = ChatGroq(
            model="llama-3.1-8b-instant",  # Fixed: removed "groq/" prefix
            temperature=0.1,
            max_tokens=4096,
            timeout=30
        )
        
        logger.debug("LLM initialized for data parsing")
        
        # Create parsing prompt
        parsing_prompt = f"""Extract employee payroll information from the following document analysis.

VLM Analysis:
{extracted_text}

Please extract ALL employee payroll information and return it as a JSON array. Each employee should have the following structure:

{{
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
}}

Important:
- Return ONLY the JSON array, no additional text
- Use null for missing values
- Extract all employees found in the document
- Be accurate with numbers - don't guess
- If no employees found, return empty array []

JSON Array:"""
        
        logger.debug("Parsing prompt created")
        
        # Get structured response
        response = await llm.ainvoke([HumanMessage(content=parsing_prompt)])
        response_text = response.content.strip()
        
        logger.debug(f"LLM response: {len(response_text)} characters")
        logger.debug(f"LLM response preview: {response_text[:200]}...")
        
        # Clean and parse JSON
        try:
            # Remove any markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            logger.debug(f"Cleaned response: {response_text[:200]}...")
            
            # Parse JSON
            employees_data = json.loads(response_text)
            
            logger.debug(f"Parsed JSON: {len(employees_data)} employees")
            
            # Convert to EmployeePayInfo objects
            employees = []
            for emp_data in employees_data:
                try:
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
                    employees.append(employee)
                    logger.debug(f"Created employee: {employee.name}")
                except Exception as e:
                    logger.error(f"❌ Error creating employee object: {e}")
                    continue
            
            logger.info(f"✅ Successfully parsed {len(employees)} employees")
            return employees
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error: {e}")
            logger.error(f"Response text: {response_text}")
            return []
        
    except Exception as e:
        logger.error(f"❌ Error parsing employee data: {str(e)}", exc_info=True)
        return []


# NOTE: In the new streamlined workflow, we don't use tools.
# VLM processing happens directly in the graph nodes.
# These functions are used as helper functions, not LangChain tools.
