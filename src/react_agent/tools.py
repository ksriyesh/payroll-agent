"""Tools for the payroll agent."""

import os
import asyncio
import logging

# Configure logging
logger = logging.getLogger(__name__)

from typing import List, Dict, Any
from langchain_core.tools import tool
from .state import EmployeeData


def merge_employees(existing_employees: List[EmployeeData], updated_employees: List[EmployeeData]) -> List[EmployeeData]:
    """
    Merge existing employees with updated employees using automatic conflict resolution.
    - Missing employees: Add from existing_employees with same data
    - Conflicts: updated_employees takes precedence 
    - New employees: Add from updated_employees
    
    Args:
        existing_employees: Current employee list
        updated_employees: New/updated employee data
        
    Returns:
        List of merged EmployeeData objects
    """
    logger.info(f"Merging {len(existing_employees)} existing + {len(updated_employees)} updates")
    
    merged = {}
    
    # First, add all existing employees
    for emp in existing_employees:
        emp_dict = emp.model_dump() if hasattr(emp, 'model_dump') else emp.__dict__
        merged[emp_dict["name"]] = EmployeeData(**emp_dict)
    
    # Then, update/add from updated_employees (takes precedence)
    for emp in updated_employees:
        emp_dict = emp.model_dump() if hasattr(emp, 'model_dump') else emp.__dict__
        
        if emp_dict["name"] in merged:
            # Employee exists - merge with updates taking precedence
            existing_emp = merged[emp_dict["name"]]
            merged_emp_data = {
                "name": emp_dict["name"],
                "regular_hours": emp_dict["regular_hours"],  # Updates take precedence
                "overtime_hours": emp_dict["overtime_hours"],  # Updates take precedence
                "payrate": emp_dict["payrate"] if emp_dict["payrate"] > 0 else existing_emp.payrate  # Keep existing payrate if updates has 0
            }
            merged[emp_dict["name"]] = EmployeeData(**merged_emp_data)
        else:
            # New employee from updated_employees
            merged[emp_dict["name"]] = EmployeeData(**emp_dict)
    
    result = list(merged.values())
    logger.info(f"Merge complete: {len(result)} total employees")
    return result


@tool
def update_state(employees: List[Dict[str, Any]], target_list: str) -> str:
    """Update employee state list."""
    return f"Updated {target_list} with {len(employees)} employees"


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
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # Render page to pixmap (image)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                    img_data = pix.tobytes("png")
                    images.append(img_data)
                    logger.debug(f"Converted page {page_num + 1} to image")
                
                doc.close()
                return images
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(None, process_pdf)
            logger.info(f"✅ PDF conversion complete: {len(images)} pages")
            return images
            
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            logger.info("🖼️  Processing image file")
            # For image files, just return the file content
            with open(file_path, 'rb') as f:
                image_data = f.read()
            return [image_data]
            
        else:
            logger.warning(f"⚠️  Unsupported file type: {file_ext}")
            return []
            
    except Exception as e:
        logger.error(f"❌ Error converting document: {str(e)}")
        return []
