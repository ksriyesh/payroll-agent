"""Tools for the payroll agent."""

import os
import asyncio
import logging

# Configure logging
logger = logging.getLogger(__name__)

from typing import List, Dict, Any
from langchain_core.tools import tool
from .state import EmployeeData


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


def merge_employees(existing: List[EmployeeData], updated: List[EmployeeData]) -> List[EmployeeData]:
    """Automatically merge employee lists with conflict resolution:
    - If employee missing from updated list but exists in existing list: add to updated list
    - If updated employee has empty/zero fields: use existing employee data  
    - Updated employees take precedence over existing
    - All existing employees are preserved in final list
    """
    result = []
    existing_dict = {emp.name: emp for emp in existing}
    updated_dict = {emp.name: emp for emp in updated}
    
    # Process all existing employees first (they all get preserved)
    for existing_emp in existing:
        if existing_emp.name in updated_dict:
            # Employee exists in both lists - merge with updated taking precedence
            updated_emp = updated_dict[existing_emp.name]
            
            # Handle empty fields: use existing data if updated has empty/zero values
            final_regular_hours = updated_emp.regular_hours if updated_emp.regular_hours > 0 else existing_emp.regular_hours
            final_overtime_hours = updated_emp.overtime_hours if updated_emp.overtime_hours > 0 else existing_emp.overtime_hours
            final_payrate = updated_emp.payrate if updated_emp.payrate > 0 else existing_emp.payrate
            
            # Create merged employee with best available data
            merged_emp = EmployeeData(
                name=existing_emp.name,
                regular_hours=final_regular_hours,
                overtime_hours=final_overtime_hours,
                payrate=final_payrate
            )
            result.append(merged_emp)
        else:
            # Employee missing from updated list - add from existing list
            result.append(existing_emp)
    
    # Add any new employees from updated list that don't exist in existing
    for updated_emp in updated:
        if updated_emp.name not in existing_dict:
            result.append(updated_emp)
    
    return result
