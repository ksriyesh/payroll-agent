#!/usr/bin/env python3
"""Test document workflow with VLM processing."""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from react_agent.graph import graph
from react_agent.configuration import Configuration
from react_agent.state import EmployeeData

async def test_document_workflow():
    """Test the document processing workflow."""
    
    # Test configuration
    config = {
        "configurable": {
            "vlm_model": "openai/gpt-4o",
            "react_model": "openai/gpt-4o",
            "payroll_model": "openai/gpt-4o"
        }
    }
    
    # Test document content
    document_content = """
    Employee Hours for Week Ending 12/01/2024
    
    John Doe: 42 hours (2 overtime)
    Jane Smith: 38.5 hours 
    Bob Johnson: 48 hours (8 overtime)
    Alice Williams: 35 hours
    """
    
    # Initial state with existing employees
    initial_state = {
        "existing_employees": [
            EmployeeData(name="John Doe", payrate=25.0, regular_hours=40.0, overtime_hours=0.0),
            EmployeeData(name="Jane Smith", payrate=30.0, regular_hours=40.0, overtime_hours=0.0),
            EmployeeData(name="Bob Johnson", payrate=22.0, regular_hours=40.0, overtime_hours=0.0),
        ],
        "updated_employees": [],
        "current_employees": [],
        "document_content": document_content,
        "document_uploaded": True,
        "user_approval": False,
        "trigger_payroll": False
    }
    
    print("🚀 Testing Document Workflow")
    print("=" * 50)
    
    # Test 1: Initial state
    print("📋 Initial State:")
    print(f"  - Document content: {len(document_content)} characters")
    print(f"  - Existing employees: {len(initial_state['existing_employees'])}")
    print(f"  - Document uploaded: {initial_state['document_uploaded']}")
    
    # Test 2: Run VLM document processing
    print("\n🔍 Running VLM Document Processing...")
    try:
        # Execute the graph with document processing
        result = await graph.ainvoke(initial_state, config)
        
        print("✅ Document processing completed!")
        print(f"  - Messages generated: {len(result.get('messages', []))}")
        print(f"  - Existing employees: {len(result.get('existing_employees', []))}")
        print(f"  - Updated employees: {len(result.get('updated_employees', []))}")
        
        # Show extracted employee data
        updated_employees = result.get('updated_employees', [])
        if updated_employees:
            print(f"\n📄 Extracted Employee Hours:")
            for emp in updated_employees:
                print(f"  - {emp.name}: {emp.regular_hours} regular, {emp.overtime_hours} overtime @ ${emp.payrate}/hr")
        
        # Show final messages
        messages = result.get('messages', [])
        if messages:
            print(f"\n💬 Agent Response:")
            for msg in messages[-2:]:  # Show last 2 messages
                if hasattr(msg, 'content'):
                    print(f"  {msg.content}")
        
    except Exception as e:
        print(f"❌ Error during document processing: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("📄 Employee hours were successfully extracted from the document.")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_document_workflow())
    if success:
        print("✅ Document workflow test completed successfully!")
    else:
        print("❌ Document workflow test failed!")
        sys.exit(1) 