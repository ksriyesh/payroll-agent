#!/usr/bin/env python3
"""Test the 3-agent payroll system."""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from react_agent.graph import graph
from react_agent.configuration import Configuration
from react_agent.state import EmployeeData

async def test_agent_system():
    """Test the complete 3-agent system."""
    
    # Test configuration
    config = {
        "configurable": {
            "vlm_model": "openai/gpt-4o",
            "react_model": "openai/gpt-4o",
            "payroll_model": "openai/gpt-4o"
        }
    }
    
    print("🚀 Testing 3-Agent System")
    print("=" * 50)
    
    # Test 1: Basic agent response
    print("📋 Test 1: Basic Agent Response")
    basic_state = {
        "messages": [],
        "existing_employees": [
            EmployeeData(name="John Doe", payrate=25.0, regular_hours=40.0, overtime_hours=5.0),
            EmployeeData(name="Jane Smith", payrate=30.0, regular_hours=38.5, overtime_hours=2.5),
        ],
        "updated_employees": [],
        "current_employees": [],
        "document_uploaded": False,
        "user_approval": False,
        "trigger_payroll": False
    }
    
    try:
        result = await graph.ainvoke(basic_state, config)
        
        print("✅ Basic agent response completed!")
        print(f"  - Existing employees: {len(result.get('existing_employees', []))}")
        print(f"  - Updated employees: {len(result.get('updated_employees', []))}")
        print(f"  - Messages: {len(result.get('messages', []))}")
        
        # Show payroll data if available
        if result.get('current_pay_data'):
            payroll = result['current_pay_data']
            print(f"  - Payroll generated: {len(payroll.employees)} employees")
            print(f"  - Total payroll: ${payroll.total_payroll}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n" + "=" * 50)
    
    # Test 2: Agent with user message
    print("📋 Test 2: Agent with User Message")
    
    message_state = {
        "messages": [{"role": "user", "content": "Show me the current employee list and generate payroll"}],
        "existing_employees": [
            EmployeeData(name="John Doe", payrate=25.0, regular_hours=40.0, overtime_hours=5.0),
        ],
        "updated_employees": [
            EmployeeData(name="Alice Williams", payrate=28.0, regular_hours=35.0, overtime_hours=3.0),
        ],
        "current_employees": [],
        "document_uploaded": False,
        "user_approval": False,
        "trigger_payroll": False
    }
    
    try:
        result = await graph.ainvoke(message_state, config)
        
        print("✅ User message response completed!")
        print(f"  - Messages: {len(result.get('messages', []))}")
        
        # Show final messages
        messages = result.get('messages', [])
        if messages:
            print(f"\n💬 Agent Response:")
            for msg in messages[-2:]:
                if hasattr(msg, 'content'):
                    print(f"  {msg.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ 3-Agent system test completed successfully!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_agent_system())
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed!")
        sys.exit(1) 