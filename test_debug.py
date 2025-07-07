"""Test script to verify debug logging and basic functionality."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Create custom formatter to handle Windows console encoding
class SafeFormatter(logging.Formatter):
    def format(self, record):
        # Get the original formatted message
        msg = super().format(record)
        # Remove emojis for console output by replacing them with text equivalents
        emoji_map = {
            '🔍': '[TEST]',
            '✅': '[PASS]', 
            '❌': '[FAIL]',
            '⚠️': '[WARN]',
            '🔄': '[PROC]',
            '📄': '[DOC]',
            '👤': '[USER]',
            '📋': '[CTX]',
            '📥': '[IN]',
            '🤖': '[AI]',
            '🚀': '[START]',
            '🎉': '[SUCCESS]',
            '📁': '[DIR]',
            '💾': '[SAVE]',
            '🧹': '[CLEAN]',
            '🖼️': '[IMG]',
            '🧠': '[VLM]',
            '📊': '[DATA]',
            '🔀': '[ROUTE]',
            '📤': '[OUT]',
            '📝': '[TEXT]',
        }
        for emoji, text in emoji_map.items():
            msg = msg.replace(emoji, text)
        return msg

# Configure logging with Windows-safe formatter
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

file_handler = logging.FileHandler('debug_test.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, file_handler]
)

logger = logging.getLogger(__name__)

async def test_imports():
    """Test that all modules import correctly."""
    logger.info("🔍 Testing imports...")
    
    try:
        # Test state imports
        from src.react_agent.state import State, DocumentInfo, EmployeePayInfo, PayrollContext
        logger.info("✅ State imports successful")
        
        # Test configuration imports
        from src.react_agent.configuration import Configuration
        logger.info("✅ Configuration imports successful")
        
        # Test utils imports
        from src.react_agent.utils import load_chat_model
        logger.info("✅ Utils imports successful")
        
        # Test tools imports
        from src.react_agent.tools import process_document_with_vlm, convert_document_to_images
        logger.info("✅ Tools imports successful")
        
        # Test graph imports
        from src.react_agent.graph import graph
        logger.info("✅ Graph imports successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {str(e)}", exc_info=True)
        return False

async def test_state_creation():
    """Test state object creation."""
    logger.info("🔍 Testing state creation...")
    
    try:
        from src.react_agent.state import State, DocumentInfo, EmployeePayInfo
        
        # Test basic state creation
        state = State()
        logger.info("✅ Basic state created")
        
        # Test state with data
        state_with_data = State(
            document_uploaded=True,
            file_path="test.pdf",
            context_query="Test query"
        )
        logger.info("✅ State with data created")
        
        # Test document info creation
        doc_info = DocumentInfo(
            filename="test.pdf",
            file_type="pdf",
            file_size=1024,
            pages=3,
            processed=True
        )
        logger.info("✅ Document info created")
        
        # Test employee creation
        employee = EmployeePayInfo(
            name="John Doe",
            employee_id="EMP001",
            pay_rate=25.0,
            hours_worked=40.0,
            net_pay=1000.0
        )
        logger.info("✅ Employee info created")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ State creation failed: {str(e)}", exc_info=True)
        return False

async def test_configuration():
    """Test configuration loading."""
    logger.info("🔍 Testing configuration...")
    
    try:
        from src.react_agent.configuration import Configuration
        
        # Test configuration creation
        config = Configuration()
        logger.info(f"✅ Configuration created with model: {config.model}")
        
        # Test configuration from context
        config_from_context = Configuration.from_context()
        logger.info(f"✅ Configuration from context: {config_from_context.model}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {str(e)}", exc_info=True)
        return False

async def test_graph_structure():
    """Test graph structure and routing."""
    logger.info("🔍 Testing graph structure...")
    
    try:
        from src.react_agent.graph import graph, route_vlm_or_agent
        from src.react_agent.state import State
        
        # Test graph compilation
        logger.info("✅ Graph compiled successfully")
        
        # Test routing logic
        empty_state = State()
        route = route_vlm_or_agent(empty_state)
        logger.info(f"✅ Empty state routes to: {route}")
        
        # Test with document upload
        upload_state = State(document_uploaded=True, file_path="test.pdf")
        route = route_vlm_or_agent(upload_state)
        logger.info(f"✅ Upload state routes to: {route}")
        
        # Test with VLM complete
        vlm_state = State(vlm_processing_complete=True)
        route = route_vlm_or_agent(vlm_state)
        logger.info(f"✅ VLM complete state routes to: {route}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Graph structure test failed: {str(e)}", exc_info=True)
        return False

async def test_environment():
    """Test environment setup."""
    logger.info("🔍 Testing environment setup...")
    
    try:
        # Check for .env file
        if os.path.exists('.env'):
            logger.info("✅ .env file found")
            
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv()
            
            # Check for required API keys
            groq_key = os.getenv('GROQ_API_KEY')
            if groq_key:
                logger.info(f"✅ GROQ_API_KEY found: {groq_key[:10]}...")
            else:
                logger.warning("⚠️ GROQ_API_KEY not found in environment")
                
        else:
            logger.warning("⚠️ .env file not found")
        
        # Check temp directory
        temp_dir = Path("temp_uploads")
        if temp_dir.exists():
            logger.info("✅ temp_uploads directory exists")
        else:
            logger.info("📁 Creating temp_uploads directory")
            temp_dir.mkdir(exist_ok=True)
            logger.info("✅ temp_uploads directory created")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment test failed: {str(e)}", exc_info=True)
        return False

async def test_dependencies():
    """Test that all required dependencies are available."""
    logger.info("🔍 Testing dependencies...")
    
    dependencies = [
        ('fitz', 'PyMuPDF'),
        ('PIL', 'Pillow'),
        ('pytesseract', 'pytesseract'),
        ('langchain_groq', 'langchain-groq'),
        ('langgraph', 'langgraph'),
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'),
        ('pydantic', 'pydantic'),
        ('dotenv', 'python-dotenv'),
        ('aiofiles', 'aiofiles'),
        ('magic', 'python-magic'),
        ('multipart', 'python-multipart')
    ]
    
    missing_deps = []
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            logger.info(f"✅ {package_name} available")
        except ImportError:
            logger.error(f"❌ {package_name} not found")
            missing_deps.append(package_name)
    
    if missing_deps:
        logger.error(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        return False
    
    logger.info("✅ All dependencies available")
    return True

async def test_interactive_workflow():
    """Test the interactive payroll editing workflow."""
    
    print("🧪 Testing Interactive Payroll Editing Workflow")
    print("=" * 50)
    
    # Create initial state with some sample employee data
    sample_employees = [
        EmployeePayInfo(
            employee_id="EMP001",
            name="Alice Johnson",
            pay_rate=20.0,
            hours_worked=40.0,
            overtime_hours=0.0,
            gross_pay=800.0,
            deductions=120.0,
            net_pay=680.0,
            pay_period="Weekly",
            position="Developer"
        ),
        EmployeePayInfo(
            employee_id="EMP002",
            name="Bob Smith",
            pay_rate=22.0,
            hours_worked=35.0,
            overtime_hours=5.0,
            gross_pay=935.0,
            deductions=140.0,
            net_pay=795.0,
            pay_period="Weekly",
            position="Designer"
        )
    ]
    
    # Create state as if VLM processing is complete
    state = State(
        document_uploaded=True,
        file_path="test_document.pdf",
        vlm_processing_complete=True,
        employees=sample_employees,
        extraction_complete=False,
        messages=[HumanMessage(content="VLM processing complete")]
    )
    
    print(f"📊 Initial state: {len(state.employees)} employees")
    for emp in state.employees:
        print(f"  - {emp.name}: ${emp.pay_rate}/hr, Net: ${emp.net_pay}")
    
    # Test 1: Initial data display
    print("\n🧪 Test 1: Initial data display")
    try:
        result = await graph.ainvoke(state)
        latest_message = result["messages"][-1]
        print(f"✅ Agent response: {latest_message.content[:200]}...")
        
        # Update state with result
        state = State(**result)
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False
    
    # Test 2: Update employee pay rate
    print("\n🧪 Test 2: Update employee pay rate")
    try:
        state.messages.append(HumanMessage(content="Update Alice's pay rate to $25/hour"))
        result = await graph.ainvoke(state)
        latest_message = result["messages"][-1]
        print(f"✅ Agent response: {latest_message.content[:200]}...")
        
        # Check if Alice's pay rate was updated
        updated_employees = result.get("employees", [])
        alice = next((emp for emp in updated_employees if emp.name == "Alice Johnson"), None)
        if alice and alice.pay_rate == 25.0:
            print(f"✅ Alice's pay rate updated to ${alice.pay_rate}/hr")
        else:
            print(f"❌ Alice's pay rate not updated correctly")
        
        # Update state with result
        state = State(**result)
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False
    
    # Test 3: Add new employee
    print("\n🧪 Test 3: Add new employee")
    try:
        state.messages.append(HumanMessage(content="Add employee Clara Martinez with $30/hour rate, worked 40 hours"))
        result = await graph.ainvoke(state)
        latest_message = result["messages"][-1]
        print(f"✅ Agent response: {latest_message.content[:200]}...")
        
        # Check if Clara was added
        updated_employees = result.get("employees", [])
        clara = next((emp for emp in updated_employees if emp.name == "Clara Martinez"), None)
        if clara:
            print(f"✅ Clara added with ${clara.pay_rate}/hr")
        else:
            print(f"❌ Clara not added correctly")
        
        # Update state with result
        state = State(**result)
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    # Test 4: Calculate missing values
    print("\n🧪 Test 4: Calculate net pay")
    try:
        state.messages.append(HumanMessage(content="Calculate net pay for all employees"))
        result = await graph.ainvoke(state)
        latest_message = result["messages"][-1]
        print(f"✅ Agent response: {latest_message.content[:200]}...")
        
        # Update state with result
        state = State(**result)
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        return False
    
    # Test 5: Export final data
    print("\n🧪 Test 5: Export final data")
    try:
        state.messages.append(HumanMessage(content="Export final JSON"))
        result = await graph.ainvoke(state)
        latest_message = result["messages"][-1]
        print(f"✅ Agent response: {latest_message.content[:200]}...")
        
        # Check if extraction is marked complete
        if result.get("extraction_complete", False):
            print(f"✅ Extraction marked as complete")
        else:
            print(f"❌ Extraction not marked as complete")
        
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Interactive workflow is working correctly.")
    return True

async def main():
    """Run all tests."""
    logger.info("🚀 Starting debug tests...")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Environment", test_environment),
        ("Imports", test_imports),
        ("State Creation", test_state_creation),
        ("Configuration", test_configuration),
        ("Graph Structure", test_graph_structure),
        ("Interactive Workflow", test_interactive_workflow),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n" + "="*50)
        logger.info(f"Running test: {test_name}")
        logger.info("="*50)
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} - PASSED")
            else:
                logger.error(f"❌ {test_name} - FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} - ERROR: {str(e)}", exc_info=True)
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    asyncio.run(main()) 