import streamlit as st
import asyncio
import base64
import os
from typing import Optional
from PIL import Image
import io
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the payroll agent
from src.react_agent.graph import graph
from src.react_agent.configuration import Configuration
from src.react_agent.state import EmployeeData, PayrollReport

# Page configuration
st.set_page_config(
    page_title="Payroll Agent",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Main header */
    .main-header {
        text-align: center;
        color: #4fc3f7;
        margin-bottom: 2rem;
        font-weight: bold;
        font-size: 2.5rem;
        text-shadow: 0 0 20px rgba(79, 195, 247, 0.5);
        animation: headerGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes headerGlow {
        from {
            text-shadow: 0 0 20px rgba(79, 195, 247, 0.5);
        }
        to {
            text-shadow: 0 0 30px rgba(79, 195, 247, 0.8), 0 0 40px rgba(79, 195, 247, 0.3);
        }
    }
    
    /* Upload container - dark theme */
    .upload-container {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #4fc3f7;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .upload-container:hover {
        border-color: #29b6f6;
        background-color: #262626;
        box-shadow: 0 6px 12px rgba(79, 195, 247, 0.2);
    }
    
    /* Button styling - dark theme */
    .stButton > button {
        background: linear-gradient(90deg, #4fc3f7, #29b6f6);
        color: #0e1117;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        padding: 0.6rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(79, 195, 247, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #29b6f6, #0288d1);
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(79, 195, 247, 0.4);
    }
    
    /* Success/Info messages - dark theme */
    .stSuccess {
        background-color: #1b4332;
        color: #a7f3d0;
        border: 1px solid #059669;
        border-radius: 8px;
        text-align: center;
    }
    
    .stError {
        background-color: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #dc2626;
        border-radius: 8px;
    }
    
    .stInfo {
        background-color: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #2563eb;
        border-radius: 8px;
    }
    
    /* Text elements */
    .stMarkdown, .stText {
        color: #fafafa !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    /* Payroll report - dark theme */
    .payroll-report {
        background: linear-gradient(135deg, #1e1e1e, #2a2a2a);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #4fc3f7;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    
    /* Employee cards - dark theme */
    .employee-card {
        background: linear-gradient(135deg, #262626, #303030);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #404040;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .employee-card:hover {
        border-color: #4fc3f7;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(79, 195, 247, 0.2);
    }
    
    /* Tracing info - dark theme */
    .tracing-info {
        background: linear-gradient(135deg, #0d4f3c, #10593f);
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid #10b981;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #a7f3d0;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        background-color: #1e1e1e;
        border-radius: 10px;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background-color: transparent;
    }
    
    .stFileUploader label {
        color: #fafafa !important;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background-color: #1a1a1a;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4fc3f7;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #29b6f6;
    }
    
    /* Loading spinner */
    .stSpinner {
        color: #4fc3f7 !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #1e1e1e;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Links */
    a {
        color: #4fc3f7 !important;
        text-decoration: none;
    }
    
    a:hover {
        color: #29b6f6 !important;
        text-decoration: underline;
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stTextInput, .stNumberInput {
        background-color: #1e1e1e;
        color: #fafafa;
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e1e;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #fafafa;
    }
    
    .stTabs [aria-selected="true"] {
        color: #4fc3f7 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
# Initialize session state for existing employees
if 'existing_employees' not in st.session_state:
    st.session_state.existing_employees = [
        EmployeeData(name="John Doe", payrate=25.0, regular_hours=40.0, overtime_hours=5.0),
        EmployeeData(name="Jane Smith", payrate=30.0, regular_hours=38.5, overtime_hours=2.5),
        EmployeeData(name="Bob Johnson", payrate=22.0, regular_hours=42.0, overtime_hours=8.0),
    ]
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = None
if 'current_payroll_report' not in st.session_state:
    st.session_state.current_payroll_report = None

def get_agent_config():
    """Get the standard configuration for the agent."""
    return {
        "configurable": {
            "vlm_model": "openai/gpt-4o",
            "react_model": "openai/gpt-4o",
            "max_employees": 1000,
            "currency_symbol": "$",
            "default_overtime_multiplier": 1.5
        }
    }

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract content."""
    if uploaded_file.type.startswith('image/'):
        # Convert image to base64 for vision model
        image = Image.open(uploaded_file)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    else:
        # Extract text content
        try:
            return uploaded_file.read().decode('utf-8')
        except:
            return f"[Document uploaded: {uploaded_file.name}]"

async def run_payroll_workflow_with_document(document_content: str):
    """Run the payroll agent workflow with document processing."""
    # Convert EmployeeData instances to dictionaries
    existing_employees_dict = [
        {
            "name": emp.name,
            "regular_hours": emp.regular_hours,
            "overtime_hours": emp.overtime_hours,
            "payrate": emp.payrate
        } for emp in st.session_state.existing_employees
    ]
    
    initial_state = {
        "messages": [],
        "existing_employees": existing_employees_dict,
        "updated_employees": [],
        "current_employees": [],
        "document_uploaded": True,  # This triggers the VLM processing
        "document_content": document_content,
        "user_approval": False,
        "trigger_payroll": False,
        "current_pay_data": None
    }
    
    result = await graph.ainvoke(initial_state, config=get_agent_config())
    return result

async def get_agent_response(user_message: str):
    """Get response from the React agent for a user message."""
    from langchain_core.messages import HumanMessage
    
    # Get current workflow state or create minimal initial state
    current_state = st.session_state.workflow_state if st.session_state.workflow_state else {}
    
    # Convert EmployeeData instances to dictionaries
    existing_employees_dict = [
        {
            "name": emp.name,
            "regular_hours": emp.regular_hours,
            "overtime_hours": emp.overtime_hours,
            "payrate": emp.payrate
        } for emp in st.session_state.existing_employees
    ]
    
    # Continue existing workflow state, don't create fresh state
    if current_state:
        # We have an existing workflow - continue it
        # Create a copy and add the new message
        continued_state = current_state.copy()
        
        # Add new message to existing messages
        existing_messages = continued_state.get('messages', [])
        continued_state['messages'] = existing_messages + [HumanMessage(content=user_message)]
        
        # Ensure we have the latest existing employees
        continued_state['existing_employees'] = existing_employees_dict
        
    else:
        # First interaction - create initial state
        continued_state = {
            "messages": [HumanMessage(content=user_message)],
            "existing_employees": existing_employees_dict,
            "updated_employees": [],
            "document_uploaded": False,  # For chat interactions, no document processing unless specified
            "document_content": None,
            "user_approval": False,
            "trigger_payroll": False,
            "current_pay_data": None
        }
    
    # Run the workflow with continued state
    result = await graph.ainvoke(continued_state, config=get_agent_config())
    
    # Update workflow state with the result
    if result:
        st.session_state.workflow_state = result
        
        # Check if we got a structured payroll report
        if result.get('current_pay_data'):
            st.session_state.current_payroll_report = result['current_pay_data']
    
    # Extract the agent's response from the result
    if result and result.get('messages'):
        last_message = result['messages'][-1]
        if hasattr(last_message, 'content'):
            return last_message.content
        elif isinstance(last_message, dict) and 'content' in last_message:
            return last_message['content']
    
    return "I'm here to help with your payroll needs! How can I assist you today?"

def display_payroll_report(payroll_data):
    """Display payroll report from dictionary or PayrollReport object."""
    from src.react_agent.state import PayrollReport, PayrollEmployee
    import pandas as pd
    import json
    
    # Convert dictionary to PayrollReport if needed
    if isinstance(payroll_data, dict):
        # Convert employee dictionaries back to PayrollEmployee objects
        employees = [PayrollEmployee(**emp) for emp in payroll_data.get('employees', [])]
        payroll_report = PayrollReport(
            employees=employees,
            total_payroll=payroll_data.get('total_payroll', 0),
            summary=payroll_data.get('summary', '')
        )
    else:
        payroll_report = payroll_data
    
    # Header
    st.markdown("## 📊 Payroll Report")
    st.markdown(f"**Total Payroll:** ${payroll_report.total_payroll:.2f}")
    st.markdown(f"**Summary:** {payroll_report.summary}")
    
    # Employee details in a clean table format
    st.markdown("#### Employee Details:")
    
    # Create a structured table
    employee_data = []
    for employee in payroll_report.employees:
        employee_data.append({
            "Name": employee.name,
            "Rate": f"${employee.payrate:.2f}/hr",
            "OT Rate": f"${employee.payrate * 1.5:.2f}/hr",
            "Regular Hours": f"{employee.regular_hours}",
            "Overtime Hours": f"{employee.overtime_hours}", 
            "Regular Pay": f"${employee.regular_pay:.2f}",
            "Overtime Pay": f"${employee.overtime_pay:.2f}",
            "Total Pay": f"${employee.total_pay:.2f}"
        })
    
    # Display as a dataframe
    if employee_data:
        df = pd.DataFrame(employee_data)
        st.dataframe(df, use_container_width=True)
    
    # Structured Output Options
    st.markdown("#### 📋 Structured Output Options:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Show JSON"):
            st.json(payroll_data)
    
    with col2:
        if st.button("📊 Download CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="payroll_report.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📋 Show Raw Data"):
            st.code(json.dumps(payroll_data, indent=2), language="json")

def display_tracing_info():
    """Display LangSmith tracing information."""
    st.markdown('<div class="tracing-info">', unsafe_allow_html=True)
    st.markdown("🔍 **LangSmith Tracing Active** | Project: **payroll-agent** | Real-time monitoring & debugging enabled")
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">💰 Payroll Agent Assistant</h1>', unsafe_allow_html=True)
    
    # Display tracing info
    display_tracing_info()
    
    # Document upload section
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown("**📄 Document Processing Hub**")
    st.markdown("*Drag & drop employee hours documents • AI-powered data extraction*")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['png', 'jpg', 'jpeg', 'pdf', 'txt', 'csv'],
        help="Upload documents containing employee hours for current pay period",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
        with col2:
            if st.button("🔄 Process Document", key="process_doc"):
                with st.spinner("Processing document with VLM..."):
                    document_content = process_uploaded_file(uploaded_file)
                    
                    # Run workflow with document
                    result = asyncio.run(run_payroll_workflow_with_document(document_content))
                    
                    if result:
                        st.session_state.workflow_state = result
                        
                        # Get the agent's response about document processing
                        if result.get('messages'):
                            agent_response = result['messages'][-1]
                            if hasattr(agent_response, 'content'):
                                response_content = agent_response.content
                            elif isinstance(agent_response, dict) and 'content' in agent_response:
                                response_content = agent_response['content']
                            else:
                                response_content = "Document processed successfully! I'll now help you review and approve any employee changes."
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_content
                            })
                            
                        # Check if we got a payroll report
                        if result.get('current_pay_data'):
                            st.session_state.current_payroll_report = result['current_pay_data']
                            
                        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display current payroll report if available
    if st.session_state.current_payroll_report:
        st.markdown("---")
        display_payroll_report(st.session_state.current_payroll_report)
    
    # Chat interface
    st.markdown("---")
    st.markdown("**💬 AI Assistant Chat**")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("💭 Ask about payroll, employees, or document processing..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response from React agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(get_agent_response(prompt))
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Check if we got a new payroll report
                    if st.session_state.current_payroll_report:
                        st.rerun()
                        
                except Exception as e:
                    error_msg = f"I'm sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    # Log more details for debugging
                    st.write("Error details:", e)
    
    # Add a reset button at the bottom
    if st.session_state.messages or st.session_state.current_payroll_report:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🗑️ Reset All Data"):
                st.session_state.messages = []
                st.session_state.workflow_state = None
                st.session_state.current_payroll_report = None
                st.rerun()

if __name__ == "__main__":
    main() 