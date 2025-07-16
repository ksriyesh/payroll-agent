"""Streamlit frontend for the payroll agent."""

import streamlit as st
import requests
import base64
import os
import io
import json
from typing import Optional, List, Dict, Any
from PIL import Image
import pandas as pd

# Define API URL (can be changed for production)
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Payroll Agent",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling (same as original app)
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
    st.session_state.existing_employees = []
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = {
        "updated_employees": [],
        "current_employees": [],
        "user_approval": False,
        "trigger_payroll": False,
        "current_pay_data": None
    }
if 'current_payroll_report' not in st.session_state:
    st.session_state.current_payroll_report = None
if 'api_status' not in st.session_state:
    st.session_state.api_status = None
# Initialize app mode (document processing or chat)
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "chat"  # Default to chat mode
# Define supported file types
SUPPORTED_FILE_TYPES = ["png", "jpg", "jpeg", "pdf", "docx", "txt", "xlsx", "csv"]

def check_api_status():
    """Check if the backend API is running."""
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            st.session_state.api_status = "connected"
            return True
        else:
            st.session_state.api_status = "error"
            return False
    except requests.RequestException:
        st.session_state.api_status = "error"
        return False

def process_uploaded_file(uploaded_file, mode="document"):
    """Process uploaded file by sending it to the appropriate API endpoint.
    
    Args:
        uploaded_file: The uploaded file object
        mode: Either "document" for document processing mode or "chat" for chat mode
    """
    if not check_api_status():
        st.error("❌ Cannot connect to the backend API. Please make sure the backend server is running.")
        return None
    
    try:
        # Read and encode the file
        file_bytes = uploaded_file.getvalue()
        file_data = base64.b64encode(file_bytes).decode('utf-8')
        file_path = uploaded_file.name
        file_type = uploaded_file.type
        
        # Verify file extension is supported
        file_extension = file_path.split('.')[-1].lower()
        if file_extension not in SUPPORTED_FILE_TYPES:
            st.error(f"❌ Unsupported file type: {file_extension}. Supported types: {', '.join(SUPPORTED_FILE_TYPES)}")
            return None
        
        # Process based on mode
        if mode == "document":
            # Use the /process-document endpoint
            with st.spinner("Processing document..."):
                # Prepare the file for upload
                files = {"file": (file_path, file_bytes, file_type)}
                
                # Send to process-document API
                response = requests.post(f"{API_URL}/process-document", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    if result["success"]:
                        # Update workflow state with extracted employees
                        if "data" in result and "updated_employees" in result["data"]:
                            extracted_employees = result["data"]["updated_employees"]
                            st.session_state.workflow_state["updated_employees"] = extracted_employees
                            
                            # Add the interaction to chat history
                            st.session_state.messages.append({"role": "user", "content": f"I'm uploading {file_path} for processing."})  
                            st.session_state.messages.append({"role": "assistant", "content": f"✅ Document processed successfully. Found {len(extracted_employees)} employees."})  
                            
                            return {
                                "updated_employees": extracted_employees,
                                "message": f"Document processed successfully. Found {len(extracted_employees)} employees."
                            }
                        else:
                            st.error(f"❌ Error: No employee data found in the response")
                            return None
                    else:
                        st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
                        return None
                else:
                    st.error(f"❌ Error: API returned status code {response.status_code}")
                    return None
        
        elif mode == "chat":
            # Add a default message for document processing in chat mode
            message = f"Please process this {file_type} document and extract employee data."
            
            # Call the unified chat endpoint
            with st.spinner("Processing document in chat mode..."):
                # Use our get_agent_response function which supports file uploads
                response_content = get_agent_response(message, file_data, file_path, file_type)
                
                # Add the interaction to chat history
                st.session_state.messages.append({"role": "user", "content": f"I'm uploading {file_path} for processing."})  
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                
            # Return the updated workflow state
            return {
                "updated_employees": st.session_state.workflow_state.get("updated_employees", []),
                "message": response_content
            }
        
        else:
            st.error(f"❌ Invalid mode: {mode}. Must be either 'document' or 'chat'.")
            return None
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return None

def merge_employees():
    """Merge existing and updated employee lists using the backend API."""
    if not check_api_status():
        st.error("❌ Cannot connect to the backend API. Please make sure the backend server is running.")
        return None
    
    try:
        # Prepare the request data
        data = {
            "request": {"employees": st.session_state.existing_employees},
            "updated": {"employees": st.session_state.workflow_state["updated_employees"]}
        }
        
        with st.spinner("Merging employee data..."):
            response = requests.post(f"{API_URL}/merge-employees", json=data)
            
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                # Update session state with the merged employees
                if "merged_employees" in result["data"]:
                    st.session_state.workflow_state["current_employees"] = result["data"]["merged_employees"]
                    st.session_state.workflow_state["user_approval"] = True
                return result["data"]
            else:
                st.error(f"❌ Error: {result['message']}")
                return None
        else:
            st.error(f"❌ Error: API returned status code {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Error merging employees: {str(e)}")
        return None

def generate_payroll():
    """Generate payroll report using the backend API."""
    if not check_api_status():
        st.error("❌ Cannot connect to the backend API. Please make sure the backend server is running.")
        return None
    
    try:
        # Use current employees or existing employees if no current employees
        employees = st.session_state.workflow_state["current_employees"] or st.session_state.existing_employees
        
        # Prepare the request data
        data = {"employees": employees}
        
        with st.spinner("Generating payroll report..."):
            response = requests.post(f"{API_URL}/generate-payroll", json=data)
            
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                # Update session state with the payroll report
                if "payroll_report" in result["data"]:
                    st.session_state.current_payroll_report = result["data"]["payroll_report"]
                    st.session_state.workflow_state["current_pay_data"] = result["data"]["payroll_report"]
                    st.session_state.workflow_state["trigger_payroll"] = True
                return result["data"]
            else:
                st.error(f"❌ Error: {result['message']}")
                return None
        else:
            st.error(f"❌ Error: API returned status code {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Error generating payroll: {str(e)}")
        return None

def get_agent_response(user_message, file_data=None, file_path=None, file_type=None):
    """Get response from the agent through the unified chat endpoint.
    
    Args:
        user_message: The text message from the user
        file_data: Optional base64-encoded file data
        file_path: Optional file path/name
        file_type: Optional MIME type of the file
    """
    if not check_api_status():
        st.error("❌ Cannot connect to the backend API. Please make sure the backend server is running.")
        return "I'm unable to process your request because the backend API is not available. Please make sure the backend server is running."
    
    try:
        # Verify file extension if provided
        if file_path:
            file_extension = file_path.split('.')[-1].lower()
            if file_extension not in SUPPORTED_FILE_TYPES:
                return f"❌ Unsupported file type: {file_extension}. Supported types: {', '.join(SUPPORTED_FILE_TYPES)}"
        
        # Prepare the request data for the unified chat endpoint
        data = {
            "content": user_message,
            "existing_employees": st.session_state.existing_employees,
            "updated_employees": st.session_state.workflow_state["updated_employees"],
            "current_employees": st.session_state.workflow_state["current_employees"],
            "user_approval": st.session_state.workflow_state["user_approval"],
            "trigger_payroll": st.session_state.workflow_state["trigger_payroll"],
            "current_pay_data": st.session_state.workflow_state["current_pay_data"],
            "file_data": file_data,
            "file_path": file_path,
            "file_type": file_type
        }
        
        # Show appropriate spinner message based on what's being processed
        spinner_message = "Processing"
        if file_data and user_message.strip():
            spinner_message += " message and document (multimodal input)"
        elif file_data:
            spinner_message += " document"
        else:
            spinner_message += " message"
        
        with st.spinner(f"{spinner_message}..."):
            response = requests.post(f"{API_URL}/chat", json=data)
            
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success", False):
                # Extract response data
                response_data = result.get("data", {})
                response_content = response_data.get("response", "Sorry, I couldn't process your request properly.")
                
                # Update workflow state if available
                if "workflow_state" in response_data:
                    workflow_state = response_data["workflow_state"]
                    st.session_state.workflow_state.update(workflow_state)
                    
                    # Update employee lists if needed
                    if workflow_state.get("updated_employees"):
                        st.session_state.workflow_state["updated_employees"] = workflow_state["updated_employees"]
                    
                    if workflow_state.get("current_employees"):
                        st.session_state.workflow_state["current_employees"] = workflow_state["current_employees"]
                    
                    # Update user approval and trigger payroll flags
                    st.session_state.workflow_state["user_approval"] = workflow_state.get("user_approval", False)
                    st.session_state.workflow_state["trigger_payroll"] = workflow_state.get("trigger_payroll", False)
                    
                    # Check if we got a structured payroll report
                    if workflow_state.get("current_pay_data"):
                        st.session_state.current_payroll_report = workflow_state["current_pay_data"]
                
                return response_content
            else:
                return f"Error: {result.get('message', 'Unknown error occurred')}"
        else:
            return f"Error: Failed to get response from the backend (Status code: {response.status_code})"
    except Exception as e:
        return f"Error processing your request: {str(e)}"

def display_payroll_report(payroll_data):
    """Display payroll report from dictionary."""
    # Header
    st.markdown("## 📊 Payroll Report")
    st.markdown(f"**Total Payroll:** ${payroll_data['total_payroll']:.2f}")
    st.markdown(f"**Summary:** {payroll_data['summary']}")
    
    # Employee details in a clean table format
    st.markdown("#### Employee Details:")
    
    # Create a structured table
    employee_data = []
    for employee in payroll_data['employees']:
        employee_data.append({
            "Name": employee["name"],
            "Rate": f"${employee['payrate']:.2f}/hr",
            "OT Rate": f"${employee['payrate'] * 1.5:.2f}/hr",
            "Regular Hours": f"{employee['regular_hours']}",
            "OT Hours": f"{employee['overtime_hours']}",
            "Regular Pay": f"${employee['regular_pay']:.2f}",
            "OT Pay": f"${employee['overtime_pay']:.2f}",
            "Total": f"${employee['total_pay']:.2f}"
        })
    
    # Display as a table
    df = pd.DataFrame(employee_data)
    st.table(df)
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export as CSV"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="payroll_report.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("Export as JSON"):
            json_str = json.dumps(payroll_data, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="payroll_report.json">Download JSON File</a>'
            st.markdown(href, unsafe_allow_html=True)

def display_api_status():
    """Display the API connection status."""
    if st.session_state.api_status == "connected":
        st.sidebar.success("✅ Connected to backend API")
    elif st.session_state.api_status == "error":
        st.sidebar.error("❌ Cannot connect to backend API")
        st.sidebar.info("Make sure the backend server is running on http://localhost:8000")
    else:
        check_api_status()

def main():
    """Main application function with dual mode support."""
    # Display header
    st.markdown('<h1 class="main-header">💰 Payroll Agent</h1>', unsafe_allow_html=True)
    
    # Check API status
    display_api_status()
    
    # Mode selector in sidebar - make it more prominent
    st.sidebar.markdown("<h2 style='text-align: center; color: #4fc3f7;'>🔄 MODE SELECTION</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='background-color: #1e3a8a; padding: 10px; border-radius: 10px; border: 2px solid #4fc3f7;'>", unsafe_allow_html=True)
    mode = st.sidebar.radio(
        "Select Application Mode:",
        ["Document Processing Mode", "Chat Mode (Unified)"],
        index=0 if st.session_state.app_mode == "document" else 1,
        help="Document Processing Mode: Upload files only for processing. Chat Mode: Interact with text and/or files."
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    # Add mode descriptions
    if mode == "Document Processing Mode":
        st.sidebar.info("📄 **Document Processing Mode**: Upload payroll documents to extract employee data. The system will process the document and extract structured employee information.")
    else:
        st.sidebar.info("💬 **Chat Mode**: Have a conversation with the payroll agent. You can upload documents, ask questions, or request payroll calculations - all in one place!")
    
    # Update the app mode in session state
    st.session_state.app_mode = "document" if mode == "Document Processing Mode" else "chat"
    
    # Display mode-specific interface
    if st.session_state.app_mode == "document":
        # Document Processing Mode
        st.markdown("### 📄 Document Processing Mode")
        st.markdown("Upload payroll documents to extract employee data. Supported file types: png, jpg, jpeg, pdf, docx, txt, xlsx, csv")
        
        # File uploader for document processing mode
        uploaded_file = st.file_uploader(
            "Upload a payroll document", 
            type=SUPPORTED_FILE_TYPES,
            help="Upload a document containing employee payroll information"
        )
        
        # Process button
        if uploaded_file is not None:
            if st.button("Process Document", use_container_width=True):
                # Process the document using the document processing endpoint
                with st.spinner("Processing document..."):
                    result = process_uploaded_file(uploaded_file, mode="document")
                    if result and "updated_employees" in result:
                        num_employees = len(result['updated_employees'])
                        st.success(f"✅ Document processed successfully! Found {num_employees} employees.")
                        
                        # Show a preview of extracted employees
                        if num_employees > 0:
                            st.markdown("### Preview of Extracted Employees")
                            for i, emp in enumerate(result['updated_employees'][:3]):  # Show first 3 employees
                                st.markdown(f"""
                                <div class="employee-card">
                                    <strong>{emp['name']}</strong><br>
                                    Regular Hours: {emp['regular_hours']} | Overtime: {emp['overtime_hours']}<br>
                                    Pay Rate: ${emp['payrate']:.2f}/hr
                                </div>
                                """, unsafe_allow_html=True)
                            
                            if num_employees > 3:
                                st.info(f"... and {num_employees - 3} more employees. View all in the Employee Data tab.")
                        
                        # Prompt user to review data in the Employee Data tab
                        st.info("👉 Please review the extracted employee data in the Employee Data tab and confirm if it's correct.")
                        st.rerun()
    
    else:  # Chat Mode
        # Create a unified chat interface that can handle both text messages and file uploads
        st.markdown("### 💬 Chat Mode (Unified)")
        st.markdown("Ask questions, upload documents, or request payroll calculations - all in one place!")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # File uploader in the chat interface
        uploaded_file = st.file_uploader(
            "Upload a payroll document (optional)", 
            type=SUPPORTED_FILE_TYPES,
            help="You can upload a document and/or type a message below"
        )
        
        # Chat input
        prompt = st.chat_input("Ask a question or upload a document above...")
        
        # Only process when the user sends a message, not when they just upload a file
        if prompt:
            
            # If a file was uploaded, process it
            file_data = None
            file_path = None
            file_type = None
            
            if uploaded_file is not None:
                # Track the last uploaded file to prevent duplicate processing
                st.session_state["last_uploaded_file"] = uploaded_file.name
                
                # Read and encode the file
                file_bytes = uploaded_file.getvalue()
                file_data = base64.b64encode(file_bytes).decode('utf-8')
                file_path = uploaded_file.name
                file_type = uploaded_file.type
                
                # Add user message about the file upload
                if prompt:
                    message_content = f"I'm uploading {file_path} and asking: {prompt}"
                else:
                    message_content = f"I'm uploading {file_path} for processing."
                    prompt = f"Please process this {file_type} document and extract employee data."
                
                st.session_state.messages.append({"role": "user", "content": message_content})
            else:
                # Just a regular text message
                st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process the request using the unified chat endpoint
            response = get_agent_response(prompt, file_data, file_path, file_type)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force a rerun to update the UI
            st.rerun()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["👥 Employee Data", "📊 Payroll Report"])
    
    with tab1:
        st.markdown("### Current Employee Data")
        
        # Display existing employees if any
        if st.session_state.existing_employees:
            st.markdown("#### Existing Employees:")
            for i, emp in enumerate(st.session_state.existing_employees):
                st.markdown(f"""
                <div class="employee-card">
                    <strong>{emp['name']}</strong><br>
                    Regular Hours: {emp['regular_hours']} | Overtime: {emp['overtime_hours']}<br>
                    Pay Rate: ${emp['payrate']:.2f}/hr
                </div>
                """, unsafe_allow_html=True)
            
            # Generate payroll section when we have existing employees
            st.markdown("### 💰 Generate Payroll")
            st.info("Generate a payroll report using the current employee data.")
            
            if st.button("Generate Payroll Report", use_container_width=True):
                payroll_result = generate_payroll()
                if payroll_result and "payroll_report" in payroll_result:
                    st.success("✅ Payroll report generated successfully!")
                    st.rerun()
        else:
            # No existing employees yet
            st.info("No employee data available yet. Upload a document in either Document Processing Mode or Chat Mode to extract employee data.")
            
            # Provide guidance based on current mode
            if st.session_state.app_mode == "document":
                st.markdown("**Tip:** Use the document uploader above to process payroll documents.")
            else:
                st.markdown("**Tip:** Use the chat interface to upload documents or ask questions about payroll processing.")
        
        # Display updated employees if any were extracted
        if st.session_state.workflow_state["updated_employees"]:
            st.markdown("#### Recently Extracted Employees:")
            for emp in st.session_state.workflow_state["updated_employees"]:
                st.markdown(f"""
                <div class="employee-card">
                    <strong>{emp['name']}</strong><br>
                    Regular Hours: {emp['regular_hours']} | Overtime: {emp['overtime_hours']}<br>
                    Pay Rate: ${emp['payrate']:.2f}/hr
                </div>
                """, unsafe_allow_html=True)
            
            # User confirmation section
            st.markdown("### 🔍 Review and Confirm")
            st.info("Please review the extracted employee data above and confirm if it's correct.")
            
            # Merge button with confirmation
            confirm_col1, confirm_col2 = st.columns(2)
            with confirm_col1:
                if st.button("✅ Confirm and Merge Data", use_container_width=True):
                    merge_result = merge_employees()
                    if merge_result and "merged_employees" in merge_result:
                        st.success(f"Successfully merged {len(merge_result['merged_employees'])} employees!")
                        st.rerun()
            
            with confirm_col2:
                if st.button("❌ Discard Extracted Data", use_container_width=True):
                    # Reset the updated employees list
                    st.session_state.workflow_state["updated_employees"] = []
                    st.success("Extracted data discarded. You can upload another document or try again.")
                    st.rerun()
    
    with tab2:
        st.markdown("### Payroll Report")
        
        if st.session_state.current_payroll_report:
            display_payroll_report(st.session_state.current_payroll_report)
            
            # Add regenerate option
            if st.button("🔄 Regenerate Payroll Report", use_container_width=True):
                payroll_result = generate_payroll()
                if payroll_result and "payroll_report" in payroll_result:
                    st.success("✅ Payroll report regenerated successfully!")
                    st.rerun()
        else:
            # No payroll report yet
            if st.session_state.existing_employees:
                # We have employees but no report yet
                st.info("No payroll report has been generated yet. You can generate one using your current employee data.")
                st.markdown("### Generate Payroll Report")
                
                if st.button("Generate Payroll Report", use_container_width=True):
                    payroll_result = generate_payroll()
                    if payroll_result and "payroll_report" in payroll_result:
                        st.success("✅ Payroll report generated successfully!")
                        st.rerun()
            else:
                # No employees and no report
                st.info("No employee data available yet. You need to process documents first to extract employee data.")
                
                # Provide guidance based on current mode
                if st.session_state.app_mode == "document":
                    st.markdown("**Steps to generate a payroll report:**")
                    st.markdown("1. Switch to the Document Processing Mode in the sidebar")
                    st.markdown("2. Upload a payroll document")
                    st.markdown("3. Process the document to extract employee data")
                    st.markdown("4. Confirm the extracted data")
                    st.markdown("5. Generate the payroll report")
                else:
                    st.markdown("**Steps to generate a payroll report:**")
                    st.markdown("1. Use the Chat Mode to upload documents or ask questions")
                    st.markdown("2. Extract employee data from documents")
                    st.markdown("3. Confirm the extracted data")
                    st.markdown("4. Generate the payroll report")

if __name__ == "__main__":
    main()
