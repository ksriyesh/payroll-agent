"""Simple Streamlit chat interface for the payroll agent."""

import streamlit as st
import requests
import base64
import hashlib
from typing import Optional

# Configuration
API_URL = "http://localhost:8000"
SUPPORTED_FILE_TYPES = ["png", "jpg", "jpeg", "pdf", "docx", "txt", "xlsx", "csv"]

# Page configuration
st.set_page_config(
    page_title="Payroll Agent",
    page_icon="💼",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = {
        "messages": [],
        "updated_employees": [],
        "existing_employees": [],
        "updates_list": [],
        "temp_merged_list": [],
        "document_uploaded": False,
        "document_processing_done": False,
        "user_approval": False,
        "trigger_payroll": False,
        "current_pay_data": None,
        "file_data": None,
        "file_path": None,
        "file_type": None
    }
if 'last_uploaded_file_hash' not in st.session_state:
    st.session_state.last_uploaded_file_hash = None
if 'processing_message' not in st.session_state:
    st.session_state.processing_message = False
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False

def get_file_hash(file_bytes):
    """Generate a hash for file bytes to track unique uploads."""
    return hashlib.md5(file_bytes).hexdigest()

def check_api_status():
    """Check if the backend API is running."""
    try:
        response = requests.get(f"{API_URL}/")
        return response.status_code == 200
    except requests.RequestException:
        return False

def get_agent_response(user_message, file_data=None, file_path=None, file_type=None):
    """Get response from the agent through the chat endpoint."""
    if not check_api_status():
        return "❌ Cannot connect to the backend API. Please make sure the backend server is running."
    
    # Prevent duplicate processing
    if st.session_state.processing_message:
        return "Please wait, still processing your previous message..."
    
    st.session_state.processing_message = True
    
    try:
        # Prepare the request data - use workflow_state structure
        data = {
            "content": user_message,
            "existing_employees": st.session_state.workflow_state.get("existing_employees", []),
            "updated_employees": st.session_state.workflow_state.get("updated_employees", []),
            "user_approval": st.session_state.workflow_state.get("user_approval", False),
            "trigger_payroll": st.session_state.workflow_state.get("trigger_payroll", False),
            "current_pay_data": st.session_state.workflow_state.get("current_pay_data"),
            "workflow_state": st.session_state.workflow_state,
            "file_data": file_data,  # Only sent when actually uploading a file
            "file_path": file_path,
            "file_type": file_type
        }
        
        response = requests.post(f"{API_URL}/chat", json=data)
            
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success", False):
                response_data = result.get("data", {})
                response_content = response_data.get("response", "Sorry, I couldn't process your request properly.")
                
                # Update workflow state if available
                if "workflow_state" in response_data:
                    new_workflow_state = response_data["workflow_state"]
                    # Update our session state with the new workflow state
                    st.session_state.workflow_state.update(new_workflow_state)
                
                return response_content
            else:
                error_msg = result.get("message", "Unknown error occurred")
                return f"❌ Error: {error_msg}"
        else:
            return f"❌ Error: API returned status code {response.status_code}"
            
    except Exception as e:
        return f"❌ Error communicating with backend: {str(e)}"
    finally:
        st.session_state.processing_message = False

def main():
    """Main application interface."""
    
    # Sidebar - Document Upload Section
    with st.sidebar:
        st.title("💼 Payroll Agent")
        
        # API status check
        if not check_api_status():
            st.error("❌ Backend API not connected")
            st.stop()
        else:
            st.success("✅ Connected")
        
        st.markdown("---")
        
        # Document Upload
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload timesheet or payroll document", 
            type=SUPPORTED_FILE_TYPES,
            help="Supported: PNG, JPG, PDF, DOCX, TXT, XLSX, CSV",
            key="document_uploader"
        )
        
        # Process uploaded file when it changes (using hash comparison)
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            current_file_hash = get_file_hash(file_bytes)
            
            # Check if this is a new file (different hash)
            is_new_file = current_file_hash != st.session_state.last_uploaded_file_hash
            
            # Only process if this is a new file AND it hasn't been processed yet
            if is_new_file:
                # Reset processing flag for new file
                st.session_state.file_processed = False
            
            # Process if it's a new file and hasn't been processed
            if is_new_file and not st.session_state.file_processed:
                with st.spinner("Processing document..."):
                    # Read and encode the file
                    file_data = f"data:{uploaded_file.type};base64,{base64.b64encode(file_bytes).decode('utf-8')}"
                    
                    # Update file hash
                    st.session_state.last_uploaded_file_hash = current_file_hash
                    
                    # Send to agent for processing
                    message = f"I've uploaded a document: {uploaded_file.name}. Please process it."
                    response = get_agent_response(message, file_data, uploaded_file.name, uploaded_file.type)
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "user", "content": f"📄 Uploaded: {uploaded_file.name}"})
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Mark this specific file as processed
                    st.session_state.file_processed = True
                    
                    st.success("✅ Document processed!")
                    # Remove st.rerun() - let natural flow handle updates
            elif not is_new_file:
                st.sidebar.text("ℹ️ Same file - no processing needed")
            elif st.session_state.file_processed:
                st.sidebar.text("✅ File already processed")
        
        st.markdown("---")
        
        # Current Status Display
        st.header("📊 Status")
        if st.session_state.workflow_state.get("current_pay_data"):
            st.success("**Payroll Generated**\nReady for final processing")
        elif st.session_state.workflow_state.get("temp_merged_list"):
            st.info("**Ready for Confirmation**\nReview merged employee data")
        elif st.session_state.workflow_state.get("updated_employees"):
            num_employees = len(st.session_state.workflow_state["updated_employees"])
            st.success(f"**{num_employees} Employees Confirmed**\nReady for modifications")
        elif st.session_state.workflow_state.get("document_processing_done"):
            st.info("**Document Processed**\nData extracted and ready")
        elif uploaded_file is not None:
            if st.session_state.file_processed:
                st.success("**Document Processed**\nReady for chat interaction")
            else:
                st.info("**Document Uploaded**\nProcessing automatically...")
        else:
            st.info("**Ready**\nUpload a document to start")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", help="Start a new conversation"):
            st.session_state.messages = []
            st.session_state.workflow_state = {
                "messages": [],
                "updated_employees": [],
                "existing_employees": [],
                "updates_list": [],
                "temp_merged_list": [],
                "document_uploaded": False,
                "document_processing_done": False,
                "user_approval": False,
                "trigger_payroll": False,
                "current_pay_data": None,
                "file_data": None,
                "file_path": None,
                "file_type": None
            }
            st.session_state.last_uploaded_file_hash = None
            st.session_state.processing_message = False
            st.session_state.file_processed = False
            st.rerun()
    
    # Main Area - Chat Interface
    st.header("💬 Chat with Agent")
    
    # Chat container with full height
    chat_container = st.container()
    with chat_container:
        # Display chat messages
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        else:
            # Welcome message when no chat history
            with st.chat_message("assistant"):
                st.markdown("👋 Hello! I'm your payroll assistant. Upload a document in the sidebar to get started, or ask me any questions about payroll processing.")
    
    # Chat input (always at bottom)
    if prompt := st.chat_input("Type your message here..."):
        # Prevent processing if already processing
        if st.session_state.processing_message:
            st.warning("Please wait, still processing your previous message...")
            st.stop()
        
        # Prevent duplicate messages
        if (st.session_state.messages and 
            st.session_state.messages[-1]["role"] == "user" and 
            st.session_state.messages[-1]["content"] == prompt):
            st.warning("Message already sent, please wait for response...")
            st.stop()
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response (NO file data for regular chat)
        with st.spinner("Agent is thinking..."):
            response = get_agent_response(prompt)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Remove st.rerun() - Streamlit will naturally update the display

if __name__ == "__main__":
    main()