"""
Streamlit App for Payroll Document Processing Agent

This app provides a user-friendly interface to interact with the payroll agent,
including chat functionality and file upload for document processing.
"""

import streamlit as st
import asyncio
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config first
st.set_page_config(
    page_title="💰 Payroll Document Parser",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    logger.info("🚀 Starting Streamlit app")
    
    # Initialize basic session state first (no imports needed)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        logger.debug("Initialized messages")
    
    if 'processed_employees' not in st.session_state:
        st.session_state.processed_employees = []
        logger.debug("Initialized processed employees")
        
    # Try to initialize more complex state with imports
    try:
        from src.react_agent.state import State
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        if 'agent_state' not in st.session_state:
            st.session_state.agent_state = State()
            logger.debug("Initialized agent state")
            
    except Exception as e:
        logger.error(f"❌ Error initializing complex session state: {str(e)}", exc_info=True)
        # Don't show error to user for complex state, just use defaults
        if 'agent_state' not in st.session_state:
            st.session_state.agent_state = {}
            logger.debug("Initialized agent state as empty dict (fallback)")
    
    logger.debug("✅ Session state initialization completed")


def get_agent_state_attr(state_or_dict: Any, attr_name: str, default: Any = None) -> Any:
    """Safely get attribute from State object or dict."""
    logger.debug(f"Getting attribute: {attr_name}")
    try:
        if hasattr(state_or_dict, attr_name):
            value = getattr(state_or_dict, attr_name)
            logger.debug(f"Found attribute {attr_name} (State): {type(value)}")
            return value
        elif isinstance(state_or_dict, dict) and attr_name in state_or_dict:
            value = state_or_dict[attr_name]
            logger.debug(f"Found attribute {attr_name} (dict): {type(value)}")
            return value
        else:
            logger.debug(f"Attribute {attr_name} not found, using default: {default}")
            return default
    except Exception as e:
        logger.error(f"❌ Error getting attribute {attr_name}: {e}")
        return default


async def run_agent(state) -> Dict[str, Any]:
    """Run the agent with the given state."""
    logger.info("🤖 Starting agent execution")
    logger.debug(f"Input state type: {type(state)}")
    logger.debug(f"Document uploaded: {getattr(state, 'document_uploaded', False)}")
    
    try:
        # Import required modules
        from src.react_agent.graph import graph
        from src.react_agent.configuration import Configuration
        
        # Create configuration
        config = Configuration()
        logger.debug(f"Configuration created: {config.model}")
        
        # Create runnable config
        runnable_config = {
            "configurable": {
                "model": config.model,
                "system_prompt": config.system_prompt
            }
        }
        
        logger.debug("Runnable config created")
        
        # Run the graph
        logger.info("🔄 Executing graph")
        result = await graph.ainvoke(state, runnable_config)
        
        logger.debug(f"Graph result type: {type(result)}")
        logger.debug(f"Graph result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        logger.info("✅ Agent execution completed")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error running agent: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "messages": [{"type": "assistant", "content": f"Error processing request: {str(e)}"}]
        }


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temp directory and return path."""
    logger.info(f"💾 Saving uploaded file: {uploaded_file.name}")
    logger.debug(f"File size: {uploaded_file.size} bytes")
    
    try:
        # Create temp_uploads directory if it doesn't exist
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(temp_dir, filename)
        
        logger.debug(f"Saving to: {file_path}")
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"✅ File saved successfully: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"❌ Error saving file: {str(e)}", exc_info=True)
        raise


def cleanup_temp_files(file_path: str):
    """Clean up temporary files."""
    logger.info(f"🧹 Cleaning up temp file: {file_path}")
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("✅ Temp file cleaned up")
        else:
            logger.warning("⚠️ Temp file already removed")
    except Exception as e:
        logger.error(f"❌ Error cleaning up temp file: {str(e)}")


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("💰 Payroll Document Parser")
    st.markdown("Upload payroll documents to extract employee pay information using AI")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📄 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a payroll document",
            type=['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Supported formats: PDF, PNG, JPG, JPEG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            logger.info(f"📁 File uploaded: {uploaded_file.name}")
            
            # Context input
            context_query = st.text_area(
                "Additional Context (Optional)",
                placeholder="e.g., This is a bi-weekly payroll report for hourly employees...",
                help="Provide any additional context about the document to improve extraction accuracy"
            )
            
            logger.debug(f"Context query: {context_query[:100]}..." if context_query else "No context provided")
            
            # Process button
            if st.button("🚀 Process Document", type="primary"):
                logger.info("🔄 Processing document button clicked")
                
                with st.spinner("Processing document..."):
                    try:
                        # Save uploaded file
                        file_path = save_uploaded_file(uploaded_file)
                        
                        logger.info(f"📄 Starting document processing: {file_path}")
                        
                        # Create state for processing with document_uploaded=True
                        from src.react_agent.state import State
                        processing_state = State(
                            document_uploaded=True,  # ✅ This is the key fix!
                            file_path=file_path,
                            context_query=context_query or "Extract all payroll information from this document.",
                            messages=[],
                            vlm_processing_complete=False,
                            document_info=None,
                            text_data={},
                            extracted_text="",
                            employees=[],
                            payroll_context=None,
                            extraction_complete=False,
                            context_gathered=False,
                            processing_errors=[]
                        )
                        
                        logger.debug(f"Processing state created: document_uploaded={processing_state.document_uploaded}")
                        
                        # Run agent
                        result = asyncio.run(run_agent(processing_state))
                        
                        if "error" in result:
                            logger.error(f"❌ Agent execution failed: {result['error']}")
                            st.error(f"Error processing document: {result['error']}")
                        else:
                            logger.info("✅ Document processing completed")
                            
                            # Update session state
                            st.session_state.agent_state = result
                            
                            # Get messages from result
                            messages = get_agent_state_attr(result, 'messages', [])
                            if messages:
                                logger.debug(f"Found {len(messages)} messages in result")
                                for msg in messages:
                                    if hasattr(msg, 'content'):
                                        st.session_state.messages.append({
                                            "type": "assistant", 
                                            "content": msg.content
                                        })
                                    else:
                                        st.session_state.messages.append({
                                            "type": "assistant", 
                                            "content": str(msg)
                                        })
                            
                            # Get employees from result
                            employees = get_agent_state_attr(result, 'employees', [])
                            if employees:
                                logger.info(f"👥 Found {len(employees)} employees")
                                st.session_state.processed_employees = employees
                            
                            st.success("Document processed successfully!")
                        
                        # Clean up temp file
                        cleanup_temp_files(file_path)
                        
                        logger.info("✅ Document processing workflow completed")
                        st.rerun()
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing document: {str(e)}", exc_info=True)
                        st.error(f"Error processing document: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat with Your Document")
        
        # Show interactive editing help if employees are processed
        if st.session_state.processed_employees:
            st.info("""
            🔧 **Interactive Editing Available!** You can now:
            - **Modify details**: "Update Alice's pay rate to $25/hour"
            - **Add information**: "Set Bob's employee ID to EMP001"
            - **Fix errors**: "Change Clara's deductions to $150"
            - **Add employees**: "Add employee David with $30/hour rate"
            - **Remove employees**: "Remove employee Alice"
            - **Calculate values**: "Calculate net pay for all employees"
            - **Export data**: "Export" or "Finalize" to get final JSON
            """)
        
        # Display chat messages
        for message in st.session_state.messages:
            logger.debug(f"Displaying message: {type(message)}")
            with st.chat_message(message.get("type", "assistant")):
                st.markdown(message.get("content", str(message)))
        
        # Chat input
        if prompt := st.chat_input("Ask about the payroll data or make changes..."):
            logger.info(f"💬 User input: {prompt[:100]}...")
            
            # Add user message
            st.session_state.messages.append({"type": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    try:
                        # Create state with user message from current agent state
                        current_state = st.session_state.agent_state
                        
                        # Create chat state - convert to proper State object
                        if isinstance(current_state, dict):
                            from src.react_agent.state import State
                            chat_state = State(
                                document_uploaded=current_state.get("document_uploaded", False),
                                file_path=current_state.get("file_path", ""),
                                context_query=current_state.get("context_query", ""),
                                vlm_processing_complete=current_state.get("vlm_processing_complete", False),
                                document_info=current_state.get("document_info"),
                                text_data=current_state.get("text_data", {}),
                                extracted_text=current_state.get("extracted_text", ""),
                                employees=current_state.get("employees", []),
                                payroll_context=current_state.get("payroll_context"),
                                extraction_complete=current_state.get("extraction_complete", False),
                                context_gathered=current_state.get("context_gathered", False),
                                processing_errors=current_state.get("processing_errors", []),
                                messages=current_state.get("messages", [])
                            )
                        else:
                            # Already a State object, use as-is
                            chat_state = current_state
                        
                        # Add new user message
                        from langchain_core.messages import HumanMessage
                        user_msg = HumanMessage(content=prompt)
                        chat_state.messages.append(user_msg)
                        
                        logger.debug(f"Chat state created: document_uploaded={chat_state.document_uploaded}")
                        
                        # Run agent
                        result = asyncio.run(run_agent(chat_state))
                        
                        if "error" in result:
                            error_message = f"Error generating response: {result['error']}"
                            st.error(error_message)
                            st.session_state.messages.append({"type": "assistant", "content": error_message})
                        else:
                            logger.info("✅ Chat response generated")
                            
                            # Get response messages
                            response_messages = get_agent_state_attr(result, 'messages', [])
                            if response_messages:
                                latest_message = response_messages[-1]
                                if hasattr(latest_message, 'content'):
                                    response_content = latest_message.content
                                else:
                                    response_content = str(latest_message)
                                
                                st.markdown(response_content)
                                
                                # Add to session state
                                st.session_state.messages.append({"type": "assistant", "content": response_content})
                                st.session_state.agent_state = result
                                
                                # Update employees if they were modified
                                updated_employees = get_agent_state_attr(result, 'employees', [])
                                if updated_employees:
                                    st.session_state.processed_employees = updated_employees
                                    logger.info(f"🔄 Updated employees in session state: {len(updated_employees)}")
                                
                                logger.debug("Chat response added to session state")
                        
                    except Exception as e:
                        logger.error(f"❌ Error generating chat response: {str(e)}", exc_info=True)
                        error_message = f"Error generating response: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({"type": "assistant", "content": error_message})
    
    with col2:
        st.header("📊 Extracted Data")
        
        if st.session_state.processed_employees:
            logger.debug(f"Displaying {len(st.session_state.processed_employees)} employees")
            
            # Summary statistics
            st.subheader("📈 Summary")
            total_employees = len(st.session_state.processed_employees)
            total_gross_pay = sum(getattr(emp, 'gross_pay', 0) or 0 for emp in st.session_state.processed_employees)
            total_net_pay = sum(getattr(emp, 'net_pay', 0) or 0 for emp in st.session_state.processed_employees)
            total_deductions = sum(getattr(emp, 'deductions', 0) or 0 for emp in st.session_state.processed_employees)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("👥 Total Employees", total_employees)
                st.metric("💰 Total Gross Pay", f"${total_gross_pay:,.2f}")
            with col_b:
                st.metric("📊 Total Net Pay", f"${total_net_pay:,.2f}")
                st.metric("📉 Total Deductions", f"${total_deductions:,.2f}")
            
            st.divider()
            
            # Display employee data
            for i, employee in enumerate(st.session_state.processed_employees):
                emp_name = getattr(employee, 'name', None) or getattr(employee, 'employee_name', f'Employee {i+1}')
                logger.debug(f"Displaying employee {i+1}: {emp_name}")
                
                with st.expander(f"👤 Employee {i+1}: {emp_name}", expanded=True):
                    # Display all available employee attributes safely
                    attrs = ['employee_id', 'position', 'pay_rate', 'hours_worked', 
                           'overtime_hours', 'gross_pay', 'deductions', 'net_pay', 'pay_period']
                    
                    for attr in attrs:
                        value = getattr(employee, attr, None)
                        if value is not None:
                            if attr in ['pay_rate', 'gross_pay', 'deductions', 'net_pay']:
                                st.write(f"**{attr.replace('_', ' ').title()}:** ${value:,.2f}")
                            else:
                                st.write(f"**{attr.replace('_', ' ').title()}:** {value}")
            
            # Export buttons
            st.divider()
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                # Export to CSV
                if st.button("📥 Export to CSV"):
                    logger.info("📥 Exporting employee data to CSV")
                    
                    try:
                        # Convert to DataFrame
                        data = []
                        for employee in st.session_state.processed_employees:
                            emp_dict = {}
                            for attr in ['employee_id', 'name', 'position', 'pay_rate', 'hours_worked', 
                                       'overtime_hours', 'gross_pay', 'deductions', 'net_pay', 'pay_period']:
                                emp_dict[attr] = getattr(employee, attr, None)
                            data.append(emp_dict)
                        
                        df = pd.DataFrame(data)
                        csv_data = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name="payroll_data.csv",
                            mime="text/csv"
                        )
                        
                        logger.info("✅ CSV export successful")
                        
                    except Exception as e:
                        logger.error(f"❌ Error exporting to CSV: {str(e)}")
                        st.error(f"Error exporting to CSV: {str(e)}")
            
            with col_export2:
                # Get final JSON
                if st.button("📊 Get Final JSON"):
                    logger.info("📊 Generating final JSON")
                    
                    try:
                        # Create final JSON
                        employee_data = []
                        for employee in st.session_state.processed_employees:
                            emp_dict = {}
                            for attr in ['employee_id', 'name', 'position', 'pay_rate', 'hours_worked', 
                                       'overtime_hours', 'gross_pay', 'deductions', 'net_pay', 'pay_period']:
                                emp_dict[attr] = getattr(employee, attr, None)
                            employee_data.append(emp_dict)
                        
                        import json
                        from datetime import datetime
                        
                        json_response = {
                            "status": "success",
                            "message": "Payroll data finalized",
                            "employees": employee_data,
                            "export_timestamp": datetime.now().isoformat(),
                            "total_employees": len(employee_data)
                        }
                        
                        json_str = json.dumps(json_response, indent=2)
                        
                        st.download_button(
                            label="📊 Download JSON",
                            data=json_str,
                            file_name="payroll_data.json",
                            mime="application/json"
                        )
                        
                        # Also show in expander
                        with st.expander("📊 View JSON Data"):
                            st.code(json_str, language="json")
                        
                        logger.info("✅ JSON export successful")
                        
                    except Exception as e:
                        logger.error(f"❌ Error generating JSON: {str(e)}")
                        st.error(f"Error generating JSON: {str(e)}")
        
        else:
            st.info("Upload and process a document to see extracted payroll data here.")
            
            # Show sample editing commands
            st.subheader("💡 Example Commands")
            st.code("""
# After processing a document, you can:
"Update Alice's pay rate to $25/hour"
"Set Bob's employee ID to EMP001"
"Add employee David with $30/hour rate"
"Calculate net pay for all employees"
"Export final data"
            """, language="text")
    
    # Footer
    st.markdown("---")
    st.markdown("**Powered by Groq VLM and ReAct Agent** | Built with Streamlit")


if __name__ == "__main__":
    main() 