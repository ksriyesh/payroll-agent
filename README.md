# 💰 Payroll Agent - AI-Powered Employee Management System

[![Open in - LangGraph Studio](
A sophisticated payroll management system that combines AI-powered document processing with interactive employee management. Built using [LangGraph](https://github.com/langchain-ai/langgraph) and featuring a modern Streamlit web interface with dark theme.

![Graph view in LangGraph studio UI](./static/studio_ui.png)

## 🚀 Features

### 🤖 AI-Powered Workflow
- **Document Processing**: Extract employee data from images, PDFs, and text files using OpenAI's vision models
- **Smart Discrepancy Detection**: Automatically identify differences between existing and updated employee records
- **Interactive Chat**: Conversational interface for payroll management and employee queries
- **Three-Agent Architecture**: Specialized agents for document processing, user interaction, and payroll generation
- **LangSmith Tracing**: Full observability and debugging with automatic trace collection

### 🌙 Modern Dark Theme Interface
- **Streamlit App**: Beautiful dark theme with modern UI aesthetics
- **Document Upload**: Drag & drop file uploads with preview and interactive hover effects
- **Real-time Processing**: Live status updates and workflow tracking with animated elements
- **Responsive Design**: Professional gradient backgrounds, glowing headers, and smooth animations
- **Enhanced UX**: Reduced eye strain with high contrast colors and modern typography

### 📊 Payroll Management
- **Employee Records**: Comprehensive employee data management with visual cards
- **Pay Rate Tracking**: Regular and overtime rate management
- **Configurable Settings**: Customizable currency and overtime multipliers
- **Workflow Automation**: Automated processing from document to final payroll report

## 🏗️ Three-Agent Architecture

The system implements a clean 3-agent architecture with specialized responsibilities:

### Agent Responsibilities

#### 1. 🔍 VLM Document Processor
- **Purpose**: Specialized document processing only
- **Input**: Document content (text, images, PDFs)
- **Output**: Structured EmployeeList with extracted data
- **Model**: OpenAI GPT-4o with vision capabilities
- **Interaction**: No user interaction - pure data processing

```python
# Example output
EmployeeList(
    employees=[
        Employee(name="John Doe", payrate=25.0, overtime_rate=37.5),
        Employee(name="Alice Williams", payrate=28.0, overtime_rate=42.0)
    ]
)
```

#### 2. 💬 UPDATE PAY Agent (Main React Agent)
- **Purpose**: Primary user interface and workflow coordination
- **Input**: User messages, current state, employee data
- **Output**: Conversational responses and tool calls
- **Model**: OpenAI GPT-4o with full tool access
- **Interaction**: **Only agent that interacts with users**

**Key Responsibilities:**
- Handle ALL user conversations
- Compare existing vs updated employee data
- Ask users about discrepancies and unknown employees
- Use tools to update employee information
- Coordinate workflow between other agents
- Present payroll results to users

**Available Tools:**
- `update_employee_payrate` - Update existing employee rates
- `add_new_employee` - Add new employees to system
- `calculate_employee_pay` - Calculate individual employee pay
- `confirm_user_approval` - Confirm changes and proceed
- `trigger_payroll_generation` - Activate payroll generator
- `check_employee_exists` - Verify employee status
- `get_system_status` - Check workflow state

#### 3. 📊 Payroll Generator Agent
- **Purpose**: Structured payroll report generation
- **Input**: Approved employee data from state
- **Output**: Structured PayrollReport with calculations
- **Model**: OpenAI GPT-4o with structured output
- **Interaction**: No user interaction - background processing

```python
# Example output
PayrollReport(
    pay_period="December 9-15, 2024",
    employees=[
        EmployeePayData(
            name="John Doe",
            payrate=25.0,
            overtime_rate=37.5,
            regular_hours=40.0,
            overtime_hours=5.0,
            regular_pay=1000.0,
            overtime_pay=187.5,
            total_pay=1187.5
        )
    ],
    total_payroll=1187.5,
    summary="Payroll for 1 employee with overtime"
)
```

### Workflow Flow
```
__start__ → init_node → vlm_doc_processor → update_pay_agent ⟷ tool_executor ⟷ payroll_generator_agent
```

**Key Design Principle**: Only the UPDATE PAY Agent interacts with users - it's the central hub for all payroll conversations and coordination.

### State Management
```python
class State:
    existing_employees: List[Employee]      # Previous pay period data
    updated_employees: List[Employee]       # Current document data
    document_uploaded: bool                 # Document processing flag
    user_approval: bool                     # User confirmation flag
    trigger_payroll: bool                   # Payroll generation trigger
    current_pay_data: PayrollReport        # Generated payroll report
    messages: List[Message]                 # Conversation history
```

## 🛠️ Quick Setup

### Prerequisites
- Python 3.11 or higher
- OpenAI API key
- Git

### 1. Clone and Install

```bash
git clone <repository-url>
cd react-agent-main
pip install -e .
```

### 2. Set Up Environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional for tracing
```

### 3. Choose Your Interface

#### Option A: Streamlit Web App (Recommended)
```bash
python run_app.py
```

#### Option B: LangGraph Studio
```bash
# Open the project in LangGraph Studio
# Point to langgraph.json configuration
```

## 🎯 How to Use

### 🌐 Streamlit Web App

The Streamlit app provides a beautiful dark-themed interface for interacting with the payroll agent:

#### 1. **Upload Documents**
- Use the animated sidebar to upload employee documents
- Supported formats: PNG, JPG, JPEG, PDF, TXT, CSV
- Interactive drag-and-drop with hover effects
- Click "Process Document" to extract employee data

#### 2. **Chat with the Agent**
- Use the modern chat interface with dark theme
- Ask questions about payroll and employee management
- Get help with calculations and data processing
- The UPDATE PAY agent handles all user interactions

#### 3. **View Results**
- Check the "Workflow Status" panel with gradient backgrounds
- Review any discrepancies found between employee lists
- See the final payroll report with professional styling
- Interactive employee cards with hover effects

#### 4. **Manage Settings**
- Adjust currency symbol and overtime multiplier
- View current employees in styled cards
- Reset the workflow with animated buttons

### 📊 Document Types Supported

The app can process various document types containing employee information:

- **Images**: Screenshots of spreadsheets, handwritten lists, forms
- **PDFs**: Official HR documents, payroll reports
- **Text Files**: CSV files, plain text employee lists
- **Tables**: Excel screenshots, formatted employee data

## 🔧 Configuration

The system uses Pydantic-based configuration:

```python
# Model configuration
vlm_model: "openai/gpt-4o"          # For document processing
react_model: "openai/gpt-4o"        # For conversations
payroll_model: "openai/gpt-4o"      # For payroll generation

# Display settings
currency_symbol: "$"                # Display currency
default_overtime_multiplier: 1.5    # Overtime calculation
max_employees: 1000                  # Processing limit
```

### Dark Theme Configuration
```toml
[theme]
primaryColor = "#4fc3f7"           # Bright cyan blue
backgroundColor = "#0e1117"         # Deep dark blue-gray
secondaryBackgroundColor = "#262730" # Slightly lighter gray
textColor = "#fafafa"              # Off-white text
```

## 📊 LangSmith Tracing

The application includes comprehensive LangSmith tracing for debugging and monitoring:

### 🔍 What's Traced
- **Document Processing**: VLM model interactions and data extraction
- **Agent Conversations**: All UPDATE PAY agent interactions
- **Tool Executions**: Employee data updates and payroll calculations
- **Workflow Routing**: Flow between different system components
- **Error Tracking**: Detailed error logs with full context

### 📈 Viewing Traces
1. Visit the [LangSmith Dashboard](https://smith.langchain.com)
2. Navigate to the "payroll-agent" project
3. View real-time traces as you interact with the application

### ⚙️ Configuration
Tracing is automatically enabled with:
- **Project**: `payroll-agent`
- **Endpoint**: `https://api.smith.langchain.com`
- **Tracing**: Enabled for all agent interactions

## 🎨 Dark Theme Features

### Visual Elements

#### 🌟 Animated Header
- **Glowing cyan effect** with pulsing animation
- **Professional typography** with enhanced visibility
- **Modern branding** with AI assistant theme

#### 📱 Interactive Components
- **Gradient backgrounds** with subtle shadows
- **Hover effects** with smooth color transitions
- **Modern buttons** with lift animations
- **Professional styling** throughout

#### 💬 Enhanced Chat Interface
- **Dark message containers** with rounded corners
- **Custom scrollbars** with cyan accents
- **Professional input styling**
- **High contrast** for readability

### Benefits
- **Reduced eye strain** during extended sessions
- **Modern aesthetic** for professional environments
- **Enhanced focus** with dark backgrounds
- **Improved readability** with high contrast text

## 📁 Project Structure

```
react-agent-main/
├── src/react_agent/
│   ├── __init__.py
│   ├── graph.py           # Main workflow logic
│   ├── state.py           # Pydantic state models
│   ├── configuration.py   # System configuration
│   ├── prompts.py         # AI model prompts
│   ├── tools.py           # Utility functions
│   └── utils.py           # Model loading utilities
├── .streamlit/
│   └── config.toml        # Dark theme configuration
├── streamlit_app.py       # Dark-themed web interface
├── run_app.py             # App launcher
├── test_3_agent_system.py # System testing
├── pyproject.toml         # Dependencies and config
├── langgraph.json         # LangGraph Studio config
└── README.md             # This file
```

## 🛠️ Development

### LangGraph Studio Development
1. Install [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio)
2. Open the project folder in LangGraph Studio
3. Use the `langgraph.json` configuration for graph visualization

### Testing
```bash
# Test the three-agent system
python test_3_agent_system.py

# Run with test documents
python test_3_agent_system.py --sample-data
```

### Customization Options
1. **Add new tools**: Extend functionality in `tools.py`
2. **Modify prompts**: Update system prompts in `prompts.py`
3. **Adjust workflow**: Edit the graph logic in `graph.py`
4. **UI customization**: Modify the Streamlit interface in `streamlit_app.py`
5. **Theme changes**: Update `.streamlit/config.toml` for color scheme

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your `.env` file contains `OPENAI_API_KEY=your_key`
   - Check that your API key is valid and has sufficient credits

2. **Import Errors**
   - Run `pip install -e .` to install dependencies
   - Ensure you're using Python 3.11 or higher

3. **Document Processing Issues**
   - Verify the document contains clear employee information
   - Try different document formats if one doesn't work
   - Check that the document has readable text/data

4. **Workflow Errors**
   - Check the terminal/console for detailed error messages
   - Ensure your OpenAI API key has access to GPT-4o
   - Try resetting the workflow using the "Reset Workflow" button

5. **Dark Theme Issues**
   - Clear browser cache if styling appears broken
   - Check `.streamlit/config.toml` for theme configuration
   - Ensure all CSS is loading properly

## 📈 Advanced Features

### Architecture Benefits
- **Separation of Concerns**: Each agent has focused responsibility
- **Simplified Workflow**: Clear routing logic with specific triggers
- **Maintainability**: Easy to modify individual agent behavior
- **Scalability**: Clean data flow between components

### Payroll Calculations
The system provides utilities for:
- Regular hour calculations
- Overtime rate computations
- Currency formatting
- Employee list management

### Usage Examples

#### Document Processing
```python
# User uploads document
# VLM extracts: Alice Williams, Charlie Brown
# UPDATE PAY agent asks: "Is Alice Williams a new employee?"
# User responds: "Yes, new employee"
# UPDATE PAY agent calls: add_new_employee(name="Alice Williams", ...)
```

#### Payroll Generation
```python
# User says: "Generate payroll report"
# UPDATE PAY agent calls: trigger_payroll_generation()
# Payroll Generator creates structured report
# UPDATE PAY agent presents: "Payroll report generated: 5 employees, Total: $12,487.50"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with `test_3_agent_system.py`
5. Submit a pull request

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your environment setup
3. Review the console/terminal output for error details
4. Test with sample documents using `test_3_agent_system.py`
5. Ensure your OpenAI API key is properly configured

## 🔒 Security Notes

- Your OpenAI API key is stored locally in the `.env` file
- Documents are processed temporarily and not permanently stored
- Employee data is managed in session state (not persistent)
- Follow best practices for API key management

## 🔮 Future Enhancements

- **Multi-tenant support** with employee data persistence
- **Advanced calculations** with deductions and benefits
- **Audit trail** for all employee changes
- **Bulk operations** for large employee updates
- **Integration APIs** for external payroll systems
- **Theme toggle** for user preference
- **Additional color schemes** (blue, purple, green)

---

**Happy Payroll Processing! 💰**
