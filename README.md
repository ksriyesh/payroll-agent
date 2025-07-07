# 📊 Payroll Document Parsing Agent

[![CI](https://github.com/langchain-ai/react-agent/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/react-agent/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/react-agent/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/react-agent/actions/workflows/integration-tests.yml)

An intelligent payroll document parsing agent that combines Vision Language Models (VLM) with ReAct reasoning patterns to extract, analyze, and interactively edit payroll information from documents.

![Graph view in LangGraph studio UI](./static/studio_ui.png)

## 🎯 What it does

The Payroll Agent follows your intended workflow:

1. **Document Upload** → User uploads payroll documents (PDF, images)
2. **VLM Processing** → Converts documents to VLM-friendly format
3. **Text + Position Extraction** → VLM extracts text with spatial awareness
4. **ReAct Agent Processing** → Groq LLM processes VLM results
5. **Interactive Editing** → User can chat to modify payroll data
6. **Export Results** → Final JSON/CSV output

## 🚀 Key Features

### 📄 **Document Processing**
- **Multi-format support**: PDF, PNG, JPEG, and more
- **VLM-powered extraction**: Uses OpenAI GPT-4o-mini for vision analysis
- **Spatial awareness**: Extracts text with position context
- **Automatic conversion**: PDF → PNG for optimal VLM processing

### 💬 **Interactive Chat-Based Editing**
- **Natural language modifications**: "Update Alice's pay rate to $25/hour"
- **Real-time updates**: Changes reflected immediately
- **Continuous conversation**: Keep chatting until satisfied
- **Smart parsing**: Handles complex payroll requests

### 🧠 **Dual Model Architecture**
- **Vision Model**: OpenAI GPT-4o-mini for document analysis
- **Text Model**: Groq Llama-3.1-8b-instant for reasoning and edits
- **Fallback strategy**: Text-based processing if VLM fails

### 📊 **Data Management**
- **Structured extraction**: Employee ID, name, pay rate, hours, deductions, etc.
- **Automatic calculations**: Net pay = gross pay - deductions
- **Export options**: JSON and CSV formats
- **State persistence**: All changes maintained in conversation

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd react-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```

4. **Configure API keys** in your `.env` file:
   ```bash
   # Required for VLM processing
   OPENAI_API_KEY=your-openai-api-key
   
   # Required for text reasoning
   GROQ_API_KEY=your-groq-api-key
   ```

## 📋 API Key Setup

### OpenAI (Vision Processing)
1. Sign up at [OpenAI](https://platform.openai.com/api-keys)
2. Create an API key
3. Add to `.env`: `OPENAI_API_KEY=your-key`

### Groq (Text Processing)
1. Sign up at [Groq Console](https://console.groq.com/keys)
2. Create an API key
3. Add to `.env`: `GROQ_API_KEY=your-key`

## 🎮 Usage

### Option 1: Streamlit Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

1. **Upload Document**: Drag and drop payroll documents
2. **VLM Processing**: Automatic text and position extraction
3. **Interactive Editing**: Chat to modify payroll data
4. **Export Results**: Download JSON or CSV

### Option 2: Direct API Usage

```python
from src.react_agent.state import State
from src.react_agent.graph import graph
from langchain_core.messages import HumanMessage

# Create state with uploaded document
state = State(
    document_uploaded=True,
    file_path="payroll_document.pdf",
    messages=[HumanMessage(content="Process this payroll document")]
)

# Run the agent
result = await graph.ainvoke(state)
```

## 💬 Interactive Editing Examples

After VLM processing, you can chat to modify payroll data:

```
🤖 "Here's your extracted payroll data..."

👤 "Update Alice's pay rate to $25/hour"
🤖 "✅ Updated Alice's pay rate to $25.00/hour"

👤 "Add employee David with $30/hour rate, worked 40 hours"
🤖 "✅ Added employee David with $30.00/hour rate, 40 hours worked"

👤 "Change Clara's deductions to $150"
🤖 "✅ Changed Clara's deductions to $150.00"

👤 "Calculate net pay for all employees"
🤖 "✅ Calculated net pay for all employees"

👤 "Export final data"
🤖 "✅ Here's your final JSON: {...}"
```

## 🔧 Supported Commands

### **Employee Modifications**
- `"Update [name]'s pay rate to $[amount]"`
- `"Set [name]'s employee ID to [id]"`
- `"Change [name]'s deductions to $[amount]"`
- `"Fix [name]'s hours to [hours]"`

### **Employee Management**
- `"Add employee [name] with $[rate]/hour"`
- `"Remove employee [name]"`
- `"Add overtime hours for [name]"`

### **Calculations**
- `"Calculate net pay for all employees"`
- `"Calculate gross pay for [name]"`
- `"Update all deductions to [amount]"`

### **Export**
- `"Export final data"`
- `"Finalize"`
- `"Get JSON"`

## 🧪 Testing

Run the test suite:
```bash
python test_debug.py
```

This tests:
- VLM processing workflow
- Interactive editing functionality
- State management
- Export capabilities

## 📁 Project Structure

```
react-agent/
├── src/react_agent/
│   ├── configuration.py    # Model and processing configuration
│   ├── state.py           # State management with Pydantic models
│   ├── graph.py           # LangGraph workflow definition
│   ├── tools.py           # VLM processing and document handling
│   ├── prompts.py         # System prompts for payroll processing
│   └── utils.py           # Utility functions
├── streamlit_app.py       # Web interface
├── test_debug.py          # Test suite
└── README.md              # This file
```

## 🎯 Workflow Details

### 1. **Document Upload**
- User uploads payroll document via Streamlit
- File saved to temp directory
- Document metadata extracted

### 2. **VLM Processing Node**
- Document converted to VLM-friendly PNG format
- OpenAI GPT-4o-mini analyzes document images
- Extracts text with spatial position awareness
- Fallback to text-based processing if VLM fails

### 3. **ReAct Agent Node**
- Groq Llama processes VLM results
- Structures data into employee payroll objects
- Handles user requests for modifications
- Continues conversation until user exports

### 4. **Interactive Editing**
- Natural language parsing of modification requests
- Real-time updates to payroll data
- State persistence across conversation
- Validation and error handling

### 5. **Export & Finalization**
- JSON format with complete payroll data
- CSV export for spreadsheet compatibility
- Timestamp and metadata inclusion

## 🔧 Configuration

### Model Configuration
```python
# Default models (configurable)
text_model = "llama-3.1-8b-instant"      # Groq for reasoning
vision_model = "gpt-4o-mini"             # OpenAI for VLM
```

### Processing Limits
```python
max_file_size = 10 * 1024 * 1024  # 10MB
max_pages = 10                     # PDF page limit
supported_formats = ['.pdf', '.png', '.jpg', '.jpeg']
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
1. Check the test suite: `python test_debug.py`
2. Review logs in the terminal
3. Ensure API keys are correctly configured
4. Verify document formats are supported

## 🎉 Next Steps

1. **Start the Streamlit app**: `streamlit run streamlit_app.py`
2. **Upload a payroll document**
3. **Watch the VLM processing**
4. **Chat to edit payroll data**
5. **Export your results**

The system is ready for production use with your increased OpenAI quota!