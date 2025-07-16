# Payroll Agent - Client-Server Architecture

This project has been converted from a monolithic Streamlit application with LangGraph to a client-server architecture with:
- FastAPI backend for handling image processing and LLM operations
- Streamlit frontend for user interface

## Project Structure

```
payroll-agent/
├── backend/
│   ├── main.py           # FastAPI server implementation
│   └── requirements.txt  # Backend dependencies
├── frontend/
│   ├── streamlit_app.py  # Streamlit client application
│   └── requirements.txt  # Frontend dependencies
└── src/                  # Shared source code (original implementation)
    └── react_agent/      # LangGraph agent implementation
```

## Features

- **Separation of Concerns**: UI logic is separated from business logic and image processing
- **API-First Design**: All operations are exposed through a RESTful API
- **Scalability**: Backend can be scaled independently from the frontend
- **Image Processing on Backend**: All document processing happens on the server side

## API Endpoints

The FastAPI backend provides the following endpoints:

- `GET /` - Health check endpoint
- `POST /process-document` - Process uploaded documents (images, PDFs, etc.)
- `POST /merge-employees` - Merge existing and updated employee lists
- `POST /generate-payroll` - Generate payroll report for employees
- `POST /chat` - Chat with the payroll agent

## Setup and Installation

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the FastAPI server:
   ```
   uvicorn main:app --reload
   ```
   
   The API will be available at http://localhost:8000

4. View API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```
   
   The app will be available at http://localhost:8501

## Usage

1. Start both the backend and frontend servers
2. Access the Streamlit app in your browser
3. The app will automatically check if it can connect to the backend API
4. Use the Document Processing tab to upload and process payroll documents
5. Use the Chat tab to interact with the payroll agent
6. View and export payroll reports in the Payroll Report tab

## Environment Variables

The application uses the following environment variables:

- `OPENAI_API_KEY` - Your OpenAI API key for LLM operations

## Technical Details

### Backend

- FastAPI for high-performance API endpoints
- Image processing with Pillow and PyMuPDF
- LangGraph for agent workflow orchestration
- Pydantic for data validation

### Frontend

- Streamlit for interactive UI components
- Requests for API communication
- Pandas for data manipulation and display

## Benefits of Client-Server Architecture

1. **Improved Performance**: Heavy processing (image processing, LLM operations) happens on the server
2. **Better Scalability**: Backend can be scaled independently based on load
3. **Enhanced Security**: API can implement authentication and rate limiting
4. **Flexibility**: Frontend and backend can be developed and deployed independently
5. **Multiple Clients**: Additional clients (mobile app, desktop app) can use the same API
