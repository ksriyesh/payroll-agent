#!/usr/bin/env python3
"""
Script to run both the FastAPI backend and Streamlit frontend servers.
This script will start both servers in separate processes.
"""

import os
import sys
import subprocess
import time
import webbrowser
import signal
import platform

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

# Define server URLs
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8501"

# Store process objects
processes = []

def check_dependencies():
    """Check if required Python packages are installed."""
    try:
        import fastapi
        import uvicorn
        import streamlit
        print(f"{Colors.GREEN}✓ All core dependencies found.{Colors.ENDC}")
        return True
    except ImportError as e:
        print(f"{Colors.RED}✗ Missing dependency: {e}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Please install dependencies:{Colors.ENDC}")
        print(f"  {Colors.BLUE}Backend:{Colors.ENDC} cd {BACKEND_DIR} && pip install -r requirements.txt")
        print(f"  {Colors.BLUE}Frontend:{Colors.ENDC} cd {FRONTEND_DIR} && pip install -r requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend server."""
    print(f"{Colors.HEADER}Starting FastAPI backend server...{Colors.ENDC}")
    
    # Use python -m uvicorn to ensure it's found in the path
    cmd = [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    processes.append(process)
    print(f"{Colors.GREEN}Backend server started! API available at {BACKEND_URL}{Colors.ENDC}")
    print(f"{Colors.GREEN}API documentation available at {BACKEND_URL}/docs{Colors.ENDC}")
    
    return process

def start_frontend():
    """Start the Streamlit frontend server."""
    print(f"{Colors.HEADER}Starting Streamlit frontend server...{Colors.ENDC}")
    
    # Use streamlit run command
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    processes.append(process)
    print(f"{Colors.GREEN}Frontend server started! UI available at {FRONTEND_URL}{Colors.ENDC}")
    
    return process

def monitor_output(process, prefix):
    """Monitor and print output from a process with a prefix."""
    color = Colors.BLUE if prefix == "Backend" else Colors.GREEN
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(f"{color}[{prefix}]{Colors.ENDC} {line.rstrip()}")

def open_browser():
    """Open the browser to the frontend URL after a delay."""
    print(f"{Colors.YELLOW}Opening browser in 3 seconds...{Colors.ENDC}")
    time.sleep(3)
    webbrowser.open(FRONTEND_URL)

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shut down all processes."""
    print(f"\n{Colors.YELLOW}Shutting down servers...{Colors.ENDC}")
    
    for process in processes:
        if process.poll() is None:  # If process is still running
            if platform.system() == "Windows":
                process.send_signal(signal.CTRL_C_EVENT)
            else:
                process.send_signal(signal.SIGINT)
    
    # Wait for processes to terminate
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    
    print(f"{Colors.GREEN}All servers stopped.{Colors.ENDC}")
    sys.exit(0)

def main():
    """Main function to run both servers."""
    print(f"{Colors.BOLD}{Colors.HEADER}Payroll Agent - Client-Server Architecture{Colors.ENDC}")
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Start backend server
    backend_process = start_backend()
    
    # Wait a moment for backend to initialize
    time.sleep(2)
    
    # Start frontend server
    frontend_process = start_frontend()
    
    # Open browser after a delay
    open_browser()
    
    # Create threads to monitor output
    import threading
    backend_thread = threading.Thread(target=monitor_output, args=(backend_process, "Backend"))
    frontend_thread = threading.Thread(target=monitor_output, args=(frontend_process, "Frontend"))
    
    # Start threads
    backend_thread.daemon = True
    frontend_thread.daemon = True
    backend_thread.start()
    frontend_thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
            # Check if either process has terminated
            if backend_process.poll() is not None:
                print(f"{Colors.RED}Backend server stopped unexpectedly!{Colors.ENDC}")
                break
            if frontend_process.poll() is not None:
                print(f"{Colors.RED}Frontend server stopped unexpectedly!{Colors.ENDC}")
                break
    except KeyboardInterrupt:
        # This will be caught by our signal handler
        pass

if __name__ == "__main__":
    main()
