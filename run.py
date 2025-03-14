import subprocess
import sys
import os
import time
import threading
import webbrowser

def run_api():
    """Run the FastAPI application"""
    print("Starting API server...")
    subprocess.run(["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])
    
def run_streamlit():
    """Run the Streamlit application"""
    print("Starting Streamlit server...")
    subprocess.run(["streamlit", "run", "app/app.py"])

def open_browsers(delay=2):
    """Open browsers for both services after a delay"""
    time.sleep(delay)
    # Open API docs
    webbrowser.open("http://localhost:8000/docs")
    # Open Streamlit interface
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    
    # Check if model file exists
    model_path = os.path.join("models", "fire_best.pt")
    if not os.path.isfile(model_path) and os.path.isfile("fire_best.pt"):
        print("Moving model file to models directory...")
        os.rename("fire_best.pt", model_path)
    
    # Create and start threads for both services
    api_thread = threading.Thread(target=run_api)
    streamlit_thread = threading.Thread(target=run_streamlit)
    browser_thread = threading.Thread(target=open_browsers)
    
    api_thread.daemon = True
    streamlit_thread.daemon = True
    
    print("Starting Fire Detection System...")
    api_thread.start()
    streamlit_thread.start()
    browser_thread.start()
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down services...")
        sys.exit(0)
