#!/usr/bin/env python3
"""
Now i am starting the Job Board Vision Assistant Web Interface
"""

import subprocess
import sys
import os

def install_requirements():
    """Installing the Flask requirements if not already installed"""
    try:
        import flask
        print("Flask is already installed")
    except ImportError:
        print("Installing Flask requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_web.txt"])

def start_server():
    """Starting the Flask web server"""
    print("Job Board Vision Assistant - Web Interface")
    print("=" * 50)
    print("Starting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Now i am starting the Flask app
    os.system("python app.py")

if __name__ == "__main__":
    install_requirements()
    start_server()
