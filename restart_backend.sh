#!/bin/bash
# Script to restart the backend server and show logs

echo "Stopping any running backend_api.py processes..."
pkill -f backend_api.py

echo "Activating virtual environment and starting backend_api.py..."
source quantum_proj_env/bin/activate
python3 backend_api.py
