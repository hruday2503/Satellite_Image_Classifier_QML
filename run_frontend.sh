#!/bin/bash
# Script to start the frontend server

echo "Starting frontend server on port 8000..."
python3 -m http.server 8000 --directory frontend
