#!/bin/bash

# Install Homebrew if not installed
if ! command -v brew &> /dev/null
then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew is already installed."
fi

# Install Python via Homebrew
echo "Installing Python via Homebrew..."
brew install python

# Create virtual environment
echo "Creating virtual environment quantum_proj_env..."
python3 -m venv quantum_proj_env

# Activate virtual environment
echo "Activating virtual environment..."
source quantum_proj_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required Python packages..."
pip install numpy qiskit qiskit-aer scikit-learn pillow matplotlib scikit-image joblib

echo "Setup complete. To activate the environment later, run:"
echo "source quantum_proj_env/bin/activate"
