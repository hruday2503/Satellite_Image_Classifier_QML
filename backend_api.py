from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image                      
import io
import numpy as np
import traceback
import os

# Import necessary quantum and ML functions from notebook logic
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import qiskit.quantum_info as qi
from sklearn.ensemble import RandomForestClassifier
import joblib  # For loading saved model if needed

app = Flask(__name__)
from flask_cors import CORS

# Enable CORS with explicit configuration to allow all origins for testing
CORS(app, resources={r"/classify": {"origins": "*"}}, supports_credentials=True)

@app.before_request
def log_request_info():
    print(f"Received {request.method} request for {request.path}")

@app.before_request
def log_request_info():
    print(f"Received {request.method} request for {request.path}")

class_mapping = {}  # To be loaded or defined
model = None  # To be loaded or defined

# Placeholder functions for preprocessing and quantum encoding (to be replaced with actual notebook functions)
def preprocess_image(image, target_size=(32, 32)):
    img = image.convert('L')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    flattened = img_array.flatten()
    return flattened

def quantum_encode_image(image_data, num_qubits=10):
    if len(image_data) > 2**num_qubits:
        indices = np.linspace(0, len(image_data)-1, 2**num_qubits, dtype=int)
        reduced_data = image_data[indices]
    else:
        reduced_data = image_data
    reduced_data = reduced_data / np.linalg.norm(reduced_data)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        if i < len(reduced_data):
            qc.ry(reduced_data[i] * np.pi, i)
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            if i < len(reduced_data) and j < len(reduced_data):
                qc.rzz(reduced_data[i] * reduced_data[j] * np.pi, i, j)
    return qc

def extract_quantum_features(quantum_circuit, shots=1024):
    qc_z = quantum_circuit.copy()
    qc_z.measure_all()
    qc_x = quantum_circuit.copy()
    qc_x.h(range(qc_x.num_qubits))
    qc_x.measure_all()
    simulator = Aer.get_backend('qasm_simulator')
    job_z = simulator.run(qc_z, shots=shots).result()
    job_x = simulator.run(qc_x, shots=shots).result()
    result_z = job_z.get_counts()
    result_x = job_x.get_counts()
    z_features = np.zeros(2**quantum_circuit.num_qubits)
    x_features = np.zeros(2**quantum_circuit.num_qubits)
    for state, count in result_z.items():
        idx = int(state, 2)
        z_features[idx] = count / shots
    for state, count in result_x.items():
        idx = int(state, 2)
        x_features[idx] = count / shots
    features = np.concatenate([z_features, x_features])
    return features

def shor_based_classifier(quantum_features, num_qubits=10):
    qc = QuantumCircuit(num_qubits)
    angles = np.arcsin(quantum_features[:num_qubits] * np.pi)
    for i, angle in enumerate(angles):
        if i < num_qubits:
            qc.ry(angle, i)
    for i in range(num_qubits):
        qc.h(i)
        for j in range(i+1, num_qubits):
            qc.cp(np.pi/float(2**(j-i)), i, j)
    for i in range(num_qubits//2):
        qc.swap(i, num_qubits-i-1)
    qc.measure_all()
    simulator = Aer.get_backend('qasm_simulator')
    job = simulator.run(qc, shots=1024).result()
    result = job.get_counts()
    shor_features = np.zeros(2**num_qubits)
    for state, count in result.items():
        idx = int(state, 2)
        shor_features[idx] = count / 1024
    return shor_features

# Load the trained model and class mapping from files if available
def load_model_and_mapping():
    global model, class_mapping
    # Load Random Forest model and class mapping saved by training script
    if os.path.exists('model_rf.joblib'):
        model = joblib.load('model_rf.joblib')
        print("Model loaded successfully.")
    else:
        model = None
        print("Model file not found.")
    if os.path.exists('class_mapping.npy'):
        class_mapping = np.load('class_mapping.npy', allow_pickle=True).item()
        print("Class mapping loaded successfully.")
    else:
        class_mapping = {}
        print("Class mapping file not found.")
    print(f"Model type: {type(model)}")
    print(f"Class mapping type: {type(class_mapping)}")

load_model_and_mapping()

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'})

@app.route('/classify', methods=['POST'])
def classify():
    try:
        print("Classify endpoint called.")
        if 'image' not in request.files:
            print("No image file provided in request.")
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        print("Received image file, starting processing.")
        image = Image.open(file.stream)
        print("Image opened successfully.")
        processed_img = preprocess_image(image)
        print("Image preprocessed.")
        print(f"Processed image shape: {processed_img.shape}, dtype: {processed_img.dtype}")
        qc = quantum_encode_image(processed_img)
        print("Quantum encoding done.")
        q_features = extract_quantum_features(qc)
        print("Quantum features extracted.")
        print(f"Quantum features shape: {q_features.shape}, dtype: {q_features.dtype}")
        shor_features = shor_based_classifier(q_features)
        shor_features = np.array(shor_features).flatten()
        print("Shor based classification features computed.")
        print(f"Shor features shape: {shor_features.shape}, dtype: {shor_features.dtype}")
        if model is None or not class_mapping:
            print("Model or class mapping not loaded.")
            return jsonify({'error': 'Model or class mapping not loaded'}), 500
        classical_features = processed_img.flatten()
        combined_features = np.concatenate([classical_features, shor_features])
        print(f"Combined features shape: {combined_features.shape}, dtype: {combined_features.dtype}")
        pred_idx = model.predict([combined_features])[0]
        probs = model.predict_proba([combined_features])[0]
        confidence = probs[pred_idx]
        predicted_class = class_mapping.get(pred_idx, 'Unknown')
        print(f"Prediction done: {predicted_class} with confidence {confidence}")
        return jsonify({'predicted_class': predicted_class, 'confidence': float(confidence)})
    except Exception as e:
        error_message = traceback.format_exc()
        print(f"Exception during classification: {error_message}")
        with open('backend_api_error.log', 'a') as f:
            f.write(error_message + '\n')
        # Additional detailed error logging for diagnosis
        try:
            import sys, platform
            f.write(f"Python version: {sys.version}\n")
            f.write(f"Platform: {platform.platform()}\n")
            f.write(f"Request headers: {dict(request.headers)}\n")
            f.write(f"Request files: {list(request.files.keys())}\n")
        except Exception as log_exc:
            f.write(f"Error during additional logging: {log_exc}\n")
        return jsonify({'error': 'Load failed', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
