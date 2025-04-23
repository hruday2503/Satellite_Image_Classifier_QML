import os
import numpy as np
from PIL import Image
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import qiskit.quantum_info as qi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Dataset path
dataset_path = "UCMerced_LandUse/UCMerced_LandUse/Images_converted"

def preprocess_image(image_path, target_size=(32, 32)):
    img = Image.open(image_path).convert('L')
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
    # Clamp values to [-1, 1] before arcsin to avoid invalid values
    clamped_values = np.clip(quantum_features[:num_qubits] * np.pi, -1, 1)
    angles = np.arcsin(clamped_values)
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

def prepare_dataset(dataset_path, num_samples_per_class=10):
    X = []
    y = []
    class_mapping = {}
    label = 0
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"Classes found: {classes}")
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        class_mapping[label] = class_name
        print(f"Assigning label {label} to class '{class_name}'")
        images_processed = 0
        for file_name in os.listdir(class_dir):
            if not file_name.endswith('.jpg'):
                continue
            file_path = os.path.join(class_dir, file_name)
            try:
                processed_img = preprocess_image(file_path)
                # Classical feature extraction: flatten image pixels
                classical_features = processed_img
                # Quantum encoding and feature extraction
                qc = quantum_encode_image(processed_img)
                q_features = extract_quantum_features(qc)
                shor_features = shor_based_classifier(q_features)
                shor_features = np.array(shor_features).flatten()
                if shor_features.ndim != 1:
                    print(f"⚠️ Feature shape issue at {file_path} => shape: {shor_features.shape}")
                    continue
                # Combine classical and quantum features
                combined_features = np.concatenate([classical_features, shor_features])
                X.append(combined_features)
                y.append(label)
                images_processed += 1
                if images_processed % 5 == 0:
                    print(f"Processed {images_processed} images for class '{class_name}'")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
            if images_processed >= num_samples_per_class:
                break
        label += 1
    print(f"Class mapping keys and types: {[(k, type(k)) for k in class_mapping.keys()]}")
    print(f"Label types in y: {set(type(label) for label in y)}")
    print(f"Sample feature vector shape: {X[0].shape if X else 'No data'}")
    print(f"Sample label: {y[0] if y else 'No labels'}")
    return np.array(X), np.array(y), class_mapping

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

def main():
    print("Preparing dataset...")
    X, y, class_mapping = prepare_dataset(dataset_path, num_samples_per_class=10)
    print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Training Random Forest classifier...")
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_train, y_train)

    print("Training SVM classifier...")
    clf_svm = SVC(kernel='rbf', probability=True, random_state=42)
    clf_svm.fit(X_train, y_train)

    print("Training Gradient Boosting classifier...")
    clf_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf_gb.fit(X_train, y_train)

    print("Saving models and class mapping...")
    joblib.dump(clf_rf, 'model_rf.joblib')
    joblib.dump(clf_svm, 'model_svm.joblib')
    joblib.dump(clf_gb, 'model_gb.joblib')
    np.save('class_mapping.npy', class_mapping)

    for name, clf in [('Random Forest', clf_rf), ('SVM', clf_svm), ('Gradient Boosting', clf_gb)]:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{name} Classification accuracy: {accuracy:.4f}")

        missing_labels = set(y_test) - set(class_mapping.keys())
        if missing_labels:
            print(f"Warning: Missing labels in class mapping: {missing_labels}")

        y_test_labels = [class_mapping.get(i, 'Unknown') for i in y_test]
        y_pred_labels = [class_mapping.get(i, 'Unknown') for i in y_pred]

        print(f"{name} Classification report:")
        print(classification_report(y_test_labels, y_pred_labels))

if __name__ == "__main__":
    main()
