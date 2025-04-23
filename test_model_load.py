import joblib
import numpy as np
import os

def test_model_and_mapping():
    if not os.path.exists('model_rf.joblib'):
        print("Model file not found.")
        return
    if not os.path.exists('class_mapping.npy'):
        print("Class mapping file not found.")
        return

    model = joblib.load('model_rf.joblib')
    print(f"Model loaded: {model}")

    class_mapping = np.load('class_mapping.npy', allow_pickle=True).item()
    print(f"Class mapping loaded: {class_mapping}")

    # Create dummy input matching expected feature size
    # Assuming model expects 2 * 32*32 features (classical + quantum)
    dummy_features = np.random.rand(2 * 32 * 32)
    dummy_features = dummy_features.reshape(1, -1)

    try:
        pred = model.predict(dummy_features)
        proba = model.predict_proba(dummy_features)
        print(f"Dummy prediction: {pred}")
        print(f"Dummy prediction probabilities: {proba}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    test_model_and_mapping()
