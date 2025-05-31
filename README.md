# Satellite_Image_Classifier_QML

This project implements Image classification system using Quantum Machine Learning . It combines classical image processing, quantum feature extraction using quantum circuits, and machine learning classification to categorize images from a dataset of aerial or satellite images.

![Image 30-04-25 at 10 17 AM](https://github.com/user-attachments/assets/69a2848c-83c8-4eb3-8978-8eb99946f6ce)

🖼️ Classical Image Preprocessing
	•	Images are converted to grayscale, resized (e.g., 32x32 pixels), normalized, and flattened into feature vectors.

⚛️ Quantum Feature Extraction
	•	Parameterized quantum circuits encode classical image data into quantum states.

	•	Quantum gates (rotations and entanglement) capture high-dimensional correlations.
	•	Measurements generate quantum features that are difficult to model classically.

🔗 Hybrid Feature Vector
	•	Classical and quantum features are concatenated into a hybrid vector.
	•	This vector represents each image in a combined classical-quantum space.

🌲 Machine Learning Model
	•	A Random Forest classifier is trained on the hybrid vectors.
	•	The model predicts land use classes such as agricultural, beach, forest, etc.
	•	Trained models are persisted for fast inference.

🧪 Backend API (Flask)
	•	Endpoint: /classify
	•	Accepts image uploads, processes them, and returns:
	•	Predicted class
	•	Confidence score (in JSON)!


💻 Frontend
	•	A simple web interface for:
	•	Uploading images
	•	Viewing prediction results
	•	Error handling

🚀 Significance

This project showcases a practical integration of quantum computing and classical AI in real-world tasks. Quantum-enhanced feature extraction may provide richer data representations, leading to improved classification performance.
