import os
import requests

dataset_dir = "UCMerced_LandUse/UCMerced_LandUse/Images"
url = "http://127.0.0.1:5000/classify"

def classify_image(image_path):
    with open(image_path, 'rb') as f:
        files = {'image': f}
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    print(f"Error for {image_path}: {data['error']}")
                else:
                    print(f"Prediction for {image_path}: {data['predicted_class']} (Confidence: {data['confidence']:.2f})")
            else:
                print(f"Failed request for {image_path}: Status code {response.status_code}")
        except Exception as e:
            print(f"Exception for {image_path}: {e}")

def main():
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                image_path = os.path.join(root, file)
                classify_image(image_path)

if __name__ == "__main__":
    main()
