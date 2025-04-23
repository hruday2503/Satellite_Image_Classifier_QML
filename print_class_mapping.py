import numpy as np

def print_class_mapping():
    try:
        class_mapping = np.load('class_mapping.npy', allow_pickle=True).item()
        print("Class mapping loaded:")
        for idx, class_name in class_mapping.items():
            print(f"{idx}: {class_name}")
    except Exception as e:
        print(f"Failed to load class mapping: {e}")

if __name__ == "__main__":
    print_class_mapping()
