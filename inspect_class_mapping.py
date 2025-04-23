import numpy as np

def main():
    class_mapping = np.load('class_mapping.npy', allow_pickle=True).item()
    print("Class mapping:")
    for k, v in class_mapping.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
