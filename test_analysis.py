from ultralytics import YOLO
import os

def train_kalahari_model():
    # 1. Load a pre-trained YOLOv8 classification model
    # 'yolov8n-cls.pt' is the 'Nano' version—very fast and great for starters
    model = YOLO('yolov8n.pt') 

    # 2. Define your dataset path
    # The folder should contain 'train' and 'val' subdirectories
    dataset_path = os.path.abspath("")

    print(f">>> Starting training on dataset at: {dataset_path}")

    # 3. Start Fine-Tuning
    results = model.train(
        data='./yolo_dataset/kalahari-wild.yaml',    # Path to your data
        epochs=10,            # Number of times the model sees the whole dataset
        imgsz=224,            # SpeciesNet standard size
        batch=5,             # Number of images processed at once (adjust based on GPU VRAM)
        name='Kalahari_Classifier_v1' # Name of the output folder
    )

    # 4. Export the model for your workflow script
    # This will create 'kalahari_best.pt'
    success = model.export(format='pt')
    print(f">>> Training Complete! Model saved to runs/classify/Kalahari_Classifier_v1/weights/best.pt")


def classification_train():
# Load the classification model
    model = YOLO('yolov8n-cls.pt') 

    # Train using the folder path
    results = model.train(
        data='./kalahari_classification_dataset', # Root folder of the splits
        epochs=50,
        imgsz=224,
        batch=16,
        name='Kalahari_Classifier_v1',
        augment=True
    )

if __name__ == "__main__":
    classification_train()