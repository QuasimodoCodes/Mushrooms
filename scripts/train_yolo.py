from ultralytics import YOLO
import os

def main():
    print("=========================================")
    print("   Starting YOLO Classification Training ")
    print("=========================================")

    # 1. Load a pre-trained YOLO Nano classification model
    # Why Nano (n)? It is the smallest and fastest model in the YOLOv8 family. 
    # It allows us to verify our pipeline works quickly without waiting hours for training.
    print("Loading YOLOv8n-cls model...")
    model = YOLO('yolov8n-cls.pt')

    # 2. Define where our dataset is located
    # Note: We are running this script from the 'scripts/' folder, so the data is in '../data/dataset_split'
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dataset_split"))
    
    # 3. Start the Training Process
    print(f"\nTraining on dataset located at: {data_dir}")
    print("Please watch the terminal output. You want to see the 'loss' value decrease over time!")
    
    # We set epochs=3 just to do a fast "dry run" and prove the architecture works before committing to a long training cycle.
    results = model.train(
        data=data_dir, 
        epochs=3, 
        imgsz=224, # Standard image size for YOLO classification 
        project=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "yolo_runs")), # Where to save logs/graphs
        name="mushroom_classifier_v1"
    )

    print("\nTraining Complete!")
    print("All charts, loss graphs, and the final model weights have been saved to the 'docs/yolo_runs' directory.")

if __name__ == "__main__":
    main()
