from ultralytics import YOLO
import os

import torch

def main():
    print("=========================================")
    print("   Starting YOLO Classification Training ")
    print("=========================================")

    # 1. Hardware Check (GPU vs CPU)
    # Why is this important? AI models are essentially millions of math equations. 
    # A CPU calculates them sequentially, but an Nvidia GPU uses "CUDA cores" to calculate thousands of them simultaneously. 
    # This makes training exponentially faster.
    if torch.cuda.is_available():
        device = 'cuda'
        print(">> SUCCESS: Nvidia GPU (CUDA) detected! Training will be hardware-accelerated.")
    else:
        device = 'cpu'
        print(">> WARNING: No Nvidia GPU detected or PyTorch isn't configured for CUDA. Falling back to slow CPU training.")

    # 2. Load a pre-trained YOLO Nano classification model
    # Why Nano (n)? It is the smallest and fastest model in the YOLOv8 family. 
    print("Loading YOLOv8n-cls model...")
    model = YOLO('yolov8n-cls.pt')

    # 3. Define where our dataset is located
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset_split"))
    
    # 4. Start the Training Process
    print(f"\nTraining on dataset located at: {data_dir}")
    print("Please watch the terminal output. You want to see the 'loss' value decrease over time!")
    
    # We pass device=device to force YOLO to use the best hardware available
    results = model.train(
        data=data_dir, 
        epochs=3, 
        imgsz=224, 
        device=device,
        exist_ok=True, # This tells YOLO to overwrite the 'mushroom_classifier_v1' folder instead of making v2, v3, etc.
        project=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "yolo_runs")), 
        name="mushroom_classifier_v1"
    )

    print("\nTraining Complete!")
    print("All charts, loss graphs, and the final model weights have been saved to the 'docs/yolo_runs' directory.")

if __name__ == "__main__":
    main()
