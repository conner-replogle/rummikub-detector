import cv2
from ultralytics import YOLO
import sys
import os

def detect(image_path, model_path='runs/detect/train/weights/best.pt'):
    """
    Runs detection on a single image and displays results.
    """
    # Allow command line override of model_path if provided as 2nd arg
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
        
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run train.py first.")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Processing image: {image_path}")
    results = model(image_path)
    
    # Process results
    for result in results:
        # Visualize the results
        # This will save a plotted image to 'runs/detect/predict/' by default
        result.save()  # save to disk
        
        # Or we can show it (might not work well in headless environment, so we stick to saving)
        # result.show() 
        
        print("Detections:")
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            # x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f" - Found {class_name} with confidence {conf:.2f}")

    print(f"Results saved to {results[0].save_dir}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python detect.py <path_to_image>")
    else:
        detect(sys.argv[1])
