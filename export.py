from ultralytics import YOLO

def export_model(model_path):
    """
    Exports a trained YOLOv8 model to CoreML format.
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print("Exporting to CoreML...")
    # nms=True includes Non-Maximum Suppression in the model pipeline itself,
    # which simplifies the iOS code significantly.
    model.export(format='coreml', nms=True)
    
    print("Export complete.")

if __name__ == '__main__':
    # Default to the best model from the standard training run
    # You might need to update this path if you run training multiple times
    model_path = 'runs/detect/train/weights/best.pt'
    
    try:
        export_model(model_path)
    except Exception as e:
        print(f"Export failed: {e}")
        print(f"Did you run train.py first? Check if {model_path} exists.")
