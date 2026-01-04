import os
import yaml
from ultralytics import YOLO

def main():
    # Define dataset path
    dataset_path = os.path.abspath('datasets/downloaded_rummikub')
    yaml_path = os.path.join(dataset_path, 'data.yaml')

    # Update data.yaml to use absolute paths to avoid confusion
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    data_config['path'] = dataset_path
    data_config['train'] = 'train/images'
    data_config['val'] = 'valid/images'
    data_config['test'] = 'test/images'
    
    # Save the corrected yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
        
    print(f"Updated config at {yaml_path}")

    # Load the Model
    # Using yolov8n (nano) for iPhone optimization
    print("Loading YOLOv8 Nano model...")
    model = YOLO('yolov8n.pt') 

    # Train
    print("Starting training on downloaded dataset...")
    # 50 epochs is usually plenty for a transfer learning task on a good dataset
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        device='cpu', # CPU for now
        plots=True,
        project='runs/detect',
        name='rummikub_downloaded'
    )
    
    print("Training finished.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()
