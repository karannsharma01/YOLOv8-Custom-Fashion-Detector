import os
import sys

try:
    from ultralytics import YOLO
    import yaml
except ImportError:
    print("Installing required libraries...")
    os.system(f'{sys.executable} -m pip install ultralytics pyyaml')
    from ultralytics import YOLO
    import yaml

dataset_dir = os.path.abspath("dataset")
print(f"Targeting dataset at: {dataset_dir}")

yaml_path = os.path.join(dataset_dir, "data.yaml")

if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"Could not find data.yaml in {dataset_dir}. Check your folder structure!")

print("\nConfiguring data.yaml paths...")
with open(yaml_path, 'r') as f:
    yaml_data = yaml.safe_load(f)

if 'path' in yaml_data:
    del yaml_data['path']

yaml_data['train'] = os.path.join(dataset_dir, "images", "train")

val_path = os.path.join(dataset_dir, "images", "val")
valid_path = os.path.join(dataset_dir, "images", "valid")

if os.path.exists(val_path):
    yaml_data['val'] = val_path
elif os.path.exists(valid_path):
    yaml_data['val'] = valid_path
else:
    print("Warning: Could not find validation images! Falling back to train data.")
    yaml_data['val'] = yaml_data['train']

if 'test' in yaml_data:
    del yaml_data['test']

with open(yaml_path, 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f" -> Train mapped strictly to: {yaml_data['train']}")
print(f" -> Val mapped strictly to: {yaml_data['val']}")

print("\nStarting L4 training run for 20 epochs...")
model = YOLO('bbox.pt')

results = model.train(
    data=yaml_path,
    epochs=10,                 
    imgsz=640,                  
    batch=64,                  
    workers=8,                  
    device=0,                   
    patience=10,                
    optimizer="AdamW",          
    lr0=0.01,                   
    cos_lr=True,                
    amp=True,                   
    project="finetune", 
    name="yolov8n_20_epochs"       
)

print("\n Success! Training completed.")
print("Your production weights file is ready at: fashion_finetune/yolov8n_20_epochs/weights/best.pt")
