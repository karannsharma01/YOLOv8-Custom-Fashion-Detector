import os
import requests
from ultralytics import YOLO

MODEL_PATH = "best.pt" 

IMAGE_URL = "https://images.pexels.com/photos/30244535/pexels-photo-30244535.jpeg"

CONFIDENCE = 0.30  

def download_image(url, save_path="temp_test.jpg"):
    """Downloads an image securely from a URL."""
    print("🌐 Downloading image from URL...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Could not find model at {MODEL_PATH}")
        print("Check your folder structure and update the MODEL_PATH variable.")
        return

    print("🧠 Loading your custom YOLOv8 model...")
    model = YOLO(MODEL_PATH)

    try:
        local_img = download_image(IMAGE_URL)
    except Exception as e:
        print(f"❌ Failed to download image: {e}")
        return

    print("🔍 Running inference...\n")
    
    results = model(local_img, conf=CONFIDENCE)
    
    result = results[0]

    if len(result.boxes) == 0:
        print("⚠️ No clothing detected above the confidence threshold.")
    else:
        print(f"✅ Detected {len(result.boxes)} item(s):")
        print("-" * 50)
        
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            confidence = float(box.conf[0].item())
            
            coords = box.xyxy[0].tolist()
            xmin, ymin, xmax, ymax = [int(c) for c in coords]
            
            print(f"👕 Label: {class_name.upper()}")
            print(f"📊 Confidence: {confidence * 100:.1f}%")
            print(f"📍 Bounding Box (Pixels): Top-Left({xmin}, {ymin}) to Bottom-Right({xmax}, {ymax})")
            print("-" * 50)

    output_filename = "inference_result.jpg"
    result.save(filename=output_filename)
    print(f"🎨 Saved visual verification with bounding boxes to: {output_filename}")

    if os.path.exists(local_img):
        os.remove(local_img)

if __name__ == "__main__":
    main()
