from PIL import Image
import os

from ResNet50_Main.ResNet50_DIP.train_resnet50_DIP import DIP_CONFIGS, apply_filter

img_path = "00000020_000.png"

# Check if the image exists
if not os.path.exists(img_path):
    raise FileNotFoundError(f"{img_path} not found")

image = Image.open(img_path).convert("RGB")

# Create folder if it doesn't exist
os.makedirs("filter_examples", exist_ok=True)

for config in DIP_CONFIGS:
    filtered = apply_filter(image, config['filter_type'], config['params'])
    output_path = os.path.join("filter_examples", f"{config['name']}.png")
    filtered.save(output_path)
    print(f"Saved {output_path}")

print("All filters applied successfully!")