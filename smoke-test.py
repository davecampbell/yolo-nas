import torch
from super_gradients.training import models

# Pick best device: MPS (Apple GPU) if available, else CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

# Load a small YOLO-NAS with COCO weights
model = models.get("yolo_nas_s", pretrained_weights="coco").to(device)

# Option A: run on a local image file
img_path = "test.jpg"   # change to your image path
preds = model.predict(img_path, conf=0.25)
preds.save("output.jpg")  # saves the result image with bounding boxes

print("OK:", type(model).__name__)