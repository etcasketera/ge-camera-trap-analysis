import torch
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife import utils as pw_utils
from PIL import Image

# print(dir(pw_classification))
# 1. Initialize the model (Weights download automatically on first run)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = pw_classification.AI4GAmazonRainforest()

# 2. Load your image
img_path = "animal_crops/crop_3_D33S006.jpg"

# 3. Perform Inference
# Pytorch-Wildlife's internal method handles resizing and tensor conversion
results = model.single_image_classification(img_path)

# 4. View Results
# 'results' usually contains the top predicted label and confidence score
print(results)