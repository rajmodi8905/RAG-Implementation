# download_models.py

from transformers import CLIPProcessor, CLIPModel

# 1. Define the model name from Hugging Face and the local path to save it to.
model_name = "openai/clip-vit-base-patch32"
local_dir = "./models/clip-vit-base-patch32"

print(f"Downloading model '{model_name}'...")

# 2. Download the model and processor from the internet.
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 3. Save the downloaded files to your specified local directory.
model.save_pretrained(local_dir)
processor.save_pretrained(local_dir)

print(f"Model saved successfully to '{local_dir}'")