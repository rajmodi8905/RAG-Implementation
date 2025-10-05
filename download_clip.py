from transformers import CLIPModel, CLIPProcessor
import os

# The model identifier from Hugging Face
MODEL_NAME = "openai/clip-vit-base-patch32"
# The local directory where we want to save the model
LOCAL_MODEL_PATH = "./clip-model"

# Create the directory if it doesn't exist
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

# Download and save the model and processor
print(f"Downloading model '{MODEL_NAME}' to '{LOCAL_MODEL_PATH}'...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.save_pretrained(LOCAL_MODEL_PATH)
processor.save_pretrained(LOCAL_MODEL_PATH)

print("✅ Model and processor downloaded and saved successfully!")