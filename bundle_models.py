import os
import torch
from torchvision import models
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageClassification, AutoImageProcessor
import requests

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Download Models ---
MODELS_TO_DOWNLOAD = [
    {
        "id": "microsoft/swin-base-patch4-window7-224-in22k",
        "name": "Swin-B (IN-22k)",
        "type": "vision"
    },
    {
        "id": "instruction-pretrain/InstructLM-1.3B",
        "name": "InstructLM 1.3B",
        "type": "language"
    },
    {
        "id": "gpt2-large",
        "name": "GPT-2 Large",
        "type": "language"
    },
]

for model in MODELS_TO_DOWNLOAD:
    model_id = model["id"]
    name = model["name"]
    model_type = model["type"]

    print(f"\nDownloading {name}...")
    path = os.path.join(MODEL_DIR, model_id.replace('/', '_'))
    os.makedirs(path, exist_ok=True)

    if model_type == "language":
        model_obj = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_obj.save_pretrained(path)
        tokenizer.save_pretrained(path)

    elif model_type == "vision":
        model_obj = AutoModelForImageClassification.from_pretrained(model_id)
        processor = AutoImageProcessor.from_pretrained(model_id)
        model_obj.save_pretrained(path)
        processor.save_pretrained(path)

    print(f"{name} saved to {path}")


# --- Download and Save MobileNetV2 ---
print("\nDownloading MobileNetV2...")
mobilenet = models.mobilenet_v2(pretrained=True)
torch.save(mobilenet.state_dict(), os.path.join(MODEL_DIR, "mobilenet_v2.pt"))
print("MobileNetV2 weights saved to ./models/mobilenet_v2.pt")


# --- Download ImageNet class labels ---
print("\nDownloading ImageNet class labels...")

# --- Download ImageNet-1K labels ---
labels_url_1k = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels_path_1k = os.path.join(MODEL_DIR, "imagenet1k_labels.txt")
response_1k = requests.get(labels_url_1k)
response_1k.raise_for_status()
with open(labels_path_1k, "w", encoding="utf-8") as f:
    f.write(response_1k.text)
print(f"ImageNet-1K labels saved to {labels_path_1k}")
