import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Directory containing the images
images_path = "images/"  # Replace with your actual directory path

# Define the models and their human-readable names spaghetti
models = {
    # "google/vit-base-patch16-224-in21k": "ViT-B/16 (ImageNet-21k)", # NOT WORKING
    "google/vit-base-patch16-224": "ViT-B/16 (ImageNet-1k)",
    "facebook/convnext-base-224-22k": "ConvNeXt-Base (ImageNet-22k)", # BETTER THAN 1K
    "microsoft/swin-base-patch4-window7-224-in22k": "Swin-B (ImageNet-22k)", # BEST
    "ashaduzzaman/vit-finetuned-food101": "ViT-Food101 (Food101)",
    "ewanlong/food_type_image_detection": "ViT-Food101-2 (Food101)"
}

# Loop through each model, run inference, and print the top-1 prediction
for model_id, model_name in models.items():
    # Load the image processor and model from Hugging Face
    print(f"\nLoading: {model_name}")

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.to("cpu")
    model.eval()      # Set model to evaluation mode

    # Loop through each image in the directory
    for image_file in os.listdir(images_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
            image_path = os.path.join(images_path, image_file)
            image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format

            # Extract the expected label from the filename (excluding the extension)
            expected_label = os.path.splitext(image_file)[0]

            print(f"\nProcessing image: {image_file}")
            print(f"Expected label: {expected_label}")

            # Preprocess the image for this model (resizing, normalization, etc.)
            inputs = image_processor(image, return_tensors="pt")

            # Run the model on the input image and get the output logits
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits

            # Get the index of the highest probability (top-1 prediction)
            predicted_label_idx = logits.argmax(-1).item()
            # Convert that index to the actual label string using the model's config
            predicted_label = model.config.id2label[predicted_label_idx]

            # Print the result for this model
            print(f"{model_name} top-1 prediction: {predicted_label}")
