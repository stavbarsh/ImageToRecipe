import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
import json

# Directory to store downloaded models
MODEL_DIR = os.path.expanduser("~/.food_recipe_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Full Model Catalog
MODEL_CATALOG = {
    "classifier": [
        {"name": "MobileNetV2", "id": "mobilenet_v2", "type": "torchvision", "size": "14MB"},
        {"name": "InceptionV3", "id": "inception_v3", "type": "torchvision", "size": "92MB"},
        {"name": "VGG16", "id": "vgg16", "type": "torchvision", "size": "528MB"},
        {"name": "ViT-B/16 (IN-1k)", "id": "google/vit-base-patch16-224", "type": "huggingface_vision", "size": "330MB"},
        {"name": "ConvNeXt-Base (IN-22k)", "id": "facebook/convnext-base-224-22k", "type": "huggingface_vision", "size": "340MB"},
        {"name": "Swin-B (IN-22k)", "id": "microsoft/swin-base-patch4-window7-224-in22k", "type": "huggingface_vision", "size": "340MB"}, # This model works best
        {"name": "ViT-Food101 (Food101)", "id": "ashaduzzaman/vit-finetuned-food101", "type": "huggingface_vision", "size": "330MB"},
        {"name": "ViT-Food101-2 (Food101)", "id": "ewanlong/food_type_image_detection", "type": "huggingface_vision", "size": "330MB"}
    ],
    "language": [
        {"name": "GPT-2 Small", "id": "gpt2", "type": "huggingface", "size": "2GB"},
        {"name": "GPT-2 Medium", "id": "gpt2-medium", "type": "huggingface", "size": "4GB"},
        {"name": "GPT-2 Large", "id": "gpt2-large", "type": "huggingface", "size": "8GB"},
        {"name": "LLaMA 7B", "id": "NousResearch/Llama-2-7b-hf", "type": "huggingface", "size": "13.5GB"},
        {"name": "LLaMA 7B Chat", "id": "NousResearch/Llama-2-7b-chat-hf", "type": "huggingface", "size": "13.5GB"}, # This model works best
        {"name": "InstructLM 1.3B", "id": "instruction-pretrain/InstructLM-1.3B", "type": "huggingface", "size": "5.4GB"}
    ]
}

# Application State
app_state = {
    "classifier_model": MODEL_CATALOG["classifier"][0],
    "language_model": MODEL_CATALOG["language"][0]
}

# Load ImageNet labels
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

LABELS_1K = load_labels("models/imagenet1k_labels.txt")


# --- Model Functions ---
def load_classifier(model_info):
    model_type = model_info["type"]
    model_id = model_info["id"]
    local_path = os.path.join("models", model_id.replace('/', '_'))

    if model_type == "torchvision":
        if model_id == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=True)
        elif model_id == "inception_v3":
            model = models.inception_v3(pretrained=True)
        elif model_id == "vgg16":
            model = models.vgg16(pretrained=True)
        else:
            raise ValueError("Unknown torchvision model.")
        model.eval()
        return model

    if model_type == "huggingface_vision":
        if os.path.exists(local_path):
            model = AutoModelForImageClassification.from_pretrained(local_path)
            processor = AutoImageProcessor.from_pretrained(local_path)
        else:
            model = AutoModelForImageClassification.from_pretrained(model_id)
            processor = AutoImageProcessor.from_pretrained(model_id)
        model.eval()
        return (model, processor)
    else:
        raise ValueError("Unsupported model type.")


def load_language_model(model_info):
    model_id = model_info["id"]
    local_path = os.path.join("models", model_id.replace('/', '_'))
    if os.path.exists(local_path):
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForCausalLM.from_pretrained(local_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval().to(torch.device("cpu"))
    # model.eval().to(torch.device("gpu"))
    return tokenizer, model

def classify_image(model_info, model_obj, image):
    model_type = model_info["type"]

    if model_type == "torchvision":
        # Preprocess using torchvision defaults
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model_obj(input_tensor)
            _, predicted = output.max(1)
            index = predicted.item()
            label = LABELS_1K[index]
        return label

    elif model_type == "huggingface_vision":
        model, processor = model_obj
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        index = torch.argmax(logits, dim=-1).item()
        label = model.config.id2label[index]
        return label

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def generate_text(tokenizer, model, prompt, max_new_tokens=50, do_sample=False):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    if do_sample:
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    else:
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,  # deterministic; use True with temperature for variation
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()

def is_food_item(tokenizer, model, label):
    # prompt = f"Is '{label}' considered food? Answer Yes or No."
    prompt = f"Is '{label}' edible? Answer Yes or No."

    response = generate_text(tokenizer, model, prompt, max_new_tokens=50, do_sample=False)
    print(f"Is food: {response}")
    return "yes" in response.lower()

def suggest_dishes(tokenizer, model, label):
    prompt = (
        f"List several common dishes that could be described as '{label}'. "
        f"Respond with dish names only, separated by commas. Do not include any explanations."
    )
    response = generate_text(tokenizer, model, prompt, max_new_tokens=50, do_sample=False)
    print(f"Suggested dishes: {response}")
    lines = [s.strip() for s in response.splitlines() if s.strip()]
    if len(lines) > 1:
        return lines
    else:
        return [s.strip() for s in response.split(',') if s.strip()]

def generate_recipe(tokenizer, model, dish):
    prompt = f"Write a detailed recipe for {dish}."
    response = generate_text(tokenizer, model, prompt, max_new_tokens=200, do_sample=False)
    print(f"Recipe: {response}")
    return response[len(prompt):].strip()


# --- GUI Classes ---
class ModelSelectionWindow(tk.Toplevel):
    def __init__(self, parent, update_callback):
        super().__init__(parent)
        self.title("Select Models")
        self.update_callback = update_callback
        tk.Label(self, text="Select Classifier Model").pack()
        self.classifier_var = tk.StringVar(value=app_state["classifier_model"]["name"])
        for model in MODEL_CATALOG["classifier"]:
            tk.Radiobutton(self, text=f"{model['name']} ({model['size']})", variable=self.classifier_var, value=model["name"]).pack(anchor=tk.W)

        tk.Label(self, text="\nSelect Language Model").pack()
        self.language_var = tk.StringVar(value=app_state["language_model"]["name"])
        for model in MODEL_CATALOG["language"]:
            tk.Radiobutton(self, text=f"{model['name']} ({model['size']})", variable=self.language_var, value=model["name"]).pack(anchor=tk.W)

        tk.Button(self, text="Confirm", command=self.select_models).pack(pady=10)

    def select_models(self):
        app_state["classifier_model"] = next(m for m in MODEL_CATALOG["classifier"] if m["name"] == self.classifier_var.get())
        app_state["language_model"] = next(m for m in MODEL_CATALOG["language"] if m["name"] == self.language_var.get())
        self.update_callback()
        self.destroy()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Food Image to Recipe")

        # Center window on screen using default size
        self.root.update_idletasks()  # ensure winfo_width/height returns valid values
        width = 600
        height = 400
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")

        # Auto-select default models
        app_state["classifier_model"] = {
            "name": "Swin-B (IN-22k)",
            "id": "microsoft/swin-base-patch4-window7-224-in22k",
            "type": "huggingface_vision",
            "size": "340MB"
        }
        app_state["language_model"] = {
            "name": "InstructLM 1.3B",
            "id": "instruction-pretrain/InstructLM-1.3B",
            "type": "huggingface",
            "size": "5.4GB"
        }

        # Welcome Window
        self.label = tk.Label(root, text="Welcome to the Food Image to Recipe Generator")
        self.label.pack()
        tk.Button(root, text="Select Models", command=self.open_model_selection).pack()
        tk.Button(root, text="Continue", command=self.setup_main_window).pack()

    def open_model_selection(self):
        ModelSelectionWindow(self.root, self.load_models)

    def load_models(self):
        self.classifier = load_classifier(app_state["classifier_model"])
        self.tokenizer, self.lm = load_language_model(app_state["language_model"])

    def setup_main_window(self):
        self.load_models()
        self.clear_widgets()
        self.canvas = tk.Label(self.root)
        self.canvas.pack()
        tk.Button(self.root, text="Upload Image", command=self.upload_image).pack()
        self.generate_button = tk.Button(self.root, text="Generate Recipe", command=self.select_dish)
        self.save_button = tk.Button(self.root, text="Save Result", command=self.save_result)

    def clear_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def upload_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        self.clear_dynamic_elements()
        self.image_path = path
        self.original_image = Image.open(path).convert("RGB")
        self.display_image(self.original_image)

        self.result_label = tk.Label(self.root, text="Image loaded. Click 'Analyze Image' to continue.")
        self.result_label.pack()

        self.analyze_button = tk.Button(self.root, text="Analyze Image", command=self.analyze_image)
        self.analyze_button.pack()

    def analyze_image(self):
        self.clear_dynamic_elements()

        label = classify_image(app_state["classifier_model"], self.classifier, self.original_image)

        if not is_food_item(self.tokenizer, self.lm, label):
            messagebox.showinfo("Result",
                                f"Detected item: {label}\nThis is not recognized as food using the language model."
                                f"\n Continue to generate the recipe if desired or try changing the language model")

            self.change_models_button = tk.Button(self.root, text="Change Models", command=self.change_models)
            self.change_models_button.pack()
            # return

        self.detected_label = label  # store for later use

        # Show detected food label
        self.food_label = tk.Label(self.root, text=f"Detected food: {label}")
        self.food_label.pack()

        # Suggestion button (optional)
        self.suggestions_button = tk.Button(self.root, text="Suggest similar dishes", command=self.show_suggestions)
        self.suggestions_button.pack()

        # Generate button (use label directly)
        self.generate_button = tk.Button(self.root, text="Generate Recipe", command=self.select_dish)
        self.generate_button.pack()

    def show_suggestions(self):
        suggestions = suggest_dishes(self.tokenizer, self.lm, self.detected_label)

        if hasattr(self, 'listbox'):
            self.listbox.destroy()
        if hasattr(self, 'generate_button'):
            self.generate_button.destroy()

        self.instruction_label = tk.Label(self.root, text="Select a dish below and click 'Generate Recipe'.")
        self.instruction_label.pack()

        self.listbox = tk.Listbox(self.root, width=50, height=6)
        self.listbox.insert(tk.END, self.detected_label)  # always include original label

        for item in suggestions:
            item_clean = item.strip()
            if item_clean and item_clean.lower() != self.detected_label.lower():
                self.listbox.insert(tk.END, item_clean)

        self.listbox.pack()

        self.generate_button = tk.Button(self.root, text="Generate Recipe", command=self.select_dish)
        self.generate_button.pack()

    def select_dish(self):
        if hasattr(self, 'listbox') and self.listbox.size() > 0:
            selection = self.listbox.get(tk.ACTIVE)
        else:
            selection = self.detected_label

        self.recipe = generate_recipe(self.tokenizer, self.lm, selection)
        messagebox.showinfo("Recipe", self.recipe)
        self.save_button.pack()

        self.change_models_button = tk.Button(self.root, text="Change Models", command=self.change_models)
        self.change_models_button.pack()

    def change_models(self):
        self.clear_dynamic_elements()
        self.display_image(self.original_image)
        self.analyze_button = tk.Button(self.root, text="Analyze Image", command=self.analyze_image)
        self.analyze_button.pack()
        self.open_model_selection()

    def clear_dynamic_elements(self):
        self.generate_button.pack_forget()
        self.save_button.pack_forget()
        if hasattr(self, 'listbox'):
            self.listbox.destroy()
        if hasattr(self, 'analyze_button'):
            self.analyze_button.destroy()
            self.result_label.destroy()
        if hasattr(self, 'suggestions_button'):
            self.suggestions_button.destroy()
        if hasattr(self, 'instruction_label'):
            self.instruction_label.destroy()
        if hasattr(self, 'food_label'):
            self.food_label.destroy()
        if hasattr(self, 'change_models_button'):
            self.change_models_button.destroy()

    def save_result(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt")
        if file_path:
            with open(file_path, "w") as f:
                f.write("--- Recipe ---\n")
                f.write(self.recipe + "\n\n")
                f.write("--- Model Info ---\n")
                json.dump({"classifier": app_state["classifier_model"], "language_model": app_state["language_model"]}, f, indent=2)

    def display_image(self, image):
        max_dim = 224
        w, h = image.size
        scale = min(max_dim / w, max_dim / h)
        new_size = (int(w * scale), int(h * scale))
        img = image.resize(new_size)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.photo)
        self.canvas.image = self.photo


root = tk.Tk()
app = App(root)
root.mainloop()

