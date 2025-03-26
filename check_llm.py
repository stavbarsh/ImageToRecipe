from transformers import AutoTokenizer, AutoModelForCausalLM

language_models = [
    {"name": "GPT-2 Small", "id": "gpt2"}, # NOT GOOD
    {"name": "GPT-2 Medium", "id": "gpt2-medium"}, # NOT GOOD
    {"name": "GPT-2 Large", "id": "gpt2-large"}, # NOT GOOD

    # {"name": "LLaMA 3.2 1B Instruct", "id": "unsloth/Llama-3.2-1B-Instruct"}, # NOT WORKING
    # {"name": "LLaMA 3.2 1B Instruct2", "id": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"}, # NOT WORKING
    # {"name": "Gemma 3 4B GGUF", "id": "unsloth/gemma-3-4b-it-GGUF"}, # NOT WORKING

    {"name": "LLaMA 7B", "id": "NousResearch/Llama-2-7b-hf"},  # NOT GOOD
    {"name": "LLaMA 7B Chat", "id": "NousResearch/Llama-2-7b-chat-hf"}, # NOT GOOD

    # {"name": "LLaMA 13B Chat", "id": "NousResearch/Llama-2-13b-chat-hf"} # NOT ENOUGH RAM

    {"name": "InstructLM 1.3B", "id": "instruction-pretrain/InstructLM-1.3B"}, # GOOD

]

prompts = ["Can you eat a cheeseburger? Answer Yes or No.",
           "Is cheeseburger considered food? Answer Yes or No.",
           "Generate text that describes if you can eat a cheeseburger or not.",
           "Is cheeseburger edible? Answer True or False.",
           ]

# prompt = "What is the capital of Israel?"
results = {}

for model_info in language_models:
    model_id = model_info["id"]
    print(f"\nLoading: {model_info['name']}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        model.eval()

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = generated[len(prompt):].strip()
            results[f"{model_info['name']} | {prompt}"] = result
    except Exception as e:
        results[f"{model_info['name']}"] = f"Error: {str(e)}"

# Print results
for name, output in results.items():
    print(f"{name}: {output}")