# Capstone Project: AI-Powered Text Completion
# Mandy Lubinski

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

max_prompt_tokens = 100

# Handles user errors when inputting temperature and top_p values.
def get_float(prompt_text, default, min_val=0.0, max_val=1.0):
    while True:
        user_input = input(f"{prompt_text} (default: {default}): ").strip()
        if not user_input:
            return default
        try:
            value = float(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Handles user errors when inputting max length values.
def get_int(prompt_text, default, min_val=1, max_val=1024):
    while True:
        user_input = input(f"{prompt_text} (default: {default}): ").strip()
        if not user_input:
            return default
        try:
            value = int(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

# Menu function to run program
def menu():
    print("Welcome to the GPT-2 Text Completion App.")
    print("You may adjust generation parameters before entering prompts.")

    temperature = get_float("Enter temperature (0.0 = conservative, 1.0 = creative)", default=0.7)
    top_p = get_float("Enter top_p (nucleus sampling, typical: 0.9)", default=0.9)
    max_length = get_int("Enter max length for output (includes prompt tokens)", default=150)

    while True:
        prompt = input("\nEnter a prompt (or type 'exit' to quit): ").strip()

        if prompt.lower() == 'exit':
            print("Goodbye!")
            break

        if not prompt:
            print("Warning: Please enter a non-empty prompt.")
            continue

        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        if input_ids.shape[1] > max_prompt_tokens:
            print(f"Warning: Prompt too long ({input_ids.shape[1]} tokens). Please shorten it to under {max_prompt_tokens} tokens.")
            continue

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print("\nGenerated Response:\n", response)

        except Exception as e:
            print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    menu()