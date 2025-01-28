import sys
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_source = "huggingface"

model_names = (
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Zero",
)


# Load the model and tokenizer
model_name = model_names[3]
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Loaded model: {model_name}", file=sys.stderr)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", file=sys.stderr)
model.to(device)


# Define the generation function
def generate_text(input: str, options: list[dict]):
    """
    Normally (assuming type is set to "messages"), the function should accept two parameters:
    a str representing the input message and list of openai-style dictionaries:
    {"role": "user" | "assistant", "content": str | {"path": str} | gr.Component} representing the chat history.
    The function should return/yield a str (for a simple message), a supported Gradio component
    (e.g. gr.Image to return an image), a dict (for a complete openai-style message response), or a list of such messages.
    """
    # Extract options
    max_length = 1024
    num_return_sequences = 1
    top_p = 0.95
    top_k = 50
    temperature = 0.6

    # Tokenize the input text
    input_ids = tokenizer.encode(
        input,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    input_ids = input_ids.to(device)

    # Generate text based on the input
    output = model.generate(
        input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    output = output.to("cpu")

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def generate_multimodal(inputs: dict, options: list[dict]):
    """
    the first argument of fn should accept not a str message but a dict message with keys "text" and "files"
    """

    return generate_text(inputs["text"], options)


# Create the Gradio interface
interface = gr.ChatInterface(
    fn=generate_text,
    cache_mode="lazy",
    cache_examples=True,
    examples=[
        "What can you do?",
        "What is 2+2?",
    ],
    type="messages",
    title="DeepSeek AI",
    description="DeepSeek AI is a text generation model trained on a large corpus of text. It can generate text based on the input you provide.",
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(pwa=True)
