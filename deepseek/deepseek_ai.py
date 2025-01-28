from pprint import pformat, pprint
import sys
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

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
def generate_text(
    message: str, history: list, temperature: float = 0.6, max_new_tokens: int = 1024
):
    """
    Args:
        message (str): The input message.
        history (list): The conversation history used by ChatInterface.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    """
    # Options
    # num_return_sequences = 1
    top_p = 0.95
    top_k = 50

    max_conversation_length = 128
    conversation = []
    print(pformat(history), file=sys.stderr)
    for user, assistant in history:
        conversation.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )
    conversation.append({"role": "user", "content": message})
    if len(conversation) > max_conversation_length:
        conversation = conversation[:max_conversation_length]

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(
        model.device
    )

    input_ids = input_ids.to(device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=15.0, skip_prompt=True, skip_special_tokens=True
    )

    generated_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        # num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Prevent crashing when the temperature is 0.
    if temperature == 0:
        generated_kwargs["do_sample"] = False

    t = Thread(target=model.generate, kwargs=generated_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        # print(outputs)
        yield "".join(outputs)


def generate_multimodal(inputs: dict, options: list[dict]):
    """
    the first argument of fn should accept not a str message but a dict message with keys "text" and "files"
    """

    return generate_text(inputs["text"], options)


# Create the Gradio interface
interface = gr.ChatInterface(
    fn=generate_text,
    additional_inputs_accordion=gr.Accordion(
        label="⚙️ Parameters", open=False, render=False
    ),
    additional_inputs=[
        gr.Slider(
            minimum=0,
            maximum=1,
            step=0.1,
            value=0.6,
            label="Temperature",
            render=False,
        ),
        gr.Slider(
            minimum=1,
            maximum=8192,
            step=1,
            value=512,
            label="Max new tokens",
            render=False,
        ),
    ],
    cache_mode="lazy",
    cache_examples=True,
    examples=[
        [
            "What can you do?",
            0.6,
            128,
        ],
        [
            "What is 2+2?",
            0.5,
            16,
        ],
    ],
    type="messages",
    title="DeepSeek AI",
    description="This is a demo of DeepSeek AI. It is not an official project.",
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(pwa=True)
