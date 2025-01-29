from pprint import pformat
import sys
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

max_format_width = 120

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

prev_context = ""


# Define the generation function
def generate_text(
    message: str,
    history: list,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    system_context: str,
):
    """
    Args:
        message (str): The input message.
        history (list): The conversation history used by ChatInterface.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
        system_context (str): The system context to guide the response generation.
    """
    if not message:
        yield "Please provide a message."

    global prev_context
    max_history_size = 16
    conversation = []
    if len(history) > max_history_size:  # Keep the history size in check
        history = history[:max_history_size]
    conversation.extend(history)
    print(
        f"Conversation:\n{pformat(conversation, width=max_format_width)}",
        file=sys.stderr,
    )
    print(f"Message: {pformat(message, width=max_format_width)}", file=sys.stderr)
    if len(conversation) == 0 or system_context != prev_context:
        message = system_context + "\n" + message
        prev_context = system_context
    conversation.append({"role": "user", "content": message})
    conversation.append({"role": "assistant", "content": ""})

    print(
        f"Input options: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}",
        file=sys.stderr,
    )

    input_ids = tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    input_ids = input_ids.to(device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True
    )

    generated_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=input_ids.ne(tokenizer.eos_token_id),
    )
    # Prevent crashing when the temperature is 0.
    if temperature == 0:
        generated_kwargs["do_sample"] = False

    t = Thread(target=model.generate, kwargs=generated_kwargs)
    t.start()

    outputs = []
    print(f"Output: {pformat(outputs, width=max_format_width)}", file=sys.stderr)
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


def generate_multimodal(inputs: dict, history: list[dict]):
    """
    the first argument of fn should accept not a str message but a dict message with keys "text" and "files"
    """

    text = generate_text(inputs["text"], history)
    return text


# Create the Gradio interface
interface = gr.ChatInterface(
    fn=generate_text,
    show_progress="full",
    additional_inputs_accordion=gr.Accordion(
        label="⚙️ Parameters", open=False, render=False
    ),
    additional_inputs=[
        gr.Slider(
            minimum=0.5,
            maximum=0.7,
            step=0.05,
            value=0.6,
            label="Temperature",
            render=False,
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1,
            step=0.05,
            value=0.95,
            label="Top p",
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
        gr.Textbox(
            label="System context",
            placeholder="Provide background information for the bot.",
            value="You are a helpful assistant that answers questions in a friendly and concise manner.",
            render=False,
        ),
    ],
    # cache_mode="lazy",
    # cache_examples=True,
    examples=[
        [
            "What can you do?",
            0.6,
            0.9,
            256,
            "You are a helpful assistant that answers questions in a friendly and concise manner.",
        ],
        [
            "What is 2+2?",
            0.5,
            0.95,
            128,
            "You are a math expert who solves problems using formal methods. Please reason step by step, and put your final answer within \\boxed{}.",
        ],
    ],
    type="messages",
    title="DeepSeek AI on local machine",
    description=f"Running {model_name} on {device}.",
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(pwa=True)
