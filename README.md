# [Deepseek AI](https://huggingface.co/deepseek-ai)

> [!IMPORTANT]
> This is not an official Deepseek project.
> I do not own any of the models or datasets used in this project.

## How to use

You need the UV package/project manager to install the dependencies.  
You can get it from [here](https://docs.astral.sh/uv/getting-started/installation/).

Set up the environment. (Only once)

```bash
uv venv
.venv/Scripts/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --link-mode=symlink
uv sync
```

Run the script.

```bash
.venv/Scripts/activate
uv run python ./deepseek/deepseek_ai.py
```

Models will be downloaded on the first run.

## Models available

Models are fetched from HuggingFace.

1. deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
2. deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
3. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
4. deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
5. deepseek-ai/DeepSeek-R1-Distill-Llama-70B
6. deepseek-ai/DeepSeek-R1-Distill-Llama-8B
7. deepseek-ai/DeepSeek-R1
8. deepseek-ai/DeepSeek-R1-Zero
