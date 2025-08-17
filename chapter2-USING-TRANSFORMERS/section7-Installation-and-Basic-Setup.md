# نصب و راه‌اندازی اولیه

فریم‌ورک **TGI** نصب و استفاده از آن بسیار آسان است و به‌طور عمیق در اکوسیستم **Hugging Face** یکپارچه شده است.

ابتدا سرور **TGI** را با استفاده از Docker راه‌اندازی کنید:

```bash
docker run --gpus all \
    --shm-size 1g \
    -p 8080:80 \
    -v ~/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id HuggingFaceTB/SmolLM2-360M-Instruct
```

سپس با استفاده از **InferenceClient** از **Hugging Face** با آن تعامل داشته باشید:

```python
from huggingface_hub import InferenceClient

# Initialize client pointing to TGI endpoint
client = InferenceClient(
    model="http://localhost:8080",  # URL to the TGI server
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
    stop_sequences=[],
)
print(response.generated_text)

# For chat format
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```
به‌طور جایگزین، می‌توانید از **OpenAI client** استفاده کنید:
```python
from openai import OpenAI

# Initialize client pointing to TGI endpoint
client = OpenAI(
    base_url="http://localhost:8080/v1",  # Make sure to include /v1
    api_key="not-needed",  # TGI doesn't require an API key by default
)

# Chat completion
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```
## نصب و راه‌اندازی llama.cpp

فریم‌ورک **llama.cpp** نصب و استفاده از آن بسیار آسان است و به حداقل وابستگی‌ها نیاز دارد و از استنتاج (**inference**) با **CPU** و **GPU** پشتیبانی می‌کند.

ابتدا **llama.cpp** را نصب و بسازید:
```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build the project
make

# Download the SmolLM2-1.7B-Instruct-GGUF model
curl -L -O https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct.Q4_K_M.gguf
```
سپس، سرور را راه‌اندازی کنید (با سازگاری با API **OpenAI**):
```bash
# Start the server
./server \
    -m smollm2-1.7b-instruct.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 4096 \
    --n-gpu-layers 0  # Set to a higher number to use GPU
```
با استفاده از **InferenceClient** از **Hugging Face** با سرور تعامل داشته باشید:
```python
from huggingface_hub import InferenceClient

# Initialize client pointing to llama.cpp server
client = InferenceClient(
    model="http://localhost:8080/v1",  # URL to the llama.cpp server
    token="sk-no-key-required",  # llama.cpp server requires this placeholder
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
)
print(response.generated_text)

# For chat format
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```
به‌طور جایگزین، می‌توانید از **OpenAI client** استفاده کنید:
```python
from openai import OpenAI

# Initialize client pointing to llama.cpp server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required",  # llama.cpp server requires this placeholder
)

# Chat completion
response = client.chat.completions.create(
    model="smollm2-1.7b-instruct",  # Model identifier can be anything as server only loads one model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```
## نصب و راه‌اندازی vLLM

فریم‌ورک **vLLM** نصب و استفاده از آن بسیار آسان است و هم با **API OpenAI** سازگاری دارد و هم دارای رابط **Python** بومی است.

ابتدا سرور **vLLM** با سازگاری **OpenAI** را راه‌اندازی کنید:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --host 0.0.0.0 \
    --port 8000
```
سپس با استفاده از **InferenceClient** از **Hugging Face** با آن تعامل داشته باشید:
```python
from huggingface_hub import InferenceClient

# Initialize client pointing to vLLM endpoint
client = InferenceClient(
    model="http://localhost:8000/v1",  # URL to the vLLM server
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
)
print(response.generated_text)

# For chat format
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```
به‌طور جایگزین، می‌توانید از **OpenAI client** استفاده کنید:
```python
from openai import OpenAI

# Initialize client pointing to vLLM endpoint
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM doesn't require an API key by default
)

# Chat completion
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```
## تولید متن پایه

بیایید به نمونه‌هایی از تولید متن با استفاده از فریم‌ورک‌ها نگاه کنیم:

فریم‌ورک **TGI** را ابتدا با پارامترهای پیشرفته راه‌اندازی کنید:
```bash
docker run --gpus all \
    --shm-size 1g \
    -p 8080:80 \
    -v ~/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
    --max-total-tokens 4096 \
    --max-input-length 3072 \
    --max-batch-total-tokens 8192 \
    --waiting-served-ratio 1.2
```
از **InferenceClient** برای تولید متن انعطاف‌پذیر استفاده کنید:
```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")

# Advanced parameters example
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

# Raw text generation
response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    details=True,
)
print(response.generated_text)
```
یا از **OpenAI client** استفاده کنید:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# Advanced parameters example
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,  # Higher for more creativity
)
print(response.choices[0].message.content)
```
برای **llama.cpp**، می‌توانید هنگام راه‌اندازی سرور، پارامترهای پیشرفته را تنظیم کنید:
```bash
./server \
    -m smollm2-1.7b-instruct.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 4096 \            # Context size
    --threads 8 \        # CPU threads to use
    --batch-size 512 \   # Batch size for prompt evaluation
    --n-gpu-layers 0     # GPU layers (0 = CPU only)
```
از **InferenceClient** استفاده کنید:
```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080/v1", token="sk-no-key-required")

# Advanced parameters example
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

# For direct text generation
response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
    details=True,
)
print(response.generated_text)
```
یا از **OpenAI client** برای تولید متن با کنترل بر روی پارامترهای نمونه‌گیری استفاده کنید:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key-required")

# Advanced parameters example
response = client.chat.completions.create(
    model="smollm2-1.7b-instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Nucleus sampling probability
    frequency_penalty=0.5,  # Reduce repetition of frequent tokens
    presence_penalty=0.5,  # Reduce repetition by penalizing tokens already present
    max_tokens=200,  # Maximum generation length
)
print(response.choices[0].message.content)
```
همچنین می‌توانید از **کتابخانه بومی llama.cpp** برای کنترل بیشتر استفاده کنید:
```python
# Using llama-cpp-python package for direct model access
from llama_cpp import Llama

# Load the model
llm = Llama(
    model_path="smollm2-1.7b-instruct.Q4_K_M.gguf",
    n_ctx=4096,  # Context window size
    n_threads=8,  # CPU threads
    n_gpu_layers=0,  # GPU layers (0 = CPU only)
)

# Format prompt according to the model's expected format
prompt = """<|im_start|>system
You are a creative storyteller.
<|im_end|>
<|im_start|>user
Write a creative story
<|im_end|>
<|im_start|>assistant
"""

# Generate response with precise parameter control
output = llm(
    prompt,
    max_tokens=200,
    temperature=0.8,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["<|im_end|>"],
)

print(output["choices"][0]["text"])
```
برای استفاده پیشرفته از **vLLM**، می‌توانید از **InferenceClient** استفاده کنید:
```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8000/v1")

# Advanced parameters example
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

# For direct text generation
response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    details=True,
)
print(response.generated_text)
```
همچنین می‌توانید از **OpenAI client** استفاده کنید:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Advanced parameters example
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    top_p=0.95,
    max_tokens=200,
)
print(response.choices[0].message.content)
```
فریم‌ورک **vLLM** همچنین یک رابط بومی **Python** با کنترل دقیق‌تر ارائه می‌دهد:
```python
from vllm import LLM, SamplingParams

# Initialize the model with advanced parameters
llm = LLM(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    gpu_memory_utilization=0.85,
    max_num_batched_tokens=8192,
    max_num_seqs=256,
    block_size=16,
)

# Configure sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Consider top 95% probability mass
    max_tokens=100,  # Maximum length
    presence_penalty=1.1,  # Reduce repetition
    frequency_penalty=1.1,  # Reduce repetition
    stop=["\n\n", "###"],  # Stop sequences
)

# Generate text
prompt = "Write a creative story"
outputs = llm.generate(prompt, sampling_params)
print(outputs[0].outputs[0].text)

# For chat-style interactions
chat_prompt = [
    {"role": "system", "content": "You are a creative storyteller."},
    {"role": "user", "content": "Write a creative story"},
]
formatted_prompt = llm.get_chat_template()(chat_prompt)  # Uses model's chat template
outputs = llm.generate(formatted_prompt, sampling_params)
print(outputs[0].outputs[0].text)
```
### کنترل پیشرفته تولید متن

**انتخاب توکن و نمونه‌گیری**

فرآیند تولید متن شامل انتخاب توکن بعدی در هر مرحله است. این فرآیند انتخاب را می‌توان از طریق پارامترهای مختلف کنترل کرد:

* **Raw Logits**: احتمال‌های اولیه خروجی برای هر توکن
* **Temperature**: میزان تصادفی بودن انتخاب را کنترل می‌کند (بالا = خلاقانه‌تر)
* **Top-p (Nucleus) Sampling**: فیلتر کردن توکن‌ها برای انتخاب آن‌هایی که X درصد از جرم احتمالات را تشکیل می‌دهند
* **Top-k Filtering**: محدود کردن انتخاب به **k** توکن با بیشترین احتمال

در اینجا نحوه تنظیم این پارامترها آورده شده است:

