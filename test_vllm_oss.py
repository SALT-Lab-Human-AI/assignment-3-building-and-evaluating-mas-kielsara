import os
from openai import OpenAI

# Read credentials from environment. Do NOT hardcode secrets.
BASE_URL = os.getenv("VLLM_BASE_URL", "https://vllm.salt-lab.org/v1")
API_KEY = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise SystemExit("No API key found. Set VLLM_API_KEY or OPENAI_API_KEY in your environment.")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

result = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what MXFP4 quantization is."},
    ],
)

print(result.choices[0].message.content)

response = client.responses.create(
    model="openai/gpt-oss-20b",
    instructions="You are a helfpul assistant.",
    input="Explain what MXFP4 quantization is.",
)

print(response.output_text)
