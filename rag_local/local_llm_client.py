# rag_local/llm_router.py
import os, requests

# install ollama: service windows sur http://localhost:11434
# ollama pull mistral (llama3.1:8b, qwen2.5, etc.) ou 
# 
# Ollama expose une API HTTP locale (par défaut http://127.0.0.1:11434).
def call_llm_local(prompt: str, model: str = "llama3.1:8b", temperature: float = 0.2) -> str:
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    response=data.get("response", "").strip()
    return response

# avec llama-server (fourni par llama.cpp), vous pouvez exposer une API compatible OpenAI.
# build: ./llama-server -m ./models/model.gguf --port 8080
def call_client_http_minimal(prompt: str, model: str = "local-gguf", temperature: float = 0.2) -> str:
    url = "http://127.0.0.1:8080/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": 600
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    
    return data["choices"][0]["text"].strip()