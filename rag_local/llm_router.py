# rag_local/llm_router.py
import os

def call_llm(prompt: str, model: str):
    backend = os.getenv("LLM_BACKEND", "ollama")  # openai|ollama|llamacpp|hf

    if backend == "ollama":
        from rag_local.local_llm_client import call_ollama_llm_generate
        return call_ollama_llm_generate(prompt, model=model)
    elif backend == "llamacpp":
        from rag_local.llamacpp_client import call_llm_local
        return call_ollama_llm_generate(prompt, model=model)
    elif backend == "hf":
        from rag_local.hf_client import call_llm_local
        return call_ollama_llm_generate(prompt)
    else:
        from rag_local.openai_client import call_llm as call_llm_openai
        return call_llm_openai(prompt, model=model)
