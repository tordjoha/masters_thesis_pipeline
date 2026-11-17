import requests
import re
import os
import json
from config import LLAMA_URL, MODEL_NAME, N_SAMPLES, LLM_TIMEOUT, DEFAULT_TEMPERATURE
from models import GeneratedTest

TOKENIZE_URL = LLAMA_URL.rsplit('/', 1)[0] + '/tokenize'


def count_tokens(text):
    """
    Count tokens using the model's actual tokenizer via llama.cpp's /tokenize endpoint.
    Raises on failure rather than silently falling back, since a wrong count
    defeats the purpose of this function.
    """
    try:
        response = requests.post(TOKENIZE_URL, json={"content": text}, timeout=30)
        response.raise_for_status()
        return len(response.json()["tokens"])
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Token counting failed - is llama.cpp server running at {TOKENIZE_URL}? ({e})")


def call_llm(prompt, temperature=DEFAULT_TEMPERATURE):
    """
    Call the llama.cpp server with the given prompt.
    Wraps the prompt in ChatML format for the local llm-model.
    Returns the generated text, or None on failure.
    """

    # starting instructions for the llm
    formatted_prompt = (
        f"<|im_start|>system\n"
        f"You are an expert Java developer writing JUnit 4 unit tests.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    payload = {
        "prompt": formatted_prompt,
        "temperature": temperature,
        "n_predict": 1024,
        "stop": ["<|im_end|>", "<|im_start|>", "[/im_end|>", "[/im_start|>"],
        "cache_prompt": True,
        "stream": False
    }

    try:
        response = requests.post(LLAMA_URL, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status()
        return response.json()["content"]
    except requests.exceptions.Timeout:
        print("ERROR: llama.cpp request timed out")
        return None
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to llama.cpp server - is it running?")
        return None
    except Exception as e:
        print(f"ERROR: LLM call failed: {e}")
        return None


def clean_generated_test(raw_output):
    """
    Clean LLM output to extract raw Java test code.
    """
    cleaned = raw_output.strip()
    was_wrapped = False

    # remove markdown code fences
    fence_pattern = r'^```(?:java)?\s*\n(.*?)\n```\s*$'
    match = re.match(fence_pattern, cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
        was_wrapped = True

    # remove ChatML artifacts
    stop_artifacts = ['<|im_end|>', '<|im_start|>', '[/im_end|>', '[/im_start|>']
    for artifact in stop_artifacts:
        cleaned = cleaned.replace(artifact, '').strip()

    return cleaned, was_wrapped


def generate_tests(prompt, bug_id, output_dir, n_samples=N_SAMPLES, temperature=DEFAULT_TEMPERATURE):
    """
    Generate N test samples for a given prompt.
    Saves each sample and raw output to disk.
    Returns list of GeneratedTest objects.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    n_wrapped = 0

    for i in range(n_samples):
        print(f"  Generating sample {i+1}/{n_samples}...")
        result = call_llm(prompt, temperature=temperature)

        if result is None:
            print(f"  WARNING: sample {i+1} failed, skipping")
            continue

        cleaned, was_wrapped = clean_generated_test(result)
        if was_wrapped:
            n_wrapped += 1

        sample_path = os.path.join(output_dir, f"sample_{i+1}.java")
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        raw_path = os.path.join(output_dir, f"sample_{i+1}_raw.java")
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(result)

        generated.append(GeneratedTest(
            sample_id=i + 1,
            content=cleaned,
            path=sample_path,
            was_wrapped=was_wrapped
        ))

    meta = {
        "bug_id": bug_id,
        "model": MODEL_NAME,
        "temperature": temperature,
        "n_requested": n_samples,
        "n_generated": len(generated),
        "n_markdown_wrapped": n_wrapped
    }
    with open(os.path.join(output_dir, "generation_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    return generated
