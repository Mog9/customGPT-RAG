import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from retrieval_func.retrieval import retrieve

model_name = "Qwen/Qwen2-1.5B-Instruct"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float32, device_map="auto", low_cpu_mem_usage=True
)
model.eval()


def rag_response(query, top_k=4):
    top_chunks = retrieve(query, top_k)

    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    messages = [
        {
            "role": "system",
            "content": "you are a helpful AI assistant. answer questions accurately and concisely using the provided context.",
        },
        {
            "role": "user",
            "content": f"context:\n{context}\n\nquestion: {query}\n\nprovide a brief, direct answer.",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # token and gen
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = outputs[0][inputs.input_ids.shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"\nQuestion: {query}")
    print(f"Answer: {answer}\n")


if __name__ == "__main__":
    while True:
        query = input("Ask question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break
        rag_response(query, top_k=4)
