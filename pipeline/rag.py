import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from retrieval_func.retrieval import retrieve

model_name = "Qwen/Qwen2-1.5B-Instruct"
device = ("cuda" if torch.cuda.is_available() else "cpu",)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)
model.eval()


def rag_response(query, top_k=4):
    top_chunks = retrieve(query, top_k)

    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    prompt = f"""
        you are a helpful AI assistant.
        use the context below to answer the questions accurately and clearly.
        context:
        {context}

        Question: {query}
        Answer:
    """

    # token and gen
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n------------ generated answer ------------")
    print(answer)
    print(answer.strip())
    print("\n------------------------------------------")


if __name__ == "__main__":
    while True:
        query = input("ask question(or type 'exit'): ").strip()
        if query.lower() == "exit":
            break
        rag_response(query, top_k=4)
