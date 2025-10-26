import json
from numpy._core.numeric import full
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from retrieval_func.retrieval import retrieve
from context.main import ConversationContext
from context.math import handle_math_query


model_name = "Qwen/Qwen2-1.5B-Instruct"
device = "cpu"
torch.set_num_threads(8)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float32, low_cpu_mem_usage=True
)
model.to(device)
model.eval()

conversation = ConversationContext(max_history=3)


def rag_response(query, top_k=5, relevance_threshold=0.5):
    is_math, math_answer = handle_math_query(query)
    if is_math:
        print("------------------------")
        print(f"Answer: {math_answer}")
        print("------------------------\n")
        conversation.add_exchange(query, math_answer)
        return

    top_chunks = retrieve(query, top_k)
    max_score = max([chunk["score"] for chunk in top_chunks]) if top_chunks else 0

    use_rag = max_score > relevance_threshold

    if use_rag:
        context = "\n\n".join([chunk["text"] for chunk in top_chunks])

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant with access to private project documentation. Your task is to answer questions only based on the provided context. If the answer is contained in the context, use that information to respond clearly and accurately. If the question asks about topics not included in the context, do not invent explanations or provide unrelated technical details; instead, respond with “The information is not available in the document.” Always think carefully before answering, ensure your response is factually grounded in the context, and avoid making assumptions or adding external knowledge that is not present in the supplied text. Accuracy and relevance are more important than verbosity.",
            },
        ]

        for exchange in conversation.get_history():
            messages.append({"role": "user", "content": exchange["question"]})
            messages.append({"role": "assistant", "content": exchange["answer"]})

        messages.append(
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        )
    else:
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant with access to private project documentation. Your task is to answer questions only based on the provided context. If the answer is contained in the context, use that information to respond clearly and accurately. If the question asks about topics not included in the context, do not invent explanations or provide unrelated technical details; instead, respond with “The information is not available in the document.” Always think carefully before answering, ensure your response is factually grounded in the context, and avoid making assumptions or adding external knowledge that is not present in the supplied text. Accuracy and relevance are more important than verbosity.",
            },
        ]

        for exchange in conversation.get_history():
            messages.append({"role": "user", "content": exchange["question"]})
            messages.append({"role": "assistant", "content": exchange["answer"]})

        messages.append({"role": "user", "content": query})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    generated_ids = outputs[0][inputs.input_ids.shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("------------------------")
    print(f"Answer: {answer}")
    print("------------------------\n")

    conversation.add_exchange(query, answer)


if __name__ == "__main__":
    print("loading model...(first query will be slower)")
    rag_response("test", top_k=1)
    print("ready! Commands: 'exit' to quit, 'clear' to clear history\n")
    while True:
        query = input("Ask Question: ").strip()
        if query.lower() == "exit":
            break
        elif query.lower() == "clear":
            conversation.clear()
            print("conversation history cleared!\n")
            continue
        elif not query:
            continue

        rag_response(query, top_k=5)
