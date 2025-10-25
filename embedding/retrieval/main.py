import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

embed_file = "../gen/embeddings.json"
model_name = "Qwen/Qwen2-1.5B"
top_k = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name, dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
model.eval()

with open(embed_file, "r") as f:
    data = json.load(f)

texts = [d["text"] for d in data]
embeddings = [torch.tensor(d["embedding"], dtype=torch.float32) for d in data]
embeddings = torch.stack(embeddings).to(device)


def retrieve(query: str, top_k: int):
    input = tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(
        device
    )
    with torch.no_grad():
        query_emb = model(**input).last_hidden_state.mean(dim=1)
    query_emb = F.normalize(query_emb, p=2, dim=-1)  # normalize for cosine similarity

    all_embs = F.normalize(embeddings, p=2, dim=-1)
    query_emb = query_emb.float()
    all_embs = all_embs.float()
    similarity = torch.matmul(query_emb, all_embs.T).squeeze(0)  # cosine similarity

    topk_values, topk_indices = torch.topk(similarity, top_k)
    results = []
    for score, idx in zip(topk_values.tolist(), topk_indices.tolist()):
        results.append({"score": score, "text": texts[idx]})
    return results


# test
if __name__ == "__main__":
    query = input("enter your query: ")
    top_chunks = retrieve(query, top_k)
    print("top relevent chunks:\n")

    for i, item in enumerate(top_chunks):
        print(f"{i + 1}. (score={item['score']:.4f})\n{item['text']}\n")
