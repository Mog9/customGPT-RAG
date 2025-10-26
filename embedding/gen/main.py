import json
from transformers import AutoTokenizer, AutoModel
import torch

with open("../splitting_chunks/split_data.json") as f:
    split_chunks = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
model = AutoModel.from_pretrained(
    "Qwen/Qwen2-1.5B", device_map="auto", dtype=torch.float16
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

all_embed = []

for file_name, chunks in split_chunks.items():
    for chunk_id, text in enumerate(chunks):
        input = tokenizer(text, return_tensors="pt")  # tokenization
        input = {k: v.to(device) for k, v in input.items()}

        with torch.no_grad():  # embed
            emb_tensor = model(**input).last_hidden_state.mean(dim=1).squeeze()
            embedding = emb_tensor.cpu().tolist()

        all_embed.append(
            {
                "source": file_name,
                "chunk_id": chunk_id,
                "text": text,
                "embedding": embedding,
            }
        )
# json
with open("embeddings.json", "w") as f:
    json.dump(all_embed, f, indent=2)
