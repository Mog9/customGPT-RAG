from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json

#split my data and store in json to embed later
folder = "./data"
texts_data = {}

for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        with open(os.path.join(folder, filename), "r") as f:
            texts_data[filename] = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
splits = {}

for name, content in texts_data.items():
    splits[name] = [doc.page_content for doc in text_splitter.create_documents([content])]

#json
output_path = os.path.join(os.path.dirname(__file__), "split_data.json")
with open (output_path, "w") as f:
    json.dump(splits, f, indent=2)