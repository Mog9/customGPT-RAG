# CustomGPT-RAG
#### This is a custom RAG (Retrieval-Augmented Generation) chatbot that combines local document retrieval with generative AI to provide accurate, context-aware answers. It can answer private information by using your own text files: simply add your files, embed them, and feed them to the model to get answers from that data. The system is flexible and easily extensible, allowing you to continuously update or expand your knowledge sources.


## Usage
## Step 1: Clone the repository
##### Clone the GitHub repository to your local machine:
```bash
git clone https://github.com/Mog9/customGPT-RAG.git
cd customGPT-RAG
```
## Step 2: Add your data (optional)
##### You can provide your own private knowledge by adding `.txt` files. Place all your text files inside the `data/` folder. These files will be used by the RAG system to generate context-aware answers.
<img width="348" height="193" alt="image" src="https://github.com/user-attachments/assets/a85d441b-d5f4-4240-912d-8fa3a0a4591f" />


## Step 3: Split your data into chunks
##### Once you have added your custom `.txt` files to the `data/` folder, run the `split.py` script located in the `embedding/` folder:
```bash
python embedding/splitting_chunks/split.py
```
##### Splitting the data into chunks helps the system retrieve relevant pieces of information more efficiently, especially for large documents. In this setup, each chunk is set to 100 characters, with a chunk overlap of 20 characters. The overlap ensures that important context at the boundaries of chunks is preserved, preventing loss of meaning and improving the accuracy of embeddings and later retrieval. The splitted data will be stored in `split_data.json` for use in the embedding and retrieval steps.
<img width="1134" height="825" alt="image" src="https://github.com/user-attachments/assets/1f08d8d9-fb1d-4ae7-9736-4f5aed52f0e9" />


## Step 4: Generate embeddings for your data
##### After splitting your data into chunks, the next step is to convert these chunks into embeddings, which will be used by the model to retrieve relevant information efficiently. To do this, run the `main.py` script located in the `embedding/` folder:
```bash
python embedding/gen/main.py
```
#### What embeddings do:
##### Embeddings are numerical representations of your text chunks in a high-dimensional vector space. Each chunk is converted into a vector that captures its semantic meaning. This allows the RAG system to compare your query with all chunks and find the most relevant pieces of information during the retrieval step. By using embeddings, the chatbot can match your question to the content of your documents even if the wording is slightly different, ensuring accurate, context-aware answers. The generated embeddings will be stored in `embeddings.json` for use in the retrieval and chatbot steps.
<img width="838" height="872" alt="image" src="https://github.com/user-attachments/assets/002e0ab1-965a-4e93-a5d7-5d0a8a958365" />

## Step 5: Retrieve relevant information
##### Once your embeddings are ready, the retrieval function allows the chatbot to find the most relevant chunks from your data when a question is asked. During retrieval, the system compares the embedding of your query with all stored embeddings and selects the chunks that are most semantically similar. This ensures that the model answers based only on the context of your provided documents, improving accuracy and preventing hallucinations.
```bash
python retrieval_func/retrieval.py
```
<img width="963" height="306" alt="image" src="https://github.com/user-attachments/assets/ddb159be-051f-4750-baa5-38d12247e370" />

## Step 6: Run the RAG pipeline
##### The RAG (Retrieval-Augmented Generation) pipeline combines retrieval and generative AI to answer questions based on your private data. When a query is entered, the system first retrieves the most relevant chunks from your embedded documents and then feeds them as context to the language model. The model generates answers that are grounded in the retrieved context, ensuring accuracy and relevance.
###### To run the RAG pipeline and start interacting with the chatbot, execute:
```bash
python pipeline/rag.py
```
##### This will load the model, use your embedded data for context-aware retrieval, and allow you to ask questions interactively. The system also maintains a short conversation history to provide coherent responses while ensuring that answers are only based on the provided documents.
<img width="1573" height="192" alt="image" src="https://github.com/user-attachments/assets/c6fcd4f6-98a6-4b35-8345-7061f8c92d60" />

## Extra Info
##### - The system maintains a context memory of the last 3 questions, allowing the chatbot to provide coherent answers across a short conversation.

##### - Includes math handling functions for improved accuracy and formatting of mathematical queries.

##### - Provides a better result layout, making answers more readable and organized.

##### - The model automatically stops generating after completing a sentence if fewer tokens remain, preventing it from continuing mid-thought.

##### - Contains private project data, such as the Pluto ML project, allowing the model to give answers based on your own knowledge sources.

##### - Do not expect ChatGPT-level answers; the model used here has 1.5B parameters, so outputs may be shorter or less fluent. multiple minor improvements that enhance usability, stability, and overall experience, making it a complete RAG system.
