# Study bot
### Author: Vlad Digori

---

## Content:
1. [About the project](#about-the-project)
2. [Setting up the environment](#setting-up-the-environment)
3. [Code](#Code)

---

## About the project

The main objective of this project is to have a summary generator on your machine.  
For this project was used : 
1. python, 
2. `jupyter-notebook` to make tests of the model 
3. `LLM (llama3.2)` 

---

## Setting up the environment

## Setting up the environment
Install [Python](https://www.python.org/downloads/) IDLE.  

Now we create a virtual environment so we can run the code.  

### 1. Creating the environment:
```bash
python -m venv venv
```
### 2. Activating the environment:
#### For Mac/Linux:

```bash
source venv/bin/activate
```
#### For Windows:
```bash
.\venv\Scripts\activate
```
### 3. Installing all the necessary libraries to run the project:
```bash
pip install -r requirements.txt
```

### 4. Installing the model on your machine(llama3.2):
#### For MAC with Homebrew
```bash
brew install ollama
```
#### For Windows:
https://ollama.com/download

#### For Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 5. Run the model
```bash
ollama run llama3.2
```

### Run the code 


```bash
python bot.py
```

## Code
The main functionality of this bot is to generate summaries from your files in form of a txt file.  
The structure if my code is this:
- An API structure for easier understanding and cleaner code
- The function that reads the files (`read_files`) where we pass the auxiliary functions like `read_docx`,`read_md` and `read_pdf`.
- Separating the content in smaller parts (_chunking_) so it would be easier for the machine to process.
- Transforming the content after separating (_embedding_) in a vector .
- Make a Vector Database where we put the vectors after passing them through the transformer.
- Give the output in a file to be easier to read

First import the libraries and the API:
```python
import fitz
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docx import Document
from langchain_ollama import OllamaLLM
import uuid
import hashlib
from RAG_API import RAG

```
Create a Class `RAGProcessor` using the [API RAG](https://github.com/glemiu6/Study_bot/blob/master/RAG_API.py). Then we inicialize the collection name , the database m the embedding model and the LLM model:
```python
    def __init__(self,collection_name="my_collection",my_db="my_db",embedding_model="all-mpnet-base-v2",llm_model="llama3.2"):
        self.collection_name=collection_name
        self.client=chromadb.PersistentClient(path=my_db)
        self.model=SentenceTransformer(embedding_model)
        self.llm=OllamaLLM(model=llm_model)
        self.collection=self._get_collection_create()
```
Then we make the auxiliary function to help read the files that end in different patterns like: .docx, .md, etc
```python
    def _read_docx(self,file):
        text=''
        document=Document(file)
        for page in document.paragraphs:
            text+=page.text+'\n'
        return text

    def _read_md(self,file):
        text=''
        with open(file,'r',encoding='utf-8') as f:
            text+=f.read()
        return text


    def _read_pdf(self,file):
        text=''
        document=fitz.open(file)
        for page in document:
            text+=page.get_text()+'\n'
        return text
```

Then implement all of this functions in a function that reads the file :
```python
    def read_files(self,file):
        ext=file.split(".")[1].lower()
        try:
            if ext=="pdf":
                return self._read_pdf(file)
            elif ext=="docx":
                return self._read_docx(file)
            elif ext in ['md','txt']:
                return self._read_md(file)
        except FileNotFoundError as e:
            print(e)
            return ""
```

Create a function `chunking(file)` where we will take the file read and stored in a variable and separate it in different parts.  
This method is called chunking. Use the function from [langchain.text_splitter](https://python.langchain.com/docs/concepts/text_splitters/)  
Where we will pass 3 variables:
- `encoding_name="cl100k_base"`-> is the schema to tokenize and checks if the distribution is how it's see the text
- `chunk_size=600`, -> is the size of the chunk
- `chunk_overlap=150`-> it's the overlapping of chunks so the response will be smoother

```python
    def _chunk_text(self,text,size=600,overlap=150):
        splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=size,
            chunk_overlap=overlap
        )
        return splitter.split_text(text)
```

Then the function `transformer(chunk)` take thos chunks from the function`chunking` and makes them in vectors.
This process is called _embedding_. Use the model declared above and encode from letters to numbers.

```python
    def transformer(self,chunks):
        return self.model.encode(chunks).tolist()
```

Create a Vector Database using the [chromadb](https://pypi.org/project/chromadb/), a simple-to-use database.  
With this database we store the vectors from the transformer in the db and the chunk corresponding to the vector.

```python
    def _hash_text(self,text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()


    def vector_db(self,chunks,transform,file):
        existing = set()
        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        for m in metadatas:
            if m and "hash" in m:
                existing.add(m["hash"])

        ids=[]
        new_chunks=[]
        new_chunk_transformed=[]
        metadatas=[]

        for chunk, emb in zip(chunks, transform):
            h=self._hash_text(chunk)
            if h in existing:
                continue
            ids.append(str(uuid.uuid4()))
            new_chunks.append(chunk)
            new_chunk_transformed.append(emb)
            metadatas.append({"hash":h,
                              "file":file})

        if new_chunks:
            self.collection.add(
                documents=new_chunks,
                embeddings=new_chunk_transformed,
                ids=ids,
                metadatas=metadatas
            )
```
To get a better response and more accurate , we will reform the question.
After the vector database is made, is needed to transform the question/query like we did with the files.  
With this function we put the question in the DB and we pull the most similar and correct response for the question by giving the llm a prompt to answer the question using the context.
```python
    def query(self,file,question,top_k=5,chat_history=None):
        reform_question=self.llm.invoke(f"Reformat this question to be easier to search{questions}")
        query_embeddings = self.model.encode(reform_question).tolist()
        result=self.collection.query(query_embeddings=[query_embeddings],n_results=top_k,where={"file":file})
        relevant_text=result['documents'][0]
        context="\n".join(relevant_text)

        memory=''
        if chat_history:
            for turn in chat_history:
                user_text=turn.get("User","")
                bot_text=turn.get("Bot","")
                memory+=f"User:{user_text}\n Bot:{bot_text}\n"
        prompt =(
            f"You are answering questions about a file. Use the context below and previous turns.\n"
            f"{memory}\n"
            f"Context:\n{context}\n"
            f"Question: {question}")
        return self.llm.invoke(prompt)
```

Also added a function where the functions are called so the `main`will look cleaner
```python
    def process(self,file):
        text=self.read_files(file)
        print("Reading file...")
        chunks=self._chunk_text(text)
        print("Chunking the text...")
        emb=self.transformer(chunks)
        print("Embedding...")
        self.vector_db(chunks,emb,file)
        print("Storing in DB")
        return self.collection
```

To make it all work, combine them in the main function.  
Make a empty list `ch` and with the new upgrade , we can add multiple questions about the file and save them in the memory for improvement.

```python
if __name__ == "__main__":
    rag = RAGProcessor()
    file_path = "files/bioinformatics_28_7_991.pdf"
    rag.process(file_path)

    chat_history = []
    questions = [
        "How is this article structured?",
        "What is bioinformatics?"
    ]

    for q in questions:
        answer = rag.query(file_path,q, chat_history=chat_history)
        chat_history.append({"User": q, "Bot": answer})
        print(f"Q: {q}\nA: {answer}\n")


```
And to make the reading easier, make an output file.
```python
    with open(f"respons.txt", "w") as f:
        for turn in chat_history:
            f.write(f"Q: {turn['User']}\nA: {turn['Bot']}\n\n")

    print("All done")
```