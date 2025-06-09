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
- The function that reads the files (`read_files`) where we pass the auxiliary functions like `read_docx`,`read_md` and `read_pdf`.
- Separating the content in smaller parts (_chunking_) so it would be easier for the machine to process.
- Transforming the content after separating (_embedding_) in a vector .
- Make a Vector Database where we put the vectors after passing them through the transformer.
- Give the output in a file to be easier to read

First import the libraries :
```python
import fitz
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docx import Document
from langchain_ollama import OllamaLLM
import uuid
import hashlib
```
Then we declare the model to use in encoding , the llm and the database:
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
llm = OllamaLLM(model="llama3.2")

client = chromadb.PersistentClient(path='my_db')
```
Then we make the auxiliary function to help read the files that end in different patterns like: .docx, .md, etc
```python
def read_docx(file):
    text=''
    document=Document(file)
    for page in document.paragraphs:
        text+=page.text+'\n'
    return text

def read_md(file):
    text=''
    with open(file,'r',encoding='utf-8') as f:
        text+=f.read()
    return text


def read_pdf(file):
    text=''
    document=fitz.open(file)
    for page in document:
        text+=page.get_text()+'\n'
    return text
```

Then implement all of this functions in a function that reads the file :
```python
def read_files(file):

    if file.endswith('.docx'):
        text=read_docx(file)
    elif file.endswith('.pdf'):
        text=read_pdf(file)
    else:
        text=read_md(file)
    return text
```

Create a function `chunking(file)` where we will take the file read and stored in a variable and separate it in different parts.  
This method is called chunking. Use the function from [langchain.text_splitter](https://python.langchain.com/docs/concepts/text_splitters/)  
Where we will pass 3 variables:
- `encoding_name="cl100k_base"`-> is the schema to tokenize and checks if the distribution is how it's see the text
- `chunk_size=500`, -> is the size of the chunk
- `chunk_overlap=100`-> it's the overlapping of chunks so the response will be smoother

```python
def chunking(text):
    text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base",chunk_size=500,chunk_overlap=100)
    texts=text_splitter.split_text(text)
    return texts
```

Then the function `transformer(chunk)` take thos chunks from the function`chunking` and makes them in vectors.
This process is called _embedding_. Use the model declared above and encode from letters to numbers.

```python
def transformer(chunk):

    embeddings = model.encode(chunk).tolist()
    return embeddings
```

Create a Vector Database using the [chromadb](https://pypi.org/project/chromadb/), a simple-to-use database.  
With this database we store the vectors from the transformer in the db and the chunk corresponding to the vector.

```python
def hash_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def vector_db(chunks, chunk_transformed):
    try:
        collection = client.get_collection('my_collection')
        print("got collection")
    except:
        collection = client.create_collection('my_collection')
        print("created collection")

    existing = set()
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    for m in metadatas:
        if m and "hash" in m:
            existing.add(m["hash"])

    ids=[]
    new_chunks=[]
    new_chunk_transformed=[]
    metadatas=[]

    for chunk, emb in zip(chunks, chunk_transformed):
        hash=hash_text(chunk)
        if hash in existing:
            continue
        ids.append(str(uuid.uuid4()))
        new_chunks.append(chunk)
        new_chunk_transformed.append(emb)
        metadatas.append({"hash":hash})


    collection.add(
        documents=new_chunks,
        embeddings=new_chunk_transformed,
        ids=ids,
        metadatas=metadatas
    )

    return collection
```
After the vector database is made, is needed to transform the question/query like we did with the files.  
With this function we put the question in the DB and we pull the most similar and correct response for the question by giving the llm a prompt to answer the question using the context.
```python
def query_vector_db(collection,query,chat_history=None,top_k=3):
    query_embeddings = model.encode(query).tolist()
    result=collection.query(query_embeddings=[query_embeddings],n_results=top_k)
    relevant_text=result['documents'][0]
    context="\n".join(relevant_text)
    
    memory=""
    if chat_history:
        for turn in chat_history:
            user_text=turn.get("User","")
            bot_text=turn.get("Bot","")
            memory+=f"User:{user_text}\n Bot:{bot_text}"
    prompt =(
        f"You are answering questions about a file. Use the context below and previous turns.\n"
        f"{memory}\n"
        f"Context:\n{context}\n"
        f"Question: {query}")
    
    return llm.invoke(prompt)
```

Also added a function where the functions are called so the `main`will look cleaner
```python
def process(texts):
    chunks=chunking(texts)
    chunk_transformed=transformer(chunks)
    return vector_db(chunks, chunk_transformed)
```

To make it all work, combine them in the main function.  
Make a empty list `ch` and with the new upgrade , we can add multiple questions about the file and save them in the memory for improvement.
```python
ch=[]
    file_path = "bioinformatics_28_7_991.pdf"
    text = read_files(file_path)
    collection = process(text)

    Question = ("Whats the main idea from the file?",
                "Give me some exemples from the most important part")
    for q in Question:
        ans = query_vector_db(collection, q,chat_history=ch)
        ch.append({"User":q,
                   "Bot":ans}
                  )

    
```
And to make the reading easier, make an output file.
```python
    with open(f"responss.txt", "w") as f:
        for turn in ch:
            f.write(f"Q: {turn['User']}\nA: {turn['Bot']}\n\n")
        print("All done")
```