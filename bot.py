import fitz
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docx import Document
from langchain_ollama import OllamaLLM
import uuid
import hashlib

model = SentenceTransformer('all-MiniLM-L6-v2')
llm = OllamaLLM(model="llama3.2")

client = chromadb.PersistentClient(path='my_db')

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

def read_files(file):

    if file.endswith('.docx'):
        text=read_docx(file)
    elif file.endswith('.pdf'):
        text=read_pdf(file)
    else:
        text=read_md(file)
    return text



def chunking(text):
    text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base",chunk_size=500,chunk_overlap=100)
    texts=text_splitter.split_text(text)
    return texts



def transformer(chunk):

    embeddings = model.encode(chunk).tolist()
    return embeddings


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
        h=hash_text(chunk)
        if h in existing:
            continue
        ids.append(str(uuid.uuid4()))
        new_chunks.append(chunk)
        new_chunk_transformed.append(emb)
        metadatas.append({"hash":h})

    if new_chunks:
        collection.add(
            documents=new_chunks,
            embeddings=new_chunk_transformed,
            ids=ids,
            metadatas=metadatas
        )

    return collection




def query_vector_db(collection,query,chat_history=None,top_k=3):
    query_embeddings = model.encode(query).tolist()
    result=collection.query(query_embeddings=[query_embeddings],n_results=top_k)
    relevant_text=result['documents'][0]
    context="\n".join(relevant_text)

    memory=''
    if chat_history:
        for turn in chat_history:
            user_text=turn.get("user","")
            bot_text=turn.get("bot","")
            memory+=f"User:{user_text}\n Bot:{bot_text}\n"
    prompt =(
        f"You are answering questions about a file. Use the context below and previous turns.\n"
        f"{memory}\n"
        f"Context:\n{context}\n"
        f"Question: {query}")
    return llm.invoke(prompt)

def process(texts):
    chunks=chunking(texts)
    chunk_transformed=transformer(chunks)
    return vector_db(chunks, chunk_transformed)

if __name__ == "__main__":
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

    with open(f"responss.txt", "w") as f:
        for turn in ch:
            f.write(f"Q: {turn['User']}\nA: {turn['Bot']}\n\n")
        print("All done")