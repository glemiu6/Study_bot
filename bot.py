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




def query_vector_db(collection,query,top_k=3):
    query_embeddings = model.encode(query).tolist()
    result=collection.query(query_embeddings=[query_embeddings],n_results=top_k)
    relevant_text=result['documents'][0]
    context="\n".join(relevant_text)
    prompt = f"Respond in the language the file is with the following question using this context:\n{context}\n Question: {query}"
    return llm.invoke(prompt)

def process(texts):
    chunks=chunking(texts)
    chunk_transformed=transformer(chunks)
    return vector_db(chunks, chunk_transformed)

if __name__ == "__main__":
    file_path = "Laboratory 6.1-6.2.pdf"
    text = read_files(file_path)
    collection = process(text)

    Question = ("Whats the file main topic?")
    ans = query_vector_db(collection, Question)

    with open(f"responss.txt", "w") as f:
        f.write(ans+'\n')
        print("All done")