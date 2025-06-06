import fitz
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docx import Document
from langchain_ollama import OllamaLLM

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



def vector_db(chunks,chunk_transformed):
    try:
        collection = client.get_collection('my_collection')
        print("got collection")

    except Exception as e:
        collection = client.create_collection('my_collection')
        print("created collection")


    collection.add(
        documents=chunks,
        embeddings=chunk_transformed,
        ids=[str(i) for i in range(len(chunks))]
        )

    return collection




def query_vector_db(collection,query,top_k=3):
    query_embeddings = model.encode(query).tolist()
    result=collection.query(query_embeddings=[query_embeddings],n_results=top_k)
    relevant_text=result['documents'][0]
    context="\n".join(relevant_text)
    prompt = f"Respond to the following question using this context:\n{context}\nQuestion: {query}"
    return llm.invoke(prompt)


if __name__ == "__main__":
    file_path = "07_Pares_AleatoÌrios (3).pdf"
    text = read_files(file_path)
    chunks = chunking(text)
    embeddings = transformer(chunks)
    collection = vector_db(chunks, embeddings)

    Question = ("What is Pares Aleatorias?")
    ans = query_vector_db(collection, Question)

    with open(f"responss.txt", "w") as f:
        f.write(ans)
        print("All done")