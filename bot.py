import fitz
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docx import Document
from langchain_ollama import OllamaLLM

model = SentenceTransformer('all-MiniLM-L6-v2')
llm = OllamaLLM(model="llama3.2")
client=chromadb.Client()


def get_pdf(file):
    text = ""
    if file.endswith(".pdf"):
        document = fitz.open(file)

        for page in document:
            text+=page.get_text()
    elif file.endswith(".DOCX"):
        document = Document(file)
        for page in document.paragraphs:
            text+=page.text+'\n'
    return text


def chunking(text):
    text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base",chunk_size=500,chunk_overlap=50)
    texts=text_splitter.split_text(text)
    return texts



def transformer(chunk):

    embeddings = model.encode(chunk).tolist()
    return embeddings



def vector_db(chunks,chunk_transformed):
    try:
        collection=client.get_collection('my_collection')
    except Exception as e:
        collection=client.create_collection('my_collection')


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
    prompt = f"Răspunde la următoarea întrebare folosind contextul:\n{context}\nÎntrebare: {query}"
    return llm.invoke(prompt)


if __name__ == "__main__":
    file_path = "T04_Camada_Rede.pdf"
    text = get_pdf(file_path)
    chunks = chunking(text)
    embeddings = transformer(chunks)
    collection = vector_db(chunks, embeddings)

    intrebare = ("Spunemi ce este in acest file?")
    raspuns = query_vector_db(collection, intrebare)
    print("Răspuns:", raspuns)