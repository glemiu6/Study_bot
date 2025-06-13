import fitz
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docx import Document
from langchain_ollama import OllamaLLM
import uuid
import hashlib
from RAG_API import RAG

class RAGProcessor(RAG):
    def __init__(self,collection_name="my_collection",my_db="my_db",embedding_model="all-mpnet-base-v2",llm_model="llama3.2"):
        self.collection_name=collection_name
        self.client=chromadb.PersistentClient(path=my_db)
        self.model=SentenceTransformer(embedding_model)
        self.llm=OllamaLLM(model=llm_model)
        self.collection=self._get_collection_create()

    def _get_collection_create(self):
        try:
            return self.client.get_collection(self.collection_name)
        except:
            return self.client.create_collection(self.collection_name)

    def _hash_text(self,text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _read_pdf(self,file):
        text=''
        document=fitz.open(file)
        for page in document:
            text+=page.get_text()+'\n'
        return text

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

    def _chunk_text(self,text,size=600,overlap=150):
        splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=size,
            chunk_overlap=overlap
        )
        return splitter.split_text(text)

    def transformer(self,chunks):
        return self.model.encode(chunks).tolist()

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

    with open(f"respons.txt", "w") as f:
        for turn in chat_history:
            f.write(f"Q: {turn['User']}\nA: {turn['Bot']}\n\n")

    print("All done")