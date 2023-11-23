from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from glob import glob


DB_FAISS_PATH = "YOURVECTORDBPATH"

def create_vector_db():
    loader = DirectoryLoader(path= "YOURDOCUMENTPATH", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    embeddings = LlamaCppEmbeddings(model_path='YOUREMBEDDINGSMODELPATH', f16_kv = True)
    vector_store = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory=DB_FAISS_PATH)
    vector_store.persist()
create_vector_db()
