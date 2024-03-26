from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os

class DocumentLoader:
    def save_file(self, file):
        # Create a temporary folder
        folder = "tmp"
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Save the file to the temporary folder
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path

    def load_file(self, uploaded_file):
        file_path = self.save_file(uploaded_file) # Save in temporary folder
        file_type = uploaded_file.type # Determine file type
        try:
            if 'text/plain' in file_type:
                loader = TextLoader(file_path)
                data = loader.load()
            elif 'application/pdf' in file_type:
                loader = PyPDFLoader(file_path)
                data = loader.load_and_split()
            else:
                raise ValueError("Unsupported file type")
        finally:
            os.remove(file_path)
        return data

    def split_document(self, doc):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(doc)
        return splits

    def embed_store_document(self, splits):
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(splits, embedding_function, persist_directory="./chroma_db")
        #return db