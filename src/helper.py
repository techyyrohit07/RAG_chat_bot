from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings



# extracting data from pdf

def load_pdf(data):
    loader = DirectoryLoader(
        data,
        loader_cls=PyPDFLoader,
        glob="*.pdf"
    )

    document = loader.load()
    return document


# remove unnecessary metadata 
def filter_to_minimal_docs(docs : List[Document]) -> List[Document]:
    minimal_docs : List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata= {"source" : src}
            )
        )

    return minimal_docs


# chunking
def text_split(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )

    text_chunks = text_splitter.split_documents(docs)
    return text_chunks


def download_embeddings():  
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )

    return embeddings

