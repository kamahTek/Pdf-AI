# Import the load_pdf function
from load_docs import load_pdf 

# Import necessary libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader

# Load HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 

# initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(load_pdf())

print(all_splits[:2])  # Print the first split document for verification
print("-------------------------next step---------------------------------")

# Initialize the Chroma vector store
vector_store = Chroma(
    collection_name="DataScience_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_pdf_db",  # Where to save data locally, remove if not necessary
)

print("chroma initialized")
print("-------------------------next step---------------------------------")

try:
    # Add the split documents to the vector store
    ids = vector_store.add_documents(documents=all_splits)
    print("Documents added successfully.")
except Exception as e:
    print(f"An error occurred while adding documents: {e}")

# embedded_query = embeddings.embed_query("who are the publishers?")

print("after embedding query")

results = vector_store.similarity_search("who are the publishers?")
print(results[0])