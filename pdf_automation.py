# Import the load_pdf function
from load_docs import load_pdf 

# Import necessary libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader

# Load HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 
print("HuggingFace embeddings object created:", embeddings)

print(len(load_pdf()))
print("-------------------------next step---------------------------------")

# initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50, chunk_overlap=20, add_start_index=True, separators=[" ", "", "\n\n", "\n", "."]  # Split by spaces or no separator
)
all_splits = text_splitter.split_documents(load_pdf())

print(len(all_splits)) # Print the number of split documents
print(all_splits)  # Print the split document for verification
print("-------------------------next step---------------------------------")

# Initialize the Chroma vector store
# vector_store = Chroma(
#     collection_name="DataScience_collection",
#     embedding_function=embeddings,
#     # persist_directory="./chroma_pdf_db",  # Where to save data locally, remove if not necessary
# )

vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings
)

print("chroma initialized")
print("-------------------------next step---------------------------------")

# Add the split documents to the vector store
# ids = vector_store.from_documents(documents=all_splits)

# embedded_query = embeddings.embed_query("who are the publishers?")

print("after embedding query")

results = vector_store.similarity_search("what are the four class types covering all the basic machine learning functionalities?")
print(results[0].page_content)