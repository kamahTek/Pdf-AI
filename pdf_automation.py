# Import the load_pdf function
from load_docs import load_pdf 

# Import necessary libraries
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 
print("HuggingFace embeddings object created:", embeddings)

print(len(load_pdf()))
print("-------------------------next step---------------------------------")

# initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=30, add_start_index=True, separators=[" ", "", "\n\n", "\n", "."]  # Split by spaces or no separator
)
all_splits = text_splitter.split_documents(load_pdf())

print(len(all_splits)) # Print the number of split documents
print(all_splits)  # Print the split document for verification
print("-------------------------next step---------------------------------")

# Initialize the Chroma vector store
# print("chromadb")
# vector_store = Chroma(
#     collection_name="DataScience_collection",
#     embedding_function=embeddings,
#     persist_directory="./chroma_pdf_db",  # Where to save data locally, remove if not necessary
# )
# print("chroma initialized")

# Initialize the Chroma vector store
print("InMemoryVextorStore")
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

print("InMemoryVectorStore initialized")
print("-------------------------next step---------------------------------")

# Add the split documents to the vector store
print("before adding documents to vector store" )
# vector_store.from_documents(documents=all_splits, embedding=embeddings)
ids = vector_store.add_documents(documents=all_splits)

print("after adding documents to vector store" )
print("-------------------------next step---------------------------------")

# User Query
print("user query to search for similarity")

results = vector_store.similarity_search("what are the base classes scikit_learn features?")
print(results[0])