from langchain_community.document_loaders import PyPDFLoader

# Define the path to the PDF file
FILE_PATH = "Docs_files\DeepLearning.pdf"

# Function to load PDF documents using PyPDFLoader
def load_pdf():
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()
    return docs
