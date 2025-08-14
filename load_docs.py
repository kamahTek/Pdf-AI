from langchain_community.document_loaders import PyPDFLoader

# Define the path to the PDF file
FILE_PATH = "Data\DataScience.pdf"

# Function to load PDF documents using PyPDFLoader
def load_pdf():
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()

    # return the loaded documents
    return docs

# print(len(load_pdf()))