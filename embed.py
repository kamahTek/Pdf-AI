from sentence_transformers import SentenceTransformer
import time

print("Starting a simple embedding test...")
start_time = time.time()

# Load the model directly using sentence-transformers library
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Embed a simple sentence
embedding = model.encode("This is a test sentence.")

end_time = time.time()
print(f"Test completed in {end_time - start_time:.2f} seconds.")
print(f"Embedding shape: {embedding.shape}")