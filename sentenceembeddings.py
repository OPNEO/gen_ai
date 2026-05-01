from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
model_name='all-MiniLM-L6-v2'
model=SentenceTransformer('all-MiniLM-L6-v2')
docs = [
    "Kafka is a distributed streaming system",
    "FAISS performs similarity search",
    "Python is used in data engineering",
    "Spark handles big data processing"
]

embeddings=model.encode(docs)
print('Embeddings Shape',embeddings.shape)
print(embeddings[0][:5])
embeddings=np.array(embeddings).astype('float32')

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)

index.add(embeddings)

query = "What is FAISS used for?"
q_vec = model.encode([query]).astype('float32')
k = 2
distances, indices = index.search(q_vec, k)

# Step 9: Output results
print("Query:", query)
print("\nTop matches:\n")

for i in indices[0]:
    print(docs[i])