import google.generativeai as genai
import numpy as np
import numpy as np
import faiss

genai.configure(api_key="AIzaSyBKmdG0lR9TfwrwpZmkXdXniJv3Z-YyjM4")

texts = [
    "kafka handles real time data",
    "kafka is a streaming platform",
    "spark processes big data",
    "databricks is analytics platform",
    "pizza is tasty"
]

embeddings=[]
for text in texts:
    response = genai.embed_content(
        model="gemini-embedding-2",
        content=text
)

    emb = response["embedding"]
    embeddings.append(emb)


    print("\nText:", text)
    print("Length:", len(emb))
    print("First 5 values:", emb[:5])


def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("\n--- Similarity Scores ---")

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        score = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{texts[i]}  <-->  {texts[j]}  =  {score}")


embedding_matrix = np.array(embeddings).astype("float32")

print("Shape of embedding matrix:", embedding_matrix.shape)

dimension = embedding_matrix.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embedding_matrix)

print("Total vectors stored:", index.ntotal)

query = "real time streaming systems"

response = genai.embed_content(
    model="gemini-embedding-2",
    content=query
)

query_embedding = np.array([response["embedding"]]).astype("float32")

k = 2  # number of nearest results

distances, indices = index.search(query_embedding, k)

print("\n--- FAISS Search Results ---")

for i, dist in zip(indices[0], distances[0]):
    print(f"{texts[i]} → distance: {dist}")