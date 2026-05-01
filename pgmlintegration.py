"""
    Connect to PostgreSQL and fetch data
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="dvdrental",
    user="postgres",
    password="1234",
    port="5432"
)

cursor = conn.cursor()

cursor.execute("SELECT * FROM customer")

rows = cursor.fetchall()


texts = []

for row in rows:
    text = f"""
    Customer Details:
    First Name: {row[2]}
    Last Name: {row[3]}
    Customer ID: {row[0]}
    Email: {row[4]}
    """
    texts.append(text.strip())

print(texts[:3])
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts)

print("Embedding shape:", embeddings.shape)
print("First vector (first 5 values):", embeddings[0][:5])


"""
    Step 4: Cosine similarity (learning purpose)
"""
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


print("\n--- Similarity ---")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        score = cosine_similarity(embeddings[i], embeddings[j])
        # print(f"{texts[i]} <--> {texts[j]} = {score:.4f}")


"""
    Step 5: FAISS index
"""
embedding_matrix = embeddings.astype("float32")

dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

print("\nVectors stored:", index.ntotal)


"""
    Step 6: Query
"""
query = "Give details of all the customers who's first name is Mary"

query_embedding = embedding_model.encode([query]).astype("float32")

k = 2

distances, indices = index.search(query_embedding, k)

print("\n--- FAISS Results ---")
for i, dist in zip(indices[0], distances[0]):
    print(f"{texts[i]} → distance: {dist}")


"""
    Step 7: Build context
"""
retrieved_docs = [texts[i] for i in indices[0]]
context = "\n".join(retrieved_docs)

print("\n--- Context ---")
print(context)


"""
    
"""
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")


"""
    Step 9: RAG Prompt
"""
prompt = f"""
You are a strict assistant.

Answer ONLY using the given context.

If the answer is not present in the context, reply:
"I don't know"

Context:
{context}

Question:
Give details of customer Mary?
Answer:
"""


"""
    Step 10: Generate answer
"""
inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=False
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)


print("\n--- Final Answer ---")
print(answer)