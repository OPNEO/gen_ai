"""
    PDF RAG SYSTEM (FINAL STABLE VERSION)
"""

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re
import os

# NEW Gemini SDK
from google.genai import Client

# -------------------------------
# STEP 1: Setup Gemini
# -------------------------------
client = Client(api_key="AIzaSyAMfbadIrd-ZnviynZ05jShUCM4VJMtOCY")

# -------------------------------
# STEP 2: Clean Text
# -------------------------------
def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------------
# STEP 3: Load PDF
# -------------------------------
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text

pdf_text = clean_text(load_pdf("resume.pdf"))

# -------------------------------
# STEP 4: Chunking
# -------------------------------
def chunk_text(text, chunk_size=300):
    sentences = text.split(".")
    
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

chunks = chunk_text(pdf_text)

# -------------------------------
# STEP 5: Embeddings
# -------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedding_model.encode(chunks)
embedding_matrix = np.array(embeddings).astype("float32")

print("Embedding shape:", embedding_matrix.shape)

# -------------------------------
# STEP 6: FAISS (Cosine Similarity)
# -------------------------------
faiss.normalize_L2(embedding_matrix)

dimension = embedding_matrix.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embedding_matrix)

print("Vectors stored:", index.ntotal)

# -------------------------------
# STEP 7: Query
# -------------------------------
query = "Extract name, role, skills and experience from this resume"
query_embedding = embedding_model.encode([query]).astype("float32")
faiss.normalize_L2(query_embedding)

k = 10

distances, indices = index.search(query_embedding, k)

retrieved_chunks = [chunks[i] for i in indices[0]]

context = "\n\n".join(retrieved_chunks)

print("\n--- Retrieved Context ---")
print(context)

# -------------------------------
# STEP 8: Gemini LLM
# -------------------------------
prompt = f"""
You are a resume analysis assistant.

Strictly use ONLY the provided context.

Extract the following in structured format:

Name:
Role:
Skills:
Experience:

Rules:
- Combine information across lines
- Do NOT hallucinate
- The name is usually at the top of the resume
- If missing, write "Not found"

Context:
{context}

Question:
{query}

Answer:
"""

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt
)

print("\n--- Final Answer ---")
print(response.text)

"""
    STEP 9: Candidate Scoring
"""

scoring_prompt = f"""
You are a senior hiring manager.

Based ONLY on the context below, evaluate the candidate.

Give:
1. Score out of 10
2. Strengths
3. Weaknesses
4. Final recommendation

Context:
{context}

Answer format:

Score: X/10
Strengths:
- ...
Weaknesses:
- ...
Recommendation:
...
"""

score_response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=scoring_prompt
)

print("\n--- Candidate Evaluation ---")
print(score_response.text)