"""
    FINAL: RAG + Gemini (Single Call + JD Matching + Memory)
"""

import chainlit as cl
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re
import os
from google.genai import Client

# ---------------------------
# Setup
# ---------------------------
client = Client(api_key="")
MODEL = "gemini-3-flash-preview"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Helpers
# ---------------------------
def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=300):
    sentences = text.split(".")
    chunks, current = [], ""

    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "
    if current:
        chunks.append(current.strip())

    return chunks


# ---------------------------
# Upload PDF
# ---------------------------
@cl.on_chat_start
async def start():

    files = await cl.AskFileMessage(
        content="Upload a resume PDF",
        accept=["application/pdf"],
        max_size_mb=10
    ).send()

    if not files:
        await cl.Message(content="Upload required").send()
        return

    file = files[0]

    pdf_text = clean_text(load_pdf(file.path))
    chunks = chunk_text(pdf_text)

    embeddings = embedding_model.encode(chunks)
    embedding_matrix = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embedding_matrix)
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    cl.user_session.set("chunks", chunks)
    cl.user_session.set("index", index)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("job_description", None)

    await cl.Message(
        content="PDF loaded. You can also paste a Job Description."
    ).send()


# ---------------------------
# Main Chat
# ---------------------------
@cl.on_message
async def main(message: cl.Message):

    query = message.content

    chunks = cl.user_session.get("chunks")
    index = cl.user_session.get("index")
    history = cl.user_session.get("chat_history")
    jd = cl.user_session.get("job_description")

    if chunks is None:
        await cl.Message(content="Upload PDF first").send()
        return

    # ---------------------------
    # Detect JD input
    # ---------------------------
    if "job description:" in query.lower():
        jd = query.replace("job description:", "").strip()
        cl.user_session.set("job_description", jd)

        await cl.Message(content="Job description saved").send()
        return

    # ---------------------------
    # Retrieval
    # ---------------------------
    query_embedding = embedding_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)

    _, indices = index.search(query_embedding, 8)

    retrieved = [chunks[i] for i in indices[0]]

    if chunks[0] not in retrieved:
        retrieved.append(chunks[0])

    context = "\n\n".join(retrieved)

    # ---------------------------
    # Memory (last 3 messages)
    # ---------------------------
    memory = "\n".join(history[-3:])

    # ---------------------------
    # Single Unified Prompt
    # ---------------------------
    prompt = f"""
You are an AI resume assistant.

Use ONLY the provided context.

--- CONTEXT ---
{context}

--- CHAT HISTORY ---
{memory}

--- JOB DESCRIPTION ---
{jd if jd else "Not provided"}

--- USER QUESTION ---
{query}

--- TASK ---
1. Answer the question
2. Extract relevant details if needed
3. If JD is provided → match candidate with JD
4. Provide score (out of 10)
5. Give strengths, weaknesses, recommendation

--- OUTPUT FORMAT ---
Answer:
...

Score:
...

JD Match:
...

Strengths:
...

Weaknesses:
...

Recommendation:
...
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    answer = response.text

    # ---------------------------
    # Save memory
    # ---------------------------
    history.append(f"User: {query}")
    history.append(f"Assistant: {answer}")
    cl.user_session.set("chat_history", history)

  
    await cl.Message(content=answer).send()