

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
client = Client(api_key="AIzaSyAFvMwIJyPOkqSu3f_7L6Rk8eaiPiPbV4w")
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

    memory = ""

    for chat in history[-3:]:
        memory += f"""
        User: {chat['user']}
        Assistant: {chat['assistant']}
        """

    # ---------------------------
    # Single Unified Prompt
    # ---------------------------
    query_lower = query.lower()

# Dynamic retrieval size
    k = 3 if "hire" in query_lower else 8

    _, indices = index.search(query_embedding, k)

    retrieved = [chunks[i] for i in indices[0]]

    if chunks[0] not in retrieved:
        retrieved.append(chunks[0])

    context = "\n\n".join(retrieved)

# ---------------------------
# Dynamic Prompt Building
# ---------------------------

    if "score" in query_lower or "rate" in query_lower:

        prompt = f"""
        You are a technical hiring manager.

        Evaluate the candidate using ONLY the context.

        Context:
        {context}

        Job Description:
        {jd if jd else "Not provided"}

        Give:
        - Score out of 10
        - JD Match Percentage
        - Strengths
        - Weaknesses
        - Recommendation
        """

    elif "hire" in query_lower or "suitable" in query_lower:

        prompt = f"""
        You are a hiring manager.

        Based ONLY on the resume context,
        tell whether this candidate should be hired.

        Keep the answer concise and practical.

        Context:
        {context}

        Job Description:
        {jd if jd else "Not provided"}

        Question:
        {query}
        """

    elif "skill" in query_lower:

        prompt = f"""
        Extract only the technical skills from the resume.

        Context:
        {context}
        """

    elif "experience" in query_lower:

        prompt = f"""
        Summarize the candidate's experience clearly.

        Context:
        {context}
        """

    elif "extract" in query_lower or "full" in query_lower:

        prompt = f"""
        Extract:
        - Name
        - Role
        - Skills
        - Experience

        Context:
        {context}
        """

    else:

        prompt = f"""
        You are a helpful AI resume assistant.

        Use ONLY the provided context.

        Previous Conversation:
        {memory}

        Context:
        {context}

        Question:
        {query}

        Answer naturally and concisely.
        """

    msg = cl.Message(content="")

    full_answer = ""

    stream = client.models.generate_content_stream(
        model=MODEL,
        contents=prompt,
        config={
            "temperature": 0.3
        }
    )

    async for chunk in stream:

        if chunk.text:
            full_answer += chunk.text
            await msg.stream_token(chunk.text)

    # Finalize message
    await msg.send()

    # ---------------------------
    # Save memory
    # ---------------------------
    history.append({
        "user": query,
        "assistant": full_answer
    })

    history = history[-5:]

    cl.user_session.set("chat_history", history)