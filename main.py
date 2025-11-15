from fastapi import FastAPI, Query
import httpx
import unicodedata
from transformers import pipeline

app = FastAPI()

# ---------------------------------------------------------
# Normalize text
# ---------------------------------------------------------
def norm(x):
    return unicodedata.normalize("NFKD", x).replace("–", "-").replace("—", "-").strip()

# ---------------------------------------------------------
# Load messages
# ---------------------------------------------------------
MESSAGES_URL = "https://november7-730026606190.europe-west1.run.app/messages"

def load_messages():
    headers = {"X-API-Key": "november7"}  # Required API key
    r = httpx.get(MESSAGES_URL, headers=headers, follow_redirects=True)
    r.raise_for_status()

    data = r.json()
    items = data.get("items", [])

    for m in items:
        m["user_name"] = norm(m["user_name"])
        m["message"] = norm(m["message"])

    print("Loaded:", len(items), "messages")
    return items

MESSAGES = load_messages()
ALL_USERS = list({m["user_name"] for m in MESSAGES})

# ---------------------------------------------------------
# Load lightweight LLM for deployment
# IMPORTANT: LLaMA 8B will NOT run on Render free tier
# ---------------------------------------------------------
qa_model = pipeline(
    "text-generation",
    model="distilgpt2",       # <--- SMALL MODEL THAT WORKS ON RENDER
)

# ---------------------------------------------------------
# Extract person intelligently
# ---------------------------------------------------------
def find_person(question: str):
    q = norm(question.lower())

    # full-name match
    for user in ALL_USERS:
        if norm(user.lower()) in q:
            return user

    # first-name match
    for user in ALL_USERS:
        first = user.split()[0].lower()
        if first in q:
            return user

    return None

# ---------------------------------------------------------
# Filter messages for user
# ---------------------------------------------------------
def get_user_messages(person: str):
    p = norm(person.lower())
    return [m for m in MESSAGES if p in norm(m["user_name"].lower())]

# ---------------------------------------------------------
# LLM Strict inference
# ---------------------------------------------------------
def ask_llm(context: str, question: str):
    prompt = f"""
You are an information extraction system.

Only use facts explicitly written in the messages.
Do NOT guess or infer anything.
If the answer is not stated, reply exactly:
No information available.

MESSAGES:
{context}

QUESTION:
{question}

ANSWER:
"""

    out = qa_model(
        prompt,
        max_new_tokens=60,
        do_sample=False
    )[0]["generated_text"]

    # Extract answer
    answer = out.split("ANSWER:")[-1].strip()

    if not answer or answer.lower().startswith("no information"):
        return "No information available."

    return answer

# ---------------------------------------------------------
# FastAPI endpoint
# ---------------------------------------------------------
@app.get("/ask")
def ask(question: str = Query(...)):
    person = find_person(question)
    if not person:
        return {"answer": "No information available."}

    msgs = get_user_messages(person)
    if not msgs:
        return {"answer": "No information available."}

    # Build context from user messages
    context = "\n".join([f"{m['user_name']}: {m['message']}" for m in msgs])

    answer = ask_llm(context, question)
    return {"answer": answer}
