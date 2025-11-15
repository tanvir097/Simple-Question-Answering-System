from fastapi import FastAPI, Query
import httpx
import unicodedata
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load env vars
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_KEY)

app = FastAPI()

# ----------------------------------------
# Normalize text
# ----------------------------------------
def norm(x):
    return unicodedata.normalize("NFKD", x).replace("–", "-").replace("—", "-").strip()

# ----------------------------------------
# Load messages with API key
# ----------------------------------------
MESSAGES_URL = "https://november7-730026606190.europe-west1.run.app/messages"

def load_messages():
    headers = {"X-API-Key": "november7"}
    r = httpx.get(MESSAGES_URL, headers=headers, follow_redirects=True)
    r.raise_for_status()

    data = r.json()
    msgs = data.get("items", [])

    for m in msgs:
        m["user_name"] = norm(m["user_name"])
        m["message"] = norm(m["message"])

    print("Loaded", len(msgs), "messages")
    return msgs

MESSAGES = load_messages()
ALL_USERS = list({m["user_name"] for m in MESSAGES})


# ----------------------------------------
# Person extraction
# ----------------------------------------
def extract_person(question: str):
    q = norm(question.lower())

    # Full name match
    for user in ALL_USERS:
        if user.lower() in q:
            return user

    # First name match
    for user in ALL_USERS:
        first = user.split()[0].lower()
        if first in q:
            return user

    return None


# ----------------------------------------
# Get messages for user
# ----------------------------------------
def get_user_messages(person: str):
    return [
        m for m in MESSAGES
        if norm(m["user_name"].lower()) == norm(person.lower())
    ]


# ----------------------------------------
# Ask OpenAI
# ----------------------------------------
def ask_openai(context: str, question: str):

    prompt = f"""
You are an information extraction assistant. Use ONLY the facts provided in MESSAGES. 
Do NOT guess or infer. If the answer is missing, reply exactly: "No information available."

MESSAGES:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=128,
    )

    ans = response.choices[0].message.content.strip()

    # normalize bad LLM behavior
    if not ans or "no information" in ans.lower():
        return "No information available."

    return ans


# ----------------------------------------
# Root endpoint
# ----------------------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Use /ask?question=Your question"
    }


# ----------------------------------------
# Ask endpoint
# ----------------------------------------
@app.get("/ask")
def ask(question: str = Query(...)):
    person = extract_person(question)

    if not person:
        return {"answer": "No information available."}

    msgs = get_user_messages(person)

    if not msgs:
        return {"answer": "No information available."}

    context = "\n".join([f"{m['user_name']}: {m['message']}" for m in msgs])

    answer = ask_openai(context, question)

    return {"answer": answer}
