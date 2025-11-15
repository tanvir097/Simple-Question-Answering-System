import os
import unicodedata
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_KEY)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

MESSAGES_URL = "https://november7-730026606190.europe-west1.run.app/messages"
MESSAGES = []
ALL_USERS = []


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------
def norm(x: str):
    return unicodedata.normalize("NFKD", x).replace("–", "-").replace("—", "-").strip()


# ---------------------------------------------------------
# Async message loader
# ---------------------------------------------------------
async def load_messages_async():
    global MESSAGES, ALL_USERS

    headers = {"X-API-Key": "november7"}

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(MESSAGES_URL, headers=headers)
            r.raise_for_status()
            data = r.json()

        items = data.get("items", [])

        for m in items:
            m["user_name"] = norm(m["user_name"])
            m["message"] = norm(m["message"])

        MESSAGES = items
        ALL_USERS = list({m["user_name"] for m in items})

        print(f"[STARTUP] Loaded {len(MESSAGES)} messages")

    except Exception as e:
        print("[ERROR] Could not load messages:", e)


# ---------------------------------------------------------
# FastAPI startup event (non-blocking)
# ---------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    await load_messages_async()


# ---------------------------------------------------------
# User extraction
# ---------------------------------------------------------
def extract_person(question: str):
    q = norm(question.lower())

    # Exact full-name match
    for user in ALL_USERS:
        if norm(user.lower()) in q:
            return user

    # First-name match
    for user in ALL_USERS:
        first = user.split()[0].lower()
        if first in q:
            return user

    return None


# ---------------------------------------------------------
# Filter user messages
# ---------------------------------------------------------
def get_user_messages(person: str):
    return [m for m in MESSAGES if norm(m["user_name"].lower()) == norm(person.lower())]


# ---------------------------------------------------------
# Ask OpenAI
# ---------------------------------------------------------
def ask_openai(context: str, question: str):
    prompt = f"""
You are an information extraction assistant. Use ONLY the facts provided in MESSAGES.
If the answer is missing, reply exactly: "No information available."
Do NOT guess or invent new facts.

MESSAGES:
{context}

QUESTION:
{question}

ANSWER:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=120,
        )

        ans = response.choices[0].message.content.strip()

        if not ans or "no information" in ans.lower():
            return "No information available."

        return ans

    except Exception as e:
        print("[OpenAI ERROR]:", e)
        return "No information available."


# ---------------------------------------------------------
# UI Home Page
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------
# /ask API endpoint
# ---------------------------------------------------------
@app.get("/ask")
def ask(question: str = Query(...)):
    if not MESSAGES:
        return {"answer": "Service not ready. Try again in a moment."}

    person = extract_person(question)

    if not person:
        return {"answer": "No information available."}

    msgs = get_user_messages(person)

    if not msgs:
        return {"answer": "No information available."}

    context = "\n".join(f"{m['user_name']}: {m['message']}" for m in msgs)

    answer = ask_openai(context, question)
    return {"answer": answer}
