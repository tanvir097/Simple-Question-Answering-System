from fastapi import FastAPI, Query
import httpx
import unicodedata
from transformers import pipeline

app = FastAPI()

# ---------------------------------------------------------
# Normalize text
# ---------------------------------------------------------
def norm(x):
    return unicodedata.normalize("NFKD", x).replace("–","-").replace("—","-").strip()


# ---------------------------------------------------------
# Load messages
# ---------------------------------------------------------
MESSAGES_URL = "https://november7-730026606190.europe-west1.run.app/messages"

def load_messages():
    headers = {"X-API-Key": "november7"}  # ADD THIS
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
# Load LLM pipeline (simple, stable)
# ---------------------------------------------------------
qa_model = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    torch_dtype="auto"
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
You are a precise information-extraction assistant.

ONLY use the facts given inside the MESSAGES.
If the answer is not explicitly stated in the messages,
reply EXACTLY with: "No information available."
Do NOT guess or infer.

MESSAGES:
{context}

QUESTION:
{question}

ANSWER:
"""

    out = qa_model(
        prompt,
        max_new_tokens=80,
        do_sample=False,         # deterministic inference
        temperature=0.2,         # strict extraction
        top_p=1.0
    )[0]["generated_text"]

    # Extract the answer after ANSWER:
    ans = out.split("ANSWER:")[-1].strip()

    if ans == "" or ans.lower().startswith("no information"):
        return "No information available."

    return ans


# ---------------------------------------------------------
# FastAPI endpoint: /ask
# ---------------------------------------------------------
@app.get("/ask")
def ask(question: str = Query(...)):

    person = find_person(question)
    if not person:
        return {"answer": "No information available."}

    msgs = get_user_messages(person)
    if not msgs:
        return {"answer": "No information available."}

    # Build context ONLY with relevant messages
    context = "\n".join([f"{m['user_name']}: {m['message']}" for m in msgs])

    # Get LLM answer
    answer = ask_llm(context, question)

    return {"answer": answer}
