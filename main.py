from fastapi import FastAPI, Query
import httpx
import re
import unicodedata

app = FastAPI()

# ----------------------------
# Helpers
# ----------------------------
def norm(x):
    return unicodedata.normalize("NFKD", x).strip()


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


# ----------------------------
# Extract which member is asked about
# ----------------------------
def find_person(question: str):
    q = question.lower()

    # full name
    for u in ALL_USERS:
        if u.lower() in q:
            return u

    # first name only
    for u in ALL_USERS:
        first = u.split()[0].lower()
        if first in q:
            return u

    return None


# ----------------------------
# Message filtering
# ----------------------------
def get_user_messages(user):
    u = user.lower()
    return [m for m in MESSAGES if m["user_name"].lower() == u]


# ----------------------------
# Extract structured info
# ----------------------------
DATE_REGEX = r"\b(?:\d{4}-\d{2}-\d{2}|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|tonight|tomorrow|this friday|this saturday|next week|next monday|first week of [a-z]+)\b"
LOCATION_REGEX = r"\b(in|to)\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)"
COUNT_REGEX = r"\b(\d+)\b"

def extract_date(text):
    m = re.search(DATE_REGEX, text, re.IGNORECASE)
    return m.group(0) if m else None

def extract_location(text):
    m = re.search(LOCATION_REGEX, text)
    return m.group(2) if m else None

def extract_numbers(text):
    return re.findall(COUNT_REGEX, text)


# ----------------------------
# Main QA Logic
# ----------------------------
def answer_question(question, msgs):
    q = question.lower()

    # 1) Travel questions
    if any(w in q for w in ["travel", "trip", "going to", "visit", "flight", "going"]):
        for m in msgs:
            date = extract_date(m["message"])
            loc = extract_location(m["message"])
            if date or loc:
                text_parts = []
                if loc:
                    text_parts.append(f"Destination: {loc}")
                if date:
                    text_parts.append(f"When: {date}")
                return ", ".join(text_parts)
        return "No travel information available."

    # 2) Cars / counts
    if "car" in q or "cars" in q:
        for m in msgs:
            nums = extract_numbers(m["message"])
            if nums:
                return f"{nums[0]}"

        return "No information about cars."

    # 3) Favorite restaurants
    if "restaurant" in q or "favorite" in q:
        favorites = []
        for m in msgs:
            if "restaurant" in m["message"].lower():
                favorites.append(m["message"])
        if favorites:
            return favorites[0]
        return "No restaurant information available."

    # 4) Generic fallback — return relevant message if any
    for m in msgs:
        if any(word in m["message"].lower() for word in ["book", "reserve", "appointment", "tickets", "confirm"]):
            return m["message"]

    return "No information available."


# ----------------------------
# API Endpoint
# ----------------------------
@app.get("/ask")
def ask(question: str = Query(...)):
    person = find_person(question)
    if not person:
        return {"answer": "No information available."}

    msgs = get_user_messages(person)
    if not msgs:
        return {"answer": "No information available."}

    answer = answer_question(question, msgs)
    return {"answer": answer}
