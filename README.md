A lightweight API service that answers natural-language questions about member messages retrieved from the public dataset.
The service extracts the relevant user, filters their messages, and returns an answer strictly based on explicit information without guessing or hallucination.

🔗 Live Demo: https://simple-question-answering-system.onrender.com/

GET /ask?question=When is Sophia Al-Farsi traveling?
{ "answer": "No information available." }


/main.py
/templates/
    index.html
/requirements.txt
/Dockerfile


📝 Design Notes

I explored multiple approaches:

Rule-based extraction — simple, fast, deterministic.

LLM-based reasoning — improved flexibility but limited by quota and deployment constraints.

Hybrid extraction — name detection + template-guided inference.

Final solution uses the most stable and deploy-friendly approach.

🚢 Deployment

Deployed on Render using:

Docker

FastAPI

Port exposed at 10000
