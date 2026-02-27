from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from embedding_service import generate_embedding
from redis_service import save_embedding
import requests

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# URL LLM local Gemma 3
LLM_URL = "http://127.0.0.1:1234/v1/chat/completions"

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
def chat(req: ChatRequest):
    # 1️⃣ Génération embedding
    embedding = generate_embedding(req.message)

    # 2️⃣ Sauvegarde dans Redis
    key = save_embedding(req.message, embedding)

    # 3️⃣ Appel Gemma
    payload = {
        "model": "google/gemma-3-4b",
        "messages": [{"role": "user", "content": req.message}],
        "temperature": 0.7
    }

    try:
        response = requests.post(LLM_URL, json=payload, timeout=30)
        llm_answer = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        llm_answer = f"Erreur de connexion au LLM: {str(e)}"

    # 4️⃣ Retour JSON
    return {
        "response": llm_answer,
        "embedding": embedding[:50],  # preview pour graphique
        "redis_key": key
    }