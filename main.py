import os
import uuid
import chromadb

from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from huggingface_hub import InferenceClient

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_KEY = os.getenv("HF_API_KEY")

hf_client = InferenceClient(
    api_key=HF_API_KEY,
    provider="auto"
)

db_client = chromadb.PersistentClient(path="./chroma_data")
collection = db_client.get_or_create_collection(name="chat_memory")


class ChatRequest(BaseModel):
    message: str
    user_id: str


def save_message(user_id: str, text: str):
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[text],
        metadatas=[{"user_id": user_id}]
    )


def get_memory(user_id: str, query: str, n_results: int = 3) -> str:
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"user_id": user_id}
    )
    docs = results.get("documents", [])
    if docs and len(docs) > 0 and len(docs[0]) > 0:
        return "\n".join(docs[0])
    return ""


def load_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


@app.get("/")
def home():
    return {"msg": "working"}


@app.post("/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    user_id = req.user_id.strip()

    if not message:
        return {"reply": "Message is required"}

    if not user_id:
        return {"reply": "User ID is required"}

    if not HF_API_KEY:
        return {"reply": "Missing HF_API_KEY in .env"}

    try:
        memory = get_memory(user_id, message)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Use the previous memory only if it is relevant to the user's current question."
            }
        ]

        if memory:
            messages.append({
                "role": "system",
                "content": f"Relevant memory:\n{memory}"
            })

        messages.append({
            "role": "user",
            "content": message
        })

        completion = hf_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=messages,
            max_tokens=200
        )

        reply = completion.choices[0].message.content if completion.choices else "No response"

        save_message(user_id, f"User: {message}")
        save_message(user_id, f"Assistant: {reply}")

        return {"reply": reply}

    except Exception as e:
        return {"reply": f"Hugging Face error: {str(e)}"}


@app.post("/upload-pdf")
async def upload_pdf(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    if not user_id.strip():
        return {"message": "User ID is required"}

    try:
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{file.filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        pdf_text = load_pdf_text(file_path)

        if not pdf_text:
            return {"message": "No text found in PDF"}

        chunks = [pdf_text[i:i+1500] for i in range(0, min(len(pdf_text), 12000), 1500)]

        for chunk in chunks:
            save_message(user_id, f"PDF Content: {chunk}")

        return {
            "message": "PDF uploaded and memory saved",
            "filename": file.filename
        }

    except Exception as e:
        return {"message": f"PDF upload failed: {str(e)}"}