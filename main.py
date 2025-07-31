from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

import os
from dotenv import load_dotenv 

load_dotenv()  


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

class Message(BaseModel):
    user_input: str

system_prompt = """
You are a caring and supportive human friend who chats naturally.
Keep your tone friendly and casual.
If the user's message is short or just a greeting like "hi", "hello", "hey" — reply briefly too (e.g., "Hi!" or "Hey! What can I do for you today?").
Don't over-explain or stretch answers unnecessarily.
Focus on being helpful, warm, and to-the-point like a real friend.
"""

@app.post("/chat")
def chat(message: Message):
    try:
        full_prompt = f"{system_prompt.strip()}\n\nUser: {message.user_input.strip()}"
        response = llm.invoke([HumanMessage(content=full_prompt)])
        return {"response": response.content.strip()}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}
