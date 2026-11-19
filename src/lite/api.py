"""
LITE version of src/full/api.py (small models, fits a 512 MB server). The chatbot as an API: a website (or any app) sends a question, the API sends back the answer.
Same search, cache and Groq answers as the Chainlit chat page (both use pipeline.py).

Start it from the project folder:
    uvicorn api:app --app-dir src/lite --host 0.0.0.0 --port 8000
Then open http://localhost:8000/docs to try it in the browser.

Endpoints:
    POST /ask          {"question": "...", "session_id": "..."}  -> the whole answer as JSON
    POST /ask/stream   same input -> the answer as plain text, piece by piece as it is written
                       (the session id is in the X-Session-Id response header)
    GET  /health       {"status": "ok"} once the documents and models are loaded
    GET  /refresh?key= re-reads the website now (key = the REFRESH_KEY secret)

session_id: leave it out on the first question; the reply gives you one. Send it back with
the next questions so follow-ups ("and how much does it cost?") are understood.
Each session remembers its last 3 questions and answers (in memory, gone after a restart).
"""
import hmac
import os
import threading
import time
import uuid
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Loads the website text, the search models and the answer cache (takes a minute)
from pipeline import answer_pieces, refresh_from_website, start_website_watcher

MAX_QUESTION_CHARS = 2000  # longer questions are refused (protects the Groq limits)
MAX_SESSIONS = 1000        # oldest chats are forgotten after this

app = FastAPI(title="TransAccelors chatbot API (lite)")

# Which websites may call the API from a browser. "*" = any website (fine while testing).
# Later set ALLOWED_ORIGINS, e.g. "https://transaccelors.com,https://www.transaccelors.com"
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],  # so the website's JavaScript can read it
)


# ---------- chat memory, one per session_id ----------

sessions: "OrderedDict[str, dict]" = OrderedDict()
sessions_lock = threading.Lock()


def get_session(session_id):
    """The memory of this chat. Unknown or missing id -> a new empty chat."""
    with sessions_lock:
        if session_id and session_id in sessions:
            sessions.move_to_end(session_id)
            return session_id, sessions[session_id]
        session_id = uuid.uuid4().hex
        sessions[session_id] = {"history": [], "last_prep": None}
        if len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)
        return session_id, sessions[session_id]


# ---------- endpoints ----------

class Question(BaseModel):
    question: str
    session_id: str | None = None


class Answer(BaseModel):
    answer: str
    session_id: str
    seconds: float


def check(question: str) -> str:
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="The question is empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"The question is too long (max {MAX_QUESTION_CHARS} characters).")
    return question


@app.post("/ask", response_model=Answer)
async def ask(body: Question):
    """Returns the whole answer at once."""
    question = check(body.question)
    session_id, session = get_session(body.session_id)
    start = time.perf_counter()
    answer = "".join([piece async for piece in answer_pieces(question, session)])
    return Answer(answer=answer or "Sorry, no answer came back. Please try again.",
                  session_id=session_id, seconds=round(time.perf_counter() - start, 2))


@app.post("/ask/stream")
async def ask_stream(body: Question):
    """Returns the answer piece by piece, so the website can show it while it is written."""
    question = check(body.question)
    session_id, session = get_session(body.session_id)
    return StreamingResponse(answer_pieces(question, session),
                             media_type="text/plain; charset=utf-8",
                             headers={"X-Session-Id": session_id})


# Re-reads the website now (instead of waiting for the next check): open
#     https://<server>/refresh?key=<REFRESH_KEY>
# REFRESH_KEY is a secret password set in the server's environment (Render: Environment).
# Without it the link is switched off.
REFRESH_KEY = os.getenv("REFRESH_KEY", "")

start_website_watcher()


@app.get("/refresh")
async def refresh(key: str = ""):
    if not REFRESH_KEY or not hmac.compare_digest(key.encode(), REFRESH_KEY.encode()):
        raise HTTPException(status_code=403, detail="Wrong or missing key.")
    threading.Thread(target=lambda: print(refresh_from_website("refresh link")), daemon=True).start()
    return {"status": "reading the website now; new content is used within a minute or two"}


@app.api_route("/health", methods=["GET", "HEAD"])  # HEAD too, for uptime monitors
async def health():
    return {"status": "ok"}
