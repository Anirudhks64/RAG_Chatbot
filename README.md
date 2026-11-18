---
title: TransAccelors Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# TransAccelors Chatbot

Answers questions about TransAccelors from its website (`data/websites.txt`), in the user's language.
Search runs on the server; the answers are written by Groq (`openai/gpt-oss-120b`).

## Folders

```
src/
├── shared/   used by both versions: website loader, chunker, answer cache, Groq / Ollama / Gemini code, prompts
├── full/     big version (~6-8 GB RAM): big multilingual models, Chainlit chat page, API, Dockerfile
└── lite/     small version (fits 512 MB): small 8-bit English models, API, Dockerfile
```

Questions are rewritten into English before the search, so the lite version's English-only
models still serve every language. In tests the lite answers were close to the full ones.

## Run on your Mac (from the project folder)

Put `GROQ_API_KEY=...` in `.env` first.

| What | Command |
|---|---|
| Chat page (demo, testing) | `chainlit run src/full/app.py -w` |
| Full API | `uvicorn api:app --app-dir src/full --port 8000` |
| Lite API | `uvicorn api:app --app-dir src/lite --port 8000` |

Then open http://localhost:8000/docs to try the API.

## API

- `POST /ask` with `{"question": "...", "session_id": "..."}` returns `{"answer", "session_id", "seconds"}`
- `POST /ask/stream` returns the answer as plain text, piece by piece
- `GET /health` returns `{"status": "ok"}`
- `GET /docs` lets you try it in the browser

Leave out `session_id` on the first question and send back the one you get, so follow-ups are understood.

## Hosting

**Lite on Render (free plan, 512 MB):** New Web Service from this repo with
Language `Docker`, Dockerfile Path `./src/lite/Dockerfile`, Docker Build Context `.`,
environment variable `GROQ_API_KEY`, Health Check Path `/health`.
The free plan sleeps after 15 idle minutes, so ping `/health` every 5 minutes (e.g. UptimeRobot).

**Full on a big server:** `docker build -f src/full/Dockerfile -t chatbot .` then
`docker run -p 7860:7860 --env-file .env chatbot`.
Hugging Face Spaces (Docker needs HF PRO): `src/full/deploy.sh <username>/<space-name>`.

Groq free plan limits: about 8,000 tokens per minute (roughly 2 questions per minute) and 1,000 requests per day.
