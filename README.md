---
title: TransAccelors Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# TransAccelors Chatbot API

Answers questions about TransAccelors from its website, using Groq (gpt-oss-120b).

- `POST /ask` with `{"question": "...", "session_id": "..."}` returns `{"answer", "session_id", "seconds"}`
- `POST /ask/stream` returns the answer as plain text, piece by piece
- `GET /health` returns `{"status": "ok"}`
- `GET /docs` lets you try it in the browser

Leave out `session_id` on the first question and send back the one you get, so follow-ups are understood.
