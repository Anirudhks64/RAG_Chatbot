#!/usr/bin/env bash
# Test the chatbot on this Mac: starts the chatbot code (lite version, with the Groq key
# from .env) and the test page, then opens the page in Chrome. Ctrl+C stops everything.
#
# Run from anywhere:   widget/start-local-test.sh
# Chatbot server:      http://localhost:8000      (its logs are printed in this window)
# Test page:           http://localhost:8090/test-page.html
set -uo pipefail
cd "$(dirname "$0")/.."   # the project folder

API_PORT=8000
PAGE_PORT=8090

# Stop anything left over from an earlier test run on these ports
for port in $API_PORT $PAGE_PORT; do
  pids=$(lsof -ti tcp:$port 2>/dev/null)
  [ -n "$pids" ] && echo "Stopping the old process on port $port" && kill $pids 2>/dev/null
done
sleep 1

# Stop both servers when this script is stopped (Ctrl+C or closing the window)
trap 'echo; echo "Stopping the test servers..."; kill $API_PID $PAGE_PID 2>/dev/null; exit 0' INT TERM EXIT

# 1. The test page (a tiny web server for the widget folder)
python3 -m http.server $PAGE_PORT --bind 127.0.0.1 --directory widget >/dev/null 2>&1 &
PAGE_PID=$!

# 2. The chatbot itself: the code in this folder, lite version
echo "Starting the chatbot (reads the website and loads the models, ~10-20 s)..."
.venv/bin/uvicorn api:app --app-dir src/lite --host 127.0.0.1 --port $API_PORT 2>&1 | grep --line-buffered -v "it/s\]" &
API_PID=$!

# 3. Open the page once the chatbot answers its health check
for i in $(seq 1 180); do
  curl -s "http://localhost:$API_PORT/health" >/dev/null && break
  sleep 1
done
if curl -s "http://localhost:$API_PORT/health" >/dev/null; then
  echo
  echo "Ready. Test page: http://localhost:$PAGE_PORT/test-page.html  (Ctrl+C here stops everything)"
  echo
  open -a "Google Chrome" "http://localhost:$PAGE_PORT/test-page.html" 2>/dev/null \
    || open "http://localhost:$PAGE_PORT/test-page.html"
else
  echo "The chatbot did not start. See the messages above."
fi
wait $API_PID
