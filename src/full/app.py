"""
Start the chat page (from the project folder):  chainlit run src/full/app.py -w

This file only loads the chat screen (chat_ui.py), which loads pipeline.py.
Chainlit picks up the @cl.on_chat_start / @cl.on_message handlers on import.
"""
import chat_ui  # noqa: F401
