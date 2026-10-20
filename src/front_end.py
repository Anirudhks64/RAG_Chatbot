import chainlit as cl
 
 
# ---------------------------------------------------------------------------
# YOUR PART: put your bot here.
# Chainlit gives you the user's question as text.
# You return the answer as text. That's all.
# ---------------------------------------------------------------------------
def get_answer(question: str) -> str:
    

    
    return f"You asked: {question}"
# ---------------------------------------------------------------------------
 
 
@cl.on_chat_start
async def start():
    await cl.Message(content="Hi! Ask me a question about your documents.").send()
 
 
@cl.on_message
async def main(message: cl.Message):
    # Show a "thinking" bubble while your code works
    reply = cl.Message(content="Thinking...")
    await reply.send()
 
    try:
        # Runs your function without freezing the chat page
        answer = await cl.make_async(get_answer)(message.content)
    except Exception as error:
        answer = f"Something went wrong: {error}"
 
    # Replace "Thinking..." with the answer
    reply.content = answer
    await reply.update()