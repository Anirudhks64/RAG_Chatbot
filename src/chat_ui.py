"""
The Chainlit chat screen: welcome message, robot animation, game,
and showing the answers. The actual work is done in pipeline.py.
"""
import asyncio

import chainlit as cl

from pipeline import stream_into, split_questions


# ======================================================================
# CHAINLIT UI (unchanged, except one funny line no longer mentions the cloud)
# ======================================================================

THINKING_LINES = [
    "Flipping through your PDFs . . .",
    "Hunting for the best chunks . . .",
    "Waking up the smart robot . . .",
    "Connecting the dots . . .",
    "Reading the fine print so you don't have to . . .",
    "Polishing the answer . . .",
    "Almost there . . .",
]

TRACK_LENGTH = 10  # how many steps the robot takes before it reaches the books


async def animate_thinking(reply: cl.Message, status: dict):
    """Updates the 'Thinking' bubble every moment until it is cancelled."""
    frame = 0
    while True:
        step = frame % (TRACK_LENGTH + 1)
        track = "  ".join(["·"] * step + ["🤖"] + ["·"] * (TRACK_LENGTH - step)) + "  📚"

        # If main() set a status text, show it. Otherwise use the funny lines.
        if status["text"]:
            line = status["text"]
        else:
            line = THINKING_LINES[(frame // 4) % len(THINKING_LINES)]

        reply.content = f"{line}\n\n{track}"
        await reply.update()

        frame += 1
        await asyncio.sleep(0.7)


@cl.on_chat_start
async def start():
    await cl.Message(content="Hi! Ask me a question about your documents.").send()


@cl.on_message
async def main(message: cl.Message):
    # Remove the game from the previous question (if there is one)
    old_game = cl.user_session.get("game")
    if old_game:
        await old_game.remove()

    questions = split_questions(message.content)
    several = len(questions) >= 2

    # 1. The "Thinking" bubble (with the animation) comes first
    reply = cl.Message(content="Thinking . . .")
    await reply.send()

    # 2. For several questions, make ONE answers message, above the game
    answers_msg = None
    if several:
        answers_msg = cl.Message(content="Answers will appear here as they are ready . . .")
        await answers_msg.send()

    # 3. The game is sent last, so it stays at the bottom
    game = cl.Message(
        content="Play while you wait! 🐦",
        elements=[
            cl.CustomElement(
                name="GameFrame",
                props={"sid": cl.context.session.id},
                display="inline",
            )
        ],
    )
    await game.send()
    cl.user_session.set("game", game)

    status = {"text": ""}
    animation = asyncio.create_task(animate_thinking(reply, status))

    async def stop_animation():
        if not animation.done():
            animation.cancel()
            try:
                await animation
            except asyncio.CancelledError:
                pass

    try:
        
        if not several:
            # One question: stop the robot when the first words arrive,
            # then the answer fills the reply bubble live
            answer = await stream_into(reply, message.content, on_first_piece=stop_animation)
            final_text = answer or "The bot did not return any text."
        else:
            total = len(questions)
            collected = []  # all finished answers so far

            for i, q in enumerate(questions, 1):
                status["text"] = f"Answering your questions one by one . . . ({i} of {total})"

                # Stream this answer below the ones already finished
                header = f"**{i}. {q}**\n\n"
                prefix = "\n\n---\n\n".join(collected + [header])
                try:
                    answer = await stream_into(answers_msg, q, prefix=prefix)
                except Exception as error:
                    answer = f"Something went wrong: {error}"

                collected.append(f"{header}{answer or 'No answer returned.'}")

                # Update the SAME message, so the game below does not move
                answers_msg.content = "\n\n---\n\n".join(collected)
                await answers_msg.update()

            final_text = f"Done! Answered {total} questions. 👇"
    except Exception as error:
        final_text = f"Something went wrong: {error}"
    finally:
        await stop_animation()

    reply.content = final_text
    await reply.update()
