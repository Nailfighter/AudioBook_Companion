import logging
import json

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI audiobook companion. The user is listening to an audiobook while talking to you.
            When the user speaks, the audiobook pauses automatically. After you finish responding, it will resume.

            You can:
            - Answer questions about the story, characters, or plot
            - Provide context or explanations about what's happening
            - Discuss themes and literary elements

            Your responses are concise, conversational, and spoken naturally without formatting symbols or emojis.
            You are knowledgeable, friendly, and enthusiastic about helping users enjoy their audiobook.""",
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Track playback state from frontend
    playback_state = {"status": "paused", "current_time": 0}

    def handle_data_received(data: rtc.DataPacket):
        """Handle data channel messages from frontend"""
        nonlocal playback_state
        try:
            message = json.loads(data.data.decode("utf-8"))
            if message.get("type") == "playback_state":
                playback_state = message
                logger.info(f"Playback state: {playback_state}")
        except Exception as e:
            logger.error(f"Error handling data: {e}")

    # Listen for data packets
    ctx.room.on("data_received", handle_data_received)

    async def send_audiobook_command(action: str, **kwargs):
        """Send control commands to the audiobook player"""
        command = {"action": action, **kwargs}
        data = json.dumps(command).encode("utf-8")
        await ctx.room.local_participant.publish_data(data, reliable=True)
        logger.info(f"Sent audiobook command: {command}")

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()

    # State machine for audiobook control
    was_playing_before_interruption = False
    import asyncio

    @session.on("user_state_changed")
    def on_user_state_changed(event):
        """User state changed - pause when user speaks"""
        nonlocal was_playing_before_interruption
        if event.new_state == "speaking":
            # Remember if audiobook was playing when user interrupted
            was_playing_before_interruption = playback_state.get("status") == "playing"
            if was_playing_before_interruption:
                asyncio.create_task(send_audiobook_command("pause_audiobook"))

    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        """Agent state changed - resume when agent finishes speaking"""
        nonlocal was_playing_before_interruption
        if event.new_state == "listening" and event.old_state == "speaking" and was_playing_before_interruption:
            asyncio.create_task(send_audiobook_command("resume_audiobook"))
            was_playing_before_interruption = False


if __name__ == "__main__":
    cli.run_app(server)
