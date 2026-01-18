import logging
import json
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from transcript_manager import TranscriptManager

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class ContextAwareAssistant(Agent):
    def __init__(
        self,
        instructions: str,
        transcript_manager: TranscriptManager,
        playback_state_ref: dict,
        room: rtc.Room,
        state_machine_flags: dict,
        audiobooks_list: list,
        current_audiobook_index: dict,
        transcript_managers: dict,
    ) -> None:
        super().__init__(instructions=instructions)
        self.transcript_manager = transcript_manager
        self.playback_state_ref = playback_state_ref
        self.room = room
        self.state_machine_flags = state_machine_flags
        self.audiobooks_list = audiobooks_list
        self.current_audiobook_index = current_audiobook_index
        self.transcript_managers = transcript_managers

    def get_current_audiobook(self):
        """Get the current audiobook metadata."""
        current_idx = self.current_audiobook_index["index"]
        return self.audiobooks_list[current_idx]

    @function_tool
    async def get_current_audiobook_info(self, context: RunContext) -> str:
        """Get information about the currently playing audiobook.

        Use this when you need to know which audiobook is currently playing,
        or when the user asks "What am I listening to?" or "What's this book called?"

        Returns:
            JSON string with title, author, and duration
        """
        current_audiobook = self.get_current_audiobook()
        return f"Title: {current_audiobook['title']}, Author: {current_audiobook['author']}"

    @function_tool
    async def get_story_context(self, context: RunContext) -> str:
        """Get current story context based on playback position.

        Use this tool when you need to reference what has happened in the story.
        Returns the text that has been heard so far (last 3 minutes of content).
        """
        current_time = self.playback_state_ref.get("current_time", 0)
        current_audiobook = self.get_current_audiobook()

        logger.info(
            f"get_story_context called for '{current_audiobook['title']}' "
            f"at playback time: {current_time:.1f}s"
        )

        story_context = self.transcript_manager.get_context_at_time(
            current_time, context_window_seconds=180
        )

        logger.info(
            f"Context: {story_context['word_count']} words, "
            f"{story_context['estimated_position']:.1f}% through story"
        )

        return story_context["heard_so_far"]

    @function_tool
    async def check_if_character_appeared(
        self, context: RunContext, character_name: str
    ) -> str:
        """Check if a character has appeared in the story so far.

        Args:
            character_name: Name of character to check (e.g., "queen", "prince", "dwarfs")

        Returns:
            "yes" if mentioned, "no" if not yet appeared
        """
        current_time = self.playback_state_ref.get("current_time", 0)
        has_appeared = self.transcript_manager.check_character_appeared(
            current_time, character_name
        )

        if has_appeared:
            return f"yes, {character_name} has appeared in the story"
        else:
            return f"no, {character_name} has not been mentioned yet (avoid spoilers!)"

    @function_tool
    async def pause_audiobook(self, context: RunContext) -> str:
        """Pause the audiobook immediately.

        Use when user says: "pause", "stop", "wait", "hold on"

        Returns:
            Confirmation message
        """
        command = {"action": "pause_audiobook"}
        data = json.dumps(command).encode("utf-8")
        await self.room.local_participant.publish_data(data, reliable=True)

        # Clear state machine flags to prevent auto-resume
        self.state_machine_flags["was_playing"] = False
        self.state_machine_flags["in_conversation"] = False
        self.state_machine_flags["resume_pending"] = False

        logger.info("🎵 User requested pause via voice command (cleared state machine)")
        return "Paused"

    @function_tool
    async def resume_audiobook(self, context: RunContext) -> str:
        """Resume playing the audiobook.

        Use when user says: "play", "resume", "continue", "start"

        Returns:
            Confirmation message
        """
        command = {"action": "resume_audiobook"}
        data = json.dumps(command).encode("utf-8")
        await self.room.local_participant.publish_data(data, reliable=True)

        # Clear state machine flags to prevent conflicts
        self.state_machine_flags["was_playing"] = False
        self.state_machine_flags["in_conversation"] = False
        self.state_machine_flags["resume_pending"] = False

        logger.info("🎵 User requested resume via voice command (cleared state machine)")
        return "Playing"

    @function_tool
    async def set_playback_speed(self, context: RunContext, speed: float) -> str:
        """Set the audiobook playback speed.

        Args:
            speed: Playback speed from 0.25x to 2.0x

        Use when user says: "speed up", "slow down", "play faster", "play slower", "set speed to 1.5x"

        Returns:
            Confirmation message
        """
        # Clamp speed between 0.25 and 2.0
        speed = max(0.25, min(2.0, speed))

        command = {"action": "set_speed", "speed": speed}
        data = json.dumps(command).encode("utf-8")
        await self.room.local_participant.publish_data(data, reliable=True)

        logger.info(f"🎵 User requested speed: {speed}x")

        return f"Speed {speed}x"

    @function_tool
    async def skip_time(self, context: RunContext, seconds: int) -> str:
        """Skip forward or backward in the audiobook.

        Args:
            seconds: Positive = skip forward, Negative = rewind backward

        Examples:
            - "rewind 30 seconds" → seconds=-30
            - "skip ahead 1 minute" → seconds=60
            - "go back 2 minutes" → seconds=-120

        Returns:
            Confirmation message
        """
        current_time = self.playback_state_ref.get("current_time", 0)
        new_time = max(0, current_time + seconds)  # Don't go below 0

        command = {"action": "seek", "time": new_time}
        data = json.dumps(command).encode("utf-8")
        await self.room.local_participant.publish_data(data, reliable=True)

        direction = "forward" if seconds > 0 else "back"
        abs_seconds = abs(seconds)

        logger.info(f"🎵 User requested skip {direction}: {abs_seconds}s (from {current_time:.1f}s to {new_time:.1f}s)")

        if abs_seconds >= 60:
            minutes = abs_seconds // 60
            return f"Skipped {direction} {minutes} minute{'s' if minutes > 1 else ''}"
        else:
            return f"Skipped {direction} {abs_seconds} seconds"

    @function_tool
    async def next_audiobook(self, context: RunContext) -> str:
        """Switch to the next audiobook in the library.

        Use when user says: "next book", "next audiobook", "skip to next book"

        Returns:
            Confirmation message with new audiobook title
        """
        command = {"action": "next_audiobook"}
        data = json.dumps(command).encode("utf-8")
        await self.room.local_participant.publish_data(data, reliable=True)

        # Update current audiobook index
        current_idx = self.current_audiobook_index["index"]
        next_idx = (current_idx + 1) % len(self.audiobooks_list)
        self.current_audiobook_index["index"] = next_idx

        # Switch to the new transcript
        audiobook_id = self.audiobooks_list[next_idx]["id"]
        if audiobook_id in self.transcript_managers:
            self.transcript_manager = self.transcript_managers[audiobook_id]
            logger.info(f"🔄 Switched to transcript for: {self.audiobooks_list[next_idx]['title']}")

        logger.info("🎵 User requested next audiobook")
        return "Next audiobook"

    @function_tool
    async def previous_audiobook(self, context: RunContext) -> str:
        """Switch to the previous audiobook in the library.

        Use when user says: "previous book", "previous audiobook", "go back to previous book"

        Returns:
            Confirmation message with new audiobook title
        """
        command = {"action": "previous_audiobook"}
        data = json.dumps(command).encode("utf-8")
        await self.room.local_participant.publish_data(data, reliable=True)

        # Update current audiobook index
        current_idx = self.current_audiobook_index["index"]
        prev_idx = (current_idx - 1 + len(self.audiobooks_list)) % len(self.audiobooks_list)
        self.current_audiobook_index["index"] = prev_idx

        # Switch to the new transcript
        audiobook_id = self.audiobooks_list[prev_idx]["id"]
        if audiobook_id in self.transcript_managers:
            self.transcript_manager = self.transcript_managers[audiobook_id]
            logger.info(f"🔄 Switched to transcript for: {self.audiobooks_list[prev_idx]['title']}")

        logger.info("🎵 User requested previous audiobook")
        return "Previous audiobook"

    @function_tool
    async def navigate_to_scene(self, context: RunContext, description: str) -> str:
        """Navigate to a specific scene or moment in the audiobook.

        Use this when the user asks to find or go to a specific scene, event,
        character moment, or topic in the audiobook.

        Args:
            description: Natural language description of the scene to find.
                        Examples: "poison apple", "queen asks the mirror",
                                 "dwarfs find snow white", "huntsman scene"

        Returns:
            Confirmation message with scene preview or "not found yet" message
        """
        current_time = self.playback_state_ref.get("current_time", 0)
        current_audiobook = self.get_current_audiobook()

        logger.info(
            f"🔍 Searching for scene: '{description}' "
            f"in '{current_audiobook['title']}' at {current_time:.1f}s"
        )

        # Search entire transcript using LLM-based semantic search (demo mode - no spoiler prevention)
        result = await self.transcript_manager.semantic_search(
            query=description,
            current_time=current_time
        )

        if not result.found:
            logger.info(f"❌ Scene not found: '{description}'")
            return "I couldn't find that scene in the audiobook."

        # Send seek command to frontend
        command = {"action": "seek", "time": result.time}
        data = json.dumps(command).encode("utf-8")
        await self.room.local_participant.publish_data(data, reliable=True)

        logger.info(
            f"✅ Navigated to scene at {result.time:.1f}s "
            f"(chunk {result.chunk_id}, confidence {result.confidence:.2f})"
        )

        # Return brief confirmation
        preview = result.context_preview.replace("\n", " ")[:80]
        return f"Found it: {preview}..."

    @function_tool
    async def search_earlier_context(self, context: RunContext, topic: str) -> str:
        """Search for earlier mentions of a topic, character, or event.

        Use this when the user asks about something that happened earlier
        in the story and it's not in the recent context.

        Args:
            topic: What to search for (e.g., "huntsman", "queen's mirror")

        Returns:
            Earlier context mentioning the topic, or "not found" message
        """
        current_time = self.playback_state_ref.get("current_time", 0)

        logger.info(f"🔍 Searching earlier context for: '{topic}'")

        result = await self.transcript_manager.semantic_search(
            query=topic,
            current_time=current_time
        )

        if result.found:
            logger.info(f"✅ Found earlier mention at {result.time:.1f}s")
            return f"Earlier mention: {result.context_preview}"
        else:
            logger.info(f"❌ No earlier mention found for: '{topic}'")
            return "I don't recall that being mentioned yet in what we've heard."

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

    # Agent reference (will be set after agent is created)
    agent_ref = {"agent": None}

    def handle_data_received(data: rtc.DataPacket):
        """Handle data channel messages from frontend"""
        nonlocal playback_state
        try:
            message = json.loads(data.data.decode("utf-8"))
            if message.get("type") == "playback_state":
                # Update the dict in place to preserve reference
                playback_state.clear()
                playback_state.update(message)
                current_time = playback_state.get("current_time", 0)
                status = playback_state.get("status", "unknown")
                logger.info(f"📊 Playback state update: {current_time:.1f}s ({status})")
            elif message.get("type") == "audiobook_changed":
                # Frontend changed audiobook via button click (will be handled after loading audiobooks)
                pass  # Will be updated below after loading transcript_managers
        except Exception as e:
            logger.error(f"Error handling data: {e}")

    # Listen for data packets
    ctx.room.on("data_received", handle_data_received)

    async def send_audiobook_command(action: str, **kwargs):
        """Send control commands to the audiobook player"""
        command = {"action": action, **kwargs}
        data = json.dumps(command).encode("utf-8")
        await ctx.room.local_participant.publish_data(data, reliable=True)

    # Load audiobook metadata
    audiobooks_json_path = os.path.join(
        os.path.dirname(__file__),
        "../../agent-starter-react/public/audiobooks.json",
    )
    try:
        with open(audiobooks_json_path, "r") as f:
            audiobooks = json.load(f)
            logger.info(f"Loaded {len(audiobooks)} audiobooks from metadata")
    except Exception as e:
        logger.error(f"Failed to load audiobook metadata: {e}")
        audiobooks = [{
            "id": "unknown",
            "title": "Unknown",
            "author": "Unknown",
            "duration": 0
        }]

    # Load transcript managers for all audiobooks
    transcript_managers = {}
    transcript_dir = os.path.join(
        os.path.dirname(__file__),
        "../../agent-starter-react/public/transcript"
    )
    captions_dir = os.path.join(
        os.path.dirname(__file__),
        "../../agent-starter-react/public/captions"
    )

    for audiobook in audiobooks:
        audiobook_id = audiobook.get("id", "unknown")

        # Try to get transcript filename from metadata, otherwise derive from ID
        transcript_filename = audiobook.get("transcript_file")

        if not transcript_filename:
            # Derive from ID: "snow-white-001" -> "snow_white_trans.txt"
            base_name = audiobook_id.rsplit("-", 1)[0].replace("-", "_")
            transcript_filename = f"{base_name}_trans.txt"

        # Check if it's a VTT file (look in captions directory)
        if transcript_filename.lower().endswith('.vtt'):
            transcript_path = os.path.join(captions_dir, transcript_filename)
        else:
            # Plain text file (look in transcript directory)
            transcript_path = os.path.join(transcript_dir, transcript_filename)

        # Load transcript if it exists
        if os.path.exists(transcript_path):
            try:
                manager = TranscriptManager(
                    transcript_path,
                    estimated_wpm=120,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
                transcript_managers[audiobook_id] = manager
                logger.info(
                    f"Loaded transcript for '{audiobook['title']}': "
                    f"{manager.total_words} words, "
                    f"~{manager.get_total_duration_estimate():.0f}s"
                )
            except Exception as e:
                logger.error(f"Failed to load transcript for {audiobook_id}: {e}")
        else:
            logger.warning(f"Transcript not found for {audiobook_id} at {transcript_path}")

    # Use first audiobook as default
    current_audiobook_index = {"index": 0}
    audiobook_metadata = audiobooks[0]
    audiobook_id = audiobook_metadata.get("id")

    # Get transcript manager for first audiobook
    if audiobook_id in transcript_managers:
        transcript_manager = transcript_managers[audiobook_id]
    else:
        # Fallback: create empty transcript manager
        logger.warning("No transcript available for first audiobook")
        transcript_manager = TranscriptManager(
            "",
            estimated_wpm=120,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    # Now update the data handler to handle audiobook changes
    def handle_audiobook_change_message(message):
        """Handle audiobook change from physical button clicks"""
        new_index = message.get("index")
        audiobook_id = message.get("audiobook_id")

        if new_index is not None and agent_ref["agent"] is not None:
            # Update the agent's current audiobook index
            current_audiobook_index["index"] = new_index

            # Switch the agent's transcript manager
            if audiobook_id in transcript_managers:
                agent_ref["agent"].transcript_manager = transcript_managers[audiobook_id]
                audiobook_title = audiobooks[new_index]["title"]
                logger.info(f"🔄 Physical button pressed - switched to: {audiobook_title}")
            else:
                logger.warning(f"No transcript found for audiobook ID: {audiobook_id}")

    # Update the data handler with the audiobook change logic
    original_handler = handle_data_received
    def handle_data_received_with_audiobook(data: rtc.DataPacket):
        nonlocal playback_state
        try:
            message = json.loads(data.data.decode("utf-8"))
            if message.get("type") == "playback_state":
                playback_state.clear()
                playback_state.update(message)
                current_time = playback_state.get("current_time", 0)
                status = playback_state.get("status", "unknown")
                logger.info(f"📊 Playback state update: {current_time:.1f}s ({status})")
            elif message.get("type") == "audiobook_changed":
                handle_audiobook_change_message(message)
        except Exception as e:
            logger.error(f"Error handling data: {e}")

    # Replace the handler
    ctx.room.off("data_received", handle_data_received)
    ctx.room.on("data_received", handle_data_received_with_audiobook)

    def create_context_aware_instructions() -> str:
        """Generate instructions that rely on function tools for current context"""

        instructions = """You are a helpful voice AI audiobook companion.
The user is listening to audiobooks and can switch between different books.

**IMPORTANT - Know What You're Discussing:**
- ALWAYS use get_current_audiobook_info() tool when answering questions about the story
- The audiobook can change when the user switches tracks
- Use the tool to know the current title and author before answering questions

**CRITICAL SPOILER PREVENTION RULE:**
- Only discuss events, characters, and plot points from the story that have been heard so far
- NEVER mention anything that hasn't happened yet in the audiobook
- If asked about future events, say: "No spoilers! Keep listening to find out!"

**IMPORTANT - Using the get_story_context Tool:**
- ALWAYS use the get_story_context tool when the user asks about the story
- This tool gives you the last 3 minutes of story content based on current playback position
- It automatically prevents spoilers by only including what has been heard

**How to Answer Questions:**
- Answer the SPECIFIC question the user asks
- Don't just summarize what happened - address their actual question
- Use the story context to find relevant information
- If the answer isn't in what they've heard yet, say so without spoiling

**Response Guidelines:**
- Be concise and conversational (1-2 sentences for simple questions)
- Answer exactly what was asked, not more
- Speak naturally without formatting symbols or emojis
- Reference specific story details when relevant

**Question Types & How to Handle:**
- "What happened?" or "What's going on?" → Briefly summarize recent key events
- "Who is [character]?" → Explain that character based on what's been heard about them
- "Why did [X] happen?" → Explain the reason if it was mentioned
- "What's [character] doing?" → Describe their current situation in the story
- "Where is [character]?" → Answer based on what's been heard
- "What happens next/at the end?" → "No spoilers! Keep listening to find out!"

**Example Responses:**
- User: "Who is the huntsman?"
  → "The huntsman is the one the queen sent to take Snow White into the woods."

- User: "Why did Snow White run away?"
  → "The huntsman showed her mercy and told her to flee because the queen wanted her dead."

- User: "What just happened?"
  → "Snow White just found the dwarves' cottage in the woods and went inside."

**Playback Control Commands:**
You can control audiobook playback when the user requests it. Listen for these phrases:

- **Pause/Stop**: "pause", "stop", "wait", "hold on"
  → Use pause_audiobook() tool

- **Play/Resume**: "play", "resume", "continue", "start"
  → Use resume_audiobook() tool

- **Speed Up**: "speed up", "play faster", "faster"
  → Get current speed from playback_state: self.playback_state_ref.get("playback_speed", 1.0)
  → Use set_playback_speed() with current_speed + 0.25
  → If user says specific speed like "1.5x" or "2x", use that exact value
  → Speed range: 0.25x to 2.0x (will be clamped automatically)

- **Slow Down**: "slow down", "play slower", "slower"
  → Get current speed from playback_state: self.playback_state_ref.get("playback_speed", 1.0)
  → Use set_playback_speed() with current_speed - 0.25
  → Speed range: 0.25x to 2.0x (will be clamped automatically)

- **Set Specific Speed**: "set speed to 1.5", "play at 2x speed"
  → Use set_playback_speed() with the requested speed
  → Speed range: 0.25x to 2.0x

- **Skip Forward**: "skip ahead 1 minute", "fast forward 30 seconds"
  → Use skip_time(seconds=60) or skip_time(seconds=30)
  → Parse the time: "1 minute" = 60, "30 seconds" = 30

- **Rewind**: "rewind 30 seconds", "go back a bit", "go back 2 minutes"
  → Use skip_time(seconds=-30) or skip_time(seconds=-120)
  → "a bit" = 30 seconds by default

- **Next Audiobook**: "next book", "next audiobook", "skip to next book"
  → Use next_audiobook() tool

- **Previous Audiobook**: "previous book", "previous audiobook", "go back to previous book"
  → Use previous_audiobook() tool

- **Navigate to Scene**: "take me to [scene]", "skip to [moment]", "find [event]"
  → Use navigate_to_scene(description="...") tool
  → Examples:
    - "Take me to the poison apple scene" → navigate_to_scene(description="poison apple")
    - "Skip to when the queen asks the mirror" → navigate_to_scene(description="queen mirror")
    - "Go back to the huntsman" → navigate_to_scene(description="huntsman")
  → This searches the entire audiobook transcript, so you can jump to any scene

- **Search Earlier Context**: When user asks about earlier events not in recent context
  → Use search_earlier_context(topic="...") tool
  → Examples:
    - "Remind me why the queen hates Snow White?" → search_earlier_context(topic="queen hates snow white")
    - "What did the huntsman do earlier?" → search_earlier_context(topic="huntsman")
  → Returns earlier context mentioning the topic

**Important for time parsing:**
- Convert "X minutes" to seconds: multiply by 60
- Convert "X seconds" to seconds: use as-is
- Default amounts: "a bit" = 30 seconds, "ahead" = 60 seconds

**Important for speed control:**
- Default increment/decrement: 0.25x
- Minimum speed: 0.25x (don't go below)
- Maximum speed: 2.0x (don't go above)
- Common speeds: 0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 1.75x, 2.0x

**Response style for commands:**
- Keep confirmations VERY brief: just one word like "Paused", "Playing", "Muted"
- For skip commands: "Skipped forward 1 minute" or "Skipped back 30 seconds"
- For scene navigation: Just say what you found: "Found it: [preview]..."
- NEVER add extra phrases like "Let me know if...", "Just ask if...", "Would you like..."
- NEVER offer additional options after executing a command
- Just confirm what was done and stop talking

Remember: Answer the specific question asked. Be helpful and informative without spoiling future events!"""

        return instructions

    # Create context-aware agent with dynamic instructions
    initial_instructions = create_context_aware_instructions()
    logger.info("Created context-aware instructions")

    # State machine flags (shared with agent for voice control)
    state_machine_flags = {
        "was_playing": False,
        "in_conversation": False,
        "resume_pending": False
    }

    # Create the agent
    agent = ContextAwareAssistant(
        initial_instructions,
        transcript_manager,
        playback_state,
        ctx.room,
        state_machine_flags,
        audiobooks,
        current_audiobook_index,
        transcript_managers,
    )

    # Store agent reference for physical button handler
    agent_ref["agent"] = agent

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=agent,
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
    conversation_timeout_task = None
    import asyncio

    async def reset_conversation_after_timeout():
        """Reset conversation state after user stops talking for a while"""
        await asyncio.sleep(3.0)  # Wait 3 seconds of silence
        logger.info("⏱️ Conversation timeout - resetting state")
        state_machine_flags["in_conversation"] = False
        state_machine_flags["was_playing"] = False
        state_machine_flags["resume_pending"] = False

    @session.on("user_state_changed")
    def on_user_state_changed(event):
        """User state changed - pause when user speaks"""
        nonlocal conversation_timeout_task

        logger.info(f"👤 User state: {event.old_state} → {event.new_state}")

        if event.new_state == "speaking":
            # Cancel any pending timeout
            if conversation_timeout_task and not conversation_timeout_task.done():
                conversation_timeout_task.cancel()

            # Only remember playback state on FIRST interruption
            if not state_machine_flags["in_conversation"]:
                state_machine_flags["was_playing"] = playback_state.get("status") == "playing"
                logger.info(f"🎙️ User started speaking. Audiobook was: {'playing' if state_machine_flags['was_playing'] else 'paused'}")

            state_machine_flags["in_conversation"] = True

            # Pause audiobook if it's playing OR if resume is pending
            if playback_state.get("status") == "playing" or state_machine_flags["resume_pending"]:
                logger.info(f"⏸️ Pausing audiobook (playing={playback_state.get('status')}, resume_pending={state_machine_flags['resume_pending']})")
                asyncio.create_task(send_audiobook_command("pause_audiobook"))
                # Remember we were playing if resume was pending
                if state_machine_flags["resume_pending"] and not state_machine_flags["was_playing"]:
                    state_machine_flags["was_playing"] = True
                    logger.info("📝 Updated was_playing=True (interrupted during resume)")
                state_machine_flags["resume_pending"] = False

        elif event.new_state == "listening":
            # User stopped speaking - start timeout to reset conversation
            logger.info("👂 User stopped speaking, waiting for agent response...")

    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        """Agent state changed - resume when agent finishes speaking"""
        nonlocal conversation_timeout_task

        logger.info(f"🤖 Agent state: {event.old_state} → {event.new_state}")

        if event.new_state == "listening" and event.old_state == "speaking":
            # Agent finished speaking
            logger.info(f"✅ Agent finished. Should resume? was_playing={state_machine_flags['was_playing']}, in_convo={state_machine_flags['in_conversation']}")

            if state_machine_flags["was_playing"] and state_machine_flags["in_conversation"]:
                logger.info("▶️ Resuming audiobook after agent response")
                state_machine_flags["resume_pending"] = True  # Mark that resume is pending
                asyncio.create_task(send_audiobook_command("resume_audiobook"))
                # Reset flags
                state_machine_flags["was_playing"] = False
                state_machine_flags["in_conversation"] = False

                # Cancel any pending timeout
                if conversation_timeout_task and not conversation_timeout_task.done():
                    conversation_timeout_task.cancel()

                # Clear resume_pending after the delay (2.5s + buffer)
                async def clear_resume_pending():
                    await asyncio.sleep(3.0)
                    state_machine_flags["resume_pending"] = False
                    logger.info("✓ Resume completed, cleared pending flag")

                asyncio.create_task(clear_resume_pending())
            else:
                # Start timeout to reset conversation state
                conversation_timeout_task = asyncio.create_task(reset_conversation_after_timeout())


if __name__ == "__main__":
    cli.run_app(server)
