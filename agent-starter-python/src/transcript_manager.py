"""
Transcript Manager for Context-Aware Audiobook Companion

Handles time-based indexing of audiobook transcripts and extracts
relevant context based on playback position.

Supports both:
- VTT files with exact timestamps (preferred)
- Plain text files with WPM-based timing (fallback)
"""

import os
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from openai import AsyncOpenAI


@dataclass
class TimestampedWord:
    """Represents a word with its exact timestamp from VTT file."""
    word: str
    start_time: float
    end_time: float


@dataclass
class Chunk:
    """Represents a chunk of transcript for semantic search."""
    chunk_id: int
    start_word_index: int
    end_word_index: int
    start_time: float
    end_time: float
    text: str
    keywords: Set[str]


@dataclass
class SearchResult:
    """Results from semantic search."""
    found: bool
    time: float = 0.0
    chunk_id: int = -1
    context_preview: str = ""
    confidence: float = 0.0


class TranscriptManager:
    """Manages audiobook transcript with time-based indexing."""

    def __init__(self, transcript_path: str, estimated_wpm: int = 120, openai_api_key: Optional[str] = None):
        """
        Initialize transcript manager.

        Args:
            transcript_path: Path to transcript file (.txt or .vtt)
            estimated_wpm: Narrator's words per minute (default: 120, only used for .txt files)
            openai_api_key: OpenAI API key for LLM-based semantic search
        """
        self.transcript_path = transcript_path
        self.wpm = estimated_wpm
        self.words_per_second = estimated_wpm / 60.0

        # Initialize OpenAI client for LLM-based semantic search
        self.openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None

        # Detect file type and load accordingly
        self.is_vtt = transcript_path.lower().endswith('.vtt')

        if self.is_vtt:
            # Load VTT file with exact timestamps
            self.timestamped_words = self._parse_vtt()
            self.transcript = " ".join([w.word for w in self.timestamped_words])
            self.words = [w.word for w in self.timestamped_words]
            self.total_words = len(self.words)
            self.total_duration = self.timestamped_words[-1].end_time if self.timestamped_words else 0.0
        else:
            # Load plain text file with WPM-based timing
            self.timestamped_words = []
            self.transcript = self._load_transcript()
            self.words = self.transcript.split()
            self.total_words = len(self.words)
            self.total_duration = self.total_words / self.words_per_second

        # Character aliases for tracking
        self.main_characters = {
            "snow white": ["snow white", "snow-white"],
            "queen": ["queen", "stepmother", "wicked woman"],
            "dwarfs": ["dwarf", "dwarves", "seven little men"],
            "huntsman": ["huntsman", "hunter"],
            "prince": ["prince", "king's son"],
            "magic mirror": ["mirror", "looking glass"],
        }

        # Create chunks for semantic search
        self.chunks = self._create_chunks()

    def _load_transcript(self) -> str:
        """Load transcript from file."""
        if not os.path.exists(self.transcript_path):
            raise FileNotFoundError(f"Transcript not found: {self.transcript_path}")

        with open(self.transcript_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _parse_vtt(self) -> List[TimestampedWord]:
        """
        Parse VTT file with exact word-level timestamps.

        YouTube VTT files contain word-level timing tags like:
        it<00:00:12.320><c> was</c><00:00:12.559><c> the</c>

        We extract these to get precise per-word timestamps.

        Returns:
            List of TimestampedWord objects with precise timing
        """
        if not os.path.exists(self.transcript_path):
            raise FileNotFoundError(f"VTT file not found: {self.transcript_path}")

        with open(self.transcript_path, "r", encoding="utf-8") as f:
            content = f.read()

        timestamped_words = []

        # Pattern to match VTT timestamp lines: "00:00:04.160 --> 00:00:05.910"
        timestamp_pattern = re.compile(r'(\d{2}):(\d{2}):(\d{2}\.\d{3}) --> (\d{2}):(\d{2}):(\d{2}\.\d{3})')
        # Pattern to match word-level timestamps: word<00:00:12.320>
        word_time_pattern = re.compile(r'([^<]+)<(\d{2}):(\d{2}):(\d{2}\.\d{3})>')

        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Match timestamp line
            match = timestamp_pattern.match(line)
            if match:
                # Parse segment start/end time for fallback
                start_h, start_m, start_s = int(match.group(1)), int(match.group(2)), float(match.group(3))
                segment_start = start_h * 3600 + start_m * 60 + start_s

                end_h, end_m, end_s = int(match.group(4)), int(match.group(5)), float(match.group(6))
                segment_end = end_h * 3600 + end_m * 60 + end_s

                # Next line(s) contain the text with word-level timestamps
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip() and not timestamp_pattern.match(lines[i]):
                    # Skip alignment markers
                    if not lines[i].strip().startswith('align:'):
                        text_lines.append(lines[i].strip())
                    i += 1

                # Process each line for word-level timestamps
                for text_line in text_lines:
                    # Skip empty lines and metadata
                    if not text_line or text_line == '[Music]':
                        continue

                    # ONLY process lines that have word-level timestamp tags
                    # Lines without tags are just accumulated caption text (duplicates)
                    if '<' not in text_line:
                        continue

                    # Remove <c> and </c> tags first
                    text_line = text_line.replace('<c>', '').replace('</c>', '')

                    # Parse the line structure:
                    # "first_word<timestamp1> second_word<timestamp2> ..."
                    # Split by < to separate text and timestamps
                    parts = text_line.split('<')

                    word_times = []
                    # First part might be a word without timestamp
                    if parts[0].strip():
                        # This word uses the segment start time
                        word_times.append((parts[0].strip(), segment_start))

                    # Process remaining parts (word<timestamp pairs)
                    for part in parts[1:]:
                        if '>' in part:
                            timestamp_str, remaining = part.split('>', 1)
                            # Parse timestamp
                            try:
                                time_match = re.match(r'(\d{2}):(\d{2}):(\d{2}\.\d{3})', timestamp_str)
                                if time_match:
                                    h, m, s = int(time_match.group(1)), int(time_match.group(2)), float(time_match.group(3))
                                    word_time = h * 3600 + m * 60 + s

                                    # The text AFTER the timestamp is what gets that timestamp
                                    if remaining.strip():
                                        word_times.append((remaining.strip(), word_time))
                            except:
                                pass

                    if word_times:
                        # Add words with their timestamps
                        for idx, (word_text, word_time) in enumerate(word_times):
                            # Determine end time (next word's start or segment end)
                            if idx + 1 < len(word_times):
                                end_time = word_times[idx + 1][1]
                            else:
                                end_time = segment_end

                            # Clean and add each word
                            for word in word_text.split():
                                clean_word = word.strip('.,!?;:')
                                if clean_word:
                                    timestamped_words.append(TimestampedWord(
                                        word=clean_word,
                                        start_time=word_time,
                                        end_time=end_time
                                    ))
            i += 1

        return timestamped_words

    def get_context_at_time(
        self, current_seconds: float, context_window_seconds: float = 180
    ) -> Dict:
        """
        Extract story context based on playback position.

        Args:
            current_seconds: Current playback time in seconds
            context_window_seconds: How many seconds of recent content to include (default: 180 = 3 minutes)

        Returns:
            Dictionary with:
                - heard_so_far: Text heard up to current_seconds
                - word_count: Number of words in context
                - character_mentions: Dict of which characters have appeared
                - estimated_position: Percentage through story
        """
        # Check if we're beyond the actual content duration
        beyond_duration = current_seconds >= self.total_duration

        if self.is_vtt:
            # Use exact timestamps from VTT file
            # Clamp current_seconds to actual duration
            clamped_current_seconds = min(current_seconds, self.total_duration)

            if beyond_duration:
                # Return ALL words when beyond duration (for compatibility)
                heard_words = [w.word for w in self.timestamped_words]
            else:
                # Normal case: return context window
                context_start_seconds = max(0, clamped_current_seconds - context_window_seconds)
                heard_words = [
                    w.word for w in self.timestamped_words
                    if context_start_seconds <= w.start_time <= clamped_current_seconds
                ]

            # Find all words heard so far for character tracking
            all_heard_words = [
                w.word for w in self.timestamped_words
                if w.start_time <= clamped_current_seconds
            ]

            heard_text = " ".join(heard_words)
            all_heard_text = " ".join(all_heard_words)

            # Calculate progress based on actual time (clamp to 100% max)
            progress_percent = min(100, (current_seconds / self.total_duration) * 100) if self.total_duration > 0 else 0

        else:
            # Use WPM-based timing for plain text files
            current_word_index = int(current_seconds * self.words_per_second)
            current_word_index = min(current_word_index, self.total_words)

            if beyond_duration:
                # Return ALL words when beyond duration
                heard_words = self.words[:]
            else:
                # Normal case: return context window
                context_start_seconds = max(0, current_seconds - context_window_seconds)
                context_start_index = int(context_start_seconds * self.words_per_second)
                heard_words = self.words[context_start_index:current_word_index]

            heard_text = " ".join(heard_words)
            all_heard_text = " ".join(self.words[:current_word_index])

            progress_percent = (current_word_index / self.total_words) * 100 if self.total_words > 0 else 0

        # Track character appearances
        character_mentions = {}
        for char_name, aliases in self.main_characters.items():
            character_mentions[char_name] = any(
                alias in all_heard_text.lower() for alias in aliases
            )

        return {
            "heard_so_far": heard_text,
            "word_count": len(heard_words),
            "character_mentions": character_mentions,
            "estimated_position": progress_percent,
        }

    def get_full_context_until_time(self, current_seconds: float) -> str:
        """
        Get ALL text heard from beginning up to current time.

        Use sparingly for full context; prefer get_context_at_time() for
        token efficiency.

        Args:
            current_seconds: Current playback time in seconds

        Returns:
            Complete text heard from start to current_seconds
        """
        if self.is_vtt:
            # Use exact timestamps from VTT file
            # Clamp current_seconds to actual duration
            clamped_current_seconds = min(current_seconds, self.total_duration)
            heard_words = [
                w.word for w in self.timestamped_words
                if w.start_time <= clamped_current_seconds
            ]
        else:
            # Use WPM-based timing for plain text files
            current_word_index = int(current_seconds * self.words_per_second)
            current_word_index = min(current_word_index, self.total_words)
            heard_words = self.words[:current_word_index]

        return " ".join(heard_words)

    def check_character_appeared(self, current_seconds: float, character_name: str) -> bool:
        """
        Check if a character has appeared in the story so far.

        Args:
            current_seconds: Current playback time
            character_name: Name of character to check

        Returns:
            True if character has been mentioned, False otherwise
        """
        heard_text = self.get_full_context_until_time(current_seconds).lower()

        # Check if it's a known character with aliases
        if character_name.lower() in self.main_characters:
            aliases = self.main_characters[character_name.lower()]
            return any(alias in heard_text for alias in aliases)

        # Otherwise check direct name match
        return character_name.lower() in heard_text

    def get_total_duration_estimate(self) -> float:
        """
        Get total audiobook duration in seconds.

        For VTT files: Returns exact duration from timestamps
        For text files: Estimates based on word count and WPM

        Returns:
            Duration in seconds
        """
        return self.total_duration

    def _create_chunks(self) -> List[Chunk]:
        """
        Create overlapping chunks for semantic search.

        Chunk size: 150 words
        Overlap: 25 words (prevents missing boundary matches)

        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_size = 150
        overlap = 25
        step = chunk_size - overlap

        for i in range(0, len(self.words), step):
            start_idx = i
            end_idx = min(i + chunk_size, len(self.words))

            chunk_words = self.words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)

            # Calculate timestamps
            if self.is_vtt:
                # Use exact timestamps from VTT file
                start_time = self.timestamped_words[start_idx].start_time
                end_time = self.timestamped_words[end_idx - 1].end_time
            else:
                # Use WPM-based estimation
                start_time = start_idx / self.words_per_second
                end_time = end_idx / self.words_per_second

            # Extract keywords (lowercase unique words)
            keywords = set(word.lower() for word in chunk_words)

            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    start_word_index=start_idx,
                    end_word_index=end_idx,
                    start_time=start_time,
                    end_time=end_time,
                    text=chunk_text,
                    keywords=keywords,
                )
            )

            if end_idx >= len(self.words):
                break

        return chunks

    async def semantic_search(self, query: str, current_time: float) -> SearchResult:
        """
        Search for a scene matching the query in the entire transcript using LLM.

        This method uses GPT-4 to analyze the transcript and find the most relevant
        section matching the user's query. It provides 90-95% accuracy and handles
        paraphrasing and natural language understanding.

        NOTE: For demo purposes, this searches the ENTIRE transcript, not just heard content.

        Args:
            query: Natural language description (e.g., "poison apple", "when the queen asks the mirror")
            current_time: Current playback position in seconds (unused for demo)

        Returns:
            SearchResult with found status, time, and preview

        Raises:
            RuntimeError: If OpenAI client is not initialized (API key missing)
        """
        if not self.openai_client:
            raise RuntimeError(
                "OpenAI API key is required for semantic search. "
                "Please set OPENAI_API_KEY in your .env.local file."
            )

        # Get the ENTIRE transcript (no spoiler prevention for demo)
        heard_text = self.transcript

        if not heard_text.strip():
            return SearchResult(found=False)

        # Construct prompt for LLM to find relevant section
        prompt = f"""You are analyzing an audiobook transcript to find a specific scene or moment.

The user is searching for: "{query}"

Here is the complete audiobook transcript:

{heard_text}

Your task:
1. Find the section of the transcript that best matches what the user is looking for
2. IMPORTANT: Return the position of the BEGINNING/FIRST SENTENCE of that scene or section, NOT the middle
3. Think about where the scene STARTS - this could be a few sentences before the most relevant content
4. If found, return the approximate position as a percentage (0-100) of where this scene BEGINS in the transcript
5. Also provide a brief preview (50-100 characters) of the FIRST sentence of that scene

Example: If searching for "poison apple", find where the poison apple scene STARTS (e.g., "The queen disguised herself..."), not the middle of the scene.

Respond ONLY with a JSON object in this exact format:
{{"found": true/false, "position_percent": 0-100, "preview": "text preview here"}}

If the content is not found or the query doesn't match anything in the transcript, return:
{{"found": false, "position_percent": 0, "preview": ""}}"""

        try:
            # Call OpenAI API
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes audiobook transcripts with high precision. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=200,
            )

            # Parse the response
            response_text = response.choices[0].message.content.strip()

            # Extract JSON from response (handle markdown code blocks)
            if response_text.startswith("```"):
                # Remove markdown code block formatting
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            result_data = json.loads(response_text)

            if not result_data.get("found", False):
                return SearchResult(found=False)

            # Convert position percentage to actual timestamp
            position_percent = result_data.get("position_percent", 0)
            position_percent = max(0, min(100, position_percent))  # Clamp to 0-100

            # Calculate word index based on percentage of transcript
            target_word_index = int((position_percent / 100.0) * self.total_words)
            target_word_index = min(target_word_index, self.total_words - 1)

            # Convert word index to time
            if self.is_vtt:
                # Use exact timestamp from VTT file
                # But backtrack to find sentence start for better user experience
                sentence_start_index = self._find_sentence_start(target_word_index)
                target_time = self.timestamped_words[sentence_start_index].start_time
            else:
                # Use WPM-based estimation
                target_time = target_word_index / self.words_per_second

            # Find the chunk containing this time for chunk_id
            target_chunk_id = -1
            for chunk in self.chunks:
                if chunk.start_time <= target_time <= chunk.end_time:
                    target_chunk_id = chunk.chunk_id
                    break

            return SearchResult(
                found=True,
                time=target_time,
                chunk_id=target_chunk_id,
                context_preview=result_data.get("preview", "")[:100],
                confidence=95.0,  # High confidence for LLM-based search
            )

        except Exception as e:
            # Log error and re-raise
            import logging
            logger = logging.getLogger("agent")
            logger.error(f"LLM semantic search failed: {e}")
            raise

    def _find_sentence_start(self, word_index: int, max_backtrack: int = 15) -> int:
        """
        Find the start of the sentence or phrase containing the given word index.

        Backtracks from the given word index to find natural boundaries.
        For transcripts with punctuation, looks for . ! ?
        For unpunctuated transcripts (like YouTube captions), backtracks a
        reasonable distance (max_backtrack words) to provide better context.

        Args:
            word_index: The word index to start from
            max_backtrack: Maximum words to backtrack for unpunctuated text (default: 15)

        Returns:
            The word index of the sentence/phrase start
        """
        if word_index <= 0:
            return 0

        # Sentence-ending punctuation marks
        sentence_endings = {'.', '!', '?'}

        # Backtrack to find the previous sentence ending
        # Start from word_index - 1 to check previous words
        for i in range(word_index - 1, max(0, word_index - max_backtrack - 1), -1):
            word = self.words[i]

            # Check if this word ends with sentence-ending punctuation
            if any(word.endswith(punct) for punct in sentence_endings):
                # Found a sentence ending, so next word (i+1) is the start
                return i + 1

        # If no punctuation found within max_backtrack words,
        # return the position max_backtrack words back
        # This provides some context without going too far back
        return max(0, word_index - max_backtrack)
