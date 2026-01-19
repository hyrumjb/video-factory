"""
Clause segmentation module.

Takes a script and word-level timestamps from TTS, then uses an LLM to split
the script into single-idea clauses with accurate timing.
"""

import json
import re
from typing import List, Optional
from pydantic import BaseModel
from config import xai_client, XAI_MODEL


class Clause(BaseModel):
    """A single clause from the segmented script."""
    clause_id: int
    text: str
    idea_type: str  # hook, fact, explanation, example, transition, conclusion
    start_time: float
    next_start_time: float  # End time (start of next clause or end of audio)


class ClauseSegmentationResult(BaseModel):
    """Result from clause segmentation."""
    clauses: List[Clause]
    total_duration: float


def _build_word_timing_text(script: str, alignment: dict) -> str:
    """
    Build a text representation of words with their timestamps.

    Args:
        script: The raw script text
        alignment: ElevenLabs alignment data with 'characters' and 'character_start_times_seconds'

    Returns:
        Formatted string like:
        [0.00] Word1
        [0.25] Word2
        ...
    """
    if not alignment:
        # Fallback: estimate timing based on word count
        words = script.split()
        estimated_duration = len(words) * 0.4  # ~0.4s per word average
        lines = []
        for i, word in enumerate(words):
            time = i * (estimated_duration / len(words))
            lines.append(f"[{time:.2f}] {word}")
        return '\n'.join(lines)

    # Parse ElevenLabs alignment format
    characters = alignment.get('characters', [])
    char_times = alignment.get('character_start_times_seconds', [])

    if not characters or not char_times or len(characters) != len(char_times):
        # Fallback
        words = script.split()
        estimated_duration = len(words) * 0.4
        lines = []
        for i, word in enumerate(words):
            time = i * (estimated_duration / len(words))
            lines.append(f"[{time:.2f}] {word}")
        return '\n'.join(lines)

    # Reconstruct words with their start times
    words_with_times: List[tuple] = []
    current_word = ""
    word_start_time = 0.0

    for i, char in enumerate(characters):
        if char == ' ':
            if current_word:
                words_with_times.append((word_start_time, current_word))
                current_word = ""
        else:
            if not current_word:
                word_start_time = char_times[i]
            current_word += char

    # Don't forget the last word
    if current_word:
        words_with_times.append((word_start_time, current_word))

    # Format as timestamped lines
    lines = []
    for time, word in words_with_times:
        lines.append(f"[{time:.2f}] {word}")

    return '\n'.join(lines)


def _get_audio_duration(alignment: dict) -> float:
    """Get total audio duration from alignment data."""
    if not alignment:
        return 60.0  # Default fallback

    char_times = alignment.get('character_start_times_seconds', [])
    char_durations = alignment.get('character_durations_seconds', [])

    if char_times and char_durations and len(char_times) == len(char_durations):
        # Last character start time + its duration
        return char_times[-1] + char_durations[-1]
    elif char_times:
        # Estimate: last char time + small buffer
        return char_times[-1] + 0.5

    return 60.0


async def segment_script_into_clauses(
    script: str,
    alignment: Optional[dict] = None,
    audio_duration: Optional[float] = None
) -> ClauseSegmentationResult:
    """
    Segment a script into single-idea clauses with timing information.

    Args:
        script: The raw script text
        alignment: Word-level timing from TTS (ElevenLabs format)
        audio_duration: Total audio duration in seconds (optional, calculated from alignment)

    Returns:
        ClauseSegmentationResult with list of clauses and total duration
    """
    if not xai_client:
        raise RuntimeError("xAI client not available")

    # Build the word-timing text representation
    word_timing_text = _build_word_timing_text(script, alignment)

    # Get total duration
    total_duration = audio_duration or _get_audio_duration(alignment)

    system_prompt = """You are a script analyzer. Your job is to split a script into single-idea clauses and assign accurate start times using word-level TTS timestamps.

INPUT FORMAT:
You will receive the script as timestamped words:
[0.00] First
[0.15] word
[0.30] of
[0.42] the
[0.55] script
...

OUTPUT FORMAT:
Return a JSON array of clause objects. Each clause should contain ONE complete idea:
[
  {
    "clause_id": 1,
    "text": "The exact words from the script",
    "idea_type": "hook",
    "start_time": 0.00,
    "next_start_time": 2.35
  },
  ...
]

IDEA TYPES:
- hook: Opening attention-grabber
- fact: A piece of information or statistic
- explanation: Explaining or elaborating on something
- example: A specific example or illustration
- transition: Connecting two ideas
- conclusion: Closing thought or summary

RULES:
1. Each clause should contain ONE complete thought/idea (usually 5-15 words)
2. The start_time is the timestamp of the FIRST word in that clause
3. The next_start_time is the timestamp of the FIRST word of the NEXT clause
4. For the last clause, next_start_time should be the total audio duration
5. Preserve the EXACT text from the input (don't paraphrase)
6. Split at natural pause points (after complete sentences or phrases)
7. Aim for 8-15 clauses for a 30-60 second script

EXAMPLE:
Input:
[0.00] The
[0.12] human
[0.35] brain
[0.58] uses
[0.82] twenty
[1.05] percent
[1.32] of
[1.45] your
[1.60] body's
[1.88] energy.
[2.15] That's
[2.38] more
[2.55] than
[2.70] any
[2.85] other
[3.02] organ.

Output:
[
  {"clause_id": 1, "text": "The human brain uses twenty percent of your body's energy.", "idea_type": "hook", "start_time": 0.00, "next_start_time": 2.15},
  {"clause_id": 2, "text": "That's more than any other organ.", "idea_type": "fact", "start_time": 2.15, "next_start_time": 3.50}
]"""

    user_message = f"""Split this script into clauses. Total audio duration: {total_duration:.2f} seconds.

{word_timing_text}

Return ONLY the JSON array, no other text."""

    print(f"📝 Segmenting script into clauses...")

    response = xai_client.chat.completions.create(
        model=XAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,  # Lower temp for more consistent parsing
        max_tokens=2000
    )

    response_content = response.choices[0].message.content
    if not response_content:
        raise RuntimeError("Failed to get clause segmentation from LLM")

    # Parse the JSON response
    try:
        # Extract JSON array from response
        json_match = re.search(r'\[[\s\S]*\]', response_content)
        if not json_match:
            raise ValueError("No JSON array found in response")

        clauses_data = json.loads(json_match.group())

        clauses = []
        for item in clauses_data:
            clause = Clause(
                clause_id=item['clause_id'],
                text=item['text'],
                idea_type=item.get('idea_type', 'fact'),
                start_time=float(item['start_time']),
                next_start_time=float(item['next_start_time'])
            )
            clauses.append(clause)

        print(f"✓ Segmented into {len(clauses)} clauses")
        for c in clauses:
            duration = c.next_start_time - c.start_time
            print(f"   [{c.clause_id}] {c.idea_type}: {c.start_time:.2f}s - {c.next_start_time:.2f}s ({duration:.2f}s)")
            print(f"       \"{c.text[:50]}{'...' if len(c.text) > 50 else ''}\"")

        return ClauseSegmentationResult(
            clauses=clauses,
            total_duration=total_duration
        )

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"⚠ Error parsing clause response: {e}")
        print(f"   Response was: {response_content[:500]}")
        raise RuntimeError(f"Failed to parse clause segmentation: {e}")


async def generate_scene_queries_for_clauses(
    clauses: List[Clause],
    topic: str
) -> List[dict]:
    """
    Generate stock video search queries for each clause.

    Args:
        clauses: List of clauses from segmentation
        topic: The original video topic

    Returns:
        List of dicts with clause_id and search_query
    """
    if not xai_client:
        raise RuntimeError("xAI client not available")

    # Build clause list for the prompt
    clause_texts = []
    for c in clauses:
        clause_texts.append(f"{c.clause_id}. [{c.idea_type}] \"{c.text}\"")

    clauses_formatted = '\n'.join(clause_texts)

    system_prompt = """You are a stock video search expert for Pexels API.

For each clause, create a 1-3 word search query that returns good stock video footage.

PEXELS SEARCH RULES:
1. Use 1-3 simple, common English words
2. Focus on FILMABLE subjects: people, places, objects, actions
3. NO abstract concepts (success, happiness, controversy)
4. NO proper nouns or brand names

GOOD QUERIES: "money", "person typing", "ocean waves", "city street", "hands phone"
BAD QUERIES: "shocking truth", "financial freedom", "controversial topic"

Return a JSON array with clause_id and search_query for each clause."""

    user_message = f"""Topic: {topic}

Clauses:
{clauses_formatted}

Return ONLY a JSON array like:
[{{"clause_id": 1, "search_query": "query here"}}, ...]"""

    print(f"🔍 Generating search queries for {len(clauses)} clauses...")

    response = xai_client.chat.completions.create(
        model=XAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.5,
        max_tokens=1000
    )

    response_content = response.choices[0].message.content
    if not response_content:
        # Fallback: generic queries
        return [{"clause_id": c.clause_id, "search_query": "people"} for c in clauses]

    try:
        json_match = re.search(r'\[[\s\S]*\]', response_content)
        if json_match:
            queries = json.loads(json_match.group())
            print(f"✓ Generated {len(queries)} search queries")
            for q in queries:
                print(f"   [{q['clause_id']}] \"{q['search_query']}\"")
            return queries
    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠ Error parsing search queries: {e}")

    # Fallback
    return [{"clause_id": c.clause_id, "search_query": "people"} for c in clauses]
