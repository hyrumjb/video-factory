"""
Script Generation Module

Generates engaging TikTok-style scripts from a topic.
Outputs pure flowing text without section labels.
"""

import re
from fastapi import HTTPException
from models import ScriptRequest, ScriptResponse
from config import xai_client, XAI_MODEL


async def generate_script_text(topic: str) -> str:
    """
    Generate a clean script text without section labels.
    The script is well-structured internally but outputs as pure flowing text.
    """
    if not xai_client:
        raise HTTPException(status_code=503, detail="xAI client not available")

    system_prompt = """You are a script writer creating 60-second TikTok-length videos. You want them to go viral so they must be edgy and engaging.

CRITICAL REQUIREMENTS:
- Start with an immediately ENGAGING hook - the strongest, most attention-grabbing content first
- Build through 2-3 body points with engaging, surprising, or little-known facts
- End with a memorable closing point or twist
- Use a casual, conversational tone as if spoken by a charismatic person
- NO dashes, NO questions, NO call to action at the end
- Total script: 100-120 words (60 seconds when spoken)

GRAMMAR REQUIREMENTS:
- Use PROPER grammar and punctuation
- ALWAYS use apostrophes in contractions (don't, can't, won't, it's, that's, they're, etc.)
- NEVER write contractions without apostrophes (WRONG: dont, cant, wont, its when meaning "it is")
- Use correct possessives with apostrophes (world's, Earth's, history's)

OUTPUT FORMAT:
- Output ONLY the script text itself
- NO section labels (no "HOOK:", "BODY 1:", etc.)
- NO formatting markers
- Just pure, flowing spoken text ready for TTS"""

    user_message = f"""Create an 100-120 word TikTok video script about: {topic}"""

    print(f"🎬 Generating script for topic: {topic}")

    response = xai_client.chat.completions.create(
        model=XAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.8,
        max_tokens=800
    )

    script = response.choices[0].message.content
    if not script:
        raise HTTPException(status_code=500, detail="Failed to generate script")

    # Clean up any formatting artifacts
    script = script.strip().strip('"').strip("'").strip('`').strip()

    # Remove any section labels if the LLM still includes them
    script = re.sub(r'\b(HOOK|BODY\s*\d?|INTRO|OUTRO|CLOSING):\s*', '', script, flags=re.IGNORECASE)
    script = ' '.join(script.split())  # Normalize whitespace

    print(f"✓ Script generated ({len(script.split())} words)")

    return script


async def generate_script(request: ScriptRequest) -> ScriptResponse:
    """
    Generate a video script from a topic.
    Returns pure script text - clauses are generated after TTS.
    """
    if not request.topic or not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required")

    topic = request.topic.strip()

    # Generate the clean script (no section labels)
    script = await generate_script_text(topic)

    print(f"✅ Script generation complete: {len(script.split())} words")

    return ScriptResponse(
        script=script,
        scenes=[],
        sections=None
    )
