import re
import json
from typing import List, Dict, Tuple
from fastapi import HTTPException
from models import ScriptRequest, ScriptResponse, VideoScene, ScriptSection
from config import openai_client


def sanitize_search_query(query: str) -> str:
    """
    Clean and optimize search query for Pexels/Pixabay API.
    Returns 1-3 simple, generic words that work well with stock video search.
    """
    if not query:
        return "people"

    # Convert to lowercase and split
    words = query.lower().split()

    # Words to remove completely
    stop_words = {
        'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
        'is', 'are', 'was', 'were', 'and', 'or', 'but', 'so', 'yet',
        'this', 'that', 'these', 'those', 'it', 'its'
    }

    # Abstract words that don't return good video results
    abstract_words = {
        'amazing', 'awesome', 'incredible', 'shocking', 'surprising',
        'interesting', 'important', 'controversial', 'secret', 'hidden',
        'unknown', 'famous', 'popular', 'successful', 'powerful',
        'best', 'worst', 'top', 'ultimate', 'real', 'true', 'actual',
        'fact', 'facts', 'truth', 'story', 'reason', 'reasons', 'way', 'ways'
    }

    # Filter words
    clean_words: List[str] = []
    for word in words:
        # Remove punctuation
        word = ''.join(c for c in word if c.isalnum())

        if not word:
            continue
        if word in stop_words:
            continue
        if word in abstract_words:
            continue
        if len(word) < 2:
            continue

        clean_words.append(word)

    # Limit to 3 words max for best Pexels results
    result = ' '.join(clean_words[:3])

    # If we filtered out everything, use generic fallback
    if not result:
        return "people"

    return result


def parse_script_sections(raw_script: str) -> Tuple[Dict[str, str], List[ScriptSection], str]:
    """
    Parse the raw script into sections (HOOK, BODY 1-4).
    Returns:
    - sections dict: {section_name: text}
    - section_list: List of ScriptSection with word boundaries
    - clean_script: The script without labels for TTS
    """
    sections: Dict[str, str] = {}

    # Pattern to match section headers like "HOOK:", "BODY 1:", etc.
    pattern = r'(HOOK|BODY\s*\d?):\s*'

    # Split by section headers while keeping the headers
    parts = re.split(pattern, raw_script, flags=re.IGNORECASE)

    # Process parts - they come as [before, header1, content1, header2, content2, ...]
    current_header = None
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check if this is a header
        if re.match(r'^(HOOK|BODY\s*\d?)$', part, re.IGNORECASE):
            current_header = part.upper().replace(' ', '')  # Normalize: "BODY 1" -> "BODY1"
        elif current_header:
            sections[current_header] = part
            current_header = None

    # Build clean script and track word boundaries
    ordered_sections = ['HOOK', 'BODY1', 'BODY2', 'BODY3', 'BODY4']
    script_parts: List[str] = []
    section_list: List[ScriptSection] = []
    current_word_index = 0

    for section_name in ordered_sections:
        if section_name in sections:
            text = sections[section_name]
            word_count = len(text.split())

            section_list.append(ScriptSection(
                name=section_name,
                text=text,
                word_start=current_word_index,
                word_end=current_word_index + word_count
            ))

            script_parts.append(text)
            current_word_index += word_count

    clean_script = ' '.join(script_parts)

    print(f"✓ Parsed {len(sections)} script sections: {list(sections.keys())}")
    print(f"✓ Section boundaries:")
    for s in section_list:
        print(f"   {s.name}: words {s.word_start}-{s.word_end} ({s.word_end - s.word_start} words)")

    return sections, section_list, clean_script


async def generate_script_text(topic: str) -> str:
    """
    Generate the raw script text with section labels.
    """
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI client not available")

    system_prompt = """You are a script writer creating 30 to 60-second TikTok-length videos. You want them to go viral so they must be edgy and engaging.

CRITICAL REQUIREMENTS:
- The script MUST have exactly 4 sections: HOOK, BODY 1, BODY 2, and BODY 3
- The hook must be immediately ENGAGING and ENTERTAINING -- start with the strongest piece of content
- Each BODY section should build upon the previous one with engaging, surprising, or little-known facts
- Use a casual, conversational tone as if spoken by a charismatic person
- NO dashes, NO questions, NO call to action at the end
- Total script: 80-120 words (30-60 seconds when spoken)

OUTPUT FORMAT (you MUST include all 4 sections):

HOOK:
[2-3 sentences - the attention grabber]

BODY 1:
[2-3 sentences - first key point]

BODY 2:
[2-3 sentences - second key point]

BODY 3:
[2-3 sentences - closing point or twist]"""

    user_message = f"""Create a 30 to 60-second TikTok video script about: {topic}"""

    print(f"🎬 Generating script for topic: {topic}")

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
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

    script = script.strip().strip('"').strip("'").strip('`').strip()

    print(f"✓ Raw script generated ({len(script.split())} words)")

    return script


async def generate_scene_for_section(section: ScriptSection, scene_number: int) -> VideoScene:
    """
    Generate a single scene based on a script section.
    Optimized for Pexels/Pixabay API searches.
    """
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI client not available")

    system_prompt = """You are a stock video search expert for Pexels API. Create search queries that return good results.

PEXELS SEARCH QUERY RULES:
1. Use 1-3 simple, common English words
2. Queries can be BROAD like: Nature, Tigers, People, Ocean, Money, City
3. Or SPECIFIC like: Group of people working, Person typing laptop, Hands holding phone
4. Focus on FILMABLE subjects: people, places, objects, actions
5. NO abstract concepts (success, happiness, controversy, shocking)
6. NO adjectives that aren't visual (surprising, interesting, controversial)
7. NO proper nouns or brand names

WHAT WORKS WELL ON PEXELS:
- Single nouns: "money", "ocean", "city", "forest", "office", "phone"
- Person + action: "person walking", "woman talking", "man working"
- Object close-ups: "hands typing", "coffee cup", "phone screen"
- Locations: "city street", "beach sunset", "office interior"
- Nature: "ocean waves", "forest trees", "sky clouds"

WHAT DOESN'T WORK:
- Abstract: "success story", "financial freedom", "shocking truth"
- Too specific: "businessman in New York making deal"
- Adjectives: "amazing sunset", "controversial topic"

Pick the MOST VISUAL element from the script section."""

    user_message = f"""Script section ({section.name}):
"{section.text}"

Create a Pexels search query (1-3 words) for stock video footage.
Think: what would a videographer film to illustrate this?

Respond with ONLY a JSON object:
{{"description": "what the video shows", "search_query": "1-3 word query"}}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.5,
        max_tokens=150
    )

    response_content = response.choices[0].message.content
    if not response_content:
        # Fallback based on section type
        fallback_query = _get_fallback_query_for_section(section.name)
        return VideoScene(
            scene_number=scene_number,
            description=section.text[:100],
            search_keywords=fallback_query,
            search_query=fallback_query,
            section_name=section.name,
            word_start=section.word_start,
            word_end=section.word_end
        )

    try:
        # Parse JSON response
        json_match = re.search(r'\{[\s\S]*\}', response_content)
        if json_match:
            data = json.loads(json_match.group())
            search_query = sanitize_search_query(data.get('search_query', ''))

            return VideoScene(
                scene_number=scene_number,
                description=data.get('description', section.text[:100]),
                search_keywords=search_query,
                search_query=search_query,
                section_name=section.name,
                word_start=section.word_start,
                word_end=section.word_end
            )
    except Exception as e:
        print(f"⚠ Error parsing scene response: {e}")

    # Fallback based on section type
    fallback_query = _get_fallback_query_for_section(section.name)
    return VideoScene(
        scene_number=scene_number,
        description=section.text[:100],
        search_keywords=fallback_query,
        search_query=fallback_query,
        section_name=section.name,
        word_start=section.word_start,
        word_end=section.word_end
    )


def _get_fallback_query_for_section(section_name: str) -> str:
    """Get a generic fallback query based on section type."""
    fallbacks = {
        "HOOK": "person talking",
        "BODY1": "people working",
        "BODY2": "city street",
        "BODY3": "office interior",
        "BODY4": "nature landscape",
    }
    return fallbacks.get(section_name, "people")


async def generate_scenes_from_sections(section_list: List[ScriptSection]) -> List[VideoScene]:
    """
    Generate scenes for each script section.
    Each scene includes word boundary info for timing.
    We expect 4 sections: HOOK, BODY1, BODY2, BODY3
    """
    scenes: List[VideoScene] = []

    # Expected sections in order
    section_priority = ['HOOK', 'BODY1', 'BODY2', 'BODY3', 'BODY4']

    # Create a map for quick lookup
    section_map = {s.name: s for s in section_list}

    print(f"\n📝 Generating scenes for {len(section_list)} sections...")

    scene_num = 1
    for section_name in section_priority:
        if section_name in section_map:
            section = section_map[section_name]
            print(f"   [{scene_num}] {section_name} (words {section.word_start}-{section.word_end})...")

            scene = await generate_scene_for_section(section, scene_num)
            scenes.append(scene)

            print(f"       → search query: \"{scene.search_query}\"")
            scene_num += 1

    print(f"✅ Generated {len(scenes)} scenes\n")

    return scenes


async def generate_script(request: ScriptRequest) -> ScriptResponse:
    """
    Generate a video script with scenes and section timing info.
    1. Generate raw script with section labels
    2. Parse sections and calculate word boundaries
    3. Generate optimized scene searches for each section
    4. Return clean script (no labels) + scenes with word boundaries + section list
    """
    if not request.topic or not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required")

    topic = request.topic.strip()

    # Step 1: Generate the raw script with section labels
    raw_script = await generate_script_text(topic)

    # Step 2: Parse into sections with word boundaries
    _, section_list, clean_script = parse_script_sections(raw_script)

    if len(section_list) < 4:
        print(f"⚠ Only got {len(section_list)} sections, expected 4. Raw script:\n{raw_script[:500]}")

    # Step 3: Generate scenes from sections (with word boundary info)
    scenes = await generate_scenes_from_sections(section_list)

    print("✅ Script generation complete:")
    print(f"   Clean script: {len(clean_script.split())} words")
    print(f"   Sections: {len(section_list)}")
    print(f"   Scenes: {len(scenes)}")
    for scene in scenes:
        print(f"   - Scene {scene.scene_number} ({scene.section_name}): words {scene.word_start}-{scene.word_end}, query=\"{scene.search_query}\"")

    return ScriptResponse(
        script=clean_script,
        scenes=scenes,
        sections=section_list
    )
