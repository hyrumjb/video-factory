import re
import json
from typing import List, Dict, Tuple
from fastapi import HTTPException
from models import ScriptRequest, ScriptResponse, VideoScene, ScriptSection
from config import openai_client


def sanitize_search_query(query: str) -> str:
    """
    Remove proper nouns and optimize for stock video API searches.
    Pexels and Pixabay work best with simple, generic terms.
    """
    if not query:
        return query

    words = query.split()

    # Words to remove (articles, prepositions, etc.)
    stop_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'is', 'are', 'was', 'were'}

    # Common capitalized words that are generic (not names)
    common_generic_caps = {
        'basketball', 'football', 'soccer', 'tennis', 'baseball', 'hockey',
        'beach', 'ocean', 'mountain', 'city', 'street', 'park', 'forest',
        'sunset', 'sunrise', 'night', 'day', 'morning', 'evening',
        'person', 'people', 'crowd', 'team', 'player', 'athlete',
        'money', 'cash', 'dollar', 'business', 'office', 'computer'
    }

    generic_words: List[str] = []
    for word in words:
        word_lower = word.lower()

        # Skip stop words
        if word_lower in stop_words:
            continue

        # If word is capitalized and not in our common generic list, it's likely a name
        if word and word[0].isupper() and len(word) > 1 and word_lower not in common_generic_caps:
            continue

        generic_words.append(word_lower)

    # Limit to 4 words max for best API results
    result = ' '.join(generic_words[:4])

    # If we filtered out everything, use a fallback
    if not result:
        return ' '.join(w.lower() for w in words[:4] if w)

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
- The script should have a story-like structure with a hook at the start and then each idea shared after building upon the earlier ones.
- The hook must be immediately ENGAGING and ENTERTAINING -- start with the strongest piece of content that will draw in the viewer.
- The rest of the script should consist of facts that are engaging, surprising, little-known, or controversial.
- The response should use a casual, conversational tone as if spoken by a charismatic person and should contain NO dashes or questions.
- There should be NO call to action at the end, NO questions, and NO invitation to learn more.
- The final script must be between 80 and 120 words in length (about 30-60 seconds when spoken).

OUTPUT FORMAT:
- Return only the script text in the below format with no additional commentary or text:

HOOK:
[Engaging hook text here: 2-4 sentences]

BODY 1:
[Engaging connected body idea here: 2-4 sentences]

BODY 2:
[Another interesting body idea here: 2-4 sentences]

BODY 3:
[Something slightly controversial as well here: 2-4 sentences]

BODY 4:
[Another closing body idea here if needed: 2-4 sentences]"""

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

    system_prompt = """You are a stock video search expert. Your job is to create the BEST search query to find relevant stock footage on Pexels or Pixabay.

CRITICAL RULES FOR STOCK VIDEO SEARCHES:
- Use 2-4 simple, generic English words
- Focus on VISUAL elements that can be filmed (actions, objects, settings)
- NO abstract concepts, emotions, or ideas that can't be visually shown
- NO proper nouns, brand names, or specific people
- NO adjectives unless they describe something visual (like "dark", "bright", "slow")
- Think: what would a videographer actually film?

GOOD EXAMPLES:
- "person typing laptop" (visual action)
- "money falling slow motion" (filmable)
- "city skyline night" (visual scene)
- "hands holding phone" (specific visual)

BAD EXAMPLES:
- "controversial business practices" (too abstract)
- "shocking revelation" (not visual)
- "surprising facts about money" (can't film "surprising")"""

    user_message = f"""Script section ({section.name}):
"{section.text}"

Create a stock video search query (2-4 words) that shows something VISUAL related to this content.

Respond with ONLY a JSON object:
{{"description": "brief visual description", "search_query": "2-4 word search"}}"""

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
        # Fallback
        return VideoScene(
            scene_number=scene_number,
            description=section.text[:100],
            search_keywords="person talking camera",
            search_query="person talking camera",
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

    # Fallback
    return VideoScene(
        scene_number=scene_number,
        description=section.text[:100],
        search_keywords="person talking camera",
        search_query="person talking camera",
        section_name=section.name,
        word_start=section.word_start,
        word_end=section.word_end
    )


async def generate_scenes_from_sections(section_list: List[ScriptSection]) -> List[VideoScene]:
    """
    Generate scenes for each script section.
    Each scene includes word boundary info for timing.
    """
    scenes: List[VideoScene] = []

    # We need 4 scenes - use HOOK, BODY1, BODY2, and either BODY3 or BODY4
    # Priority: HOOK, BODY1, BODY2, BODY3 (or BODY4 if BODY3 missing)
    section_priority = ['HOOK', 'BODY1', 'BODY2', 'BODY3', 'BODY4']

    # Create a map for quick lookup
    section_map = {s.name: s for s in section_list}

    scene_num = 1
    for section_name in section_priority:
        if section_name in section_map and scene_num <= 4:
            section = section_map[section_name]
            print(f"   Generating scene {scene_num} from {section_name} (words {section.word_start}-{section.word_end})...")

            scene = await generate_scene_for_section(section, scene_num)
            scenes.append(scene)

            print(f"   ✓ Scene {scene_num}: \"{scene.search_query}\"")
            scene_num += 1

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
    sections_dict, section_list, clean_script = parse_script_sections(raw_script)

    if len(section_list) < 4:
        print(f"⚠ Only got {len(section_list)} sections, expected 5. Raw script:\n{raw_script[:500]}")

    # Step 3: Generate scenes from sections (with word boundary info)
    print("🎬 Generating scenes from script sections...")
    scenes = await generate_scenes_from_sections(section_list)

    print(f"\n✅ Script generation complete:")
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
