import re
import json
from fastapi import HTTPException
from models import ScriptRequest, ScriptResponse, VideoScene
from config import openai_client

def sanitize_search_query(query: str) -> str:
    """
    Remove proper nouns (capitalized words that look like names) from search queries 
    to make them stock-video-friendly. Keeps only generic terms.
    """
    if not query:
        return query
    
    words = query.split()
    # Common capitalized words that are generic (not names)
    common_generic_caps = {
        'basketball', 'football', 'soccer', 'tennis', 'baseball', 'hockey',
        'beach', 'ocean', 'mountain', 'city', 'street', 'park', 'forest',
        'sunset', 'sunrise', 'night', 'day', 'morning', 'evening',
        'person', 'people', 'crowd', 'team', 'player', 'athlete'
    }
    
    generic_words = []
    for word in words:
        word_lower = word.lower()
        # If word is capitalized and not in our common generic list, it's likely a name
        if word and word[0].isupper() and len(word) > 1 and word_lower not in common_generic_caps:
            # This looks like a proper noun/name, skip it
            continue
        # Keep the word (convert to lowercase for consistency)
        generic_words.append(word_lower)
    
    # If we filtered out everything, use a fallback - just lowercase everything
    if not generic_words:
        return ' '.join(w.lower() for w in words if w)
    
    return ' '.join(generic_words)

async def generate_script(request: ScriptRequest) -> ScriptResponse:
    """
    Generate a 30-second edgy, controversial video script with 4 scenes.
    """
    if not request.topic or not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required")

    # System prompt for 30-second edgy, controversial video script
    system_prompt = """You are a script writer creating 30-second TikTok-length videos. Make them as EDGY and CONTROVERSIAL as possible about the given topic.
CRITICAL REQUIREMENTS:
- Start with a CRISP HOOK that grabs attention immediately
- DO RESEARCH and provide ACTUAL FACTS and little-known information that people wouldn't know
- Every sentence must be a STATEMENT OF FACT - NO questions whatsoever
- Every sentence should build on the previous ones, creating a chain of controversial revelations
- NO dashes (use commas or periods instead - Google TTS doesn't pause properly for dashes)
- NO question marks - only statements of fact
- Spout EDGY, CONTROVERSIAL, and obscure facts/ideas about the topic
- NO call-to-action, NO "find out more", NO "learn anything new" - just pure edgy information dumping
- Pack it with crazy, interesting, provocative statements - make it as controversial as possible
- Script should be exactly 30 seconds when spoken at normal pace
- Write ONLY the raw spoken script - no quotes, no visual descriptions, no stage directions, no speaker names
- Output pure dialogue/narration only

Provide exactly 4 video scenes formatted as a JSON array. Each scene needs:
- scene_number: 1-4
- description: Visual content description (generic, stock-video-friendly)
- search_keywords: Comma-separated generic keywords
- search_query: 3-5 word generic search query for stock video APIs (NO specific names/brands/celebrities, NO filler words like 'a', 'the'). Example: "basketball player court" not "Ricky Rubio basketball"."""

    # Generate script using OpenAI with structured output
    user_message = f"""Create a 30-second TikTok-length video script about: {request.topic}.

DO RESEARCH and find ACTUAL FACTS and little-known information about this topic that people wouldn't know. Make it as EDGY and CONTROVERSIAL as possible.

CRITICAL FORMATTING RULES:
- Start with a crisp hook, then spout crazy, edgy, little-known facts for 30 seconds
- NO dashes - use commas or periods instead (dashes don't work with TTS)
- NO questions - every sentence must be a statement of fact
- Every sentence should build on the previous ones, revealing controversial facts
- No learning, no discovery, no call-to-action - just pure provocative information dumping
- Write ONLY statements of fact, one building on the next

Output format:
SCRIPT:
[Raw spoken script only - no descriptions, no formatting, no dashes, no questions]

SCENES:
[JSON array with exactly 4 scenes, each with scene_number (1-4), description, search_keywords (comma-separated), search_query (3-5 generic words, no names/brands/celebrities)]"""

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.8,
        max_tokens=1500
    )

    response_content = response.choices[0].message.content

    if not response_content:
        raise HTTPException(status_code=500, detail="Failed to generate script")

    # Debug: Print the raw response to see what we're getting
    print(f"📝 Raw AI response (first 1000 chars): {response_content[:1000]}")

    # Parse the response to extract script and scenes
    script = ""
    scenes = []

    # Try to split by SCRIPT: and SCENES: markers (case-insensitive)
    script_marker = "SCRIPT:" if "SCRIPT:" in response_content else "script:" if "script:" in response_content else None
    scenes_marker = "SCENES:" if "SCENES:" in response_content else "scenes:" if "scenes:" in response_content else None
    
    if script_marker and scenes_marker:
        # Split by scenes marker (case-insensitive)
        parts = re.split(r'SCENES?:', response_content, flags=re.IGNORECASE)
        if len(parts) >= 2:
            script = parts[0].replace("SCRIPT:", "").replace("script:", "").strip()
            # Remove any quotes that might wrap the script
            script = script.strip('"').strip("'").strip('`').strip()
            scenes_text = parts[1].strip()
        else:
            scenes_text = ""
            script = response_content
        
        # Try to extract JSON from the scenes text
        if scenes_text:
            try:
                scenes_data = []
                
                # First try: Find JSON array in the text - try multiple patterns
                json_match = re.search(r'\[.*?\]', scenes_text, re.DOTALL)
                if not json_match:
                    # Try to find JSON that might be wrapped in code blocks
                    code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', scenes_text, re.DOTALL)
                    if code_block_match:
                        json_match = code_block_match
                        json_str = code_block_match.group(1)
                    else:
                        # Try to find JSON array that spans multiple lines
                        json_match = re.search(r'\[[\s\S]*?\]', scenes_text)
                
                if json_match:
                    json_str = json_match.group() if hasattr(json_match, 'group') else json_match
                    if isinstance(json_str, re.Match):
                        json_str = json_str.group()
                    try:
                        scenes_data = json.loads(json_str)
                        print(f"✓ Successfully parsed {len(scenes_data)} scenes from JSON")
                    except json.JSONDecodeError as e:
                        print(f"⚠ JSON parse error: {e}")
                        print(f"   JSON string: {json_str[:200]}")
                        scenes_data = []
                else:
                    # Second try: Parse numbered list format
                    scenes_data = []
                    
                    # Split by numbered items (1., 2., 3., etc.)
                    scene_blocks = re.split(r'^\d+\.\s*\n?', scenes_text, flags=re.MULTILINE)
                    
                    for block in scene_blocks:
                        if not block.strip():
                            continue
                        
                        scene_obj = {}
                        # Extract description
                        desc_match = re.search(r'[-•]\s*description:\s*(.+?)(?:\n|$)', block, re.IGNORECASE | re.MULTILINE)
                        if desc_match:
                            scene_obj['description'] = desc_match.group(1).strip()
                        else:
                            desc_match = re.search(r'description:\s*(.+?)(?:\n|$)', block, re.IGNORECASE | re.MULTILINE)
                            if desc_match:
                                scene_obj['description'] = desc_match.group(1).strip()
                        
                        # Extract search_keywords
                        keywords_match = re.search(r'[-•]\s*search_keywords:\s*(.+?)(?:\n|$)', block, re.IGNORECASE | re.MULTILINE)
                        if keywords_match:
                            scene_obj['search_keywords'] = keywords_match.group(1).strip()
                        else:
                            keywords_match = re.search(r'search_keywords:\s*(.+?)(?:\n|$)', block, re.IGNORECASE | re.MULTILINE)
                            if keywords_match:
                                scene_obj['search_keywords'] = keywords_match.group(1).strip()
                        
                        # Extract search_query
                        query_match = re.search(r'[-•]\s*search_query:\s*(.+?)(?:\n|$)', block, re.IGNORECASE | re.MULTILINE)
                        if query_match:
                            scene_obj['search_query'] = query_match.group(1).strip()
                        else:
                            query_match = re.search(r'search_query:\s*(.+?)(?:\n|$)', block, re.IGNORECASE | re.MULTILINE)
                            if query_match:
                                scene_obj['search_query'] = query_match.group(1).strip()
                        
                        # Only add if we have at least a description
                        if scene_obj.get('description'):
                            if 'scene_number' not in scene_obj:
                                scene_obj['scene_number'] = len(scenes_data) + 1
                            scenes_data.append(scene_obj)
                    
                    # If we still don't have scenes, try line-by-line parsing
                    if not scenes_data:
                        lines = scenes_text.split('\n')
                        current_scene = {}
                        scene_num = 1
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Check if this is a new scene (starts with number)
                            num_match = re.match(r'^(\d+)\.', line)
                            if num_match:
                                if current_scene.get('description'):
                                    current_scene['scene_number'] = scene_num
                                    scenes_data.append(current_scene)
                                    scene_num += 1
                                current_scene = {}
                                continue
                            
                            # Parse key-value pairs with optional dash/bullet
                            if ':' in line:
                                line_clean = re.sub(r'^[-•]\s*', '', line)
                                key, value = line_clean.split(':', 1)
                                key = key.strip().lower().replace('-', '').replace('_', '')
                                value = value.strip()
                                
                                if 'description' in key:
                                    current_scene['description'] = value
                                elif 'searchkeywords' in key or 'keywords' in key:
                                    current_scene['search_keywords'] = value
                                elif 'searchquery' in key or 'query' in key:
                                    current_scene['search_query'] = value
                        
                        # Add last scene
                        if current_scene.get('description'):
                            current_scene['scene_number'] = scene_num
                            scenes_data.append(current_scene)
                
                # Process scenes_data if we found any
                if scenes_data:
                    # Ensure we have a list
                    if not isinstance(scenes_data, list):
                        scenes_data = [scenes_data]
                    
                    # Ensure each scene has all required fields
                    for idx, scene_data in enumerate(scenes_data):
                        # Set scene_number if missing
                        if 'scene_number' not in scene_data:
                            scene_data['scene_number'] = idx + 1
                        
                        if 'search_query' not in scene_data or not scene_data.get('search_query'):
                            # Generate search_query from description if missing
                            desc = scene_data.get('description', '')
                            if desc:
                                words = desc.split()[:5]
                                scene_data['search_query'] = ' '.join(words)
                        else:
                            # Sanitize the search query to remove proper nouns
                            scene_data['search_query'] = sanitize_search_query(scene_data.get('search_query', ''))
                        if 'search_keywords' not in scene_data or not scene_data.get('search_keywords'):
                            scene_data['search_keywords'] = scene_data.get('search_query', '')
                    scenes = [VideoScene(**scene) for scene in scenes_data]
                else:
                    print(f"Could not parse scenes from text. First 500 chars: {scenes_text[:500]}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Scenes text snippet: {scenes_text[:500] if 'scenes_text' in locals() else 'N/A'}...")
                scenes = []
            except Exception as e:
                print(f"Error parsing scenes: {e}")
                import traceback
                traceback.print_exc()
                print(f"Scenes text snippet: {scenes_text[:500] if 'scenes_text' in locals() else 'N/A'}...")
                scenes = []
    elif script_marker:
        # Only script marker found, try to extract script
        parts = re.split(r'SCRIPT?:', response_content, flags=re.IGNORECASE)
        script = parts[-1].strip().strip('"').strip("'").strip('`').strip()
        scenes_text = ""
    else:
        # Fallback: use entire response as script if format is wrong
        script = response_content.strip()
        scenes_text = ""
        # Try to find JSON array anywhere in the response
        json_match = re.search(r'\[.*?\]', response_content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group()
                scenes_data = json.loads(json_str)
                print(f"✓ Found JSON array in response, parsed {len(scenes_data)} scenes")
                for scene_data in scenes_data:
                    try:
                        scene = VideoScene(**scene_data)
                        scenes.append(scene)
                    except Exception as e:
                        print(f"⚠ Error creating scene object: {e}")
            except json.JSONDecodeError:
                print(f"⚠ Could not parse JSON array from response")

    if not script:
        raise HTTPException(status_code=500, detail="Failed to generate script")

    # Debug: Print what we found
    print(f"📊 Parsed {len(scenes)} scenes from response")
    if len(scenes) > 0:
        print(f"   First scene: {scenes[0].dict() if hasattr(scenes[0], 'dict') else scenes[0]}")

    # Validate scenes - if we don't have 4 valid scenes, raise an error instead of using placeholders
    if len(scenes) != 4:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate scenes. Expected 4 scenes but got {len(scenes)}. The AI response format may be incorrect. Check backend logs for details."
        )

    # Validate each scene has required fields and sanitize search queries
    for scene in scenes:
        if not scene.search_query:
            # Generate search_query from description if missing
            words = scene.description.split()[:5]
            scene.search_query = ' '.join(words)
        else:
            # Sanitize the search query to remove proper nouns
            scene.search_query = sanitize_search_query(scene.search_query)
        if not scene.search_keywords:
            scene.search_keywords = scene.search_query

    return ScriptResponse(script=script, scenes=scenes)
