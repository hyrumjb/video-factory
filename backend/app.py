from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from openai import OpenAI
import os
import base64
import re
import json
import requests
import tempfile
import subprocess
from typing import Optional, List
from urllib.parse import quote_plus
from pathlib import Path
from dotenv import load_dotenv

# Get ffmpeg executable path from imageio-ffmpeg
try:
    import imageio_ffmpeg
    FFMPEG_EXECUTABLE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    # Fallback to system ffmpeg if imageio-ffmpeg not available
    FFMPEG_EXECUTABLE = 'ffmpeg'
    FFPROBE_EXECUTABLE = 'ffprobe'
else:
    # Use ffprobe from the same location
    ffmpeg_dir = os.path.dirname(FFMPEG_EXECUTABLE)
    FFPROBE_EXECUTABLE = os.path.join(ffmpeg_dir, 'ffprobe' + ('.exe' if os.name == 'nt' else ''))

# Try to import Google TTS, but make it optional
# Use v1beta1 for timepointing support (required for caption timing)
try:
    from google.cloud import texttospeech_v1beta1 as texttospeech
    TTS_AVAILABLE = True
except ImportError:
    # Fallback to stable v1 if beta not available (no timepointing)
    try:
        from google.cloud import texttospeech
        TTS_AVAILABLE = True
        print("⚠ Warning: Using stable v1 API - timepointing not available. Install latest google-cloud-texttospeech for caption timing support.")
    except ImportError:
        TTS_AVAILABLE = False
        print("⚠ Warning: google-cloud-texttospeech not installed. TTS functionality will be disabled.")

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Google TTS client
tts_client = None
google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Try to find credentials file if not set in env
if not google_creds_path:
    # Check common locations
    possible_paths = [
        "./google-credentials.json",
        "./credentials.json",
        "../google-credentials.json",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(path)
            google_creds_path = os.path.abspath(path)
            break

if TTS_AVAILABLE:
    try:
        if google_creds_path and os.path.exists(google_creds_path):
            tts_client = texttospeech.TextToSpeechClient()
            print(f"✓ Google TTS client initialized with credentials: {google_creds_path}")
        else:
            print("⚠ Warning: Google TTS credentials not found. TTS will not work.")
            print("   To enable TTS, download credentials from Google Cloud Console and:")
            print("   1. Save the JSON file as 'google-credentials.json' in the backend folder")
            print("   2. Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            tts_client = None
    except Exception as e:
        print(f"⚠ Warning: Could not initialize Google TTS client: {e}")
        print("   TTS functionality will be disabled until credentials are configured.")
        tts_client = None
else:
    tts_client = None
    print("⚠ Google TTS package not installed. Install with: pip install google-cloud-texttospeech")

# Request models
class ScriptRequest(BaseModel):
    topic: str

class TTSRequest(BaseModel):
    text: str
    voice_name: Optional[str] = "en-US-Neural2-H"  # Default voice

class MultiVoiceTTSRequest(BaseModel):
    text: str
    voices: Optional[List[str]] = None  # If None, use all Neural2 voices

class VideoSearchRequest(BaseModel):
    search_query: str

class VideoItem(BaseModel):
    id: str
    url: str
    thumbnail_url: str
    source: str  # "pexels" or "pixabay"
    duration: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

class VideoSearchResponse(BaseModel):
    videos: list[VideoItem]

class CompileVideoRequest(BaseModel):
    video_urls: List[str]  # List of video URLs to combine
    audio_url: str  # Base64 audio data URL
    script: str  # Script text for captions
    scene_durations: Optional[List[float]] = None  # Duration for each scene in seconds
    voice_name: Optional[str] = "en-US-Neural2-H"  # Voice used for TTS

class CompileVideoResponse(BaseModel):
    video_url: str  # Base64 encoded video or URL to final video

# Response models
class VideoScene(BaseModel):
    scene_number: int
    description: str
    search_keywords: str
    search_query: str  # Optimized 3-5 word query for stock video APIs

class ScriptResponse(BaseModel):
    script: str
    scenes: list[VideoScene]

class TTSResponse(BaseModel):
    audio_url: str

class VoiceAudio(BaseModel):
    voice_name: str
    audio_url: str
    gender: str

class MultiVoiceTTSResponse(BaseModel):
    voices: List[VoiceAudio]

@app.get("/")
def read_root():
    return {"message": "Video Factory API"}

@app.get("/api/list-voices")
async def list_voices(language_code: str = "en-US"):
    """
    List all available Google TTS voices for a given language code.
    Useful for finding voice names to use.
    """
    try:
        voices = list_available_voices(language_code)
        return {"voices": voices, "language_code": language_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing voices: {str(e)}")

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

@app.post("/api/generate-script", response_model=ScriptResponse)
async def generate_script(request: ScriptRequest):
    try:
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
            import re
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
                        # Pattern: "1. \n- description: ...\n- search_keywords: ...\n- search_query: ..."
                        scenes_data = []
                        
                        # Split by numbered items (1., 2., 3., etc.) - match number followed by newline or space
                        scene_blocks = re.split(r'^\d+\.\s*\n?', scenes_text, flags=re.MULTILINE)
                        
                        for block in scene_blocks:
                            if not block.strip():
                                continue
                            
                            scene_obj = {}
                            # Extract description - match "- description:" or "description:" with optional dash
                            desc_match = re.search(r'[-•]\s*description:\s*(.+?)(?:\n|$)', block, re.IGNORECASE | re.MULTILINE)
                            if desc_match:
                                scene_obj['description'] = desc_match.group(1).strip()
                            else:
                                # Try without dash
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
                                # Set scene_number if not present
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
                                    # Remove leading dash/bullet if present
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
                    # Only log parsing errors, not debug info
                    print(f"Error parsing scenes: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"Scenes text snippet: {scenes_text[:500] if 'scenes_text' in locals() else 'N/A'}...")
                    scenes = []
        elif script_marker:
            # Only script marker found, try to extract script
            import re
            parts = re.split(r'SCRIPT?:', response_content, flags=re.IGNORECASE)
            script = parts[-1].strip().strip('"').strip("'").strip('`').strip()
            scenes_text = ""
        else:
            # Fallback: use entire response as script if format is wrong
            script = response_content.strip()
            scenes_text = ""
            # Try to find JSON array anywhere in the response
            import re
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
        else:
            print(f"⚠ No scenes found! Response content length: {len(response_content)}")
            print(f"   Response snippet: {response_content[:500]}")

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

    except Exception as e:
        # Handle OpenAI API errors
        if hasattr(e, 'status_code'):
            raise HTTPException(status_code=e.status_code, detail=str(e))
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

def list_available_voices(language_code: str = "en-US"):
    """
    List all available voices for a given language code.
    Useful for finding voice names to use.
    """
    if not TTS_AVAILABLE or not tts_client:
        return []
    
    try:
        voices = tts_client.list_voices(language_code=language_code)
        voice_list = []
        for voice in voices.voices:
            voice_list.append({
                'name': voice.name,
                'ssml_gender': voice.ssml_gender.name,
                'natural_sample_rate_hertz': voice.natural_sample_rate_hertz
            })
        return voice_list
    except Exception as e:
        print(f"Error listing voices: {e}")
        return []

def generate_tts_with_timing(text: str, voice_name: str = "en-US-Neural2-H"):
    """
    Generate TTS audio with word-level timing information using SSML marks.
    Returns audio content and timepoints for subtitle generation.
    
    Following Google's recommended approach: insert <mark/> tags for EACH WORD
    to get accurate word-level timing for subtitles.
    
    Args:
        text: Text to synthesize
        voice_name: Voice name to use (default: "en-US-Neural2-H")
                    Options include:
                    - en-US-Neural2-H (female, default)
                    - en-US-Neural2-D (male)
                    - en-US-Neural2-J (male)
                    - en-US-Neural2-A (male)
                    - en-US-Neural2-C (female)
                    - en-US-Neural2-E (female)
                    - en-US-Neural2-G (female)
                    - en-US-Neural2-H (female)
                    - en-US-Neural2-I (male)
    """
    if not TTS_AVAILABLE or not tts_client:
        return None, None
    
    try:
        import re
        import html
        
        # Split text into words while preserving punctuation and spacing
        # This regex splits on word boundaries but keeps punctuation attached
        words = re.findall(r'\S+|\s+', text)  # Matches words (non-whitespace) or whitespace sequences
        
        # Build SSML with mark tags for EACH WORD (as per Google's recommendation)
        # Also add natural breaks between sentences
        ssml_parts = ['<speak>']
        mark_count = 0
        
        for i, word in enumerate(words):
            if word.strip():  # Only mark actual words, not pure whitespace
                # Escape HTML entities but preserve SSML structure
                escaped_word = html.escape(word)
                ssml_parts.append(escaped_word)
                # Add mark after each word for word-level timing
                ssml_parts.append(f'<mark name="{mark_count}"/>')
                mark_count += 1
                
                # Check if this word ends a sentence (ends with . ! ?)
                # Add a natural break after sentence-ending punctuation
                if re.search(r'[.!?]$', word):
                    # Add a light break (0.3 seconds) for natural sentence pauses
                    # Check if there's more content after this sentence
                    remaining_text = ''.join(words[i+1:])
                    if remaining_text.strip():  # Only add break if there's more content
                        ssml_parts.append('<break time="0.3s"/>')
            else:
                # Preserve whitespace as-is
                ssml_parts.append(word)
        
        ssml_parts.append('</speak>')
        ssml_text = ''.join(ssml_parts)
        
        # Configure synthesis with timing enabled
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
        
        # Determine gender from voice name (F = female, others typically male)
        ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-F") or voice_name.endswith("-C") or voice_name.endswith("-E") or voice_name.endswith("-G") or voice_name.endswith("-H") else texttospeech.SsmlVoiceGender.MALE
        
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice_name,
            ssml_gender=ssml_gender,
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=0.97,
            pitch=0.0,
        )
        
        # Generate speech with timepointing enabled using SynthesizeSpeechRequest object
        # This is the crucial part - must use the request object format
        timepoint_type = texttospeech.SynthesizeSpeechRequest.TimepointType.SSML_MARK
        
        request = texttospeech.SynthesizeSpeechRequest(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
            enable_time_pointing=[timepoint_type]
        )
        
        response = tts_client.synthesize_speech(request=request)
        
        # Extract timepoints - these now contain word-level timing
        timepoints = response.timepoints if hasattr(response, 'timepoints') else None
        
        if timepoints:
            print(f"✓ Generated audio with {len(timepoints)} word-level timepoints")
        
        return response.audio_content, timepoints
        
    except (AttributeError, TypeError) as e:
        print(f"Timepointing not available: {e}")
        print("Make sure you're using google-cloud-texttospeech with v1beta1 support")
        raise
    except Exception as e:
        print(f"Error in generate_tts_with_timing: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to simple TTS without timing
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            # Determine gender from voice name
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-F") or voice_name.endswith("-C") or voice_name.endswith("-E") or voice_name.endswith("-G") or voice_name.endswith("-H") else texttospeech.SsmlVoiceGender.MALE
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice_name,
                ssml_gender=ssml_gender,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0,
            )
            response = tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            return response.audio_content, None
        except Exception as fallback_error:
            print(f"Fallback TTS also failed: {fallback_error}")
            raise

@app.post("/api/generate-tts")
async def generate_tts(request: TTSRequest):
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        if not TTS_AVAILABLE or not tts_client:
            raise HTTPException(status_code=503, detail="Google TTS is not available. Please install google-cloud-texttospeech package.")

        # Try to use the new function with timing, but fallback to simple TTS if it fails
        voice_name = request.voice_name or "en-US-Neural2-H"
        try:
            audio_content, timepoints = generate_tts_with_timing(request.text.strip(), voice_name=voice_name)
            if not audio_content:
                raise ValueError("No audio content returned")
        except Exception as timing_error:
            print(f"Warning: TTS with timing failed: {timing_error}, falling back to simple TTS")
            # Fallback to simple TTS without timing
            synthesis_input = texttospeech.SynthesisInput(text=request.text.strip())
            # Use selected voice for fallback
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-F") or voice_name.endswith("-C") or voice_name.endswith("-E") or voice_name.endswith("-G") or voice_name.endswith("-H") else texttospeech.SsmlVoiceGender.MALE
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice_name,
                ssml_gender=ssml_gender,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0,
            )
            response = tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            audio_content = response.audio_content

        # Return audio as base64 encoded string
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        return TTSResponse(audio_url=f"data:audio/wav;base64,{audio_base64}")

    except Exception as e:
        print(f"TTS generation error: {e}")
        import traceback
        traceback.print_exc()
        if hasattr(e, 'status_code'):
            raise HTTPException(status_code=e.status_code, detail=str(e))
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/api/generate-tts-multi-voice", response_model=MultiVoiceTTSResponse)
async def generate_tts_multi_voice(request: MultiVoiceTTSRequest):
    """
    Generate TTS audio for multiple voices. Useful for testing different voice options.
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        if not TTS_AVAILABLE or not tts_client:
            raise HTTPException(status_code=503, detail="Google TTS is not available. Please install google-cloud-texttospeech package.")

        # Default to all Neural2 voices if not specified
        if request.voices is None or len(request.voices) == 0:
            voices_to_generate = [
                "en-US-Neural2-H",
                "en-US-Neural2-D",
                "en-US-Neural2-J",
                "en-US-Neural2-A",
                "en-US-Neural2-C",
                "en-US-Neural2-E",
                "en-US-Neural2-G",
                "en-US-Neural2-H",
                "en-US-Neural2-I",
            ]
        else:
            voices_to_generate = request.voices

        voice_audios = []
        
        for voice_name in voices_to_generate:
            try:
                print(f"🎤 Generating TTS with voice: {voice_name}")
                
                # Determine gender from voice name
                ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-F") or voice_name.endswith("-C") or voice_name.endswith("-E") or voice_name.endswith("-G") or voice_name.endswith("-H") else texttospeech.SsmlVoiceGender.MALE
                gender_str = "Female" if ssml_gender == texttospeech.SsmlVoiceGender.FEMALE else "Male"
                
                # Generate TTS (without timing for faster generation in testing)
                synthesis_input = texttospeech.SynthesisInput(text=request.text.strip())
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US",
                    name=voice_name,
                    ssml_gender=ssml_gender,
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=1.0,
                    pitch=0.0,
                )
                
                response = tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                
                # Convert to base64
                audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
                audio_url = f"data:audio/wav;base64,{audio_base64}"
                
                voice_audios.append(VoiceAudio(
                    voice_name=voice_name,
                    audio_url=audio_url,
                    gender=gender_str
                ))
                
                print(f"✓ Generated audio for {voice_name}")
                
            except Exception as voice_error:
                print(f"⚠ Error generating audio for {voice_name}: {voice_error}")
                # Continue with other voices even if one fails
                continue
        
        if len(voice_audios) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate audio for any voices")
        
        return MultiVoiceTTSResponse(voices=voice_audios)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/api/search-videos", response_model=VideoSearchResponse)
async def search_videos(request: VideoSearchRequest):
    """
    Search for videos from Pexels and Pixabay based on optimized search query.
    Returns videos from both sources combined.
    """
    try:
        if not request.search_query or not request.search_query.strip():
            raise HTTPException(status_code=400, detail="Search query is required")
        
        all_videos = []
        search_query = request.search_query.strip()
        
        # Get API keys from environment
        pexels_api_key = os.getenv("PEXELS_API_KEY")
        pixabay_api_key = os.getenv("PIXABAY_API_KEY")
        
        # Search Pexels
        if pexels_api_key:
            try:
                
                pexels_url = "https://api.pexels.com/videos/search"
                headers = {"Authorization": pexels_api_key}
                params = {
                    "query": search_query,
                    "per_page": 1,  # Only need 1 video per source
                    "orientation": "vertical"  # Good for short-form content
                }
                
                pexels_response = requests.get(pexels_url, headers=headers, params=params, timeout=10)
                
                if pexels_response.status_code == 200:
                    pexels_data = pexels_response.json()
                    # Limit to 1 video per source
                    videos_list = pexels_data.get("videos", [])
                    video_url = None
                    
                    # First, try to find videos that are vertical (check video files)
                    for video in videos_list:
                        video_files = video.get("video_files", [])
                        # Look for vertical videos (height > width)
                        vertical_videos = [vf for vf in video_files if vf.get("height", 0) > vf.get("width", 0)]
                        if vertical_videos:
                            # Get the best quality vertical video
                            for vf in sorted(vertical_videos, key=lambda x: x.get("height", 0), reverse=True):
                                if vf.get("link"):
                                    video_url = vf["link"]
                                    break
                            if video_url:
                                video = video  # Keep reference to the selected video
                                break
                    
                    # If still no vertical video found, check video dimensions from API response
                    if not video_url:
                        for video in videos_list:
                            # Some APIs provide width/height at video level
                            video_width = video.get("width", 0)
                            video_height = video.get("height", 0)
                            if video_height > video_width:
                                video_files = video.get("video_files", [])
                                # Get best quality video file
                                for vf in sorted(video_files, key=lambda x: x.get("height", 0), reverse=True):
                                    if vf.get("link"):
                                        video_url = vf["link"]
                                        break
                                if video_url:
                                    break
                    
                    # Final fallback: use first video if no vertical found
                    if not video_url and videos_list:
                        video = videos_list[0]
                        video_files = video.get("video_files", [])
                        for vf in sorted(video_files, key=lambda x: x.get("height", 0), reverse=True):
                            if vf.get("link"):
                                video_url = vf["link"]
                                break
                        
                        if video_url:
                            all_videos.append(VideoItem(
                                id=f"pexels_{video.get('id')}",
                                url=video_url,
                                thumbnail_url=video.get("image", ""),
                                source="pexels",
                                duration=video.get("duration"),
                                width=video_files[0].get("width") if video_files else None,
                                height=video_files[0].get("height") if video_files else None
                            ))
            except Exception as e:
                print(f"Error fetching Pexels videos: {e}")
        
        # Search Pixabay
        if pixabay_api_key:
            try:
                # Try multiple search strategies
                search_queries = [search_query]
                
                # If original query has multiple words, try with just the first 2-3 words
                words = search_query.split()
                if len(words) > 2:
                    search_queries.append(' '.join(words[:2]))  # First 2 words
                if len(words) > 1:
                    search_queries.append(words[0])  # Just first word
                
                pixabay_url = "https://pixabay.com/api/videos/"
                
                for query_attempt in search_queries:
                    # URL encode the search query
                    encoded_query = quote_plus(query_attempt)
                    
                    params = {
                        "key": pixabay_api_key,
                        "q": encoded_query,
                        "video_type": "all",
                        "per_page": 3,  # Request 3 but only use 1
                        "safesearch": "true",
                        "order": "popular"  # Get popular videos first
                    }
                    
                    pixabay_response = requests.get(pixabay_url, params=params, timeout=10)
                    
                    if pixabay_response.status_code == 200:
                        pixabay_data = pixabay_response.json()
                        
                        # Check for errors in response
                        if "error" in pixabay_data:
                            print(f"Pixabay API error in response: {pixabay_data.get('error')}")
                            continue
                        
                        # Limit to 1 video per source
                        hits = pixabay_data.get("hits", [])
                        total_hits = pixabay_data.get("totalHits", 0)
                        
                        if hits:
                            # First, try to find a vertical video (height > width)
                            vertical_video = None
                            for video in hits:
                                videos_obj = video.get("videos", {})
                                # Check if any video size is vertical
                                is_vertical = False
                                for size_key in ["large", "medium", "small", "tiny"]:
                                    if size_key in videos_obj:
                                        v_width = videos_obj[size_key].get("width", 0)
                                        v_height = videos_obj[size_key].get("height", 0)
                                        if v_height > v_width:
                                            is_vertical = True
                                            break
                                
                                if is_vertical:
                                    vertical_video = video
                                    break
                            
                            # Use vertical video if found, otherwise use first video
                            video = vertical_video if vertical_video else hits[0]
                            
                            # Get the video URL (prefer medium quality, then small, then tiny)
                            video_url = None
                            thumbnail_url = ""
                            videos_obj = video.get("videos", {})
                            
                            # Prefer vertical video files if available
                            vertical_video_file = None
                            for size_key in ["medium", "small", "tiny", "large"]:
                                if size_key in videos_obj:
                                    vf = videos_obj[size_key]
                                    if vf.get("url") and vf.get("height", 0) > vf.get("width", 0):
                                        vertical_video_file = vf
                                        break
                            
                            if vertical_video_file:
                                video_url = vertical_video_file.get("url")
                                thumbnail_url = vertical_video_file.get("thumbnail", "")
                            elif "medium" in videos_obj and videos_obj["medium"].get("url"):
                                video_url = videos_obj["medium"].get("url")
                                thumbnail_url = videos_obj["medium"].get("thumbnail", "")
                            elif "small" in videos_obj and videos_obj["small"].get("url"):
                                video_url = videos_obj["small"].get("url")
                                thumbnail_url = videos_obj["small"].get("thumbnail", "")
                            elif "tiny" in videos_obj and videos_obj["tiny"].get("url"):
                                video_url = videos_obj["tiny"].get("url")
                                thumbnail_url = videos_obj["tiny"].get("thumbnail", "")
                            
                            if video_url:
                                all_videos.append(VideoItem(
                                    id=f"pixabay_{video.get('id')}",
                                    url=video_url,
                                    thumbnail_url=thumbnail_url,
                                    source="pixabay",
                                    duration=video.get("duration"),
                                    width=videos_obj.get("medium", {}).get("width") if "medium" in videos_obj else None,
                                    height=videos_obj.get("medium", {}).get("height") if "medium" in videos_obj else None
                                ))
                                # Success! Break out of loop
                                break
                        else:
                            # No hits for this query attempt, try next one
                            if query_attempt == search_queries[-1]:
                                # Last attempt, log it
                                print(f"Pixabay: No hits found for any query variant. Total available: {total_hits}")
                    elif pixabay_response.status_code == 429:
                        print(f"Pixabay: Rate limit exceeded")
                        break  # Don't retry if rate limited
                    else:
                        print(f"Pixabay API error: {pixabay_response.status_code} - {pixabay_response.text[:200]}")
                        # Try next query variant
                        continue
            except Exception as e:
                # Log errors so we can debug issues
                print(f"Error fetching Pixabay videos: {e}")
                import traceback
                traceback.print_exc()
        
        if not all_videos and not pexels_api_key and not pixabay_api_key:
            raise HTTPException(
                status_code=503, 
                detail="Video search APIs not configured. Please set PEXELS_API_KEY and/or PIXABAY_API_KEY in your environment variables."
            )
        
        # Return all videos (max 2 per scene: 1 from Pexels, 1 from Pixabay)
        return VideoSearchResponse(videos=all_videos)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video search failed: {str(e)}")

def create_srt_from_timepoints(script: str, timepoints: list, audio_duration: float, style: str = "3words") -> str:
    """
    Create SRT subtitle file from script text and Google TTS word-level timepoints.
    
    Args:
        script: The script text
        timepoints: List of timepoints from Google TTS
        audio_duration: Total audio duration in seconds
        style: Caption style - "1word", "3words", "5words", or "varying"
    """
    import re
    
    if not timepoints or len(timepoints) == 0:
        # Fallback: create simple subtitles with even timing
        return create_srt_fallback(script, audio_duration)
    
    # CRITICAL: Split script into words EXACTLY the same way as in generate_tts_with_timing
    # This must match exactly or timepoint indices will be wrong
    words = re.findall(r'\S+|\s+', script)  # Same regex as in generate_tts_with_timing
    # Filter to get only actual words (not pure whitespace) - MUST match TTS logic exactly
    word_list = []
    for item in words:
        if item.strip():  # Only add non-whitespace items
            word_list.append(item.strip())
    
    # Debug: Verify word count matches timepoint count
    if len(timepoints) != len(word_list):
        print(f"⚠ WARNING: Word count mismatch! Words: {len(word_list)}, Timepoints: {len(timepoints)}")
        print(f"   First 10 words: {word_list[:10]}")
        print(f"   First 10 timepoints: {[(tp.mark_name, tp.time_seconds) for tp in timepoints[:10]]}")
    
    # Create timepoint map: mark_name -> time_seconds
    timepoint_map = {}
    for tp in timepoints:
        mark_name = tp.mark_name
        # Mark names are numeric strings like "0", "1", "2", etc.
        try:
            mark_num = int(mark_name)
            timepoint_map[mark_num] = tp.time_seconds
        except (ValueError, AttributeError):
            continue
    
    # Debug: Log timepoint map for first few words
    print(f"   Timepoint map sample: {dict(list(timepoint_map.items())[:10])}")
    print(f"   Word list sample: {word_list[:10]}")
    
    # Determine grouping strategy based on style
    if style == "1word":
        max_words_per_subtitle = 1
        min_words_per_subtitle = 1
    elif style == "5words":
        max_words_per_subtitle = 5
        min_words_per_subtitle = 5
    elif style == "varying":
        # Smart varying: try to fit 5-7 words or ~42 chars per line, max 2 lines
        max_words_per_subtitle = 7
        min_words_per_subtitle = 3
        max_chars_per_line = 42
    else:  # Default: 3 words
        max_words_per_subtitle = 3
        min_words_per_subtitle = 3
    
    # Group words into subtitle lines
    srt_lines = []
    subtitle_index = 1
    
    i = 0
    while i < len(word_list):
        # CRITICAL: For the FIRST subtitle, it MUST start at word 0
        if subtitle_index == 1 and i != 0:
            print(f"⚠⚠⚠ CRITICAL ERROR: First subtitle starting at word index {i} instead of 0!")
            print(f"   Forcing first subtitle to start at word 0")
            i = 0  # Force to start at word 0
        
        # Ensure we don't go out of bounds
        if i >= len(word_list):
            break  # No more words
        
        if style == "varying":
            # Smart varying length - build subtitle based on character count
            words_for_subtitle = []
            char_count = 0
            line_breaks = 0
            start_word_idx = i
            
            while i < len(word_list) and len(words_for_subtitle) < max_words_per_subtitle * 2:
                word = word_list[i]
                word_len = len(word)
                
                # Check if adding this word would exceed line length
                if char_count + word_len > max_chars_per_line and len(words_for_subtitle) >= min_words_per_subtitle:
                    # Start a second line if we haven't already
                    if line_breaks == 0:
                        words_for_subtitle.append('\n')
                        char_count = 0
                        line_breaks = 1
                    else:
                        # Already have 2 lines, break
                        break
                
                words_for_subtitle.append(word)
                char_count += word_len
                i += 1
            
            # Ensure we have at least min_words
            if len([w for w in words_for_subtitle if w != '\n']) < min_words_per_subtitle:
                while i < len(word_list) and len([w for w in words_for_subtitle if w != '\n']) < min_words_per_subtitle:
                    words_for_subtitle.append(word_list[i])
                    i += 1
            
            last_word_idx = i - 1
            subtitle_text = ''.join(words_for_subtitle).strip()
        else:
            # Fixed word count per subtitle
            words_for_subtitle = word_list[i:min(i + max_words_per_subtitle, len(word_list))]
            if not words_for_subtitle:
                break  # No words to add
            start_word_idx = i
            last_word_idx = i + len(words_for_subtitle) - 1
            # Ensure last_word_idx is valid
            if last_word_idx >= len(word_list):
                last_word_idx = len(word_list) - 1
                words_for_subtitle = word_list[start_word_idx:last_word_idx+1]
            subtitle_text = ' '.join(words_for_subtitle)
            i += len(words_for_subtitle)
        
        # CRITICAL: Google TTS mark timing works like this:
        # - mark "0" = time when word 0 FINISHES (and word 1 STARTS)
        # - mark "1" = time when word 1 FINISHES (and word 2 STARTS)
        # So for words 0, 1, 2:
        #   - Start time = 0.0 (first word) OR mark[i-1] (subsequent words)
        #   - End time = mark[last_word_idx] (when last word finishes)
        # CAPTIONS APPEAR 0.1 SECONDS EARLY for better readability
        
        # CRITICAL: Ensure word indices are valid BEFORE calculating timing
        if start_word_idx >= len(word_list) or last_word_idx >= len(word_list):
            print(f"⚠ ERROR: Invalid word indices for subtitle {subtitle_index}: start={start_word_idx}, last={last_word_idx}, word_list_length={len(word_list)}")
            continue  # Skip this subtitle
        
        # CRITICAL: Validate and fix subtitle text FIRST, then calculate timing
        # This ensures timing is always calculated for the correct words
        expected_words = word_list[start_word_idx:last_word_idx+1]
        if style == "varying":
            # For varying, we already have the subtitle text built
            pass
        else:
            expected_text = ' '.join(expected_words)
            if subtitle_text != expected_text:
                print(f"⚠⚠⚠ CRITICAL: Subtitle text mismatch at index {subtitle_index}:")
                print(f"   Expected (words {start_word_idx}-{last_word_idx}): '{expected_text[:50]}'")
                print(f"   Got: '{subtitle_text[:50]}'")
                # CRITICAL: Fix the text AND ensure indices match
                subtitle_text = expected_text
                # Re-verify the word list slice matches what we expect
                words_for_subtitle = expected_words
                # Ensure last_word_idx is correct based on the fixed words
                if len(expected_words) > 0:
                    # Recalculate last_word_idx based on actual words
                    actual_last_idx = start_word_idx + len(expected_words) - 1
                    if actual_last_idx != last_word_idx:
                        print(f"   Fixing last_word_idx from {last_word_idx} to {actual_last_idx}")
                        last_word_idx = actual_last_idx
        
        # NOW calculate timing using the CORRECT word indices
        if start_word_idx == 0:
            # First subtitle starts at 0.0, but show 0.1s early (clamp to 0.0 minimum)
            start_time = max(0.0, 0.0 - 0.1)
        else:
            # Start time is when the previous word finished (mark[start_word_idx - 1]), minus 0.1s to show early
            if (start_word_idx - 1) not in timepoint_map:
                print(f"⚠ WARNING: Missing timepoint for word {start_word_idx - 1}, using previous end time")
            prev_end_time = timepoint_map.get(start_word_idx - 1, 0.0)
            start_time = max(0.0, prev_end_time - 0.1)
        
        # End time is when the last word in this subtitle finishes (mark[last_word_idx]), also 0.1s early
        if last_word_idx not in timepoint_map:
            print(f"⚠ WARNING: Missing timepoint for word {last_word_idx}, using audio duration")
        word_end_time = timepoint_map.get(last_word_idx, audio_duration)
        end_time = max(start_time + 0.2, word_end_time - 0.1)  # End 0.1s early, but ensure minimum duration
        
        # Ensure minimum subtitle duration (0.2 seconds for 1 word, 0.3 for others) and max (3 seconds)
        min_duration = 0.2 if style == "1word" else 0.3
        if end_time - start_time < min_duration:
            end_time = start_time + min_duration
        elif end_time - start_time > 3.0:
            end_time = start_time + 3.0
        
        # Clamp end time to audio duration
        end_time = min(end_time, audio_duration)
        
        # Debug first subtitle with full details
        if subtitle_index == 1:
            print(f"   ✓✓✓ FIRST SUBTITLE VERIFICATION:")
            print(f"     Words: {start_word_idx}-{last_word_idx} = '{subtitle_text[:50]}'")
            print(f"     Timing: {start_time:.3f}s -> {end_time:.3f}s")
            print(f"     Timepoint for word {last_word_idx}: {timepoint_map.get(last_word_idx, 'MISSING')}")
            print(f"     First 5 words in word_list: {word_list[:5]}")
            print(f"     Subtitle matches first 5 words: {subtitle_text.startswith(' '.join(word_list[:5]))}")
        
        # Format times as SRT (HH:MM:SS,mmm)
        def format_srt_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        if subtitle_text:
            # Final validation: ensure this subtitle doesn't overlap with previous one
            # (This check happens after we've calculated timing)
            
            srt_lines.append(str(subtitle_index))
            srt_lines.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
            srt_lines.append(subtitle_text)
            srt_lines.append("")  # Empty line between subtitles
            
            # Debug first few subtitles
            if subtitle_index <= 3:
                print(f"   Subtitle {subtitle_index}: words {start_word_idx}-{last_word_idx}, '{subtitle_text[:30]}...', {start_time:.3f}s -> {end_time:.3f}s")
            
            subtitle_index += 1
    
    # CRITICAL: Final validation - ensure no duplicate or overlapping subtitles
    # Sort by start_time to ensure proper ordering
    subtitle_entries = []
    i = 0
    while i < len(srt_lines):
        if i + 3 < len(srt_lines) and srt_lines[i].strip().isdigit():
            try:
                sub_index = int(srt_lines[i])
                time_line = srt_lines[i + 1]
                text_line = srt_lines[i + 2]
                # Parse time
                if ' --> ' in time_line:
                    start_str, end_str = time_line.split(' --> ')
                    def parse_time(t):
                        h, m, s = t.split(':')
                        sec, ms = s.split(',')
                        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0
                    start_time_parsed = parse_time(start_str)
                    end_time_parsed = parse_time(end_str)
                    subtitle_entries.append({
                        'index': sub_index,
                        'start': start_time_parsed,
                        'end': end_time_parsed,
                        'text': text_line,
                        'lines': [srt_lines[i], srt_lines[i + 1], srt_lines[i + 2], srt_lines[i + 3] if i + 3 < len(srt_lines) else ""]
                    })
                i += 4
            except:
                i += 1
        else:
            i += 1
    
    # Check for duplicates and overlaps
    seen_texts = set()
    final_lines = []
    prev_end = -1
    for entry in subtitle_entries:
        # Check for duplicate text
        text_key = entry['text'].strip()
        if text_key in seen_texts and entry['start'] < 1.0:
            print(f"⚠⚠⚠ DUPLICATE SUBTITLE DETECTED at time {entry['start']:.3f}s: '{text_key[:50]}'")
            continue  # Skip duplicates that appear early
        seen_texts.add(text_key)
        
        # Check for overlap with previous subtitle
        if entry['start'] < prev_end:
            print(f"⚠⚠⚠ OVERLAPPING SUBTITLE DETECTED: subtitle {entry['index']} starts at {entry['start']:.3f}s while previous ends at {prev_end:.3f}s")
            # Adjust start time to be right after previous
            entry['start'] = prev_end + 0.01
            # Update the time line
            def format_srt_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int((seconds % 1) * 1000)
                return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
            entry['lines'][1] = f"{format_srt_time(entry['start'])} --> {format_srt_time(entry['end'])}"
        
        prev_end = entry['end']
        final_lines.extend(entry['lines'])
    
    return '\n'.join(final_lines) if final_lines else '\n'.join(srt_lines)

def create_srt_fallback(script: str, audio_duration: float) -> str:
    """
    Fallback SRT generation when timepoints are not available.
    Creates subtitles with even timing distribution.
    """
    import re
    
    # Split into sentences
    sentences = re.split(r'([.!?]+\s+)', script)
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])
    if len(sentences) % 2 == 1:
        combined_sentences.append(sentences[-1])
    
    combined_sentences = [s.strip() for s in combined_sentences if s.strip()]
    
    if not combined_sentences:
        return ""
    
    srt_lines = []
    subtitle_index = 1
    duration_per_sentence = audio_duration / len(combined_sentences)
    
    def format_srt_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    for i, sentence in enumerate(combined_sentences):
        start_time = i * duration_per_sentence
        end_time = (i + 1) * duration_per_sentence if i < len(combined_sentences) - 1 else audio_duration
        
        srt_lines.append(str(subtitle_index))
        srt_lines.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
        srt_lines.append(sentence)
        srt_lines.append("")
        subtitle_index += 1
    
    return '\n'.join(srt_lines)

@app.post("/api/compile-video", response_model=CompileVideoResponse)
async def compile_video(request: CompileVideoRequest):
    """
    Combine multiple stock videos with audio using ffmpeg.
    Downloads videos, combines them, adds audio, burns captions, and returns the final video.
    """
    try:
        if not request.video_urls or len(request.video_urls) == 0:
            raise HTTPException(status_code=400, detail="At least one video URL is required")
        
        if not request.audio_url:
            raise HTTPException(status_code=400, detail="Audio URL is required")
        
        if not request.script or not request.script.strip():
            raise HTTPException(status_code=400, detail="Script text is required for captions")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Download audio
            audio_data = None
            if request.audio_url.startswith('data:audio'):
                # Extract base64 audio data
                audio_base64 = request.audio_url.split(',')[1]
                audio_data = base64.b64decode(audio_base64)
                audio_path = temp_path / "audio.wav"
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
            else:
                # Download audio from URL
                audio_response = requests.get(request.audio_url, timeout=30)
                if audio_response.status_code != 200:
                    raise HTTPException(status_code=500, detail="Failed to download audio")
                audio_path = temp_path / "audio.wav"
                with open(audio_path, 'wb') as f:
                    f.write(audio_response.content)
            
            # Download videos
            video_paths = []
            for i, video_url in enumerate(request.video_urls):
                try:
                    video_response = requests.get(video_url, timeout=30, stream=True)
                    if video_response.status_code != 200:
                        print(f"Warning: Failed to download video {i+1}: {video_url}")
                        continue
                    
                    video_path = temp_path / f"video_{i+1}.mp4"
                    with open(video_path, 'wb') as f:
                        for chunk in video_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    video_paths.append(str(video_path))
                except Exception as e:
                    print(f"Error downloading video {i+1}: {e}")
                    continue
            
            if not video_paths:
                raise HTTPException(status_code=500, detail="Failed to download any videos")
            
            # Calculate duration per scene (divide audio duration evenly)
            # Get audio duration using ffprobe - CRITICAL: Must get actual duration, never default to 30
            audio_duration = None
            try:
                probe_cmd = [
                    FFPROBE_EXECUTABLE, '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)
                ]
                result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True, timeout=10)
                duration_str = result.stdout.strip()
                if duration_str:
                    audio_duration = float(duration_str)
                    print(f"✓ Audio duration detected: {audio_duration:.3f}s")
                else:
                    raise ValueError("Empty duration string from ffprobe")
            except Exception as e:
                print(f"✗ ERROR: Could not get audio duration: {e}")
                print(f"   ffprobe stderr: {result.stderr if 'result' in locals() else 'N/A'}")
                # Try alternative method: use ffmpeg to get duration
                try:
                    alt_cmd = [
                        FFMPEG_EXECUTABLE, '-i', str(audio_path),
                        '-f', 'null', '-'
                    ]
                    alt_result = subprocess.run(alt_cmd, capture_output=True, text=True, timeout=10)
                    # Parse duration from ffmpeg output
                    import re
                    duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})', alt_result.stderr)
                    if duration_match:
                        hours, minutes, seconds, centiseconds = duration_match.groups()
                        audio_duration = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(centiseconds) / 100
                        print(f"✓ Audio duration detected (alternative method): {audio_duration:.3f}s")
                    else:
                        raise ValueError("Could not parse duration from ffmpeg output")
                except Exception as e2:
                    print(f"✗ ERROR: Alternative duration detection also failed: {e2}")
                    raise HTTPException(status_code=500, detail=f"Failed to determine audio duration: {e}. Please ensure audio file is valid.")
            
            if audio_duration is None or audio_duration <= 0:
                raise HTTPException(status_code=500, detail="Invalid audio duration detected. Please check audio file.")
            
            # Calculate equal duration for each video scene
            scene_duration = audio_duration / len(video_paths)
            print(f"📹 Compiling video with {len(video_paths)} videos")
            print(f"🎵 Audio duration: {audio_duration:.2f}s")
            print(f"⏱️  Scene duration per video: {scene_duration:.2f}s")
            print(f"📊 Total expected video duration: {scene_duration * len(video_paths):.2f}s")
            
            # Trim each video to exactly scene_duration (equal portion of audio length)
            trimmed_videos = []
            for i, video_path in enumerate(video_paths):
                trimmed_path = temp_path / f"trimmed_video_{i+1}.mp4"
                # Trim video to exactly scene_duration
                trim_cmd = [
                    FFMPEG_EXECUTABLE, '-y',
                    '-i', str(video_path),
                    '-t', f'{scene_duration:.3f}',  # Trim to exact scene duration (3 decimal precision)
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-an',  # Remove audio from video clips (we'll add our audio later)
                    str(trimmed_path)
                ]
                try:
                    result = subprocess.run(trim_cmd, check=True, capture_output=True, timeout=30, text=True)
                    trimmed_videos.append(trimmed_path)
                    print(f"✓ Trimmed video {i+1} to {scene_duration:.2f}s")
                except subprocess.CalledProcessError as e:
                    error_msg = e.stderr.decode() if e.stderr else 'Unknown error'
                    print(f"✗ Failed to trim video {i+1}: {error_msg[:200]}")
                    raise HTTPException(status_code=500, detail=f"Failed to trim video {i+1} to {scene_duration:.2f}s")
            
            # Output video path
            output_path = temp_path / "final_video.mp4"
            
            # Step 1: Concatenate trimmed videos and scale to vertical format
            # Use filter_complex for reliable concatenation
            if len(trimmed_videos) == 1:
                # Single video - just scale it to vertical format
                concat_video_path = temp_path / "concatenated.mp4"
                scale_cmd = [
                    FFMPEG_EXECUTABLE, '-y',
                    '-i', str(trimmed_videos[0]),
                    '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-an',  # No audio
                    str(concat_video_path)
                ]
                try:
                    result = subprocess.run(scale_cmd, check=True, capture_output=True, timeout=60, text=True)
                    print(f"✓ Scaled single video to vertical format")
                except subprocess.CalledProcessError as e:
                    error_msg = e.stderr.decode() if e.stderr else 'Unknown error'
                    print(f"✗ Failed to scale video: {error_msg[:200]}")
                    raise HTTPException(status_code=500, detail="Failed to scale video")
            else:
                # Multiple videos - concatenate them and scale to vertical
                concat_video_path = temp_path / "concatenated.mp4"
                concat_cmd = [
                    FFMPEG_EXECUTABLE, '-y',
                ]
                
                # Add all input videos
                for trimmed_path in trimmed_videos:
                    concat_cmd.extend(['-i', str(trimmed_path)])
                
                # Build filter_complex: scale each to 1080x1920, then concatenate
                filter_parts = []
                # Scale all videos to same dimensions (1080x1920 for vertical)
                for i in range(len(trimmed_videos)):
                    filter_parts.append(f"[{i}:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[v{i}]")
                
                # Concatenate all scaled videos
                concat_inputs = ''.join([f"[v{i}]" for i in range(len(trimmed_videos))])
                filter_parts.append(f"{concat_inputs}concat=n={len(trimmed_videos)}:v=1:a=0[outv]")
                
                filter_complex = ';'.join(filter_parts)
                
                concat_cmd.extend([
                    '-filter_complex', filter_complex,
                    '-map', '[outv]',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    str(concat_video_path)
                ])
                
                try:
                    result = subprocess.run(concat_cmd, check=True, capture_output=True, timeout=180, text=True)
                    expected_duration = scene_duration * len(trimmed_videos)
                    print(f"✓ Concatenated {len(trimmed_videos)} videos")
                    print(f"   Expected concatenated duration: {expected_duration:.2f}s")
                    
                    # Verify concatenated video duration
                    try:
                        probe_cmd = [
                            FFPROBE_EXECUTABLE, '-v', 'error', '-show_entries', 'format=duration',
                            '-of', 'default=noprint_wrappers=1:nokey=1', str(concat_video_path)
                        ]
                        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                        actual_duration = float(probe_result.stdout.strip())
                        print(f"   Actual concatenated duration: {actual_duration:.2f}s")
                    except:
                        pass
                    
                    if result.stderr:
                        print(f"   Concat stderr: {result.stderr[:500]}")
                except subprocess.CalledProcessError as e:
                    error_msg = e.stderr.decode() if e.stderr else 'Unknown error'
                    print(f"✗ FFmpeg concat error: {error_msg[:500]}")
                    if e.stdout:
                        print(f"   FFmpeg stdout: {e.stdout.decode()[:500]}")
                    raise HTTPException(status_code=500, detail="Failed to concatenate videos")
            
            # Step 2: Generate subtitles with timing from Google TTS
            subtitle_path = None
            timepoints = None
            if TTS_AVAILABLE and tts_client and request.script:
                try:
                    # CRITICAL: Use the EXACT same script text for both TTS and SRT
                    # Normalize the script to ensure consistency
                    script_text = request.script.strip()
                    print(f"📝 Generating subtitle timings from script (length: {len(script_text)} chars)...")
                    # Use the same voice that was used for the main audio
                    selected_voice = request.voice_name if request.voice_name else "en-US-Neural2-H"
                    _, timepoints = generate_tts_with_timing(script_text, voice_name=selected_voice)
                    
                    if timepoints:
                        print(f"   Received {len(timepoints)} timepoints from TTS")
                        # Create main subtitle file (5 words per caption)
                        # Use the EXACT same script_text that was used for TTS
                        srt_content = create_srt_from_timepoints(script_text, timepoints, audio_duration, style="5words")
                        subtitle_path = temp_path / "subtitles.srt"
                        with open(subtitle_path, 'w', encoding='utf-8') as f:
                            f.write(srt_content)
                        print(f"✓ Created subtitle file (5 words per caption) with {len(timepoints)} timepoints")
                    else:
                        print(f"⚠ No timepoints received, creating estimated subtitles")
                        # Fallback: create simple subtitles with even timing
                        srt_content = create_srt_from_timepoints(script_text, [], audio_duration, style="5words")
                        subtitle_path = temp_path / "subtitles.srt"
                        with open(subtitle_path, 'w', encoding='utf-8') as f:
                            f.write(srt_content)
                except Exception as e:
                    print(f"⚠ Failed to generate subtitles with timing: {e}")
                    # Fallback: create simple subtitles
                    try:
                        srt_content = create_srt_from_timepoints(script_text, [], audio_duration, style="5words")
                        subtitle_path = temp_path / "subtitles.srt"
                        with open(subtitle_path, 'w', encoding='utf-8') as f:
                            f.write(srt_content)
                    except:
                        subtitle_path = None
            
            
            # Step 2.5: Verify concatenated video duration and extend if needed
            # Check actual concatenated video duration
            try:
                probe_concat_cmd = [
                    FFPROBE_EXECUTABLE, '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', str(concat_video_path)
                ]
                probe_result = subprocess.run(probe_concat_cmd, capture_output=True, text=True, timeout=10)
                concat_duration = float(probe_result.stdout.strip())
                print(f"📊 Concatenated video duration: {concat_duration:.2f}s (audio: {audio_duration:.2f}s)")
                
                # If concatenated video is shorter than audio, extend it by looping the last frame
                if concat_duration < audio_duration:
                    extended_path = temp_path / "extended_concat.mp4"
                    extend_duration = audio_duration - concat_duration
                    # Use tpad to extend video with last frame
                    extend_cmd = [
                        FFMPEG_EXECUTABLE, '-y',
                        '-i', str(concat_video_path),
                        '-vf', f'tpad=stop_mode=clone:stop_duration={extend_duration:.3f}',
                        '-c:v', 'libx264',
                        '-preset', 'fast',
                        '-an',
                        str(extended_path)
                    ]
                    try:
                        subprocess.run(extend_cmd, check=True, capture_output=True, timeout=30, text=True)
                        concat_video_path = extended_path
                        print(f"✓ Extended video to match audio duration ({audio_duration:.2f}s)")
                    except Exception as e:
                        print(f"⚠ Warning: Could not extend video: {e}")
            except Exception as e:
                print(f"⚠ Warning: Could not verify concatenated video duration: {e}")
            
            # Step 3: Add audio and burn captions to concatenated video
            # Video duration should now match or exceed audio - use -t to match exactly
            if subtitle_path and subtitle_path.exists():
                # Build video filter with subtitles
                # Escape path for Windows compatibility
                subtitle_path_escaped = str(subtitle_path).replace('\\', '/').replace(':', '\\:')
                # Style: Fixed positioning to keep captions at consistent location
                # MarginV=40 adds bottom margin (fixed position), MarginL and MarginR add side margins
                # Alignment=2 centers text, which keeps it in the same horizontal position
                video_filter = f"subtitles='{subtitle_path_escaped}':force_style='FontSize=20,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2,Bold=1,Alignment=2,MarginV=40,MarginL=20,MarginR=20'"
                final_cmd = [
                    FFMPEG_EXECUTABLE, '-y',
                    '-i', str(concat_video_path),
                    '-i', str(audio_path),
                    '-vf', video_filter,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-t', f'{audio_duration:.3f}',  # Force exact audio duration
                    str(output_path)
                ]
                print(f"✓ Burning captions into video from {subtitle_path}")
            else:
                # No subtitles, just add audio
                final_cmd = [
                    FFMPEG_EXECUTABLE, '-y',
                    '-i', str(concat_video_path),
                    '-i', str(audio_path),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-t', f'{audio_duration:.3f}',  # Force exact audio duration
                    str(output_path)
                ]
            
            try:
                result = subprocess.run(final_cmd, check=True, capture_output=True, timeout=180, text=True)
                print(f"✓ Final video created with audio")
                print(f"   Target duration: {audio_duration:.2f}s")
                
                # Verify final video duration
                try:
                    probe_cmd = [
                        FFPROBE_EXECUTABLE, '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', str(output_path)
                    ]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                    final_duration = float(probe_result.stdout.strip())
                    print(f"   Actual final duration: {final_duration:.2f}s")
                except:
                    pass
                
                if result.stderr:
                    print(f"   Final stderr: {result.stderr[:500]}")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode() if e.stderr else 'Unknown error'
                print(f"✗ FFmpeg final error: {error_msg[:500]}")
                if e.stdout:
                    print(f"   FFmpeg stdout: {e.stdout.decode()[:500]}")
                raise HTTPException(status_code=500, detail="Failed to combine video and audio")
            
            # Read the final video and return as base64
            with open(output_path, 'rb') as f:
                video_data = f.read()
            
            video_base64 = base64.b64encode(video_data).decode('utf-8')
            video_data_url = f"data:video/mp4;base64,{video_base64}"
            
            return CompileVideoResponse(video_url=video_data_url)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Video compilation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video compilation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
