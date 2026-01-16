from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sse_starlette.sse import EventSourceResponse
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

# Import shared configuration
from config import (
    FFMPEG_EXECUTABLE,
    FFPROBE_EXECUTABLE,
    TTS_AVAILABLE,
    tts_client,
    texttospeech,
    openai_client,
    ELEVENLABS_AVAILABLE,
    elevenlabs_client,
)

# Import models
from models import (
    ScriptRequest,
    TTSRequest,
    MultiVoiceTTSRequest,
    VideoSearchRequest,
    VideoItem,
    VideoSearchResponse,
    CompileVideoRequest,
    CompileVideoResponse,
    VideoScene,
    ScriptResponse,
    TTSResponse,
    VoiceAudio,
    MultiVoiceTTSResponse,
    MediaSearchRequest,
    MediaSearchResponse,
    SceneMediaResult,
    AIImageRequest,
    AIImageResponse,
    AIImageResult,
)

# Import ASS subtitle functions from subtitles module
try:
    from subtitles import create_ass_from_timepoints, create_ass_fallback
except ImportError:
    # Fallback: these will be defined later in the file if module not available
    create_ass_from_timepoints = None
    create_ass_fallback = None

# Import script generation module
from script_generation import generate_script as generate_script_impl

# Import orchestrator
from orchestrator import VideoCreationOrchestrator


def calculate_scene_durations_from_alignment(
    alignment: dict,
    scenes: List[dict],
    audio_duration: float
) -> List[float]:
    """
    Calculate scene durations from ElevenLabs alignment data and scene word boundaries.

    Args:
        alignment: ElevenLabs alignment dict with 'characters', 'character_start_times_seconds', 'character_end_times_seconds'
        scenes: List of scene dicts with 'scene_number', 'word_start', 'word_end'
        audio_duration: Total audio duration in seconds

    Returns:
        List of durations in seconds for each scene
    """
    if not alignment or not scenes:
        return []

    try:
        characters = alignment.get('characters', [])
        start_times = alignment.get('character_start_times_seconds', [])
        end_times = alignment.get('character_end_times_seconds', [])

        if not characters or not start_times or not end_times:
            print("⚠ Alignment data incomplete, cannot calculate scene durations")
            return []

        # Build word boundaries from character data
        # Find where each word starts and ends in the character array
        word_boundaries: List[dict] = []  # List of {word_index, start_time, end_time}
        current_word_index = 0
        word_start_time = None
        word_end_time = None

        for i, char in enumerate(characters):
            if i >= len(start_times) or i >= len(end_times):
                break

            if char == ' ' or char == '\n':
                # End of word
                if word_start_time is not None:
                    word_boundaries.append({
                        'word_index': current_word_index,
                        'start': word_start_time,
                        'end': word_end_time
                    })
                    current_word_index += 1
                word_start_time = None
                word_end_time = None
            else:
                # Part of a word
                if word_start_time is None:
                    word_start_time = start_times[i]
                word_end_time = end_times[i]

        # Don't forget the last word
        if word_start_time is not None:
            word_boundaries.append({
                'word_index': current_word_index,
                'start': word_start_time,
                'end': word_end_time
            })

        print(f"✓ Built {len(word_boundaries)} word boundaries from alignment")

        # Create a map of word_index -> timing
        word_timing_map = {wb['word_index']: wb for wb in word_boundaries}

        # Calculate duration for each scene based on word boundaries
        scene_durations: List[float] = []

        for scene in sorted(scenes, key=lambda s: s.get('scene_number', 0)):
            word_start = scene.get('word_start', 0)
            word_end = scene.get('word_end', len(word_boundaries))

            # Get start time (from word_start)
            if word_start in word_timing_map:
                start_time = word_timing_map[word_start]['start']
            elif word_start == 0:
                start_time = 0.0
            else:
                # Estimate: use previous word's end time
                prev_words = [w for w in word_boundaries if w['word_index'] < word_start]
                start_time = prev_words[-1]['end'] if prev_words else 0.0

            # Get end time (from word_end - 1, since word_end is exclusive)
            end_word_index = word_end - 1
            if end_word_index in word_timing_map:
                end_time = word_timing_map[end_word_index]['end']
            else:
                # Estimate: use last known word's end time or audio duration
                end_time = audio_duration

            duration = max(end_time - start_time, 0.5)  # Minimum 0.5 second per scene
            scene_durations.append(duration)

            print(f"   Scene {scene.get('scene_number')}: words {word_start}-{word_end} = {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")

        return scene_durations

    except Exception as e:
        print(f"⚠ Error calculating scene durations: {e}")
        import traceback
        traceback.print_exc()
        return []

app = FastAPI()

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log TTS status
if TTS_AVAILABLE and tts_client:
    print("✓ Google TTS client initialized")
elif TTS_AVAILABLE:
    print("⚠ Warning: Google TTS credentials not found. TTS will not work.")
else:
    print("⚠ Google TTS package not installed. Install with: pip install google-cloud-texttospeech")


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
    """
    Generate a video script with scenes using the script_generation module.
    Uses HOOK/BODY structure with word boundaries for scene timing.
    """
    return await generate_script_impl(request)


@app.get("/api/create-video")
async def create_video_stream(
    topic: str,
    voice_id: str = "nPczCjzI2devNBz1zQrb",
    staged: bool = False,
    background_type: str = "videos"
):
    """
    SSE endpoint that orchestrates the entire video creation pipeline.
    Streams progress events and returns the final video URL.

    Args:
        topic: The video topic/prompt
        voice_id: ElevenLabs voice ID (default: Brian)
        staged: If true, stop after script + media search (no TTS/compile).
                Emits 'paused' event with all data for manual continuation.
        background_type: Type of background media ("videos", "images", or "ai")

    Usage:
        const eventSource = new EventSource('/api/create-video?topic=...&voice_id=...&staged=true&background_type=ai');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            // data.step: 'script' | 'videos' | 'tts' | 'compile' | 'complete' | 'paused' | 'error'
            // data.status: 'pending' | 'running' | 'done' | 'error' | 'retrying'
            // data.message: Human-readable status message
            // data.data: Step-specific data (script, video_url, ai_images, etc.)
        };
    """
    async def event_generator():
        orchestrator = VideoCreationOrchestrator()
        async for progress_event in orchestrator.create_video(
            topic,
            voice_id=voice_id,
            staged=staged,
            background_type=background_type
        ):
            yield {
                "event": "message",
                "data": progress_event.to_json()
            }

    return EventSourceResponse(event_generator())


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
        
        # Extract language code from voice name and convert to lowercase (Google API requires lowercase)
        language_code = "en-US"  # Default
        if "-" in voice_name:
            parts = voice_name.split("-")
            if len(parts) >= 2:
                language_code = f"{parts[0]}-{parts[1]}"
        language_code_lower = language_code.lower()
        
        # Determine gender based on language code and voice ending
        if language_code.upper() == "EN-GB":
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-A") or voice_name.endswith("-C") or voice_name.endswith("-F") or voice_name.endswith("-N") else texttospeech.SsmlVoiceGender.MALE
        elif language_code.upper() == "EN-AU":
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-A") or voice_name.endswith("-C") else texttospeech.SsmlVoiceGender.MALE
        else:  # American (en-US)
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-F") or voice_name.endswith("-C") or voice_name.endswith("-E") or voice_name.endswith("-G") or voice_name.endswith("-H") else texttospeech.SsmlVoiceGender.MALE
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code_lower,
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
            # Extract language code from voice name and convert to lowercase
            language_code = "en-US"  # Default
            if "-" in voice_name:
                parts = voice_name.split("-")
                if len(parts) >= 2:
                    language_code = f"{parts[0]}-{parts[1]}"
            language_code_lower = language_code.lower()
            
            # Determine gender based on language code and voice ending
            if language_code.upper() == "EN-GB":
                ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-A") or voice_name.endswith("-C") or voice_name.endswith("-F") or voice_name.endswith("-N") else texttospeech.SsmlVoiceGender.MALE
            elif language_code.upper() == "EN-AU":
                ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-A") or voice_name.endswith("-C") else texttospeech.SsmlVoiceGender.MALE
            else:  # American (en-US)
                ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-F") or voice_name.endswith("-C") or voice_name.endswith("-E") or voice_name.endswith("-G") or voice_name.endswith("-H") else texttospeech.SsmlVoiceGender.MALE
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code_lower,
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
            # Extract language code from voice name and convert to lowercase
            language_code = "en-US"  # Default
            if "-" in voice_name:
                parts = voice_name.split("-")
                if len(parts) >= 2:
                    language_code = f"{parts[0]}-{parts[1]}"
            language_code_lower = language_code.lower()
            
            # Determine gender based on language code and voice ending
            if language_code.upper() == "EN-GB":
                ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-A") or voice_name.endswith("-C") or voice_name.endswith("-F") or voice_name.endswith("-N") else texttospeech.SsmlVoiceGender.MALE
            elif language_code.upper() == "EN-AU":
                ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-A") or voice_name.endswith("-C") else texttospeech.SsmlVoiceGender.MALE
            else:  # American (en-US)
                ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-F") or voice_name.endswith("-C") or voice_name.endswith("-E") or voice_name.endswith("-G") or voice_name.endswith("-H") else texttospeech.SsmlVoiceGender.MALE
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code_lower,
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


@app.post("/api/generate-tts-elevenlabs")
async def generate_tts_elevenlabs(request: TTSRequest):
    """
    Generate TTS audio using ElevenLabs API with word-level timing.
    Returns audio as base64 encoded MP3 and alignment data for captions.
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        if not ELEVENLABS_AVAILABLE or not elevenlabs_client:
            raise HTTPException(status_code=503, detail="ElevenLabs is not available. Check API key.")

        from tts import generate_elevenlabs_tts_with_timing

        # Use default voice (Brian - narration voice)
        voice_id = "nPczCjzI2devNBz1zQrb"

        audio_content, alignment = generate_elevenlabs_tts_with_timing(request.text.strip(), voice_id=voice_id)

        # Return audio as base64 encoded string
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')

        # Serialize alignment data for JSON response
        alignment_dict = None
        if alignment:
            alignment_dict = {
                'characters': list(alignment.characters) if hasattr(alignment, 'characters') else [],
                'character_start_times_seconds': list(alignment.character_start_times_seconds) if hasattr(alignment, 'character_start_times_seconds') else [],
                'character_end_times_seconds': list(alignment.character_end_times_seconds) if hasattr(alignment, 'character_end_times_seconds') else [],
            }
            print(f"✓ ElevenLabs alignment serialized: {len(alignment_dict['characters'])} characters")

        return TTSResponse(audio_url=f"data:audio/mp3;base64,{audio_base64}", alignment=alignment_dict)

    except Exception as e:
        print(f"ElevenLabs TTS generation error: {e}")
        import traceback
        traceback.print_exc()
        if hasattr(e, 'status_code'):
            raise HTTPException(status_code=e.status_code, detail=str(e))
        raise HTTPException(status_code=500, detail=f"ElevenLabs TTS generation failed: {str(e)}")


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
                
                # Extract language code from voice name and convert to lowercase
                language_code = "en-US"  # Default
                if "-" in voice_name:
                    parts = voice_name.split("-")
                    if len(parts) >= 2:
                        language_code = f"{parts[0]}-{parts[1]}"
                language_code_lower = language_code.lower()
                
                # Determine gender based on language code and voice ending
                if language_code.upper() == "EN-GB":
                    ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-A") or voice_name.endswith("-C") or voice_name.endswith("-F") or voice_name.endswith("-N") else texttospeech.SsmlVoiceGender.MALE
                elif language_code.upper() == "EN-AU":
                    ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-A") or voice_name.endswith("-C") else texttospeech.SsmlVoiceGender.MALE
                else:  # American (en-US)
                    ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-F") or voice_name.endswith("-C") or voice_name.endswith("-E") or voice_name.endswith("-G") or voice_name.endswith("-H") else texttospeech.SsmlVoiceGender.MALE
                gender_str = "Female" if ssml_gender == texttospeech.SsmlVoiceGender.FEMALE else "Male"
                
                # Generate TTS (without timing for faster generation in testing)
                synthesis_input = texttospeech.SynthesisInput(text=request.text.strip())
                voice = texttospeech.VoiceSelectionParams(
                    language_code=language_code_lower,
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
async def compile_video_endpoint(request: CompileVideoRequest):
    """
    Combine multiple stock videos with audio using ffmpeg.
    Delegates to video_compilation module for actual processing.
    """
    from video_compilation import compile_video as compile_video_impl
    return await compile_video_impl(request)


# Video compilation logic moved to video_compilation.py module


@app.post("/api/search-stock-videos", response_model=MediaSearchResponse)
async def search_stock_videos(request: MediaSearchRequest):
    """
    Search for stock videos for the given scenes.
    Used for on-demand video search when user selects Videos mode after initial generation.
    """
    from video_search import search_pexels_videos, search_pixabay_videos

    results = []

    for scene in request.scenes:
        search_query = scene.search_query

        # Search both Pexels and Pixabay
        video_url = None
        video_source = None

        try:
            # Try Pexels first (has portrait filter)
            pexels_results = search_pexels_videos(search_query, per_page=5, orientation="portrait")
            if pexels_results:
                # Prefer vertical videos
                vertical = next((v for v in pexels_results if v.get("height", 0) > v.get("width", 0)), None)
                if vertical:
                    video_url = vertical["url"]
                    video_source = "pexels"
                else:
                    video_url = pexels_results[0]["url"]
                    video_source = "pexels"
        except Exception as e:
            print(f"Pexels search error: {e}")

        if not video_url:
            try:
                # Try Pixabay as fallback
                pixabay_results = search_pixabay_videos(search_query, per_page=5)
                if pixabay_results:
                    vertical = next((v for v in pixabay_results if v.get("height", 0) > v.get("width", 0)), None)
                    if vertical:
                        video_url = vertical["url"]
                        video_source = "pixabay"
                    else:
                        video_url = pixabay_results[0]["url"]
                        video_source = "pixabay"
            except Exception as e:
                print(f"Pixabay search error: {e}")

        results.append(SceneMediaResult(
            scene_number=scene.scene_number,
            section_name=scene.section_name,
            video_search_query=search_query,
            video_url=video_url,
            video_source=video_source,
            image_search_query=search_query,
            images=[]
        ))

    return MediaSearchResponse(results=results)


@app.post("/api/search-stock-images", response_model=MediaSearchResponse)
async def search_stock_images(request: MediaSearchRequest):
    """
    Search for Google images for the given scenes.
    Used for on-demand image search when user selects Images mode after initial generation.
    """
    from image_search import generate_image_search_queries, search_images_with_query
    import asyncio

    # Generate optimized search queries using LLM
    scenes_for_llm = [
        {
            "scene_number": s.scene_number,
            "section_name": s.section_name or f"Scene {s.scene_number}",
            "description": s.description
        }
        for s in request.scenes
    ]
    image_queries = generate_image_search_queries(request.topic, request.script, scenes_for_llm)

    results = []

    for i, scene in enumerate(request.scenes):
        image_query = image_queries[i] if i < len(image_queries) else scene.search_query

        # Search for images
        image_result = await search_images_with_query(
            search_query=image_query,
            fallback_query=scene.search_query
        )

        results.append(SceneMediaResult(
            scene_number=scene.scene_number,
            section_name=scene.section_name,
            video_search_query=scene.search_query,
            video_url=None,
            video_source=None,
            image_search_query=image_result.get("query", image_query),
            images=image_result.get("images", [])
        ))

    return MediaSearchResponse(results=results)


@app.post("/api/generate-ai-images", response_model=AIImageResponse)
async def generate_ai_images(request: AIImageRequest):
    """
    Generate AI images for the video.
    Used for on-demand AI image generation when user selects AI mode after initial generation.

    Generates images for each scene with each model (3 models x N scenes).
    """
    try:
        from image_generate import generate_image_for_video, FAL_AVAILABLE

        if not FAL_AVAILABLE:
            raise HTTPException(status_code=503, detail="FAL_KEY not configured - AI image generation unavailable")

        # Build sections list from request
        sections = None
        if request.sections:
            sections = [{"name": s.name, "text": s.text} for s in request.sections]

        result = await generate_image_for_video(
            topic=request.topic,
            script=request.script,
            first_section_text=request.first_section_text,
            sections=sections
        )

        # Convert to response format - include scene info
        images = [
            AIImageResult(
                model_name=img.get("model_name", "Unknown"),
                model_id=img.get("model_id", ""),
                url=img.get("url"),
                width=img.get("width"),
                height=img.get("height"),
                error=img.get("error"),
                scene_number=img.get("scene_number"),
                section_name=img.get("section_name")
            )
            for img in result.get("images", [])
        ]

        return AIImageResponse(
            images=images,
            prompts=result.get("prompts")
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"AI image generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI image generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


