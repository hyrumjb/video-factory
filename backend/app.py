from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
import os
import base64
import re
import json
import requests
import time
import tempfile
import subprocess
import uuid
import shutil
from typing import Optional, List, Dict, Any
from urllib.parse import quote_plus
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Import shared configuration
from config import (
    FFMPEG_EXECUTABLE,
    FFPROBE_EXECUTABLE,
    TTS_AVAILABLE,
    tts_client,
    texttospeech,
    xai_client,
    ELEVENLABS_AVAILABLE,
    elevenlabs_client,
    JOB_TTL_SECONDS,
    CACHE_DIR,
    cleanup_cache_if_needed,
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

# Import subtitle functions from subtitles module
try:
    from subtitles import (
        create_ass_from_timepoints,
        create_ass_fallback,
        create_srt_from_timepoints,
        create_srt_fallback
    )
except ImportError:
    # Fallback: subtitle functions not available
    create_ass_from_timepoints = None
    create_ass_fallback = None
    create_srt_from_timepoints = None
    create_srt_fallback = None

# Import script generation module
from script_generation import generate_script as generate_script_impl

# Import orchestrator
from orchestrator import VideoCreationOrchestrator


# =============================================================================
# JOB STATE MANAGEMENT
# =============================================================================
# In-memory job store (MVP - replace with Redis for production)

@dataclass
class JobState:
    """Tracks the state of a video creation job for reconnection support."""
    job_id: str
    topic: str
    voice_id: str
    user_ip: str  # Track user by IP for one-job-per-user limit
    created_at: datetime = field(default_factory=datetime.now)
    current_step: str = "pending"
    completed_steps: List[str] = field(default_factory=list)
    # Cached data from completed steps
    script: Optional[str] = None
    clauses: Optional[List[Dict]] = None
    audio_url: Optional[str] = None
    audio_data: Optional[Dict] = None
    media: Optional[List[Dict]] = None
    final_data: Optional[Dict] = None  # Paused/complete data
    error: Optional[str] = None
    # Track files created by this job for cleanup
    created_files: List[str] = field(default_factory=list)


# Global job store - maps job_id to JobState
JOB_STORE: Dict[str, JobState] = {}

# Track active jobs per user (by IP) - maps IP to job_id
USER_ACTIVE_JOBS: Dict[str, str] = {}


def cleanup_job_files(job: JobState):
    """Delete all files created by a job."""
    for file_path in job.created_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"  🗑 Deleted job file: {os.path.basename(file_path)}")
        except OSError:
            pass

    # Also clean the final video if it exists
    if job.final_data and job.final_data.get("video_url"):
        video_url = job.final_data["video_url"]
        # Convert URL back to file path
        if "/api/media/" in video_url:
            parts = video_url.split("/api/media/")[-1].split("/")
            if len(parts) >= 2:
                media_type, filename = parts[0], parts[1]
                file_path = os.path.join(CACHE_DIR, media_type, filename)
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"  🗑 Deleted final video: {filename}")
                except OSError:
                    pass


def cleanup_old_jobs():
    """Remove jobs older than TTL and their files."""
    now = datetime.now()
    expired = [
        job_id for job_id, job in JOB_STORE.items()
        if (now - job.created_at).total_seconds() > JOB_TTL_SECONDS
    ]
    for job_id in expired:
        job = JOB_STORE[job_id]
        cleanup_job_files(job)

        # Remove from user active jobs
        if job.user_ip in USER_ACTIVE_JOBS and USER_ACTIVE_JOBS[job.user_ip] == job_id:
            del USER_ACTIVE_JOBS[job.user_ip]

        del JOB_STORE[job_id]
        print(f"🗑 Cleaned up expired job: {job_id}")

    # Also clean disk cache if needed
    cleanup_cache_if_needed()


def cleanup_user_previous_job(user_ip: str):
    """Clean up a user's previous job when they start a new one."""
    if user_ip in USER_ACTIVE_JOBS:
        old_job_id = USER_ACTIVE_JOBS[user_ip]
        if old_job_id in JOB_STORE:
            old_job = JOB_STORE[old_job_id]
            print(f"🗑 Cleaning up previous job {old_job_id} for user {user_ip}")
            cleanup_job_files(old_job)
            del JOB_STORE[old_job_id]
        del USER_ACTIVE_JOBS[user_ip]


app = FastAPI()


# =============================================================================
# GOOGLE TTS HELPER FUNCTIONS
# =============================================================================

def get_voice_params(voice_name: str) -> tuple:
    """
    Extract language code and determine gender from a Google TTS voice name.

    Args:
        voice_name: e.g., "en-US-Neural2-H", "en-GB-Neural2-A"

    Returns:
        (language_code_lower, ssml_gender) tuple
    """
    # Extract language code from voice name
    language_code = "en-US"  # Default
    if "-" in voice_name:
        parts = voice_name.split("-")
        if len(parts) >= 2:
            language_code = f"{parts[0]}-{parts[1]}"
    language_code_lower = language_code.lower()

    # Determine gender based on language code and voice ending
    if language_code.upper() == "EN-GB":
        # British voices: A, C, F, N are female
        ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if any(
            voice_name.endswith(s) for s in ["-A", "-C", "-F", "-N"]
        ) else texttospeech.SsmlVoiceGender.MALE
    elif language_code.upper() == "EN-AU":
        # Australian voices: A, C are female
        ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if any(
            voice_name.endswith(s) for s in ["-A", "-C"]
        ) else texttospeech.SsmlVoiceGender.MALE
    else:
        # American (en-US) voices: F, C, E, G, H are female
        ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if any(
            voice_name.endswith(s) for s in ["-F", "-C", "-E", "-G", "-H"]
        ) else texttospeech.SsmlVoiceGender.MALE

    return language_code_lower, ssml_gender


# CORS middleware to allow frontend requests
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
allowed_origins = [
    "http://localhost:3000",
    FRONTEND_URL,
    "https://lightfall.ai",
    "https://www.lightfall.ai",
]
# Also allow any vercel.app subdomain
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
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


@app.post("/api/generate-script", response_model=ScriptResponse)
async def generate_script(request: ScriptRequest):
    """
    Generate a video script with scenes using the script_generation module.
    Uses HOOK/BODY structure with word boundaries for scene timing.
    """
    return await generate_script_impl(request)


@app.get("/api/create-video")
async def create_video_stream(
    request: Request,
    topic: str,
    voice_id: str = "nPczCjzI2devNBz1zQrb",
    job_id: Optional[str] = None,
    input_mode: str = "idea"
):
    """
    SSE endpoint that orchestrates the entire video creation pipeline.
    Streams progress events using semantic event types.

    Flow:
        1. Topic → Script generation
        2. Script → TTS audio with word-level timing
        3. Script + Timing → Clause segmentation
        4. Clauses → Media routing (LLM decides media type per clause)
        5. Media instructions → Parallel media retrieval
        6. Media + Audio → Final video with subtitles

    Args:
        topic: The video topic/prompt
        voice_id: ElevenLabs voice ID (default: Brian)
        job_id: Optional job ID for reconnection. If provided, resumes from cached state.

    SSE Event Types:
        - job: First event, contains job_id for reconnection
        - script: Script generation progress
        - tts: TTS generation progress
        - clauses: Clause segmentation progress
        - routing: Media routing decisions
        - media: Media retrieval progress
        - compile: Video compilation progress
        - complete: Job finished with final video
        - error: Error occurred
        - ping: Heartbeat to keep connection alive

    Usage:
        const eventSource = new EventSource('/api/create-video?topic=...&job_id=...');
        eventSource.addEventListener('job', (e) => { jobId = JSON.parse(e.data).job_id; });
        eventSource.addEventListener('script', (e) => { ... });
        eventSource.addEventListener('complete', (e) => { ... });
    """
    # Cleanup old jobs periodically
    cleanup_old_jobs()

    async def event_generator():
        nonlocal job_id

        # Check for existing job (reconnection)
        job: Optional[JobState] = None
        if job_id and job_id in JOB_STORE:
            job = JOB_STORE[job_id]
            print(f"🔄 Reconnecting to job {job_id}, current step: {job.current_step}")

            # Emit job event
            yield {"event": "job", "data": json.dumps({"job_id": job_id, "reconnected": True})}

            # Replay cached data from completed steps
            if job.script and "script" in job.completed_steps:
                yield {
                    "event": "script",
                    "data": json.dumps({
                        "step": "script",
                        "status": "done",
                        "message": "Script ready (cached)",
                        "data": {"script": job.script}
                    })
                }

            if job.audio_url and "tts" in job.completed_steps:
                yield {
                    "event": "tts",
                    "data": json.dumps({
                        "step": "tts",
                        "status": "done",
                        "message": "Audio ready (cached)",
                        "data": job.audio_data
                    })
                }

            if job.clauses and "clauses" in job.completed_steps:
                yield {
                    "event": "clauses",
                    "data": json.dumps({
                        "step": "clauses",
                        "status": "done",
                        "message": "Clauses ready (cached)",
                        "data": {"clauses": job.clauses}
                    })
                }

            if job.media and "media" in job.completed_steps:
                yield {
                    "event": "media",
                    "data": json.dumps({
                        "step": "media",
                        "status": "done",
                        "message": "Media ready (cached)",
                        "data": {"media": job.media}
                    })
                }

            if job.final_data and "complete" in job.completed_steps:
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "step": "complete",
                        "status": "done",
                        "message": "Video ready (cached)",
                        "data": job.final_data
                    })
                }
                return  # Job already complete

            if job.error:
                yield {
                    "event": "error",
                    "data": json.dumps({"step": "error", "status": "error", "message": job.error})
                }
                return

        else:
            # Get user IP for tracking
            user_ip = request.client.host if request.client else "unknown"

            # Clean up user's previous job (one job per user)
            cleanup_user_previous_job(user_ip)

            # Create new job
            job_id = str(uuid.uuid4())[:8]  # Short ID for convenience
            job = JobState(
                job_id=job_id,
                topic=topic,
                voice_id=voice_id,
                user_ip=user_ip
            )
            JOB_STORE[job_id] = job
            USER_ACTIVE_JOBS[user_ip] = job_id
            print(f"🆕 Created new job {job_id} for user {user_ip}, topic: {topic[:50]}...")

            # Emit job event first
            yield {"event": "job", "data": json.dumps({"job_id": job_id, "reconnected": False})}

        # Run orchestrator
        orchestrator = VideoCreationOrchestrator()
        # If input_mode is 'script', the topic field contains the actual script
        provided_script = topic if input_mode == "script" else None
        event_iter = orchestrator.create_video(
            topic,
            voice_id=voice_id,
            provided_script=provided_script,
        ).__aiter__()

        heartbeat_interval = 10  # seconds

        while True:
            try:
                event_task = asyncio.create_task(event_iter.__anext__())

                while True:
                    try:
                        progress_event = await asyncio.wait_for(
                            asyncio.shield(event_task),
                            timeout=heartbeat_interval
                        )

                        # Update job state based on event
                        step = progress_event.step
                        status = progress_event.status.value if hasattr(progress_event.status, 'value') else str(progress_event.status)
                        job.current_step = step

                        if status == "done":
                            job.completed_steps.append(step)

                            # Cache step data
                            if step == "script" and progress_event.data:
                                job.script = progress_event.data.get("script")

                            elif step == "tts" and progress_event.data:
                                job.audio_url = progress_event.data.get("audio_url")
                                job.audio_data = progress_event.data

                            elif step == "clauses" and progress_event.data:
                                job.clauses = progress_event.data.get("clauses")

                            elif step == "media" and progress_event.data:
                                job.media = progress_event.data.get("media")

                            elif step == "complete" and progress_event.data:
                                job.final_data = progress_event.data

                        elif status == "error":
                            job.error = progress_event.message

                        # Use semantic event type (step name)
                        yield {
                            "event": step,
                            "data": progress_event.to_json()
                        }
                        break

                    except asyncio.TimeoutError:
                        yield {"event": "ping", "data": "{}"}

            except StopAsyncIteration:
                break
            except Exception as e:
                job.error = str(e)
                yield {
                    "event": "error",
                    "data": json.dumps({"step": "error", "status": "error", "message": str(e)})
                }
                break

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
        
        language_code_lower, ssml_gender = get_voice_params(voice_name)

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
            language_code_lower, ssml_gender = get_voice_params(voice_name)

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
            language_code_lower, ssml_gender = get_voice_params(voice_name)

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

                language_code_lower, ssml_gender = get_voice_params(voice_name)
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


# SRT subtitle functions imported from subtitles module (see imports at top)


@app.post("/api/compile-video", response_model=CompileVideoResponse)
async def compile_video_endpoint(request: CompileVideoRequest):
    """
    Combine multiple stock videos with audio using ffmpeg.
    Delegates to video_compilation module for actual processing.

    Note: For the main video creation flow, use /api/create-video which
    handles the entire pipeline via the orchestrator.
    """
    from video_compilation import compile_video as compile_video_impl
    return await compile_video_impl(request)


# =============================================================================
# MEDIA FILE SERVING
# =============================================================================
# Serve locally cached media files (images, videos) that were downloaded
# during the media retrieval phase. This allows the frontend to display
# media previews while the video is being compiled.
# =============================================================================

@app.get("/api/media/{media_type}/{filename}")
async def serve_media_file(media_type: str, filename: str):
    """
    Serve a cached media file.

    Args:
        media_type: Type of media (stock_videos, web_images, ai_images, ai_videos, youtube)
        filename: The filename of the cached media

    Returns:
        The media file
    """
    import mimetypes

    # Map media types to cache subdirectories
    cache_subdirs = {
        "stock_videos": "stock_videos",
        "web_images": "web_images",
        "ai_images": "ai_images",
        "ai_videos": "ai_videos",
        "youtube": "youtube_cache"
    }

    if media_type not in cache_subdirs:
        raise HTTPException(status_code=400, detail=f"Invalid media type: {media_type}")

    # Build path to cached file
    # YouTube cache is stored in backend/youtube_cache/, others in temp dir
    if media_type == "youtube":
        cache_dir = Path(__file__).parent / "youtube_cache"
    else:
        cache_dir = Path(tempfile.gettempdir()) / "video_factory_cache" / cache_subdirs[media_type]
    file_path = cache_dir / filename

    # Security: Ensure the path is within the cache directory (prevent path traversal)
    try:
        file_path = file_path.resolve()
        cache_dir_resolved = cache_dir.resolve()
        if not str(file_path).startswith(str(cache_dir_resolved)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")

    # Determine content type
    content_type, _ = mimetypes.guess_type(str(file_path))
    if not content_type:
        if file_path.suffix.lower() in ['.mp4', '.webm', '.mkv']:
            content_type = 'video/mp4'
        elif file_path.suffix.lower() in ['.jpg', '.jpeg']:
            content_type = 'image/jpeg'
        elif file_path.suffix.lower() == '.png':
            content_type = 'image/png'
        elif file_path.suffix.lower() == '.webp':
            content_type = 'image/webp'
        else:
            content_type = 'application/octet-stream'

    return FileResponse(file_path, media_type=content_type)


# =============================================================================
# LEGACY ENDPOINTS
# =============================================================================
# These endpoints are from the old scene-based flow and are kept for
# backwards compatibility. The new flow uses /api/create-video which
# handles everything via the orchestrator (clause-based media routing).
# =============================================================================


@app.post("/api/search-stock-videos", response_model=MediaSearchResponse)
async def search_stock_videos(request: MediaSearchRequest):
    """
    [LEGACY] Search for stock videos for the given scenes.
    Used for on-demand video search when user selects Videos mode after initial generation.
    """
    from media_video_search import search_pexels_videos, search_pixabay_videos

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
    [LEGACY] Search for Google images for the given scenes.
    Used for on-demand image search when user selects Images mode after initial generation.
    """
    from media_image_search import generate_image_search_queries, search_images_with_query
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


@app.post("/api/download-youtube-clips")
async def download_youtube_clips(request: MediaSearchRequest):
    """
    [LEGACY] Download YouTube clips for the given scenes.
    Used for on-demand YouTube clip download when user selects YouTube mode.
    """
    from media_youtube_download import download_youtube_for_scenes

    # Convert scenes to dict format expected by youtube_download
    scenes_for_youtube = [
        {
            "scene_number": s.scene_number,
            "section_name": s.section_name or f"Scene {s.scene_number}",
            "description": s.description
        }
        for s in request.scenes
    ]

    result = await download_youtube_for_scenes(
        topic=request.topic,
        script=request.script,
        scenes=scenes_for_youtube
    )

    # Convert to MediaSearchResponse format
    scene_results = []
    for clip in result.get("clips", []):
        scene_results.append(SceneMediaResult(
            scene_number=clip.get("scene_number"),
            section_name=None,
            video_search_query=clip.get("query", ""),
            video_url=clip.get("video_path"),  # Local file path for YouTube clips
            video_source="youtube",
            image_search_query="",
            images=[]
        ))

    return MediaSearchResponse(results=scene_results)


@app.post("/api/generate-ai-images", response_model=AIImageResponse)
async def generate_ai_images(request: AIImageRequest):
    """
    [LEGACY] Generate AI images for the video.
    Used for on-demand AI image generation when user selects AI mode after initial generation.

    Generates images for each scene with each model (3 models x N scenes).
    """
    try:
        from media_image_generate import generate_image_for_video, FAL_AVAILABLE

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


