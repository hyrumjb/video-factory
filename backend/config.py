"""
Shared configuration for the video-factory backend.
This module contains common configuration variables used across multiple modules.
"""

import os
from typing import Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FFmpeg/FFprobe executables
FFMPEG_EXECUTABLE: str = 'ffmpeg'
FFPROBE_EXECUTABLE: str = 'ffprobe'

try:
    import imageio_ffmpeg
    FFMPEG_EXECUTABLE = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(FFMPEG_EXECUTABLE)
    FFPROBE_EXECUTABLE = os.path.join(ffmpeg_dir, 'ffprobe' + ('.exe' if os.name == 'nt' else ''))
except ImportError:
    # Fallback to system ffmpeg if imageio-ffmpeg not available
    pass

# Google TTS configuration
TTS_AVAILABLE: bool = False
texttospeech: Optional[Any] = None
tts_client: Optional[Any] = None

try:
    from google.cloud import texttospeech_v1beta1 as texttospeech
    TTS_AVAILABLE = True
except ImportError:
    # Fallback to stable v1 if beta not available (no timepointing)
    try:
        from google.cloud import texttospeech
        TTS_AVAILABLE = True
        print("Warning: Using stable v1 API - timepointing not available.")
    except ImportError:
        TTS_AVAILABLE = False
        texttospeech = None

# Initialize Google TTS client
if TTS_AVAILABLE:
    google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # Try to find credentials file if not set in env
    if not google_creds_path:
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

    try:
        if google_creds_path and os.path.exists(google_creds_path):
            tts_client = texttospeech.TextToSpeechClient()
        else:
            tts_client = None
    except Exception:
        tts_client = None

# OpenAI client
openai_client: Optional[Any] = None

try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    openai_client = None

# ElevenLabs configuration
ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_AVAILABLE: bool = False
elevenlabs_client: Optional[Any] = None

if ELEVENLABS_API_KEY:
    try:
        from elevenlabs.client import ElevenLabs
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        ELEVENLABS_AVAILABLE = True
        print("✓ ElevenLabs TTS initialized")
    except ImportError:
        print("⚠ elevenlabs package not installed. Run: pip install elevenlabs")
        ELEVENLABS_AVAILABLE = False
    except Exception as e:
        print(f"⚠ ElevenLabs initialization failed: {e}")
        ELEVENLABS_AVAILABLE = False
