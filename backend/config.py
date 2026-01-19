"""
Shared configuration for the video-factory backend.
This module contains common configuration variables used across multiple modules.
"""

import os
import shutil
import tempfile
from typing import Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# RESOURCE LIMITS
# =============================================================================

# Maximum disk usage for cache directories (in bytes)
# 2GB total limit
MAX_CACHE_SIZE_BYTES: int = 2 * 1024 * 1024 * 1024

# Job TTL - videos deleted after 30 minutes
JOB_TTL_SECONDS: int = 30 * 60  # 30 minutes

# Cache directory
CACHE_DIR = os.path.join(tempfile.gettempdir(), "video_factory_cache")


def get_cache_size() -> int:
    """Get total size of cache directory in bytes."""
    total_size = 0
    if os.path.exists(CACHE_DIR):
        for dirpath, dirnames, filenames in os.walk(CACHE_DIR):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass
    return total_size


def cleanup_cache_if_needed(max_size: int = MAX_CACHE_SIZE_BYTES) -> None:
    """Delete oldest files if cache exceeds max size."""
    current_size = get_cache_size()
    if current_size <= max_size:
        return

    print(f"⚠ Cache size ({current_size / 1024 / 1024:.1f}MB) exceeds limit ({max_size / 1024 / 1024:.1f}MB), cleaning up...")

    # Get all files with their modification times
    files = []
    if os.path.exists(CACHE_DIR):
        for dirpath, dirnames, filenames in os.walk(CACHE_DIR):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    files.append((fp, os.path.getmtime(fp), os.path.getsize(fp)))
                except OSError:
                    pass

    # Sort by modification time (oldest first)
    files.sort(key=lambda x: x[1])

    # Delete oldest files until under limit
    for fp, mtime, size in files:
        if current_size <= max_size * 0.8:  # Clean to 80% of limit
            break
        try:
            os.remove(fp)
            current_size -= size
            print(f"  🗑 Deleted: {os.path.basename(fp)}")
        except OSError:
            pass

    print(f"  ✓ Cache cleaned to {current_size / 1024 / 1024:.1f}MB")

# FFmpeg/FFprobe executables
FFMPEG_EXECUTABLE: str = 'ffmpeg'
FFPROBE_EXECUTABLE: str = 'ffprobe'

try:
    import imageio_ffmpeg
    FFMPEG_EXECUTABLE = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(FFMPEG_EXECUTABLE)
    ffprobe_path = os.path.join(ffmpeg_dir, 'ffprobe' + ('.exe' if os.name == 'nt' else ''))
    # Only use imageio_ffmpeg's ffprobe if it actually exists
    if os.path.exists(ffprobe_path):
        FFPROBE_EXECUTABLE = ffprobe_path
    # Otherwise keep default 'ffprobe' which will use system PATH
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

# xAI (Grok) client
# Uses the OpenAI SDK (xAI's API is OpenAI-compatible)
xai_client: Optional[Any] = None
XAI_MODEL: str = "grok-4-1-fast-reasoning"

try:
    from openai import OpenAI
    xai_api_key = os.getenv("XAI_API_KEY")
    if xai_api_key:
        xai_client = OpenAI(
            api_key=xai_api_key,
            base_url="https://api.x.ai/v1"
        )
        print(f"✓ xAI (Grok) client initialized with model: {XAI_MODEL}")
    else:
        print("⚠ XAI_API_KEY not set - LLM features disabled")
except ImportError:
    print("⚠ openai package not installed - needed for xAI client")
    xai_client = None


# ElevenLabs configuration
ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_AVAILABLE: bool = False
elevenlabs_client: Optional[Any] = None

# Google Programmable Search API configuration
GOOGLE_SEARCH_API_KEY: Optional[str] = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX: Optional[str] = os.getenv("GOOGLE_SEARCH_CX")  # Custom Search Engine ID
GOOGLE_SEARCH_AVAILABLE: bool = bool(GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX)

if GOOGLE_SEARCH_AVAILABLE:
    print("✓ Google Programmable Search API initialized")
else:
    if not GOOGLE_SEARCH_API_KEY:
        print("⚠ GOOGLE_SEARCH_API_KEY not set - Google image search disabled")
    if not GOOGLE_SEARCH_CX:
        print("⚠ GOOGLE_SEARCH_CX not set - Google image search disabled")

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
