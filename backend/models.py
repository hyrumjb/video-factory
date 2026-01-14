"""
Pydantic models for the video-factory backend.
"""

from pydantic import BaseModel
from typing import Optional, List


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


class SceneTimingInfo(BaseModel):
    scene_number: int
    word_start: Optional[int] = None  # Starting word index
    word_end: Optional[int] = None  # Ending word index


class CompileVideoRequest(BaseModel):
    video_urls: List[str]  # List of video URLs to combine
    audio_url: str  # Base64 audio data URL
    script: str  # Script text for captions
    scene_durations: Optional[List[float]] = None  # Duration for each scene in seconds
    voice_name: Optional[str] = "en-US-Neural2-H"  # Voice used for TTS
    alignment: Optional[dict] = None  # ElevenLabs word-level timing data for captions
    tts_provider: Optional[str] = "google"  # "google" or "elevenlabs"
    scenes: Optional[List[SceneTimingInfo]] = None  # Scene word boundaries for timing


class CompileVideoResponse(BaseModel):
    video_url: str  # Base64 encoded video or URL to final video


# Response models
class VideoScene(BaseModel):
    scene_number: int
    description: str
    search_keywords: str
    search_query: str  # Optimized 3-5 word query for stock video APIs
    section_name: Optional[str] = None  # e.g., "HOOK", "BODY1"
    word_start: Optional[int] = None  # Starting word index in script
    word_end: Optional[int] = None  # Ending word index in script


class ScriptSection(BaseModel):
    name: str  # "HOOK", "BODY1", etc.
    text: str  # The text content of this section
    word_start: int  # Starting word index (0-based)
    word_end: int  # Ending word index (exclusive)


class ScriptResponse(BaseModel):
    script: str  # Clean script for TTS (no section labels)
    scenes: list[VideoScene]
    sections: Optional[List[ScriptSection]] = None  # Section boundaries for timing


class TTSResponse(BaseModel):
    audio_url: str
    alignment: Optional[dict] = None  # ElevenLabs word-level timing data


class VoiceAudio(BaseModel):
    voice_name: str
    audio_url: str
    gender: str


class MultiVoiceTTSResponse(BaseModel):
    voices: List[VoiceAudio]
