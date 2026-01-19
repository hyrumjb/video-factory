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


class CompileVideoRequest(BaseModel):
    """Request model for video compilation endpoint."""
    video_urls: List[str] = []  # List of video/image URLs
    audio_url: str  # Base64 audio data URL
    script: str  # Script text for captions
    scene_durations: Optional[List[float]] = None  # Duration for each scene
    voice_name: Optional[str] = "en-US-Neural2-H"  # Voice used for TTS
    alignment: Optional[dict] = None  # ElevenLabs word-level timing data for captions
    tts_provider: Optional[str] = "google"  # "google" or "elevenlabs"
    use_images: Optional[bool] = False  # If True, treat video_urls as images


class CompileVideoResponse(BaseModel):
    video_url: str  # Base64 encoded video or URL to final video


# Response models
class ScriptResponse(BaseModel):
    """Response model for script generation."""
    script: str  # Clean script for TTS (no section labels)
    scenes: list = []  # Deprecated - kept for API compatibility
    sections: Optional[list] = None  # Deprecated - kept for API compatibility


class TTSResponse(BaseModel):
    audio_url: str
    alignment: Optional[dict] = None  # ElevenLabs word-level timing data


class VoiceAudio(BaseModel):
    voice_name: str
    audio_url: str
    gender: str


class MultiVoiceTTSResponse(BaseModel):
    voices: List[VoiceAudio]


# On-demand media generation models
class SceneInfo(BaseModel):
    scene_number: int
    section_name: Optional[str] = None
    description: str
    search_query: str


class MediaSearchRequest(BaseModel):
    topic: str
    script: str
    scenes: List[SceneInfo]


class SceneMediaResult(BaseModel):
    scene_number: int
    section_name: Optional[str] = None
    video_search_query: str
    video_url: Optional[str] = None
    video_source: Optional[str] = None
    image_search_query: str
    images: List[dict] = []


class MediaSearchResponse(BaseModel):
    results: List[SceneMediaResult]


class SectionInfo(BaseModel):
    name: str
    text: str


class AIImageRequest(BaseModel):
    topic: str
    script: str
    first_section_text: str  # Backward compatible
    sections: Optional[List[SectionInfo]] = None  # New: sections for multi-scene generation


class AIImageResult(BaseModel):
    model_name: str
    model_id: str
    url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    error: Optional[str] = None
    scene_number: Optional[int] = None
    section_name: Optional[str] = None


class AIImageResponse(BaseModel):
    images: List[AIImageResult]  # One image per scene
    prompts: Optional[List[dict]] = None  # Prompts for each scene
