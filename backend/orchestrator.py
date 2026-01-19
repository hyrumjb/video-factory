"""
Video Creation Orchestrator

Manages the complete video creation pipeline with progress streaming.

Flow:
1. Topic → Script
2. Script → TTS Audio (with word-level timing)
3. Script + Timing → Clauses
4. Clauses → Media Instructions (via LLM router)
5. Media Instructions → Actual Media (parallel retrieval)
6. Media + Audio → Final Video with Subtitles
"""

import asyncio
import json
import base64
import os
import tempfile
from typing import AsyncGenerator, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from models import ScriptRequest, ScriptResponse
from clause_segmentation import segment_script_into_clauses, Clause


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class ProgressEvent:
    step: str
    status: StepStatus
    message: Optional[str] = None
    progress: Optional[int] = None
    data: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        d = {"step": self.step, "status": self.status.value}
        if self.message:
            d["message"] = self.message
        if self.progress is not None:
            d["progress"] = self.progress
        if self.data:
            d["data"] = self.data
        return json.dumps(d)


def _convert_local_path_to_api_url(local_path: str) -> str:
    """
    Convert a local cache file path to an API URL for frontend display.

    Examples:
        /var/folders/.../video_factory_cache/stock_videos/pexels_abc.mp4
        -> http://localhost:8000/api/media/stock_videos/pexels_abc.mp4

        /Users/.../backend/youtube_cache/yt_clip_abc.mp4
        -> http://localhost:8000/api/media/youtube/yt_clip_abc.mp4
    """
    import re

    if not local_path:
        return local_path

    # If it's already a URL or data URI, return as-is
    if local_path.startswith(('http://', 'https://', 'data:')):
        return local_path

    # Check if it's a video_factory_cache file path
    if 'video_factory_cache' in local_path:
        # Extract media type and filename from path
        # Pattern: .../video_factory_cache/{media_type}/{filename}
        match = re.search(r'video_factory_cache/([^/]+)/([^/]+)$', local_path)
        if match:
            media_type = match.group(1)
            filename = match.group(2)
            return f"http://localhost:8000/api/media/{media_type}/{filename}"

    # Check if it's a YouTube cache file path
    if 'youtube_cache' in local_path:
        # Extract filename from path
        # Pattern: .../youtube_cache/{filename}
        match = re.search(r'youtube_cache/([^/]+)$', local_path)
        if match:
            filename = match.group(1)
            return f"http://localhost:8000/api/media/youtube/{filename}"

    # For any other local path, try to extract just the filename and serve via generic endpoint
    if local_path.startswith('/'):
        filename = os.path.basename(local_path)
        # Determine media type from extension
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.mp4', '.webm', '.mkv', '.mov']:
            return f"http://localhost:8000/api/media/stock_videos/{filename}"
        elif ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
            return f"http://localhost:8000/api/media/web_images/{filename}"

    return local_path


class VideoCreationOrchestrator:
    """
    Orchestrates the complete video creation pipeline.
    """

    def __init__(self):
        self.max_retries = 2

    async def create_video(
        self,
        topic: str,
        voice_id: str = "nPczCjzI2devNBz1zQrb",
        compile_video: bool = True,
        provided_script: Optional[str] = None
    ) -> AsyncGenerator[ProgressEvent, None]:
        """
        Main orchestration method. Yields progress events as SSE data.

        Args:
            topic: The video topic (or the script itself if provided_script is set)
            voice_id: ElevenLabs voice ID
            compile_video: If True, compile final video. If False, stop after media retrieval.
            provided_script: If provided, skip script generation and use this script directly.
        """
        # If a script is provided, derive a topic from it for media routing
        if provided_script:
            # Use first sentence or first 50 words as the derived topic
            first_sentence = provided_script.split('.')[0].strip()
            derived_topic = first_sentence if len(first_sentence) < 100 else ' '.join(first_sentence.split()[:15])
            print(f"\n🎬 Starting video creation with provided script (derived topic: {derived_topic[:50]}...)")
        else:
            derived_topic = topic
            print(f"\n🎬 Starting video creation for: {topic[:50]}...")

        try:
            # ============================================================
            # STEP 1: Generate Script (or use provided script)
            # ============================================================
            if provided_script:
                # User provided their own script, skip generation
                yield ProgressEvent(
                    step="script",
                    status=StepStatus.RUNNING,
                    message="Using provided script..."
                )
                await asyncio.sleep(0)

                script = provided_script.strip()

                yield ProgressEvent(
                    step="script",
                    status=StepStatus.DONE,
                    message=f"Script ready ({len(script.split())} words)",
                    data={"script": script}
                )
                await asyncio.sleep(0)
            else:
                # Generate script from topic
                yield ProgressEvent(
                    step="script",
                    status=StepStatus.RUNNING,
                    message="Generating script..."
                )
                await asyncio.sleep(0)

                script = await self._generate_script(topic)

                yield ProgressEvent(
                    step="script",
                    status=StepStatus.DONE,
                    message=f"Script generated ({len(script.split())} words)",
                    data={"script": script}
                )
                await asyncio.sleep(0)

            # ============================================================
            # STEP 2: Generate TTS Audio
            # ============================================================
            yield ProgressEvent(
                step="tts",
                status=StepStatus.RUNNING,
                message="Generating audio..."
            )
            await asyncio.sleep(0)

            audio_data = await self._generate_tts(script, voice_id)

            yield ProgressEvent(
                step="tts",
                status=StepStatus.DONE,
                message=f"Audio generated ({audio_data['provider']})",
                data={
                    "audio_url": audio_data["audio_url"],
                    "provider": audio_data["provider"],
                }
            )
            await asyncio.sleep(0)

            # ============================================================
            # STEP 3: Segment Script into Clauses
            # ============================================================
            yield ProgressEvent(
                step="clauses",
                status=StepStatus.RUNNING,
                message="Segmenting script into clauses..."
            )
            await asyncio.sleep(0)

            clause_result = await segment_script_into_clauses(
                script=script,
                alignment=audio_data.get("alignment")
            )
            clauses = clause_result.clauses
            total_duration = clause_result.total_duration

            yield ProgressEvent(
                step="clauses",
                status=StepStatus.DONE,
                message=f"Segmented into {len(clauses)} clauses",
                data={
                    "clauses": [c.model_dump() for c in clauses],
                    "total_duration": total_duration
                }
            )
            await asyncio.sleep(0)

            # ============================================================
            # STEP 4: Route Clauses to Media Types
            # ============================================================
            yield ProgressEvent(
                step="routing",
                status=StepStatus.RUNNING,
                message="Planning media for each clause..."
            )
            await asyncio.sleep(0)

            from media_router import route_clauses_to_media

            clause_dicts = [c.model_dump() for c in clauses]
            media_instructions = await route_clauses_to_media(derived_topic, clause_dicts)

            yield ProgressEvent(
                step="routing",
                status=StepStatus.DONE,
                message=f"Planned {len(media_instructions)} media items",
                data={
                    "instructions": [
                        {
                            "clause_id": inst.clause_id,
                            "media_type": inst.media_type.value,
                            "query": inst.query,
                            "duration": inst.duration
                        }
                        for inst in media_instructions
                    ]
                }
            )
            await asyncio.sleep(0)

            # ============================================================
            # STEP 5: Retrieve Media
            # ============================================================
            yield ProgressEvent(
                step="media",
                status=StepStatus.RUNNING,
                message="Retrieving media..."
            )
            await asyncio.sleep(0)

            from media_router import retrieve_all_media

            media_results = await retrieve_all_media(media_instructions, derived_topic)

            successful = len([r for r in media_results if r.media_url])
            failed = len([r for r in media_results if not r.media_url])

            yield ProgressEvent(
                step="media",
                status=StepStatus.DONE,
                message=f"Retrieved {successful} media items ({failed} failed)",
                data={
                    "media": [
                        {
                            "clause_id": r.clause_id,
                            "media_type": r.media_type.value,
                            "media_url": _convert_local_path_to_api_url(r.media_url),
                            "duration": r.duration,
                            "error": r.error
                        }
                        for r in media_results
                    ]
                }
            )
            await asyncio.sleep(0)

            if not compile_video:
                # Return data without compiling
                yield ProgressEvent(
                    step="complete",
                    status=StepStatus.DONE,
                    message="Media ready for compilation",
                    data={
                        "script": script,
                        "clauses": clause_dicts,
                        "media": [
                            {
                                "clause_id": r.clause_id,
                                "media_type": r.media_type.value,
                                "media_url": _convert_local_path_to_api_url(r.media_url),
                                "duration": r.duration
                            }
                            for r in media_results
                        ],
                        "audio_url": audio_data["audio_url"],
                        "alignment": audio_data.get("alignment"),
                        "total_duration": total_duration
                    }
                )
                return

            # ============================================================
            # STEP 6: Compile Final Video
            # ============================================================
            print(f"\n🎬 Starting video compilation with {len([r for r in media_results if r.media_url])} media items...")

            yield ProgressEvent(
                step="compile",
                status=StepStatus.RUNNING,
                message="Compiling final video..."
            )
            await asyncio.sleep(0)

            final_video_url = await self._compile_video(
                clauses=clauses,
                media_results=media_results,
                audio_data=audio_data,
                script=script
            )
            print(f"✓ Video compilation complete")

            yield ProgressEvent(
                step="compile",
                status=StepStatus.DONE,
                message="Video compiled successfully",
                data={"video_url": final_video_url}
            )
            await asyncio.sleep(0)

            # ============================================================
            # COMPLETE
            # ============================================================
            yield ProgressEvent(
                step="complete",
                status=StepStatus.DONE,
                message="Video creation complete!",
                data={
                    "video_url": final_video_url,
                    "script": script,
                    "total_duration": total_duration
                }
            )

        except Exception as e:
            print(f"Orchestrator error: {e}")
            import traceback
            traceback.print_exc()
            yield ProgressEvent(
                step="error",
                status=StepStatus.ERROR,
                message=str(e)
            )

    async def _generate_script(self, topic: str) -> str:
        """Generate script with retry logic."""
        from script_generation import generate_script_text

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return await generate_script_text(topic)
            except Exception as e:
                last_error = e
                print(f"Script generation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1)

        raise last_error or Exception("Script generation failed")

    async def _generate_tts(self, script: str, voice_id: str) -> Dict[str, Any]:
        """Generate TTS audio with ElevenLabs primary, Google fallback."""
        from tts import generate_elevenlabs_tts_with_timing, generate_tts_with_timing
        from config import ELEVENLABS_AVAILABLE, TTS_AVAILABLE

        if ELEVENLABS_AVAILABLE:
            try:
                print(f"Using ElevenLabs TTS with voice: {voice_id}")
                audio_content, alignment = generate_elevenlabs_tts_with_timing(script, voice_id=voice_id)
                audio_base64 = base64.b64encode(audio_content).decode('utf-8')

                alignment_dict = None
                if alignment:
                    alignment_dict = {
                        'characters': list(alignment.characters),
                        'character_start_times_seconds': list(alignment.character_start_times_seconds),
                        'character_end_times_seconds': list(alignment.character_end_times_seconds),
                    }

                return {
                    "audio_url": f"data:audio/mp3;base64,{audio_base64}",
                    "audio_content": audio_content,
                    "alignment": alignment_dict,
                    "provider": "elevenlabs",
                    "voice_id": voice_id
                }
            except Exception as e:
                print(f"ElevenLabs TTS failed: {e}, trying Google...")

        if TTS_AVAILABLE:
            try:
                print("Using Google TTS")
                audio_content, timepoints = generate_tts_with_timing(script)
                audio_base64 = base64.b64encode(audio_content).decode('utf-8')

                return {
                    "audio_url": f"data:audio/wav;base64,{audio_base64}",
                    "audio_content": audio_content,
                    "alignment": None,
                    "timepoints": timepoints,
                    "provider": "google"
                }
            except Exception as e:
                print(f"Google TTS failed: {e}")
                raise

        raise Exception("No TTS provider available")

    async def _compile_video(
        self,
        clauses: List[Clause],
        media_results: List[Any],
        audio_data: Dict[str, Any],
        script: str
    ) -> str:
        """Compile final video from media segments."""
        from video_compilation import compile_video_from_clauses

        # Build segments list for compiler
        segments = []
        for clause, media in zip(clauses, media_results):
            if media.media_url:
                segments.append({
                    "clause_id": clause.clause_id,
                    "media_url": media.media_url,
                    "media_type": media.media_type.value,
                    "start_time": clause.start_time,
                    "duration": clause.next_start_time - clause.start_time,
                    "text": clause.text
                })

        print(f"   Built {len(segments)} segments from {len(media_results)} media results")

        if not segments:
            raise Exception("No media segments available for compilation - all media retrieval failed")

        # Compile video
        result = await compile_video_from_clauses(
            segments=segments,
            audio_data=audio_data,
            script=script,
            alignment=audio_data.get("alignment")
        )

        return result
