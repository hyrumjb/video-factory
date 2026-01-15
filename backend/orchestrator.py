"""
Video Creation Orchestrator

Manages the entire video creation pipeline with progress streaming,
retry logic, and error recovery.
"""

import asyncio
import json
import base64
from typing import AsyncGenerator, Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

from models import ScriptRequest, ScriptResponse, VideoScene, ScriptSection


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    RETRYING = "retrying"


@dataclass
class ProgressEvent:
    step: str  # 'script', 'tts', 'videos', 'compile', 'complete'
    status: StepStatus
    message: Optional[str] = None
    progress: Optional[int] = None  # 0-100
    data: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        d = {
            "step": self.step,
            "status": self.status.value,
        }
        if self.message:
            d["message"] = self.message
        if self.progress is not None:
            d["progress"] = self.progress
        if self.data:
            d["data"] = self.data
        return json.dumps(d)


class VideoCreationOrchestrator:
    """
    Orchestrates the video creation pipeline:
    1. Generate script (with sections and word boundaries)
    2. Generate TTS audio (ElevenLabs primary, Google fallback)
    3. Search for stock videos (with fallback queries)
    4. Compile final video with subtitles
    """

    def __init__(self):
        self.max_retries = 2
        self.video_search_retries = 3

    async def create_video(
        self,
        topic: str,
        voice_id: str = "nPczCjzI2devNBz1zQrb"
    ) -> AsyncGenerator[ProgressEvent, None]:
        """
        Main orchestration method. Yields progress events as SSE data.

        Args:
            topic: The video topic/prompt
            voice_id: ElevenLabs voice ID (default: Brian)
        """
        script_response: Optional[ScriptResponse] = None
        audio_data: Optional[Dict[str, Any]] = None
        video_urls: List[str] = []
        scenes_with_videos: List[Dict[str, Any]] = []

        try:
            # Step 1: Generate Script
            yield ProgressEvent(
                step="script",
                status=StepStatus.RUNNING,
                message="Generating script..."
            )

            script_response = await self._generate_script_with_retry(topic)

            yield ProgressEvent(
                step="script",
                status=StepStatus.DONE,
                message=f"Script generated ({len(script_response.script.split())} words)",
                data={
                    "script": script_response.script,
                    "scenes": [s.dict() if hasattr(s, 'dict') else s.model_dump() for s in script_response.scenes],
                }
            )

            # Step 2 & 3: TTS and Video Search in parallel
            yield ProgressEvent(
                step="tts",
                status=StepStatus.RUNNING,
                message="Generating audio..."
            )
            yield ProgressEvent(
                step="videos",
                status=StepStatus.RUNNING,
                message="Searching for stock videos..."
            )

            # Run TTS and video search concurrently
            tts_task = asyncio.create_task(
                self._generate_tts_with_fallback(script_response.script, voice_id)
            )
            videos_task = asyncio.create_task(
                self._search_videos_for_scenes(script_response.scenes)
            )

            # Wait for both
            audio_data, scenes_with_videos = await asyncio.gather(tts_task, videos_task)

            # Report TTS completion
            yield ProgressEvent(
                step="tts",
                status=StepStatus.DONE,
                message=f"Audio generated ({audio_data.get('provider', 'unknown')})",
                data={"provider": audio_data.get("provider")}
            )

            # Report video search completion
            video_urls = [s["video_url"] for s in scenes_with_videos if s.get("video_url")]
            total_scenes = len(script_response.scenes)

            yield ProgressEvent(
                step="videos",
                status=StepStatus.DONE,
                message=f"Found {len(video_urls)}/{total_scenes} videos",
                data={"videos": scenes_with_videos}
            )

            if not video_urls:
                raise Exception(f"No videos found for any of the {total_scenes} scenes")

            if len(video_urls) < total_scenes:
                print(f"⚠ Warning: Only found {len(video_urls)}/{total_scenes} videos, proceeding anyway")

            # Step 4: Compile final video
            yield ProgressEvent(
                step="compile",
                status=StepStatus.RUNNING,
                message="Compiling final video..."
            )

            final_video_url = await self._compile_video(
                video_urls=video_urls,
                audio_url=audio_data["audio_url"],
                script=script_response.script,
                alignment=audio_data.get("alignment"),
                tts_provider=audio_data.get("provider", "google"),
                scenes=[{
                    "scene_number": s.scene_number,
                    "word_start": s.word_start,
                    "word_end": s.word_end
                } for s in script_response.scenes if s.word_start is not None]
            )

            yield ProgressEvent(
                step="compile",
                status=StepStatus.DONE,
                message="Video compiled successfully"
            )

            # Final completion event with video URL
            yield ProgressEvent(
                step="complete",
                status=StepStatus.DONE,
                message="Video creation complete!",
                data={
                    "video_url": final_video_url,
                    "script": script_response.script,
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

    async def _generate_script_with_retry(self, topic: str) -> ScriptResponse:
        """Generate script with retry logic."""
        from script_generation import generate_script

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                request = ScriptRequest(topic=topic)
                return await generate_script(request)
            except Exception as e:
                last_error = e
                print(f"Script generation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1)  # Brief delay before retry

        raise last_error or Exception("Script generation failed")

    async def _generate_tts_with_fallback(
        self,
        script: str,
        voice_id: str = "nPczCjzI2devNBz1zQrb"
    ) -> Dict[str, Any]:
        """Generate TTS audio with ElevenLabs primary, Google fallback."""
        from tts import generate_elevenlabs_tts_with_timing, generate_tts_with_timing
        from config import ELEVENLABS_AVAILABLE, TTS_AVAILABLE

        # Try ElevenLabs first
        if ELEVENLABS_AVAILABLE:
            try:
                print(f"Attempting ElevenLabs TTS with voice: {voice_id}...")
                audio_content, alignment = generate_elevenlabs_tts_with_timing(script, voice_id=voice_id)
                audio_base64 = base64.b64encode(audio_content).decode('utf-8')

                alignment_dict = None
                if alignment:
                    alignment_dict = {
                        'characters': list(alignment.characters) if hasattr(alignment, 'characters') else [],
                        'character_start_times_seconds': list(alignment.character_start_times_seconds) if hasattr(alignment, 'character_start_times_seconds') else [],
                        'character_end_times_seconds': list(alignment.character_end_times_seconds) if hasattr(alignment, 'character_end_times_seconds') else [],
                    }

                return {
                    "audio_url": f"data:audio/mp3;base64,{audio_base64}",
                    "alignment": alignment_dict,
                    "provider": "elevenlabs",
                    "voice_id": voice_id
                }
            except Exception as e:
                print(f"ElevenLabs TTS failed: {e}, trying Google...")

        # Fallback to Google TTS
        if TTS_AVAILABLE:
            try:
                print("Attempting Google TTS...")
                audio_content, timepoints = generate_tts_with_timing(script)
                audio_base64 = base64.b64encode(audio_content).decode('utf-8')

                return {
                    "audio_url": f"data:audio/wav;base64,{audio_base64}",
                    "alignment": None,  # Google uses timepoints, not alignment
                    "timepoints": timepoints,
                    "provider": "google"
                }
            except Exception as e:
                print(f"Google TTS also failed: {e}")
                raise

        raise Exception("No TTS provider available")

    async def _search_videos_for_scenes(
        self, scenes: List[VideoScene]
    ) -> List[Dict[str, Any]]:
        """Search for videos for each scene with fallback queries."""
        results = []

        print(f"\n🎬 Searching videos for {len(scenes)} scenes:")
        for i, scene in enumerate(scenes):
            section_name = scene.section_name or f"Scene {scene.scene_number}"
            print(f"\n[{i+1}/{len(scenes)}] {section_name} - query: \"{scene.search_query}\"")

            video_result = await self._search_video_with_fallbacks(scene)
            results.append({
                "scene_number": scene.scene_number,
                "section_name": section_name,
                "search_query": scene.search_query,
                "video_url": video_result.get("url") if video_result else None,
                "video_source": video_result.get("source") if video_result else None,
            })

        found_count = sum(1 for r in results if r.get("video_url"))
        print(f"\n✅ Video search complete: {found_count}/{len(scenes)} scenes have videos\n")

        return results

    async def _search_video_with_fallbacks(
        self, scene: VideoScene
    ) -> Optional[Dict[str, Any]]:
        """Search for a video with fallback queries if primary fails."""
        from video_search import search_pexels_videos, search_pixabay_videos

        # Build list of queries to try - from specific to generic
        queries = [
            scene.search_query,
            self._simplify_query(scene.search_query),
            self._get_generic_fallback(scene),
        ]

        # Ultimate fallbacks - these almost always return results on Pexels
        ultimate_fallbacks = ["people", "nature", "city", "ocean", "sky"]

        for query in queries:
            if not query:
                continue

            result = self._try_search(query, scene.scene_number)
            if result:
                return result

        # If all scene-specific queries failed, try ultimate fallbacks
        print(f"  Scene {scene.scene_number}: trying ultimate fallbacks...")
        for fallback in ultimate_fallbacks:
            result = self._try_search(fallback, scene.scene_number)
            if result:
                return result

        print(f"    ✗ No video found for scene {scene.scene_number} (all fallbacks exhausted)")
        return None

    def _try_search(self, query: str, scene_number: int) -> Optional[Dict[str, Any]]:
        """
        Search both Pexels and Pixabay, pick the best result.
        Preference: vertical video > more results > Pexels (has portrait filter)
        """
        from video_search import search_pexels_videos, search_pixabay_videos

        print(f"  Searching for scene {scene_number}: '{query}'")

        pexels_results = []
        pixabay_results = []

        # Query both APIs
        try:
            pexels_results = search_pexels_videos(query, per_page=5, orientation="portrait")
        except Exception as e:
            print(f"    Pexels error: {e}")

        try:
            pixabay_results = search_pixabay_videos(query, per_page=5)
        except Exception as e:
            print(f"    Pixabay error: {e}")

        # Find best vertical video from each source
        pexels_vertical = next((v for v in pexels_results if self._is_vertical(v)), None)
        pixabay_vertical = next((v for v in pixabay_results if self._is_vertical(v)), None)

        # Decision logic: prefer vertical videos, then compare result counts
        if pexels_vertical and pixabay_vertical:
            # Both have vertical - pick source with more results (more variety)
            if len(pixabay_results) > len(pexels_results):
                print(f"    ✓ Pixabay vertical ({len(pixabay_results)} results)")
                return {"url": pixabay_vertical["url"], "source": "pixabay"}
            else:
                print(f"    ✓ Pexels vertical ({len(pexels_results)} results)")
                return {"url": pexels_vertical["url"], "source": "pexels"}

        elif pexels_vertical:
            print(f"    ✓ Pexels vertical (only source with vertical)")
            return {"url": pexels_vertical["url"], "source": "pexels"}

        elif pixabay_vertical:
            print(f"    ✓ Pixabay vertical (only source with vertical)")
            return {"url": pixabay_vertical["url"], "source": "pixabay"}

        # No vertical videos - fall back to any video, prefer more results
        if pexels_results and pixabay_results:
            if len(pixabay_results) > len(pexels_results):
                print(f"    ✓ Pixabay (no vertical, {len(pixabay_results)} results)")
                return {"url": pixabay_results[0]["url"], "source": "pixabay"}
            else:
                print(f"    ✓ Pexels (no vertical, {len(pexels_results)} results)")
                return {"url": pexels_results[0]["url"], "source": "pexels"}

        elif pexels_results:
            print(f"    ✓ Pexels (only source with results)")
            return {"url": pexels_results[0]["url"], "source": "pexels"}

        elif pixabay_results:
            print(f"    ✓ Pixabay (only source with results)")
            return {"url": pixabay_results[0]["url"], "source": "pixabay"}

        return None

    def _simplify_query(self, query: str) -> str:
        """Simplify a search query to 1-2 core words for Pexels."""
        if not query:
            return ""

        # Common words that don't help with stock video search
        skip_words = {
            'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
            'person', 'people', 'man', 'woman',  # too generic alone
        }

        words = query.lower().split()
        useful_words = [w for w in words if w not in skip_words and len(w) > 2]

        if useful_words:
            # Return first useful word (most likely to be the subject)
            return useful_words[0]

        # Fallback to first word if nothing useful
        return words[0] if words else "people"

    def _get_generic_fallback(self, scene: VideoScene) -> str:
        """Get a generic fallback query based on scene description."""
        desc_lower = scene.description.lower() if scene.description else ""

        # Map themes to Pexels-friendly single/double word queries
        theme_map = [
            (["money", "cash", "dollar", "finance", "wealth", "rich"], "money"),
            (["city", "street", "urban", "downtown"], "city"),
            (["nature", "outdoor", "forest", "trees"], "forest"),
            (["ocean", "sea", "water", "beach"], "ocean"),
            (["technology", "computer", "laptop", "screen"], "laptop"),
            (["phone", "mobile", "smartphone"], "phone"),
            (["office", "work", "business", "corporate"], "office"),
            (["food", "eating", "restaurant", "cooking"], "food"),
            (["car", "driving", "road", "traffic"], "car"),
            (["sky", "clouds", "sunset", "sunrise"], "sky"),
        ]

        for keywords, query in theme_map:
            if any(word in desc_lower for word in keywords):
                return query

        # Default fallback - generic but works well on Pexels
        return "people"

    def _is_vertical(self, video: Dict[str, Any]) -> bool:
        """Check if a video is vertical (9:16 aspect ratio)."""
        width = video.get("width", 0)
        height = video.get("height", 0)
        if width and height:
            return height > width
        return False

    async def _compile_video(
        self,
        video_urls: List[str],
        audio_url: str,
        script: str,
        alignment: Optional[Dict[str, Any]],
        tts_provider: str,
        scenes: List[Dict[str, Any]]
    ) -> str:
        """Compile the final video by calling the compile endpoint logic."""
        # Import the compile logic
        import tempfile
        import subprocess
        import requests
        from pathlib import Path
        from config import FFMPEG_EXECUTABLE, FFPROBE_EXECUTABLE

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download audio
            if audio_url.startswith('data:audio'):
                audio_base64 = audio_url.split(',')[1]
                audio_data = base64.b64decode(audio_base64)
                if 'audio/mp3' in audio_url or 'audio/mpeg' in audio_url:
                    audio_path = temp_path / "audio.mp3"
                else:
                    audio_path = temp_path / "audio.wav"
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
            else:
                raise Exception("Audio URL must be base64 data URL")

            # Get audio duration
            probe_cmd = [
                FFPROBE_EXECUTABLE, '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            audio_duration = float(result.stdout.strip())
            print(f"Audio duration: {audio_duration:.2f}s")

            # Download videos
            video_paths = []
            for i, url in enumerate(video_urls):
                try:
                    response = requests.get(url, timeout=30, stream=True)
                    if response.status_code == 200:
                        video_path = temp_path / f"video_{i+1}.mp4"
                        with open(video_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        video_paths.append(str(video_path))
                except Exception as e:
                    print(f"Failed to download video {i+1}: {e}")

            if not video_paths:
                raise Exception("Failed to download any videos")

            # Calculate scene durations (equal split for now)
            scene_duration = audio_duration / len(video_paths)

            # Trim videos
            trimmed_videos = []
            for i, video_path in enumerate(video_paths):
                trimmed_path = temp_path / f"trimmed_{i+1}.mp4"
                trim_cmd = [
                    FFMPEG_EXECUTABLE, '-y',
                    '-i', video_path,
                    '-t', f'{scene_duration:.3f}',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-an',
                    str(trimmed_path)
                ]
                subprocess.run(trim_cmd, capture_output=True, timeout=30)
                trimmed_videos.append(trimmed_path)

            # Concatenate videos
            concat_path = temp_path / "concat.mp4"
            if len(trimmed_videos) == 1:
                # Single video - just scale
                scale_cmd = [
                    FFMPEG_EXECUTABLE, '-y',
                    '-i', str(trimmed_videos[0]),
                    '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-an',
                    str(concat_path)
                ]
                subprocess.run(scale_cmd, capture_output=True, timeout=60)
            else:
                # Multiple videos - concatenate with filter_complex
                filter_parts = []
                for i in range(len(trimmed_videos)):
                    filter_parts.append(f"[{i}:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[v{i}];")
                filter_parts.append("".join([f"[v{i}]" for i in range(len(trimmed_videos))]))
                filter_parts.append(f"concat=n={len(trimmed_videos)}:v=1:a=0[outv]")
                filter_complex = "".join(filter_parts)

                concat_cmd = [FFMPEG_EXECUTABLE, '-y']
                for vp in trimmed_videos:
                    concat_cmd.extend(['-i', str(vp)])
                concat_cmd.extend([
                    '-filter_complex', filter_complex,
                    '-map', '[outv]',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    str(concat_path)
                ])
                subprocess.run(concat_cmd, capture_output=True, timeout=120)

            # Add audio
            output_path = temp_path / "final.mp4"
            audio_cmd = [
                FFMPEG_EXECUTABLE, '-y',
                '-i', str(concat_path),
                '-i', str(audio_path),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                str(output_path)
            ]
            subprocess.run(audio_cmd, capture_output=True, timeout=60)

            # Read and return as base64
            with open(output_path, 'rb') as f:
                video_content = f.read()

            video_base64 = base64.b64encode(video_content).decode('utf-8')
            return f"data:video/mp4;base64,{video_base64}"
