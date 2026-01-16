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
        voice_id: str = "nPczCjzI2devNBz1zQrb",
        staged: bool = False,
        background_type: str = "videos"  # "videos", "images", or "ai"
    ) -> AsyncGenerator[ProgressEvent, None]:
        """
        Main orchestration method. Yields progress events as SSE data.

        Args:
            topic: The video topic/prompt
            voice_id: ElevenLabs voice ID (default: Brian)
            staged: If True, stop after script + media search (no TTS/compile)
            background_type: Type of background media ("videos", "images", or "ai")
        """
        script_response: Optional[ScriptResponse] = None
        audio_data: Optional[Dict[str, Any]] = None
        video_urls: List[str] = []
        scenes_with_videos: List[Dict[str, Any]] = []
        ai_images_result: Optional[Dict[str, Any]] = None

        print(f"\n🎬 Starting video creation - background_type: {background_type}")

        try:
            # Step 1: Generate Script
            yield ProgressEvent(
                step="script",
                status=StepStatus.RUNNING,
                message="Generating script..."
            )

            script_response = await self._generate_script_with_retry(topic)

            # Include sections in script data for displaying raw vs edited
            sections_data = None
            if script_response.sections:
                sections_data = [s.dict() if hasattr(s, 'dict') else s.model_dump() for s in script_response.sections]

            yield ProgressEvent(
                step="script",
                status=StepStatus.DONE,
                message=f"Script generated ({len(script_response.script.split())} words)",
                data={
                    "script": script_response.script,
                    "scenes": [s.dict() if hasattr(s, 'dict') else s.model_dump() for s in script_response.scenes],
                    "sections": sections_data,
                }
            )

            # Step 2: Search for videos/images based on background_type
            # Only search for what the user selected, not everything
            search_videos = background_type == "videos"
            search_images = background_type == "images"

            if background_type == "ai":
                search_message = "Generating AI images..."
            elif background_type == "images":
                search_message = "Searching for Google images..."
            else:  # videos
                search_message = "Searching for stock videos..."

            yield ProgressEvent(
                step="videos",
                status=StepStatus.RUNNING,
                message=search_message
            )

            # Search based on selected type (AI mode doesn't need stock video/image search initially)
            if background_type != "ai":
                scenes_with_videos = await self._search_videos_for_scenes(
                    script_response.scenes,
                    topic=topic,
                    script=script_response.script,
                    search_videos=search_videos,
                    search_images=search_images
                )
            else:
                # For AI mode, create empty scene placeholders (no video/image search)
                scenes_with_videos = [
                    {
                        "scene_number": scene.scene_number,
                        "section_name": scene.section_name or f"Scene {scene.scene_number}",
                        "video_search_query": scene.search_query,
                        "video_url": None,
                        "video_source": None,
                        "image_search_query": scene.search_query,
                        "images": [],
                    }
                    for scene in script_response.scenes
                ]

            # If AI mode, also generate AI images
            print(f"   Checking AI mode: background_type='{background_type}' (is 'ai': {background_type == 'ai'})")
            if background_type == "ai":
                print("   🎨 AI mode detected - starting AI image generation...")
                try:
                    from image_generate import generate_image_for_video, FAL_AVAILABLE
                    print(f"   FAL_AVAILABLE: {FAL_AVAILABLE}")

                    # Build sections list for AI generation (one image per section)
                    sections_for_ai = []
                    if script_response.sections and len(script_response.sections) > 0:
                        sections_for_ai = [
                            {"name": s.name, "text": s.text}
                            for s in script_response.sections
                        ]
                    else:
                        # Fallback: use first 100 words of script as single section
                        words = script_response.script.split()
                        first_section_text = " ".join(words[:min(100, len(words))])
                        sections_for_ai = [{"name": "HOOK", "text": first_section_text}]

                    ai_images_result = await generate_image_for_video(
                        topic=topic,
                        script=script_response.script,
                        first_section_text=sections_for_ai[0]["text"] if sections_for_ai else "",
                        sections=sections_for_ai
                    )
                    total_images = len(ai_images_result.get('images', []))
                    print(f"✓ AI image generation complete: {total_images} images")
                except Exception as e:
                    print(f"⚠ AI image generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    ai_images_result = {"images": [], "prompts": [], "error": str(e)}

            # Report search completion based on what was searched
            video_urls = [s["video_url"] for s in scenes_with_videos if s.get("video_url")]
            images_found = sum(1 for s in scenes_with_videos if s.get("images"))
            total_scenes = len(script_response.scenes)

            # Build completion message based on what was actually searched
            completion_parts = []
            if background_type == "videos":
                completion_parts.append(f"Found {len(video_urls)}/{total_scenes} videos")
            elif background_type == "images":
                completion_parts.append(f"Found {images_found}/{total_scenes} image sets")
            elif background_type == "ai" and ai_images_result:
                ai_success = sum(1 for img in ai_images_result.get("images", []) if img.get("url"))
                total_expected = len(ai_images_result.get("images", []))
                completion_parts.append(f"Generated {ai_success}/{total_expected} AI images")

            completion_msg = ", ".join(completion_parts) if completion_parts else "Media search complete"

            # Build video step data
            videos_data: Dict[str, Any] = {"videos": scenes_with_videos}
            if ai_images_result:
                videos_data["ai_images"] = ai_images_result.get("images", [])
                videos_data["ai_prompts"] = ai_images_result.get("prompts", [])

            yield ProgressEvent(
                step="videos",
                status=StepStatus.DONE,
                message=completion_msg,
                data=videos_data
            )

            # If staged mode, stop here and emit paused event
            if staged:
                paused_data: Dict[str, Any] = {
                    "script": script_response.script,
                    "scenes": [s.dict() if hasattr(s, 'dict') else s.model_dump() for s in script_response.scenes],
                    "sections": sections_data,
                    "videos": scenes_with_videos,
                    "voice_id": voice_id,
                }
                if ai_images_result:
                    paused_data["ai_images"] = ai_images_result.get("images", [])
                    paused_data["ai_prompt"] = ai_images_result.get("prompt")

                yield ProgressEvent(
                    step="paused",
                    status=StepStatus.DONE,
                    message="Ready for TTS and video compilation",
                    data=paused_data
                )
                return  # Stop here in staged mode

            # Step 3: TTS (only in non-staged mode)
            yield ProgressEvent(
                step="tts",
                status=StepStatus.RUNNING,
                message="Generating audio..."
            )

            audio_data = await self._generate_tts_with_fallback(script_response.script, voice_id)

            # Report TTS completion
            yield ProgressEvent(
                step="tts",
                status=StepStatus.DONE,
                message=f"Audio generated ({audio_data.get('provider', 'unknown')})",
                data={"provider": audio_data.get("provider")}
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
        self, scenes: List[VideoScene], topic: str = "", script: str = "",
        search_videos: bool = True, search_images: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for videos and/or images for each scene with fallback queries.

        Args:
            scenes: List of video scenes
            topic: Video topic for context
            script: Full script for context
            search_videos: If True, search for stock videos
            search_images: If True, search for Google images
        """
        from image_search import generate_image_search_queries, search_images_with_query

        results = []

        # Generate image search queries if needed
        image_queries = []
        if search_images:
            print(f"\n🔍 Generating image search queries for {len(scenes)} scenes...")
            scenes_for_llm = [
                {
                    "scene_number": s.scene_number,
                    "section_name": s.section_name or f"Scene {s.scene_number}",
                    "description": s.description
                }
                for s in scenes
            ]
            image_queries = generate_image_search_queries(topic, script, scenes_for_llm)

        search_type = "videos" if search_videos and not search_images else "images" if search_images and not search_videos else "videos and images"
        print(f"\n🎬 Searching {search_type} for {len(scenes)} scenes:")

        for i, scene in enumerate(scenes):
            section_name = scene.section_name or f"Scene {scene.scene_number}"
            image_query = image_queries[i] if i < len(image_queries) else scene.search_query

            print(f"\n[{i+1}/{len(scenes)}] {section_name}")
            if search_videos:
                print(f"  Video query: \"{scene.search_query}\"")
            if search_images:
                print(f"  Image query: \"{image_query}\"")

            # Build tasks based on what we need to search
            tasks = []
            task_types = []

            if search_videos:
                tasks.append(asyncio.create_task(
                    self._search_video_with_fallbacks_async(scene)
                ))
                task_types.append("video")

            if search_images:
                tasks.append(asyncio.create_task(
                    search_images_with_query(
                        search_query=image_query,
                        fallback_query=scene.search_query
                    )
                ))
                task_types.append("image")

            # Run searches in parallel
            task_results = await asyncio.gather(*tasks)

            # Parse results based on task types
            video_result = None
            image_result = {"query": image_query, "images": []}

            for j, task_type in enumerate(task_types):
                if task_type == "video":
                    video_result = task_results[j]
                elif task_type == "image":
                    image_result = task_results[j]

            results.append({
                "scene_number": scene.scene_number,
                "section_name": section_name,
                "video_search_query": scene.search_query,
                "video_url": video_result.get("url") if video_result else None,
                "video_source": video_result.get("source") if video_result else None,
                "image_search_query": image_result.get("query", image_query),
                "images": image_result.get("images", []),
            })

        found_videos = sum(1 for r in results if r.get("video_url"))
        found_images = sum(1 for r in results if r.get("images"))

        completion_parts = []
        if search_videos:
            completion_parts.append(f"{found_videos}/{len(scenes)} videos")
        if search_images:
            completion_parts.append(f"{found_images}/{len(scenes)} image sets")
        print(f"\n✅ Search complete: {', '.join(completion_parts)}\n")

        return results

    async def _search_video_with_fallbacks_async(
        self, scene: VideoScene
    ) -> Optional[Dict[str, Any]]:
        """Async wrapper for _search_video_with_fallbacks."""
        return await asyncio.to_thread(self._search_video_with_fallbacks_sync, scene)

    def _search_video_with_fallbacks_sync(
        self, scene: VideoScene
    ) -> Optional[Dict[str, Any]]:
        """Search for a video with fallback queries if primary fails (synchronous)."""
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

    def _calculate_scene_durations(
        self,
        alignment: Optional[Dict[str, Any]],
        scenes: List[Dict[str, Any]],
        audio_duration: float,
        num_videos: int
    ) -> List[float]:
        """
        Calculate scene durations from ElevenLabs alignment data and scene word boundaries.

        Args:
            alignment: ElevenLabs alignment dict with character-level timing
            scenes: List of scene dicts with word_start and word_end
            audio_duration: Total audio duration in seconds
            num_videos: Number of videos (for fallback)

        Returns:
            List of durations in seconds for each scene
        """
        if not alignment or not scenes:
            print("⚠ No alignment data or scenes, using equal duration fallback")
            equal_duration = audio_duration / num_videos
            return [equal_duration] * num_videos

        try:
            characters = alignment.get('characters', [])
            start_times = alignment.get('character_start_times_seconds', [])
            end_times = alignment.get('character_end_times_seconds', [])

            if not characters or not start_times or not end_times:
                print("⚠ Alignment data incomplete, using equal duration fallback")
                equal_duration = audio_duration / num_videos
                return [equal_duration] * num_videos

            # Build word boundaries from character data
            word_boundaries = []
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
            scene_durations = []

            for scene in sorted(scenes, key=lambda s: s.get('scene_number', 0)):
                word_start = scene.get('word_start', 0)
                word_end = scene.get('word_end', len(word_boundaries))

                if word_start is None or word_end is None:
                    # Scene missing word boundaries, use equal portion
                    scene_durations.append(audio_duration / num_videos)
                    continue

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
            equal_duration = audio_duration / num_videos
            return [equal_duration] * num_videos

    async def _compile_video(
        self,
        video_urls: List[str],
        audio_url: str,
        script: str,
        alignment: Optional[Dict[str, Any]],
        tts_provider: str,
        scenes: List[Dict[str, Any]]
    ) -> str:
        """
        Compile the final video using the video_compilation module.

        Delegates to video_compilation.compile_video() which handles:
        - Downloading and trimming videos
        - Generating subtitles (ElevenLabs alignment or fallback)
        - Burning captions with karaoke effects
        - Adding audio and encoding final video
        """
        from video_compilation import compile_video
        from models import CompileVideoRequest, SceneTimingInfo

        # Build scene timing info for duration calculation
        scene_timing = [
            SceneTimingInfo(
                scene_number=s.get('scene_number', i + 1),
                word_start=s.get('word_start'),
                word_end=s.get('word_end')
            )
            for i, s in enumerate(scenes)
        ] if scenes else None

        # Let video_compilation calculate scene durations from alignment
        # This ensures consistent timing calculation with better debugging
        scene_durations = None  # video_compilation will calculate from alignment + scenes

        # Create request for video_compilation module
        request = CompileVideoRequest(
            video_urls=video_urls,
            audio_url=audio_url,
            script=script,
            alignment=alignment,
            tts_provider=tts_provider,
            scenes=scene_timing,
            scene_durations=scene_durations
        )

        # Delegate to video_compilation module
        response = await compile_video(request)
        return response.video_url
