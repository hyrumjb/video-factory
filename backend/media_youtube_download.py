"""
YouTube video downloading module.
Downloads short clips from YouTube for video backgrounds.
"""

import os
import re
import json
import shutil
import tempfile
import subprocess
from typing import Dict, Any, List, Optional
from config import xai_client, FFMPEG_EXECUTABLE, XAI_MODEL, get_media_duration


# Persistent cache directory (survives across runs)
_CACHE_BASE = os.path.join(os.path.dirname(__file__), "youtube_cache")

# Cookie file for YouTube authentication (bypasses bot detection)
_COOKIES_FILE = os.path.join(os.path.dirname(__file__), "cookies.txt")


def _check_ffmpeg() -> bool:
    """Check if FFmpeg is available in system PATH."""
    try:
        result = subprocess.run(
            [FFMPEG_EXECUTABLE, '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def _get_cookiefile() -> Optional[str]:
    """Get the cookie file path if it exists."""
    if os.path.exists(_COOKIES_FILE):
        return _COOKIES_FILE
    return None


# Check FFmpeg on module load
if not _check_ffmpeg():
    print("⚠ CRITICAL: FFmpeg not found in system PATH!")
    print("  → Railway/Nixpacks: Add 'ffmpeg' to nixPkgs in nixpacks.toml")
    print("  → Docker: Add 'RUN apt-get install -y ffmpeg' to Dockerfile")
    print("  → Local: Install FFmpeg via your package manager")

# Cache of downloaded source videos: video_id -> local file path
_downloaded_sources: Dict[str, str] = {}

# Semaphore for parallel processing (3 concurrent jobs safe on most Macs)
import asyncio
_processing_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the processing semaphore."""
    global _processing_semaphore
    if _processing_semaphore is None:
        _processing_semaphore = asyncio.Semaphore(3)
    return _processing_semaphore


def _get_cache_dir() -> str:
    """Get or create the persistent cache directory."""
    os.makedirs(_CACHE_BASE, exist_ok=True)
    return _CACHE_BASE


def _init_cache() -> None:
    """Initialize cache from existing files on disk."""
    global _downloaded_sources
    cache_dir = _get_cache_dir()

    # Scan for existing source videos
    for filename in os.listdir(cache_dir):
        if filename.startswith("source_") and filename.endswith((".mp4", ".mkv", ".webm")):
            video_id = filename.replace("source_", "").rsplit(".", 1)[0]
            filepath = os.path.join(cache_dir, filename)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                _downloaded_sources[video_id] = filepath


def clear_cache() -> None:
    """Clear all cached videos (call manually if needed)."""
    global _downloaded_sources
    cache_dir = _get_cache_dir()

    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        try:
            os.remove(filepath)
        except Exception:
            pass

    _downloaded_sources.clear()
    print(f"   ✓ Cleared YouTube cache: {cache_dir}")


# Initialize cache on module load
_init_cache()


def _extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    import re
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_source_video(video_url: str, video_id: str) -> Optional[str]:
    """
    Download full source video (≤480p) and cache it persistently.
    Returns path to cached file, or None if failed.
    """
    global _downloaded_sources

    MAX_SOURCE_SIZE_MB = 100  # Reject sources larger than this (memory constraint)

    # Check cache first (in-memory)
    if video_id in _downloaded_sources:
        cached_path = _downloaded_sources[video_id]
        if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
            file_size_mb = os.path.getsize(cached_path) / (1024 * 1024)
            if file_size_mb > MAX_SOURCE_SIZE_MB:
                print(f"      ✗ Cached source too large ({file_size_mb:.1f} MB > {MAX_SOURCE_SIZE_MB} MB)")
                del _downloaded_sources[video_id]
                return None
            print(f"      ✓ Cache hit: {video_id} ({file_size_mb:.1f} MB)")
            return cached_path
        else:
            del _downloaded_sources[video_id]

    # Check disk cache
    cache_dir = _get_cache_dir()
    for ext in ['.mp4', '.mkv', '.webm']:
        disk_path = os.path.join(cache_dir, f"source_{video_id}{ext}")
        if os.path.exists(disk_path) and os.path.getsize(disk_path) > 0:
            file_size_mb = os.path.getsize(disk_path) / (1024 * 1024)
            if file_size_mb > MAX_SOURCE_SIZE_MB:
                print(f"      ✗ Cached source too large ({file_size_mb:.1f} MB > {MAX_SOURCE_SIZE_MB} MB)")
                # Remove oversized cached file to free disk space
                try:
                    os.remove(disk_path)
                except:
                    pass
                return None
            print(f"      ✓ Disk cache hit: {video_id} ({file_size_mb:.1f} MB)")
            _downloaded_sources[video_id] = disk_path
            return disk_path

    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        print("      ✗ yt-dlp not installed")
        return None

    try:
        output_template = os.path.join(cache_dir, f"source_{video_id}.%(ext)s")

        print(f"      Downloading source: {video_id}...")

        ydl_opts = {
            # Use mp4 container to avoid heavy merging (Railway RAM constraint)
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/mp4/best[height<=720]",
            "merge_output_format": "mp4",
            "concurrent_fragment_downloads": 4,  # Reduced for Railway memory
            "noplaylist": True,
            "writesubtitles": False,
            "writethumbnail": False,
            "quiet": True,
            "no_warnings": True,
            "outtmpl": output_template,
            "noprogress": True,  # Avoid progress bar issues
            "extractor_args": {"youtube": {"player_client": ["web"]}},
            "impersonate": "chrome",  # Mimic Chrome browser headers
        }

        # Add cookies if available (bypasses bot detection on Railway IPs)
        cookiefile = _get_cookiefile()
        if cookiefile:
            ydl_opts["cookiefile"] = cookiefile

        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as dl_error:
            # yt-dlp sometimes fails on rename but file is actually downloaded
            print(f"      ⚠ yt-dlp warning: {str(dl_error)[:100]}")

        # Check for .part files and rename them if the final file doesn't exist
        for ext in ['.mp4', '.webm', '.mkv']:
            part_path = os.path.join(cache_dir, f"source_{video_id}{ext}.part")
            final_path = os.path.join(cache_dir, f"source_{video_id}{ext}")
            if os.path.exists(part_path) and not os.path.exists(final_path):
                try:
                    import shutil
                    shutil.move(part_path, final_path)
                    print(f"      ✓ Renamed .part file to final")
                except Exception as rename_err:
                    print(f"      ⚠ Could not rename .part: {rename_err}")

        # Find downloaded file
        for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov']:
            path = os.path.join(cache_dir, f"source_{video_id}{ext}")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                file_size_mb = os.path.getsize(path) / (1024 * 1024)

                # Check size limit
                if file_size_mb > MAX_SOURCE_SIZE_MB:
                    print(f"      ✗ Downloaded source too large ({file_size_mb:.1f} MB > {MAX_SOURCE_SIZE_MB} MB)")
                    try:
                        os.remove(path)
                    except:
                        pass
                    return None

                print(f"      ✓ Downloaded: {video_id} ({file_size_mb:.1f} MB)")
                _downloaded_sources[video_id] = path
                return path

        print(f"      ✗ Download failed: no file found")
        return None

    except Exception as e:
        print(f"      ✗ Download error: {e}")
        return None


def clip_from_source(
    source_path: str,
    output_path: str,
    start_time: float,
    duration: float
) -> Optional[str]:
    """
    Create a vertical clip from a cached source video.

    Two-step process for speed:
    1. Fast clip with -c copy (no re-encode, instant)
    2. Re-encode small clip to vertical format
    """
    try:
        print(f"      Clipping {duration:.0f}s from t={start_time:.0f}s...")

        # Step 1: Fast clip with stream copy (no re-encode)
        temp_clip = output_path.replace('.mp4', '_raw.mp4')

        copy_cmd = [
            FFMPEG_EXECUTABLE, '-y',
            '-ss', str(start_time),
            '-i', source_path,
            '-t', str(duration),
            '-c', 'copy',  # No re-encode - instant
            '-an',
            temp_clip
        ]

        result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0 or not os.path.exists(temp_clip):
            print("      ⚠ Copy-clip failed, trying direct encode...")
            # Fallback: direct encode from source
            return _direct_encode_clip(source_path, output_path, start_time, duration)

        # Step 2: Re-encode small clip to vertical format
        encode_cmd = [
            FFMPEG_EXECUTABLE, '-y',
            '-i', temp_clip,
            '-vf', 'scale=1080:-2,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
            '-b:v', '3000k',
            '-pix_fmt', 'yuv420p',
            '-r', '30',
            '-an',
            output_path
        ]

        result = subprocess.run(encode_cmd, capture_output=True, text=True, timeout=60)

        # Clean up temp file
        if os.path.exists(temp_clip):
            os.remove(temp_clip)

        if result.returncode != 0 or not os.path.exists(output_path):
            print(f"      ✗ Encode failed: {result.stderr[-200:] if result.stderr else 'Unknown error'}")
            return None

        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"      ✓ Created clip ({output_size_mb:.1f} MB)")
        return output_path

    except subprocess.TimeoutExpired:
        print("      ✗ Clip timed out")
        return None
    except Exception as e:
        print(f"      ✗ Clip error: {e}")
        return None


def _direct_encode_clip(
    source_path: str,
    output_path: str,
    start_time: float,
    duration: float
) -> Optional[str]:
    """Fallback: direct encode from source (slower but more reliable)."""
    try:
        cmd = [
            FFMPEG_EXECUTABLE, '-y',
            '-ss', str(start_time),
            '-i', source_path,
            '-t', str(duration),
            '-vf', 'scale=1080:-2,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
            '-b:v', '3000k',
            '-pix_fmt', 'yuv420p',
            '-r', '30',
            '-an',
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)

        if result.returncode != 0 or not os.path.exists(output_path):
            return None

        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"      ✓ Created clip via direct encode ({output_size_mb:.1f} MB)")
        return output_path

    except Exception:
        return None


def generate_youtube_search_queries(
    topic: str,
    script: str,
    scenes: List[Dict[str, Any]]
) -> List[str]:
    """
    Generate YouTube search queries for scenes using LLM.
    Creates a general query based on topic, with variations for interesting scene details.

    Args:
        topic: The overall video topic
        script: The full script text
        scenes: List of scene dicts with 'scene_number', 'section_name', 'description'

    Returns:
        List of YouTube search queries (one per scene)
    """
    if not xai_client:
        # Fallback: use topic for all scenes
        return [f"{topic} documentary" for _ in scenes]

    try:
        scenes_info = "\n".join([
            f"Scene {s.get('scene_number', i+1)} ({s.get('section_name', 'Scene')}): {s.get('description', '')[:100]}"
            for i, s in enumerate(scenes)
        ])

        prompt = f"""Generate YouTube search queries to find documentary/educational video clips about the topic.

TOPIC: {topic}

SCRIPT:
{script[:500]}

SCENES:
{scenes_info}

CRITICAL RULES:
1. EVERY search query MUST include the topic "{topic}" or a very close variation
2. The base query should be "{topic}" plus words like "documentary", "history", "explained", or "footage"
3. Only add scene-specific details if there's a VERY important visual element unique to that scene
4. Most scenes should use the SAME base query (e.g., "{topic} documentary")
5. Keep queries 2-5 words total

GOOD EXAMPLES for topic "Andrew Carnegie":
- "Andrew Carnegie documentary"
- "Andrew Carnegie history"
- "Andrew Carnegie biography"
- "Carnegie steel industry" (only if steel is specifically mentioned)

BAD EXAMPLES (too generic, missing topic):
- "city street walking"
- "person reading book"
- "factory footage"

Return ONLY a JSON array of search queries, one per scene:
["query for scene 1", "query for scene 2", ...]"""

        response = xai_client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.5
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON array from response
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            queries = json.loads(json_match.group())
            if len(queries) == len(scenes):
                print(f"✓ Generated {len(queries)} YouTube search queries")
                for i, q in enumerate(queries):
                    print(f"   Scene {i+1}: \"{q}\"")
                return queries

        # Fallback if parsing failed
        print("⚠ Failed to parse YouTube queries, using topic-based fallback")
        return [f"{topic} documentary" for _ in scenes]

    except Exception as e:
        print(f"⚠ Error generating YouTube queries: {e}")
        return [f"{topic} documentary" for _ in scenes]


def search_youtube(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search YouTube using yt-dlp and return video info.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        List of video info dicts with 'id', 'title', 'url', 'duration'
    """
    try:
        # Use yt-dlp to search YouTube (no duration filter - we only download 30s anyway)
        cmd = [
            'yt-dlp',
            f'ytsearch{max_results}:{query}',
            '--dump-json',
            '--no-download',
            '--no-playlist',
            '--no-warnings',
            '--extractor-args', 'youtube:player_client=web',
            '--impersonate', 'chrome',  # Mimic Chrome browser headers
        ]

        # Add cookies if available (bypasses bot detection)
        cookiefile = _get_cookiefile()
        if cookiefile:
            cmd.extend(['--cookies', cookiefile])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Check for actual errors (not just warnings)
        # If we have results in stdout, use them even if returncode is non-zero (warnings are OK)
        if result.returncode != 0 and not result.stdout.strip():
            # Extract the actual error message, ignoring "WARNING:" lines
            error_lines = [line for line in result.stderr.split('\n')
                          if line.strip() and not line.strip().startswith('WARNING')]
            if error_lines:
                print(f"⚠ yt-dlp search failed: {' '.join(error_lines[:2])[:200]}")
            else:
                print(f"⚠ yt-dlp search failed (returncode {result.returncode})")
            return []

        videos = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    videos.append({
                        'id': data.get('id'),
                        'title': data.get('title', 'Unknown'),
                        'url': f"https://www.youtube.com/watch?v={data.get('id')}",
                        'duration': data.get('duration', 0),
                        'thumbnail': data.get('thumbnail'),
                    })
                except json.JSONDecodeError:
                    continue

        print(f"   Found {len(videos)} videos for \"{query}\"")
        return videos

    except subprocess.TimeoutExpired:
        print(f"⚠ YouTube search timed out for \"{query}\"")
        return []
    except Exception as e:
        print(f"⚠ YouTube search error: {e}")
        return []


def pick_start_time(duration: Optional[float], clip: float = 30.0) -> float:
    """
    Pick a smart start time to skip intros and get mid-video content.

    Args:
        duration: Total video duration in seconds
        clip: Clip duration we want to download

    Returns:
        Start time in seconds
    """
    if not duration or duration <= clip + 15:
        return 0.0
    return min(
        duration - clip,
        int(duration * 0.3)  # skip intros - start at 30% into video
    )


def download_youtube_clip(
    video_url: str,
    output_path: str,
    clip_duration: float = 30.0,
    video_duration: Optional[float] = None
) -> Optional[str]:
    """
    Download a short clip from YouTube using ffmpeg streaming.

    Uses yt-dlp to get the stream URL, then ffmpeg to download only
    the specific section needed. This is more reliable than download_sections.

    Args:
        video_url: YouTube video URL
        output_path: Path to save the output file
        clip_duration: Duration to download in seconds (default 30)
        video_duration: Total video duration (used to pick smart start time)

    Returns:
        Path to downloaded file, or None if failed
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        print("      ✗ yt-dlp not installed. Run: pip install yt-dlp")
        return None

    try:
        # Calculate smart start time to skip intros
        start = pick_start_time(video_duration, clip_duration)

        print(f"      Streaming clip: t={start:.0f}s, duration={clip_duration:.0f}s...")

        # Step 1: Get the direct stream URL using yt-dlp (no download)
        ydl_opts = {
            # Use mp4 container to avoid heavy merging (Railway RAM constraint)
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/mp4/best[height<=720]",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "extractor_args": {"youtube": {"player_client": ["web"]}},
            "impersonate": "chrome",  # Mimic Chrome browser headers
        }

        # Add cookies if available (bypasses bot detection on Railway IPs)
        cookiefile = _get_cookiefile()
        if cookiefile:
            ydl_opts["cookiefile"] = cookiefile

        stream_url = None
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            if info:
                stream_url = info.get('url')
                # If no direct URL, try to get from formats
                if not stream_url and info.get('formats'):
                    # Find best matching format
                    for fmt in reversed(info['formats']):
                        if fmt.get('url') and fmt.get('vcodec') != 'none':
                            height = fmt.get('height', 0)
                            if height and height <= 720:
                                stream_url = fmt['url']
                                break
                    # Fallback to any video format
                    if not stream_url:
                        for fmt in reversed(info['formats']):
                            if fmt.get('url') and fmt.get('vcodec') != 'none':
                                stream_url = fmt['url']
                                break

        if not stream_url:
            print("      ✗ Could not get stream URL")
            return None

        # Step 2: Use ffmpeg to download only the section we need
        # -ss before -i for fast seeking (input seeking)
        ffmpeg_cmd = [
            FFMPEG_EXECUTABLE, '-y',
            '-ss', str(start),
            '-i', stream_url,
            '-t', str(clip_duration),
            '-vf', 'scale=1080:-2,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
            '-b:v', '3000k',
            '-pix_fmt', 'yuv420p',
            '-r', '30',
            '-an',
            output_path
        ]

        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes should be enough for streaming a short clip
        )

        if result.returncode != 0 or not os.path.exists(output_path):
            stderr_lines = result.stderr.split('\n') if result.stderr else []
            error_lines = [line for line in stderr_lines if 'error' in line.lower() or 'invalid' in line.lower()]
            if error_lines:
                print(f"      ✗ Convert failed: {' | '.join(error_lines[:3])}")
            else:
                print(f"      ✗ Convert failed: {' | '.join(stderr_lines[-3:])}")
            return None

        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"      ✓ Created clip ({output_size_mb:.1f} MB)")
        return output_path

    except subprocess.TimeoutExpired:
        print("      ✗ Stream timeout (video may be geo-restricted or slow)")
        return None
    except Exception as e:
        print(f"      ✗ Download error: {e}")
        return None


async def download_youtube_for_scene(
    query: str,
    scene_number: int,
    output_dir: str,
    clip_duration: float = 30
) -> Dict[str, Any]:
    """
    Search YouTube and download a clip for a single scene.

    Args:
        query: YouTube search query
        scene_number: Scene number for naming
        output_dir: Directory to save downloaded clip
        clip_duration: Duration of clip to download

    Returns:
        Dict with scene_number, query, video_path, error
    """
    print(f"\n   [Scene {scene_number}] Searching: \"{query}\"")

    # Search YouTube
    videos = search_youtube(query, max_results=3)

    if not videos:
        return {
            "scene_number": scene_number,
            "query": query,
            "video_path": None,
            "error": "No videos found"
        }

    # Try to download from the first few results
    for video in videos:
        # Skip short videos - longer videos have more usable b-roll
        if video.get('duration', 0) and video['duration'] < 90:
            print(f"      Skipping short video ({video['duration']}s < 90s)")
            continue

        output_path = os.path.join(output_dir, f"youtube_scene_{scene_number}.mp4")

        result = download_youtube_clip(
            video_url=video['url'],
            output_path=output_path,
            clip_duration=clip_duration,
            video_duration=video.get('duration')
        )

        if result:
            return {
                "scene_number": scene_number,
                "query": query,
                "video_path": result,
                "video_title": video['title'],
                "video_url": video['url'],
                "error": None
            }

        print("      Trying next video...")

    return {
        "scene_number": scene_number,
        "query": query,
        "video_path": None,
        "error": "Failed to download any videos"
    }


async def download_youtube_for_scenes(
    topic: str,
    script: str,
    scenes: List[Dict[str, Any]],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point: Download YouTube clips for all scenes.

    Strategy:
    1. Search YouTube once for the topic
    2. Download each unique source video ONCE and cache it
    3. Create clips from cached sources (fast ffmpeg seek)

    Args:
        topic: The overall video topic
        script: The full script text
        scenes: List of scene dicts
        output_dir: Directory to save clips (uses temp dir if not provided)

    Returns:
        Dict with 'clips' (list of results), 'queries' (list of search queries)
    """
    print(f"\n🎬 Downloading YouTube clips for: {topic}")

    # Use persistent cache directory for clips
    if not output_dir:
        output_dir = _get_cache_dir()

    # Step 1: Single search for the topic
    search_query = topic
    print(f"   Step 1: Searching YouTube for \"{search_query}\"...")

    all_videos = search_youtube(search_query, max_results=10)

    # Filter out short videos (< 90s)
    videos = [v for v in all_videos if v.get('duration', 0) >= 90]
    print(f"   Found {len(videos)} usable videos (90s+ duration)")

    if not videos:
        print("   ⚠ No long videos found, using all results")
        videos = all_videos

    if not videos:
        print("   ✗ No videos found at all")
        return {"clips": [], "query": search_query, "output_dir": output_dir}

    # Step 2: Figure out which unique videos we need
    # Map scene index -> video info
    scene_video_map = []
    for i, scene in enumerate(scenes):
        video_index = i % len(videos)
        video = videos[video_index]
        video_id = _extract_video_id(video['url']) or video.get('id')
        scene_video_map.append({
            "scene": scene,
            "video": video,
            "video_id": video_id,
            "video_index": video_index
        })

    # Get unique video IDs we need to download
    unique_video_ids = list(set(m["video_id"] for m in scene_video_map if m["video_id"]))
    print(f"   Step 2: Downloading {len(unique_video_ids)} unique source videos in parallel...")

    # Download each unique source video in parallel
    async def download_source(video_id: str) -> None:
        video_info = next((m["video"] for m in scene_video_map if m["video_id"] == video_id), None)
        if video_info:
            sem = _get_semaphore()
            async with sem:
                await asyncio.to_thread(download_source_video, video_info['url'], video_id)

    download_tasks = [download_source(vid) for vid in unique_video_ids if vid]
    await asyncio.gather(*download_tasks)

    # Step 3: Create clips from cached sources (parallel processing)
    print(f"   Step 3: Creating {len(scenes)} clips in parallel (3 workers)...")

    async def process_scene(i: int, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single scene - runs in thread pool with semaphore."""
        sem = _get_semaphore()
        async with sem:
            return await asyncio.to_thread(_process_scene_sync, i, mapping, videos, search_query, output_dir)

    def _process_scene_sync(i: int, mapping: Dict[str, Any], videos: List[Dict], search_query: str, output_dir: str) -> Dict[str, Any]:
        """Synchronous scene processing (runs in thread)."""
        scene = mapping["scene"]
        video = mapping["video"]
        video_id = mapping["video_id"]
        scene_number = scene.get('scene_number', i + 1)

        print(f"\n   [Scene {scene_number}] Clipping from: \"{video['title'][:40]}...\"")

        # Check if source is cached
        if video_id not in _downloaded_sources:
            source_path = download_source_video(video['url'], video_id)
        else:
            source_path = _downloaded_sources.get(video_id)

        if not source_path or not os.path.exists(source_path):
            # Try fallback videos
            success = False
            for fallback_idx in range(mapping["video_index"] + 1, len(videos)):
                fallback_video = videos[fallback_idx]
                fallback_id = _extract_video_id(fallback_video['url']) or fallback_video.get('id')
                print(f"      Trying fallback: {fallback_video['title'][:30]}...")

                source_path = download_source_video(fallback_video['url'], fallback_id)
                if source_path:
                    video = fallback_video
                    video_id = fallback_id
                    success = True
                    break

            if not success:
                return {
                    "scene_number": scene_number,
                    "query": search_query,
                    "video_path": None,
                    "error": "Failed to download source"
                }

        # Calculate start time (skip intros, vary by scene)
        video_duration = video.get('duration', 300)
        clip_duration = 30
        base_start = pick_start_time(video_duration, clip_duration)
        start_time = base_start + (i * 15) % max(video_duration - clip_duration - base_start, 1)

        output_path = os.path.join(output_dir, f"youtube_scene_{scene_number}.mp4")

        result = clip_from_source(
            source_path=source_path,
            output_path=output_path,
            start_time=start_time,
            duration=clip_duration
        )

        if result:
            return {
                "scene_number": scene_number,
                "query": search_query,
                "video_path": result,
                "video_title": video['title'],
                "video_url": video['url'],
                "error": None
            }
        else:
            return {
                "scene_number": scene_number,
                "query": search_query,
                "video_path": None,
                "error": "Failed to create clip"
            }

    # Process all scenes in parallel
    tasks = [process_scene(i, mapping) for i, mapping in enumerate(scene_video_map)]
    results = await asyncio.gather(*tasks)

    successful = sum(1 for r in results if r.get('video_path'))
    print(f"\n   ✅ Created {successful}/{len(scenes)} YouTube clips")

    return {
        "clips": results,
        "query": search_query,
        "output_dir": output_dir
    }


async def download_single_youtube_clip(
    query: str,
    topic: str,
    target_duration: float = 30.0
) -> Optional[Dict[str, Any]]:
    """
    Search YouTube and download a single clip.
    Used by the media router for individual clause visuals.

    Args:
        query: Search query for YouTube
        topic: The overall video topic (for context)
        target_duration: Target clip duration in seconds

    Returns:
        Dict with 'video_path', 'video_title', 'video_url' or None if failed
    """
    print(f"   📺 Searching YouTube: \"{query}\"")

    # Search YouTube
    videos = search_youtube(query, max_results=5)

    # Filter to longer videos (prefer 90s+ for better b-roll)
    long_videos = [v for v in videos if v.get('duration', 0) >= 90]
    if not long_videos:
        long_videos = videos

    if not long_videos:
        print(f"   ⚠ No YouTube videos found for: {query}")
        return None

    # Try each video until one works
    sem = _get_semaphore()
    output_dir = _get_cache_dir()

    for video in long_videos:
        video_id = _extract_video_id(video['url']) or video.get('id')

        if not video_id:
            continue

        output_filename = f"yt_clip_{video_id}_{int(target_duration)}s.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # If clip already exists, return it
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"   ✓ Using cached clip: {output_filename}")
            return {
                "video_path": output_path,
                "video_title": video.get('title', ''),
                "video_url": video.get('url', '')
            }

        # Download source if not cached
        async with sem:
            source_path = await asyncio.to_thread(download_source_video, video['url'], video_id)

        if not source_path:
            print(f"      Trying next video...")
            continue

        # Get source duration and pick start time
        video_duration = video.get('duration', 300)
        start_time = pick_start_time(video_duration, target_duration)

        async with sem:
            result = await asyncio.to_thread(
                clip_from_source,
                source_path,
                output_path,
                start_time,
                target_duration
            )

        if result:
            return {
                "video_path": result,
                "video_title": video.get('title', ''),
                "video_url": video.get('url', '')
            }

        print(f"      Clip failed, trying next video...")

    print(f"   ⚠ All {len(long_videos)} videos failed for: {query}")
    return None


# For testing
if __name__ == "__main__":
    import asyncio

    async def test():
        result = await download_youtube_for_scenes(
            topic="The history of coffee",
            script="Coffee has a fascinating history. It was discovered in Ethiopia...",
            scenes=[
                {"scene_number": 1, "section_name": "HOOK", "description": "Coffee's origins"},
                {"scene_number": 2, "section_name": "BODY1", "description": "Ethiopian discovery"},
            ]
        )
        print(f"\nResults: {result}")

    asyncio.run(test())
