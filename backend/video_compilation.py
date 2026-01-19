import base64
import os
import tempfile
import subprocess
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from models import CompileVideoRequest, CompileVideoResponse
from config import FFMPEG_EXECUTABLE, TTS_AVAILABLE, tts_client, get_media_duration
from tts import generate_tts_with_timing
from subtitles import create_ass_from_timepoints, create_ass_fallback, create_ass_from_elevenlabs_alignment



async def _download_audio(audio_url: str, temp_path: Path) -> Path:
    """Download audio from URL or base64 data."""
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
        response = requests.get(audio_url, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to download audio")
        content_type = response.headers.get('content-type', '')
        if 'mp3' in content_type or 'mpeg' in content_type or audio_url.endswith('.mp3'):
            audio_path = temp_path / "audio.mp3"
        else:
            audio_path = temp_path / "audio.wav"
        with open(audio_path, 'wb') as f:
            f.write(response.content)
    return audio_path


def _get_audio_duration(audio_path: Path) -> float:
    """Get audio duration using ffmpeg (more portable than ffprobe)."""
    import re
    try:
        # Use ffmpeg -i to get duration (works without ffprobe)
        probe_cmd = [
            FFMPEG_EXECUTABLE, '-i', str(audio_path), '-f', 'null', '-'
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        # Parse duration from stderr (ffmpeg outputs info to stderr)
        output = result.stderr
        # Look for "Duration: HH:MM:SS.ms" pattern
        match = re.search(r'Duration:\s*(\d+):(\d+):(\d+)\.(\d+)', output)
        if match:
            hours, minutes, seconds, centiseconds = match.groups()
            duration = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(centiseconds) / 100
            return duration
        # Fallback: look for "time=" at the end
        match = re.search(r'time=(\d+):(\d+):(\d+)\.(\d+)', output)
        if match:
            hours, minutes, seconds, centiseconds = match.groups()
            duration = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(centiseconds) / 100
            return duration
        raise ValueError(f"Could not parse duration from ffmpeg output")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get audio duration: {e}")


async def _download_media(url: str, temp_path: Path, index: int, is_image: bool) -> Optional[Path]:
    """Download a media file (video or image)."""
    try:
        # Handle local files (YouTube clips)
        if url.startswith('/') or url.startswith('./'):
            local_path = Path(url)
            if local_path.exists():
                ext = local_path.suffix or '.mp4'
                dest_path = temp_path / f"media_{index}{ext}"
                import shutil
                shutil.copy(url, dest_path)
                return dest_path
            return None

        # Download remote file with proper headers to avoid blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=60, stream=True, headers=headers)
        if response.status_code != 200:
            print(f"      Download failed: HTTP {response.status_code}")
            return None

        content_type = response.headers.get('content-type', '')
        if is_image:
            if 'png' in content_type or url.lower().endswith('.png'):
                ext = '.png'
            elif 'webp' in content_type or url.lower().endswith('.webp'):
                ext = '.webp'
            else:
                ext = '.jpg'
        else:
            ext = '.mp4'

        media_path = temp_path / f"media_{index}{ext}"
        with open(media_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return media_path
    except Exception as e:
        print(f"      Download error: {e}")
        return None


def _process_media_segment(
    input_path: Path,
    output_path: Path,
    duration: float,
    is_image: bool
) -> bool:
    """Process a single media segment to the correct duration."""
    try:
        if is_image:
            # Convert image to video with Ken Burns effect
            fps = 30
            # Use round() to get nearest frame count, add 1 to ensure we have enough frames
            total_frames = round(duration * fps) + 1
            zoom_per_frame = 0.12 / max(total_frames, 1)

            zoompan_filter = (
                f"scale=2160:3840:force_original_aspect_ratio=increase,"
                f"crop=2160:3840,"
                f"zoompan=z='min(1+{zoom_per_frame}*on,1.12)':"
                f"x='iw/2-(iw/zoom/2)':"
                f"y='ih/2-(ih/zoom/2)':"
                f"d={total_frames}:"
                f"s=1080x1920:"
                f"fps={fps}"
            )

            cmd = [
                FFMPEG_EXECUTABLE, '-y',
                '-i', str(input_path),
                '-vf', zoompan_filter,
                '-t', f'{duration:.3f}',  # Force exact output duration
                '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
                '-b:v', '3000k',
                '-pix_fmt', 'yuv420p',
                '-an',
                str(output_path)
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=120, text=True)
                return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Fallback: simple image to video
                simple_cmd = [
                    FFMPEG_EXECUTABLE, '-y',
                    '-loop', '1',
                    '-i', str(input_path),
                    '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
                    '-b:v', '3000k',
                    '-pix_fmt', 'yuv420p',
                    '-t', f'{duration:.3f}',
                    '-r', '30',
                    '-an',
                    str(output_path)
                ]
                subprocess.run(simple_cmd, check=True, capture_output=True, timeout=90, text=True)
                return True
        else:
            # Trim video to duration
            cmd = [
                FFMPEG_EXECUTABLE, '-y',
                '-i', str(input_path),
                '-t', f'{duration:.3f}',
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
                '-b:v', '3000k',
                '-an',
                str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=60, text=True)
            return True
    except Exception as e:
        print(f"      Process error: {e}")
        return False


def _concatenate_segments(segment_paths: List[Path], output_path: Path) -> None:
    """Concatenate multiple video segments using concat demuxer (memory-efficient)."""
    if len(segment_paths) == 1:
        import shutil
        shutil.copy(segment_paths[0], output_path)
        return

    # Use concat demuxer (much more memory-efficient than filter_complex)
    # Create a concat list file
    concat_list_path = output_path.parent / "concat_list.txt"
    with open(concat_list_path, 'w') as f:
        for path in segment_paths:
            # Use absolute paths and escape single quotes
            f.write(f"file '{str(path.absolute())}'\n")

    # First try with stream copy (fastest, no re-encoding)
    cmd = [
        FFMPEG_EXECUTABLE, '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_list_path),
        '-c', 'copy',
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300, text=True)
        print(f"✓ Concatenated {len(segment_paths)} segments (stream copy)")
        return
    except subprocess.CalledProcessError:
        print("⚠ Stream copy failed, trying re-encode...")

    # Fallback: re-encode with concat demuxer (still memory-efficient)
    cmd = [
        FFMPEG_EXECUTABLE, '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_list_path),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
        '-b:v', '3000k',
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300, text=True)
        print(f"✓ Concatenated {len(segment_paths)} segments (re-encoded)")
    except subprocess.CalledProcessError as e:
        print(f"✗ Concatenation failed: {e.stderr[:500] if e.stderr else 'Unknown error'}")
        raise HTTPException(status_code=500, detail="Failed to concatenate video segments")


def _check_video_duration(video_path: Path, audio_duration: float) -> None:
    """Check video duration and log warning if shorter than audio (but don't extend - replacement happens earlier)."""
    try:
        video_duration = get_media_duration(str(video_path))

        if video_duration < audio_duration:
            diff = audio_duration - video_duration
            print(f"⚠ Video is {diff:.1f}s shorter than audio ({video_duration:.1f}s vs {audio_duration:.1f}s)")
            print(f"   Note: Media replacement should have been handled during retrieval")
        else:
            print(f"✓ Video duration OK: {video_duration:.1f}s (audio: {audio_duration:.1f}s)")
    except Exception as e:
        print(f"⚠ Could not check video duration: {e}")


def _generate_subtitles(request: CompileVideoRequest, temp_path: Path, audio_duration: float) -> Optional[Path]:
    """Generate ASS subtitles."""
    subtitle_path = None
    script_text = request.script.strip()

    if request.alignment and request.tts_provider == "elevenlabs":
        try:
            class AlignmentData:
                def __init__(self, data):
                    self.characters = data.get('characters', [])
                    self.character_start_times_seconds = data.get('character_start_times_seconds', [])
                    self.character_end_times_seconds = data.get('character_end_times_seconds', [])

            alignment_obj = AlignmentData(request.alignment)
            ass_content = create_ass_from_elevenlabs_alignment(script_text, alignment_obj, audio_duration)
            subtitle_path = temp_path / "subtitles.ass"
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            print("✓ Created subtitles from ElevenLabs alignment")
        except Exception as e:
            print(f"⚠ ElevenLabs subtitles failed: {e}")

    if subtitle_path is None and TTS_AVAILABLE and tts_client:
        try:
            selected_voice = request.voice_name or "en-US-Neural2-H"
            _, timepoints = generate_tts_with_timing(script_text, voice_name=selected_voice)
            if timepoints:
                ass_content = create_ass_from_timepoints(script_text, timepoints, audio_duration, style="3words")
                subtitle_path = temp_path / "subtitles.ass"
                with open(subtitle_path, 'w', encoding='utf-8') as f:
                    f.write(ass_content)
                print("✓ Created subtitles from Google TTS")
        except Exception as e:
            print(f"⚠ Google TTS subtitles failed: {e}")

    if subtitle_path is None:
        try:
            ass_content = create_ass_fallback(script_text, audio_duration)
            subtitle_path = temp_path / "subtitles.ass"
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            print("✓ Created fallback subtitles")
        except Exception as e:
            print(f"⚠ Fallback subtitles failed: {e}")

    return subtitle_path


def _finalize_video(
    video_path: Path,
    audio_path: Path,
    subtitle_path: Optional[Path],
    output_path: Path,
    audio_duration: float
) -> None:
    """Combine video, audio, and subtitles into final output."""
    if subtitle_path and subtitle_path.exists():
        subtitle_escaped = str(subtitle_path).replace('\\', '/').replace(':', '\\:')
        video_filter = f"ass='{subtitle_escaped}'"
        cmd = [
            FFMPEG_EXECUTABLE, '-y',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-vf', video_filter,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
            '-b:v', '3000k',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-t', f'{audio_duration:.3f}',
            str(output_path)
        ]
    else:
        cmd = [
            FFMPEG_EXECUTABLE, '-y',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
            '-b:v', '3000k',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-t', f'{audio_duration:.3f}',
            str(output_path)
        ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180, text=True)
        print("✓ Final video created")
    except subprocess.CalledProcessError as e:
        print(f"✗ Final video failed: {e.stderr[:500] if e.stderr else 'Unknown error'}")
        raise HTTPException(status_code=500, detail="Failed to create final video")


async def compile_video(request: CompileVideoRequest) -> CompileVideoResponse:
    """
    Combine media segments with audio using ffmpeg.

    Processes video_urls (or images if use_images=True) and combines
    them with audio and subtitles.
    """
    # Validate inputs
    if not request.video_urls or len(request.video_urls) == 0:
        raise HTTPException(status_code=400, detail="video_urls is required")

    if not request.audio_url:
        raise HTTPException(status_code=400, detail="Audio URL is required")

    if not request.script or not request.script.strip():
        raise HTTPException(status_code=400, detail="Script text is required for captions")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Download audio
        audio_data = None
        if request.audio_url.startswith('data:audio'):
            # Extract base64 audio data and detect format
            audio_base64 = request.audio_url.split(',')[1]
            audio_data = base64.b64decode(audio_base64)
            # Detect audio format from data URL (e.g., data:audio/mp3;base64, or data:audio/wav;base64,)
            if 'audio/mp3' in request.audio_url or 'audio/mpeg' in request.audio_url:
                audio_path = temp_path / "audio.mp3"
            else:
                audio_path = temp_path / "audio.wav"
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
        else:
            # Download audio from URL
            audio_response = requests.get(request.audio_url, timeout=30)
            if audio_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to download audio")
            # Detect format from content-type header or URL
            content_type = audio_response.headers.get('content-type', '')
            if 'mp3' in content_type or 'mpeg' in content_type or request.audio_url.endswith('.mp3'):
                audio_path = temp_path / "audio.mp3"
            else:
                audio_path = temp_path / "audio.wav"
            with open(audio_path, 'wb') as f:
                f.write(audio_response.content)

        # Download videos or images
        video_paths = []
        use_images = getattr(request, 'use_images', False)

        # Headers to avoid blocks from image hosts
        download_headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        if use_images:
            print(f"🖼️  Downloading {len(request.video_urls)} images...")
            for i, image_url in enumerate(request.video_urls):
                try:
                    response = requests.get(image_url, timeout=30, stream=True, headers=download_headers)
                    if response.status_code != 200:
                        print(f"Warning: Failed to download image {i+1} (HTTP {response.status_code}): {image_url[:50]}...")
                        continue

                    # Determine image format from content-type or URL
                    content_type = response.headers.get('content-type', '')
                    if 'png' in content_type or image_url.lower().endswith('.png'):
                        ext = 'png'
                    elif 'gif' in content_type or image_url.lower().endswith('.gif'):
                        ext = 'gif'
                    elif 'webp' in content_type or image_url.lower().endswith('.webp'):
                        ext = 'webp'
                    else:
                        ext = 'jpg'

                    image_path = temp_path / f"image_{i+1}.{ext}"
                    with open(image_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # We'll convert to video after we know the durations
                    video_paths.append(str(image_path))
                    print(f"  ✓ Downloaded image {i+1}")
                except Exception as e:
                    print(f"Error downloading image {i+1}: {e}")
                    continue
        else:
            print(f"📹 Downloading {len(request.video_urls)} videos...")
            for i, video_url in enumerate(request.video_urls):
                try:
                    video_path = temp_path / f"video_{i+1}.mp4"

                    # Check if this is a local file path (YouTube clips) or a URL
                    if video_url.startswith('/') or video_url.startswith('./'):
                        # Local file - copy it to temp directory
                        import shutil
                        if Path(video_url).exists():
                            shutil.copy(video_url, video_path)
                            video_paths.append(str(video_path))
                            print(f"  ✓ Copied local video {i+1}")
                        else:
                            print(f"Warning: Local video file not found: {video_url}")
                            continue
                    else:
                        # Remote URL - download it
                        video_response = requests.get(video_url, timeout=30, stream=True, headers=download_headers)
                        if video_response.status_code != 200:
                            print(f"Warning: Failed to download video {i+1} (HTTP {video_response.status_code}): {video_url[:50]}...")
                            continue

                        with open(video_path, 'wb') as f:
                            for chunk in video_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        video_paths.append(str(video_path))
                        print(f"  ✓ Downloaded video {i+1}")
                except Exception as e:
                    print(f"Error downloading video {i+1}: {e}")
                    continue

        if not video_paths:
            raise HTTPException(status_code=500, detail="Failed to download any media")
        
        # Get audio duration using ffmpeg
        audio_duration = None
        try:
            audio_duration = get_media_duration(str(audio_path))
            print(f"✓ Audio duration detected: {audio_duration:.3f}s")
        except Exception as e:
            print(f"✗ ERROR: Could not get audio duration: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to determine audio duration: {e}. Please ensure audio file is valid.")
        
        if audio_duration is None or audio_duration <= 0:
            raise HTTPException(status_code=500, detail="Invalid audio duration detected. Please check audio file.")
        
        # Calculate scene durations - use provided durations or equal split
        print(f"📹 Compiling video with {len(video_paths)} media files")
        print(f"🎵 Audio duration: {audio_duration:.2f}s")

        if request.scene_durations and len(request.scene_durations) == len(video_paths):
            scene_durations = request.scene_durations
            print(f"⏱️  Using provided scene durations: {[f'{d:.2f}s' for d in scene_durations]}")
        else:
            equal_duration = audio_duration / len(video_paths)
            scene_durations = [equal_duration] * len(video_paths)
            print(f"⏱️  Using equal scene duration: {equal_duration:.2f}s per video")

        # Process each media item (video or image) to its scene duration
        trimmed_videos = []
        for i, media_path in enumerate(video_paths):
            scene_duration = scene_durations[i]
            trimmed_path = temp_path / f"trimmed_video_{i+1}.mp4"

            if use_images:
                # Convert image to video segment with Ken Burns effect (slow zoom in)
                fps = 30
                total_frames = int(scene_duration * fps)
                # Zoom: start at 1.0, end at ~1.12 (12% zoom over duration)
                zoom_per_frame = 0.12 / max(total_frames, 1)

                # zoompan filter: z=zoom level, d=total frames, s=output size, fps=frame rate
                # Scale image to cover output area (force_original_aspect_ratio=increase ensures no distortion)
                # Then crop to exact size needed for zoompan input
                zoompan_filter = (
                    f"scale=2160:3840:force_original_aspect_ratio=increase,"
                    f"crop=2160:3840,"
                    f"zoompan=z='min(1+{zoom_per_frame}*on,1.12)':"
                    f"x='iw/2-(iw/zoom/2)':"
                    f"y='ih/2-(ih/zoom/2)':"
                    f"d={total_frames}:"
                    f"s=1080x1920:"
                    f"fps={fps}"
                )

                convert_cmd = [
                    FFMPEG_EXECUTABLE, '-y',
                    '-i', str(media_path),
                    '-vf', zoompan_filter,
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
                    '-b:v', '3000k',
                    '-pix_fmt', 'yuv420p',
                    '-an',
                    str(trimmed_path)
                ]
                try:
                    # Longer timeout for Ken Burns processing
                    result = subprocess.run(convert_cmd, check=True, capture_output=True, timeout=120, text=True)
                    trimmed_videos.append(trimmed_path)
                    print(f"✓ Converted image {i+1} to {scene_duration:.2f}s video with Ken Burns")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    error_msg = getattr(e, 'stderr', '') or str(e)
                    print(f"⚠ Ken Burns failed, trying simple conversion: {error_msg[:200]}")
                    # Fallback: simple image to video without zoom effect
                    # Scale to cover, crop to fit, then convert to video
                    simple_cmd = [
                        FFMPEG_EXECUTABLE, '-y',
                        '-loop', '1',
                        '-i', str(media_path),
                        '-vf', f'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
                        '-b:v', '3000k',
                        '-pix_fmt', 'yuv420p',
                        '-t', f'{scene_duration:.3f}',
                        '-r', str(fps),
                        '-an',
                        str(trimmed_path)
                    ]
                    try:
                        result = subprocess.run(simple_cmd, check=True, capture_output=True, timeout=90, text=True)
                        trimmed_videos.append(trimmed_path)
                        print(f"✓ Converted image {i+1} to {scene_duration:.2f}s video (simple)")
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
                        error_msg = getattr(e2, 'stderr', '') or str(e2)
                        print(f"✗ Failed to convert image {i+1}: {error_msg[:200]}")
                        raise HTTPException(status_code=500, detail=f"Failed to convert image {i+1}")
            else:
                # Trim video to scene duration
                trim_cmd = [
                    FFMPEG_EXECUTABLE, '-y',
                    '-i', str(media_path),
                    '-t', f'{scene_duration:.3f}',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
                    '-b:v', '3000k',
                    '-an',
                    str(trimmed_path)
                ]
                try:
                    result = subprocess.run(trim_cmd, check=True, capture_output=True, timeout=30, text=True)
                    trimmed_videos.append(trimmed_path)
                    print(f"✓ Trimmed video {i+1} to {scene_duration:.2f}s")
                except subprocess.CalledProcessError as e:
                    error_msg = e.stderr if e.stderr else 'Unknown error'
                    print(f"✗ Failed to trim video {i+1}: {error_msg[:200]}")
                    raise HTTPException(status_code=500, detail=f"Failed to trim video {i+1}")
        
        # Output video path
        output_path = temp_path / "final_video.mp4"
        
        # Step 1: Scale videos to vertical format, then concatenate
        concat_video_path = temp_path / "concatenated.mp4"

        # First, scale each video individually (memory-efficient)
        scaled_videos = []
        for i, trimmed_path in enumerate(trimmed_videos):
            scaled_path = temp_path / f"scaled_{i:03d}.mp4"
            scale_cmd = [
                FFMPEG_EXECUTABLE, '-y',
                '-i', str(trimmed_path),
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
                '-b:v', '3000k',
                '-an',
                str(scaled_path)
            ]
            try:
                subprocess.run(scale_cmd, check=True, capture_output=True, timeout=60, text=True)
                scaled_videos.append(scaled_path)
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else 'Unknown error'
                print(f"✗ Failed to scale video {i+1}: {error_msg[:200]}")
                raise HTTPException(status_code=500, detail=f"Failed to scale video {i+1}")

        print(f"✓ Scaled {len(scaled_videos)} videos to vertical format")

        # Then concatenate using concat demuxer (memory-efficient)
        if len(scaled_videos) == 1:
            import shutil
            shutil.copy(scaled_videos[0], concat_video_path)
        else:
            concat_list_path = temp_path / "concat_list.txt"
            with open(concat_list_path, 'w') as f:
                for path in scaled_videos:
                    f.write(f"file '{str(path.absolute())}'\n")

            concat_cmd = [
                FFMPEG_EXECUTABLE, '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_list_path),
                '-c', 'copy',
                str(concat_video_path)
            ]

            try:
                subprocess.run(concat_cmd, check=True, capture_output=True, timeout=180, text=True)
                print(f"✓ Concatenated {len(scaled_videos)} videos")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else 'Unknown error'
                print(f"✗ FFmpeg concat error: {error_msg[:500]}")
                raise HTTPException(status_code=500, detail="Failed to concatenate videos")
        
        # Step 2: Generate ASS subtitles with timing
        subtitle_path = None
        script_text = request.script.strip()

        # Priority: ElevenLabs alignment > Google TTS timepoints > Fallback
        if request.alignment and request.tts_provider == "elevenlabs":
            # Use ElevenLabs character-level alignment for precise timing
            try:
                print(f"📝 Creating subtitles from ElevenLabs alignment...")

                class AlignmentData:
                    def __init__(self, data):
                        self.characters = data.get('characters', [])
                        self.character_start_times_seconds = data.get('character_start_times_seconds', [])
                        self.character_end_times_seconds = data.get('character_end_times_seconds', [])

                alignment_obj = AlignmentData(request.alignment)
                ass_content = create_ass_from_elevenlabs_alignment(script_text, alignment_obj, audio_duration)
                subtitle_path = temp_path / "subtitles.ass"
                with open(subtitle_path, 'w', encoding='utf-8') as f:
                    f.write(ass_content)
                print(f"✓ Created ASS subtitles with ElevenLabs word-level timing")
            except Exception as e:
                print(f"⚠ Failed to create ElevenLabs subtitles: {e}")
                import traceback
                traceback.print_exc()

        # Fallback to Google TTS timepoints
        if subtitle_path is None and TTS_AVAILABLE and tts_client:
            try:
                print(f"📝 Generating subtitle timings from Google TTS...")
                selected_voice = request.voice_name if request.voice_name else "en-US-Neural2-H"
                _, timepoints = generate_tts_with_timing(script_text, voice_name=selected_voice)

                if timepoints:
                    print(f"   Received {len(timepoints)} timepoints from TTS")
                    ass_content = create_ass_from_timepoints(script_text, timepoints, audio_duration, style="3words")
                    subtitle_path = temp_path / "subtitles.ass"
                    with open(subtitle_path, 'w', encoding='utf-8') as f:
                        f.write(ass_content)
                    print(f"✓ Created ASS subtitles with Google TTS timing")
            except Exception as e:
                print(f"⚠ Failed to generate Google TTS subtitles: {e}")

        # Final fallback: even timing distribution
        if subtitle_path is None:
            try:
                print(f"📝 Creating fallback subtitles with even timing...")
                ass_content = create_ass_fallback(script_text, audio_duration)
                subtitle_path = temp_path / "subtitles.ass"
                with open(subtitle_path, 'w', encoding='utf-8') as f:
                    f.write(ass_content)
                print(f"✓ Created fallback ASS subtitles")
            except Exception as e:
                print(f"⚠ Failed to create fallback subtitles: {e}")
                subtitle_path = None
        
        # Step 2.5: Verify concatenated video duration (don't extend - just log)
        try:
            concat_duration = get_media_duration(str(concat_video_path))
            print(f"📊 Concatenated video duration: {concat_duration:.2f}s (audio: {audio_duration:.2f}s)")

            if concat_duration < audio_duration:
                diff = audio_duration - concat_duration
                print(f"⚠ Video is {diff:.1f}s shorter than audio - media replacement should have happened during retrieval")
        except Exception as e:
            print(f"⚠ Warning: Could not verify concatenated video duration: {e}")
        
        # Step 3: Add audio and burn captions to concatenated video
        if subtitle_path and subtitle_path.exists():
            # Build video filter with ASS subtitles (supports karaoke effects)
            subtitle_path_escaped = str(subtitle_path).replace('\\', '/').replace(':', '\\:')
            # Use ass= filter for ASS format (supports karaoke \k tags)
            video_filter = f"ass='{subtitle_path_escaped}'"
            final_cmd = [
                FFMPEG_EXECUTABLE, '-y',
                '-i', str(concat_video_path),
                '-i', str(audio_path),
                '-vf', video_filter,
                '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
                '-b:v', '3000k',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-t', f'{audio_duration:.3f}',
                str(output_path)
            ]
            print(f"✓ Burning captions into video from {subtitle_path}")
        else:
            # No subtitles, just add audio
            final_cmd = [
                FFMPEG_EXECUTABLE, '-y',
                '-i', str(concat_video_path),
                '-i', str(audio_path),
                '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
                '-b:v', '3000k',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-t', f'{audio_duration:.3f}',
                str(output_path)
            ]
        
        try:
            result = subprocess.run(final_cmd, check=True, capture_output=True, timeout=180, text=True)
            print(f"✓ Final video created with audio")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else 'Unknown error'
            print(f"✗ FFmpeg final error: {error_msg[:500]}")
            raise HTTPException(status_code=500, detail="Failed to combine video and audio")
        
        # Read the final video and return as base64
        with open(output_path, 'rb') as f:
            video_data = f.read()
        
        video_base64 = base64.b64encode(video_data).decode('utf-8')
        video_data_url = f"data:video/mp4;base64,{video_base64}"
        
        return CompileVideoResponse(video_url=video_data_url)


async def compile_video_from_clauses(
    segments: List[Dict[str, Any]],
    audio_data: Dict[str, Any],
    script: str,
    alignment: Optional[Dict[str, Any]] = None
) -> str:
    """
    Compile video from clause-based media segments.

    Args:
        segments: List of segment dicts with:
            - clause_id: int
            - media_url: str (URL or local path)
            - media_type: str (ai_video, stock_video, youtube_video, web_image, ai_image)
            - start_time: float
            - duration: float
            - text: str (clause text)
        audio_data: Dict with audio_url, audio_content, alignment
        script: Full script text
        alignment: Word-level timing from TTS

    Returns:
        Base64 data URL of the compiled video
    """
    print(f"\n🎬 Compiling video from {len(segments)} clause segments")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save audio to file
        if "audio_content" in audio_data and audio_data["audio_content"]:
            audio_path = temp_path / "audio.mp3"
            with open(audio_path, 'wb') as f:
                f.write(audio_data["audio_content"])
        elif audio_data["audio_url"].startswith('data:audio'):
            audio_base64 = audio_data["audio_url"].split(',')[1]
            audio_bytes = base64.b64decode(audio_base64)
            audio_ext = "mp3" if "mp3" in audio_data["audio_url"] else "wav"
            audio_path = temp_path / f"audio.{audio_ext}"
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
        else:
            audio_path = await _download_audio(audio_data["audio_url"], temp_path)

        audio_duration = _get_audio_duration(audio_path)
        print(f"   Audio duration: {audio_duration:.2f}s")

        # Download and process each segment
        processed_paths: List[Path] = []

        for i, seg in enumerate(segments):
            print(f"   [{i+1}/{len(segments)}] Processing {seg['media_type']}: {seg['duration']:.2f}s")

            media_url = seg["media_url"]
            is_image = seg["media_type"] in ("ai_image", "web_image")
            duration = seg["duration"]
            output_path = temp_path / f"segment_{i:03d}.mp4"

            # Get media (local path or download remote)
            media_path = None
            if media_url.startswith(('http://', 'https://')):
                media_path = await _download_media(media_url, temp_path, i, is_image)
            elif os.path.exists(media_url):
                media_path = Path(media_url)

            if not media_path or not media_path.exists():
                print(f"      ✗ Failed to get media for clause {seg['clause_id']}")
                continue

            # Process to exact duration
            success = _process_media_segment(media_path, output_path, duration, is_image)

            if success and output_path.exists():
                processed_paths.append(output_path)
                print(f"      ✓ Segment {i+1} ready ({duration:.2f}s)")
            else:
                print(f"      ✗ Failed to process segment {i+1}")

        if not processed_paths:
            raise HTTPException(status_code=500, detail="No media segments processed successfully")

        print(f"\n   Concatenating {len(processed_paths)} segments...")

        # Concatenate all segments
        concat_path = temp_path / "concatenated.mp4"
        _concatenate_segments(processed_paths, concat_path)

        # Check video duration (just log, don't extend - replacement happens during media retrieval)
        _check_video_duration(concat_path, audio_duration)
        final_video_path = concat_path

        # Generate subtitles using the subtitles module
        subtitle_path = None
        if alignment:
            from subtitles import create_ass_from_elevenlabs_alignment
            subtitle_path = temp_path / "subtitles.ass"
            try:
                ass_content = create_ass_from_elevenlabs_alignment(
                    script=script,
                    alignment=alignment,
                    audio_duration=audio_duration
                )
                with open(subtitle_path, 'w') as f:
                    f.write(ass_content)
                print(f"   ✓ Generated subtitles")
            except Exception as e:
                print(f"   ⚠ Subtitle generation failed: {e}")
                subtitle_path = None

        # Combine video + audio + subtitles
        output_path = temp_path / "final_video.mp4"
        _finalize_video(final_video_path, audio_path, subtitle_path, output_path, audio_duration)

        # Return as base64 data URL
        with open(output_path, 'rb') as f:
            video_bytes = f.read()

        video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        video_data_url = f"data:video/mp4;base64,{video_base64}"

        print(f"\n✅ Video compilation complete ({len(video_bytes) / (1024*1024):.1f} MB)")

        return video_data_url
