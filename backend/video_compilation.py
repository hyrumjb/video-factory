import base64
import tempfile
import subprocess
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from models import CompileVideoRequest, CompileVideoResponse
from config import FFMPEG_EXECUTABLE, FFPROBE_EXECUTABLE, TTS_AVAILABLE, tts_client
from tts import generate_tts_with_timing
from subtitles import create_ass_from_timepoints, create_ass_fallback, create_ass_from_elevenlabs_alignment


def calculate_scene_durations_from_alignment(
    alignment: Dict[str, Any],
    scenes: List[Dict[str, Any]],
    audio_duration: float,
    script: Optional[str] = None
) -> List[float]:
    """
    Calculate scene durations from ElevenLabs alignment data and scene word boundaries.

    Args:
        alignment: ElevenLabs alignment dict with 'characters', 'character_start_times_seconds', 'character_end_times_seconds'
        scenes: List of scene dicts with 'scene_number', 'word_start', 'word_end'
        audio_duration: Total audio duration in seconds
        script: Optional script text to verify word count matches

    Returns:
        List of durations in seconds for each scene
    """
    if not alignment or not scenes:
        print(f"⚠ Missing data: alignment={bool(alignment)}, scenes={len(scenes) if scenes else 0}")
        return []

    try:
        characters = alignment.get('characters', [])
        start_times = alignment.get('character_start_times_seconds', [])
        end_times = alignment.get('character_end_times_seconds', [])

        if not characters or not start_times or not end_times:
            print("⚠ Alignment data incomplete, cannot calculate scene durations")
            return []

        print(f"📊 SCENE TIMING DEBUG:")
        print(f"   Alignment: {len(characters)} chars, {len(start_times)} start_times, {len(end_times)} end_times")
        print(f"   Scenes received: {len(scenes)}")
        for s in scenes:
            print(f"      Scene {s.get('scene_number')}: word_start={s.get('word_start')}, word_end={s.get('word_end')}")

        # Build word boundaries from character data
        word_boundaries: List[Dict[str, Any]] = []
        current_word_index = 0
        word_start_time: Optional[float] = None
        word_end_time: Optional[float] = None
        current_word = ""

        for i, char in enumerate(characters):
            if i >= len(start_times) or i >= len(end_times):
                break

            if char == ' ' or char == '\n':
                # End of word
                if word_start_time is not None and current_word.strip():
                    word_boundaries.append({
                        'word_index': current_word_index,
                        'word': current_word.strip(),
                        'start': word_start_time,
                        'end': word_end_time
                    })
                    current_word_index += 1
                word_start_time = None
                word_end_time = None
                current_word = ""
            else:
                # Part of a word
                if word_start_time is None:
                    word_start_time = start_times[i]
                word_end_time = end_times[i]
                current_word += char

        # Don't forget the last word
        if word_start_time is not None and current_word.strip():
            word_boundaries.append({
                'word_index': current_word_index,
                'word': current_word.strip(),
                'start': word_start_time,
                'end': word_end_time
            })

        total_alignment_words = len(word_boundaries)
        print(f"✓ Built {total_alignment_words} word boundaries from alignment")

        # Print timing for first and last words
        if word_boundaries:
            print(f"   First word: '{word_boundaries[0]['word']}' at {word_boundaries[0]['start']:.3f}s")
            print(f"   Last word: '{word_boundaries[-1]['word']}' ends at {word_boundaries[-1]['end']:.3f}s")

        # Verify word count against script if provided
        if script:
            script_words = script.split()
            script_word_count = len(script_words)
            if script_word_count != total_alignment_words:
                print(f"⚠ WORD COUNT MISMATCH: script has {script_word_count} words, alignment has {total_alignment_words} words")
                print(f"   Script first 5 words: {script_words[:5]}")
                print(f"   Alignment first 5 words: {[wb['word'] for wb in word_boundaries[:5]]}")
            else:
                print(f"✓ Word counts match: {script_word_count} words")

        # Find max word_end from scenes to verify alignment coverage
        max_scene_word_end = max((s.get('word_end', 0) or 0) for s in scenes)
        if max_scene_word_end > total_alignment_words:
            print(f"⚠ Scene word_end ({max_scene_word_end}) exceeds alignment word count ({total_alignment_words})")
            print(f"   Will clamp word_end values to {total_alignment_words}")

        # Create a map of word_index -> timing
        word_timing_map = {wb['word_index']: wb for wb in word_boundaries}

        # Calculate duration for each scene based on word boundaries
        scene_durations: List[float] = []

        print(f"\n📊 SCENE TIMING CALCULATION (audio duration: {audio_duration:.3f}s):")

        # Sort scenes by scene_number to ensure correct order
        sorted_scenes = sorted(scenes, key=lambda s: s.get('scene_number', 0))

        for scene in sorted_scenes:
            word_start = scene.get('word_start')
            word_end = scene.get('word_end')

            if word_start is None or word_end is None:
                print(f"   Scene {scene.get('scene_number')}: SKIPPED (word_start={word_start}, word_end={word_end})")
                continue

            # Clamp word indices to valid range
            orig_word_start, orig_word_end = word_start, word_end
            word_start = max(0, min(word_start, total_alignment_words - 1))
            word_end = max(word_start + 1, min(word_end, total_alignment_words))

            if orig_word_start != word_start or orig_word_end != word_end:
                print(f"   ⚠ Clamped words {orig_word_start}-{orig_word_end} to {word_start}-{word_end}")

            # Get start time (from word_start)
            if word_start in word_timing_map:
                start_time = word_timing_map[word_start]['start']
            elif word_start == 0:
                start_time = 0.0
            else:
                # Estimate: use previous word's end time
                prev_words = [w for w in word_boundaries if w['word_index'] < word_start]
                start_time = prev_words[-1]['end'] if prev_words else 0.0
                print(f"   ⚠ word_start {word_start} not in map, estimated start_time={start_time:.3f}s")

            # Get end time (from word_end - 1, since word_end is exclusive)
            end_word_index = word_end - 1
            if end_word_index in word_timing_map:
                end_time = word_timing_map[end_word_index]['end']
            else:
                # For the last scene, use audio duration as end time
                end_time = audio_duration
                print(f"   ⚠ end_word_index {end_word_index} not in map, using audio_duration={audio_duration:.3f}s")

            duration = max(end_time - start_time, 0.5)  # Minimum 0.5 second per scene
            scene_durations.append(duration)

            # Get actual words for this scene
            scene_words = [wb['word'] for wb in word_boundaries if word_start <= wb['word_index'] < word_end]
            print(f"   Scene {scene.get('scene_number')}: words {word_start}-{word_end} ({len(scene_words)} words)")
            print(f"      Time: {start_time:.3f}s - {end_time:.3f}s = {duration:.3f}s")
            if scene_words:
                print(f"      First words: {' '.join(scene_words[:3])}")
                print(f"      Last words: {' '.join(scene_words[-3:])}")

        total_scene_duration = sum(scene_durations)
        diff = total_scene_duration - audio_duration
        print(f"\n   Total scene duration: {total_scene_duration:.3f}s vs audio: {audio_duration:.3f}s (diff: {diff:.3f}s)")

        # If total duration differs significantly from audio, adjust proportionally
        if abs(diff) > 0.1 and scene_durations:
            print(f"   ⚠ Adjusting durations to match audio...")
            scale_factor = audio_duration / total_scene_duration
            scene_durations = [d * scale_factor for d in scene_durations]
            print(f"   Adjusted durations: {[f'{d:.2f}s' for d in scene_durations]}")

        return scene_durations

    except Exception as e:
        print(f"⚠ Error calculating scene durations: {e}")
        import traceback
        traceback.print_exc()
        return []

async def compile_video(request: CompileVideoRequest) -> CompileVideoResponse:
    """
    Combine multiple stock videos with audio using ffmpeg.
    Downloads videos, combines them, adds audio, burns captions, and returns the final video.
    """
    if not request.video_urls or len(request.video_urls) == 0:
        raise HTTPException(status_code=400, detail="At least one video URL is required")
    
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

        if use_images:
            print(f"🖼️  Downloading {len(request.video_urls)} images...")
            for i, image_url in enumerate(request.video_urls):
                try:
                    response = requests.get(image_url, timeout=30, stream=True)
                    if response.status_code != 200:
                        print(f"Warning: Failed to download image {i+1}: {image_url[:50]}...")
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
                    video_response = requests.get(video_url, timeout=30, stream=True)
                    if video_response.status_code != 200:
                        print(f"Warning: Failed to download video {i+1}: {video_url[:50]}...")
                        continue

                    video_path = temp_path / f"video_{i+1}.mp4"
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
        
        # Get audio duration using ffprobe
        audio_duration = None
        try:
            probe_cmd = [
                FFPROBE_EXECUTABLE, '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)
            ]
            result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True, timeout=10)
            duration_str = result.stdout.strip()
            if duration_str:
                audio_duration = float(duration_str)
                print(f"✓ Audio duration detected: {audio_duration:.3f}s")
            else:
                raise ValueError("Empty duration string from ffprobe")
        except Exception as e:
            print(f"✗ ERROR: Could not get audio duration: {e}")
            try:
                alt_cmd = [
                    FFMPEG_EXECUTABLE, '-i', str(audio_path),
                    '-f', 'null', '-'
                ]
                alt_result = subprocess.run(alt_cmd, capture_output=True, text=True, timeout=10)
                import re
                duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})', alt_result.stderr)
                if duration_match:
                    hours, minutes, seconds, centiseconds = duration_match.groups()
                    audio_duration = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(centiseconds) / 100
                    print(f"✓ Audio duration detected (alternative method): {audio_duration:.3f}s")
                else:
                    raise ValueError("Could not parse duration from ffmpeg output")
            except Exception as e2:
                print(f"✗ ERROR: Alternative duration detection also failed: {e2}")
                raise HTTPException(status_code=500, detail=f"Failed to determine audio duration: {e}. Please ensure audio file is valid.")
        
        if audio_duration is None or audio_duration <= 0:
            raise HTTPException(status_code=500, detail="Invalid audio duration detected. Please check audio file.")
        
        # Calculate scene durations - use provided durations, calculate from alignment, or equal split
        print(f"📹 Compiling video with {len(video_paths)} media files")
        print(f"🎵 Audio duration: {audio_duration:.2f}s")

        if request.scene_durations and len(request.scene_durations) == len(video_paths):
            scene_durations = request.scene_durations
            print(f"⏱️  Using provided scene durations: {[f'{d:.2f}s' for d in scene_durations]}")
        elif request.alignment and request.scenes:
            # Calculate scene durations from alignment and scene word boundaries
            print(f"⏱️  Calculating scene durations from alignment...")
            scenes_data = [
                {
                    'scene_number': s.scene_number,
                    'word_start': s.word_start,
                    'word_end': s.word_end
                }
                for s in request.scenes if s.word_start is not None
            ]
            calculated_durations = calculate_scene_durations_from_alignment(
                request.alignment,
                scenes_data,
                audio_duration,
                script=request.script
            )
            if calculated_durations and len(calculated_durations) == len(video_paths):
                scene_durations = calculated_durations
                print(f"⏱️  Calculated scene durations: {[f'{d:.2f}s' for d in scene_durations]}")
            else:
                # Fallback if calculation failed or count mismatch
                print(f"⚠ Duration calculation mismatch: got {len(calculated_durations)}, need {len(video_paths)}")
                equal_duration = audio_duration / len(video_paths)
                scene_durations = [equal_duration] * len(video_paths)
                print(f"⏱️  Using equal scene duration fallback: {equal_duration:.2f}s per video")
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
                    '-c:v', 'libx264',
                    '-preset', 'fast',
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
                        '-c:v', 'libx264',
                        '-preset', 'fast',
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
                    '-c:v', 'libx264',
                    '-preset', 'fast',
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
        
        # Step 1: Concatenate trimmed videos and scale to vertical format
        if len(trimmed_videos) == 1:
            # Single video - just scale it to vertical format
            concat_video_path = temp_path / "concatenated.mp4"
            scale_cmd = [
                FFMPEG_EXECUTABLE, '-y',
                '-i', str(trimmed_videos[0]),
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-an',
                str(concat_video_path)
            ]
            try:
                result = subprocess.run(scale_cmd, check=True, capture_output=True, timeout=60, text=True)
                print(f"✓ Scaled single video to vertical format")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else 'Unknown error'
                print(f"✗ Failed to scale video: {error_msg[:200]}")
                raise HTTPException(status_code=500, detail="Failed to scale video")
        else:
            # Multiple videos - concatenate them and scale to vertical
            concat_video_path = temp_path / "concatenated.mp4"
            concat_cmd = [
                FFMPEG_EXECUTABLE, '-y',
            ]
            
            # Add all input videos
            for trimmed_path in trimmed_videos:
                concat_cmd.extend(['-i', str(trimmed_path)])
            
            # Build filter_complex: scale each to 1080x1920, then concatenate
            filter_parts = []
            for i in range(len(trimmed_videos)):
                filter_parts.append(f"[{i}:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[v{i}]")
            
            concat_inputs = ''.join([f"[v{i}]" for i in range(len(trimmed_videos))])
            filter_parts.append(f"{concat_inputs}concat=n={len(trimmed_videos)}:v=1:a=0[outv]")
            
            filter_complex = ';'.join(filter_parts)
            
            concat_cmd.extend([
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-c:v', 'libx264',
                '-preset', 'fast',
                str(concat_video_path)
            ])
            
            try:
                result = subprocess.run(concat_cmd, check=True, capture_output=True, timeout=180, text=True)
                print(f"✓ Concatenated {len(trimmed_videos)} videos")
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
        
        # Step 2.5: Verify concatenated video duration and extend if needed
        try:
            probe_concat_cmd = [
                FFPROBE_EXECUTABLE, '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(concat_video_path)
            ]
            probe_result = subprocess.run(probe_concat_cmd, capture_output=True, text=True, timeout=10)
            concat_duration = float(probe_result.stdout.strip())
            print(f"📊 Concatenated video duration: {concat_duration:.2f}s (audio: {audio_duration:.2f}s)")
            
            # If concatenated video is shorter than audio, extend it by looping the last frame
            if concat_duration < audio_duration:
                extended_path = temp_path / "extended_concat.mp4"
                extend_duration = audio_duration - concat_duration
                extend_cmd = [
                    FFMPEG_EXECUTABLE, '-y',
                    '-i', str(concat_video_path),
                    '-vf', f'tpad=stop_mode=clone:stop_duration={extend_duration:.3f}',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-an',
                    str(extended_path)
                ]
                try:
                    subprocess.run(extend_cmd, check=True, capture_output=True, timeout=30, text=True)
                    concat_video_path = extended_path
                    print(f"✓ Extended video to match audio duration ({audio_duration:.2f}s)")
                except Exception as e:
                    print(f"⚠ Warning: Could not extend video: {e}")
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
                '-c:v', 'libx264',
                '-preset', 'fast',
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
                '-c:v', 'libx264',
                '-preset', 'fast',
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
