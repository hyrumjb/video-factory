import base64
import tempfile
import subprocess
import requests
from pathlib import Path
from fastapi import HTTPException
from models import CompileVideoRequest, CompileVideoResponse
from config import FFMPEG_EXECUTABLE, FFPROBE_EXECUTABLE, TTS_AVAILABLE, tts_client
from tts import generate_tts_with_timing
from subtitles import create_ass_from_timepoints, create_ass_fallback

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
            # Extract base64 audio data
            audio_base64 = request.audio_url.split(',')[1]
            audio_data = base64.b64decode(audio_base64)
            audio_path = temp_path / "audio.wav"
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
        else:
            # Download audio from URL
            audio_response = requests.get(request.audio_url, timeout=30)
            if audio_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to download audio")
            audio_path = temp_path / "audio.wav"
            with open(audio_path, 'wb') as f:
                f.write(audio_response.content)
        
        # Download videos
        video_paths = []
        for i, video_url in enumerate(request.video_urls):
            try:
                video_response = requests.get(video_url, timeout=30, stream=True)
                if video_response.status_code != 200:
                    print(f"Warning: Failed to download video {i+1}: {video_url}")
                    continue
                
                video_path = temp_path / f"video_{i+1}.mp4"
                with open(video_path, 'wb') as f:
                    for chunk in video_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                video_paths.append(str(video_path))
            except Exception as e:
                print(f"Error downloading video {i+1}: {e}")
                continue
        
        if not video_paths:
            raise HTTPException(status_code=500, detail="Failed to download any videos")
        
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
        
        # Calculate equal duration for each video scene
        scene_duration = audio_duration / len(video_paths)
        print(f"📹 Compiling video with {len(video_paths)} videos")
        print(f"🎵 Audio duration: {audio_duration:.2f}s")
        print(f"⏱️  Scene duration per video: {scene_duration:.2f}s")
        
        # Trim each video to exactly scene_duration
        trimmed_videos = []
        for i, video_path in enumerate(video_paths):
            trimmed_path = temp_path / f"trimmed_video_{i+1}.mp4"
            trim_cmd = [
                FFMPEG_EXECUTABLE, '-y',
                '-i', str(video_path),
                '-t', f'{scene_duration:.3f}',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-an',  # Remove audio from video clips
                str(trimmed_path)
            ]
            try:
                result = subprocess.run(trim_cmd, check=True, capture_output=True, timeout=30, text=True)
                trimmed_videos.append(trimmed_path)
                print(f"✓ Trimmed video {i+1} to {scene_duration:.2f}s")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode() if e.stderr else 'Unknown error'
                print(f"✗ Failed to trim video {i+1}: {error_msg[:200]}")
                raise HTTPException(status_code=500, detail=f"Failed to trim video {i+1} to {scene_duration:.2f}s")
        
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
                error_msg = e.stderr.decode() if e.stderr else 'Unknown error'
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
                error_msg = e.stderr.decode() if e.stderr else 'Unknown error'
                print(f"✗ FFmpeg concat error: {error_msg[:500]}")
                raise HTTPException(status_code=500, detail="Failed to concatenate videos")
        
        # Step 2: Generate ASS subtitles with timing from Google TTS
        subtitle_path = None
        timepoints = None
        if TTS_AVAILABLE and tts_client and request.script:
            try:
                script_text = request.script.strip()
                print(f"📝 Generating subtitle timings from script (length: {len(script_text)} chars)...")
                selected_voice = request.voice_name if request.voice_name else "en-US-Neural2-H"
                _, timepoints = generate_tts_with_timing(script_text, voice_name=selected_voice)
                
                if timepoints:
                    print(f"   Received {len(timepoints)} timepoints from TTS")
                    # Create ASS subtitle file with karaoke effects (current word in yellow, others in white)
                    ass_content = create_ass_from_timepoints(script_text, timepoints, audio_duration, style="3words")
                    subtitle_path = temp_path / "subtitles.ass"
                    with open(subtitle_path, 'w', encoding='utf-8') as f:
                        f.write(ass_content)
                    print(f"✓ Created ASS subtitle file (3 words per caption, karaoke effects) with {len(timepoints)} timepoints")
                else:
                    print(f"⚠ No timepoints received, creating estimated subtitles")
                    ass_content = create_ass_fallback(script_text, audio_duration)
                    subtitle_path = temp_path / "subtitles.ass"
                    with open(subtitle_path, 'w', encoding='utf-8') as f:
                        f.write(ass_content)
            except Exception as e:
                print(f"⚠ Failed to generate subtitles with timing: {e}")
                try:
                    ass_content = create_ass_fallback(script_text, audio_duration)
                    subtitle_path = temp_path / "subtitles.ass"
                    with open(subtitle_path, 'w', encoding='utf-8') as f:
                        f.write(ass_content)
                except:
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
            error_msg = e.stderr.decode() if e.stderr else 'Unknown error'
            print(f"✗ FFmpeg final error: {error_msg[:500]}")
            raise HTTPException(status_code=500, detail="Failed to combine video and audio")
        
        # Read the final video and return as base64
        with open(output_path, 'rb') as f:
            video_data = f.read()
        
        video_base64 = base64.b64encode(video_data).decode('utf-8')
        video_data_url = f"data:video/mp4;base64,{video_base64}"
        
        return CompileVideoResponse(video_url=video_data_url)
