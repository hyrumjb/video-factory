"""
AI Video generation module using Fal.ai Veo 3.1.
Generates video clips for each scene using advanced cinematic prompts.
"""

import os
import json
import asyncio
import requests
from typing import Dict, Any, List, Optional
from config import xai_client, XAI_MODEL


# Check if Fal.ai is configured
FAL_API_KEY = os.getenv("FAL_KEY")
FAL_AVAILABLE = FAL_API_KEY is not None

if FAL_AVAILABLE:
    print("✓ Fal.ai API key configured for video generation")
else:
    print("⚠ Warning: FAL_KEY not set - AI video generation will not work")

# Primary model: Veo 3.1 Fast
PRIMARY_MODEL = {
    "id": "fal-ai/veo3.1/fast",
    "name": "Veo 3.1 Fast",
    "endpoint": "https://fal.run/fal-ai/veo3.1/fast",
    "allowed_durations": [4, 6, 8],  # Only these durations are allowed
}

# No fallback models - only use Veo 3.1 Fast
FALLBACK_MODELS = []


def get_closest_allowed_duration(requested: float, allowed: list) -> int:
    """
    Get the closest allowed duration to the requested duration.
    Always returns the maximum allowed if requested is higher.
    """
    if not allowed:
        return 8  # Default fallback

    # If requested is higher than max allowed, use max
    max_allowed = max(allowed)
    if requested >= max_allowed:
        return max_allowed

    # Find closest allowed duration
    closest = min(allowed, key=lambda x: abs(x - requested))
    return closest

# Speaking rate for duration estimation (words per second)
# Average narration is about 150 words/minute = 2.5 words/second
WORDS_PER_SECOND = 2.5


def estimate_scene_duration(section_text: str, min_duration: float = 3.0, max_duration: float = 10.0) -> float:
    """
    Estimate scene duration based on word count.

    Args:
        section_text: The text for this section
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds

    Returns:
        Estimated duration in seconds
    """
    word_count = len(section_text.split())
    estimated = word_count / WORDS_PER_SECOND

    # Clamp to min/max
    duration = max(min_duration, min(estimated, max_duration))
    return round(duration, 1)


def generate_advanced_video_prompt(
    topic: str,
    script: str,
    section_text: str,
    section_name: str = "HOOK",
    duration: float = 5.0
) -> str:
    """
    Generate a powerful, cinematic video prompt using the Advanced Formula:
    Subject (Description) + Scene (Description) + Motion (Description) + Aesthetic Control + Stylization

    Args:
        topic: The overall video topic
        script: The full script text (for context)
        section_text: The text of this section
        section_name: The name of the section (e.g., "HOOK", "BODY1")
        duration: Target duration in seconds

    Returns:
        Detailed cinematic prompt for video generation
    """
    if not xai_client:
        return f"Cinematic video about {topic}. {section_text[:150]}. Smooth camera movement, dramatic lighting."

    try:
        prompt = f"""You are a cinematic video prompt engineer. Generate a POWERFUL, SPECIFIC video prompt for an AI video generator (Veo 3.1).

TOPIC: {topic}
SECTION: {section_name}
TARGET DURATION: {duration} seconds
SECTION TEXT (what the video should represent):
{section_text}

FULL SCRIPT (for context):
{script[:500]}

Use this ADVANCED FORMULA to create the prompt:
Prompt = Subject (Subject Description) + Scene (Scene Description) + Motion (Motion Description) + Aesthetic Control + Stylization

REQUIRED COMPONENTS (include ALL of these):

1. SUBJECT DESCRIPTION: Detailed physical attributes of the main subject
   - Specific colors, textures, features
   - What they're wearing/holding if applicable
   - Example: "A weathered explorer with silver-streaked hair and a worn leather jacket"

2. SCENE DESCRIPTION: Vivid environment details
   - Specific location, time of day, weather
   - Background elements, atmosphere
   - Example: "Standing atop a misty mountain peak at golden hour, ancient ruins visible in the valley below"

3. MOTION DESCRIPTION: Specific movement characteristics
   - Amplitude, speed, effects of motion
   - What is moving and how
   - Example: "Camera slowly pushes in while fog swirls gently around, hair billowing in the wind"

4. AESTHETIC CONTROL: Cinematic technical elements
   - Lighting: golden hour, dramatic rim light, soft diffused, harsh shadows
   - Shot size: extreme close-up, close-up, medium shot, wide shot, establishing shot
   - Camera angle: low angle (heroic), high angle, eye level, bird's eye, dutch angle
   - Lens: 24mm wide, 35mm, 50mm, 85mm portrait, telephoto compression
   - Camera movement: slow push in, dolly, tracking, crane, handheld, static

5. STYLIZATION: Visual style
   - Cinematic look: anamorphic, film grain, color grade
   - Genre feel: documentary, thriller, epic, intimate
   - Reference style if helpful

FORMAT REQUIREMENTS:
- Write as ONE cohesive paragraph (not bullet points)
- Be VERY SPECIFIC - avoid generic descriptions
- Focus on what makes this section VISUALLY UNIQUE
- Include motion - video needs MOVEMENT
- Keep under 400 characters for best results
- Vertical format (9:16 aspect ratio) - compose for mobile viewing

Return ONLY the video prompt, no explanations or labels."""

        response = xai_client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.8
        )

        video_prompt = response.choices[0].message.content.strip()

        # Remove any quotes if the model wrapped it
        if video_prompt.startswith('"') and video_prompt.endswith('"'):
            video_prompt = video_prompt[1:-1]

        print(f"   🎬 [{section_name}] ({duration}s) Prompt: {video_prompt[:80]}...")
        return video_prompt

    except Exception as e:
        print(f"⚠ LLM prompt generation failed for {section_name}: {e}")
        # Fallback to a basic but still descriptive prompt
        return f"Cinematic {topic} scene. {section_text[:100]}. Slow dramatic camera push, golden hour lighting, shallow depth of field, 85mm lens, film grain."


def generate_prompts_for_video_scenes(
    topic: str,
    script: str,
    sections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate advanced video prompts for all scenes with estimated durations.

    Args:
        topic: The overall video topic
        script: The full script text
        sections: List of section dicts with 'name' and 'text'

    Returns:
        List of dicts with 'section_name', 'section_text', 'prompt', 'duration'
    """
    prompts = []

    print(f"\n🎬 Generating advanced video prompts for {len(sections)} scenes...")

    # Calculate durations for all sections
    total_words = sum(len(s.get("text", "").split()) for s in sections)
    print(f"   Total words: {total_words} (~{total_words/WORDS_PER_SECOND:.1f}s estimated)")

    for i, section in enumerate(sections):
        section_name = section.get("name", f"Scene {i+1}")
        section_text = section.get("text", "")

        # Estimate duration based on word count
        duration = estimate_scene_duration(section_text)

        video_prompt = generate_advanced_video_prompt(
            topic=topic,
            script=script,
            section_text=section_text,
            section_name=section_name,
            duration=duration
        )

        prompts.append({
            "section_name": section_name,
            "section_text": section_text,
            "prompt": video_prompt,
            "duration": duration
        })

    total_duration = sum(p["duration"] for p in prompts)
    print(f"   Total estimated video duration: {total_duration:.1f}s")

    return prompts


def call_veo_model(
    model: Dict[str, str],
    text_prompt: str,
    duration: float,
    aspect_ratio: str = "9:16",
    scene_number: int = 1,
    section_name: str = "Scene"
) -> Dict[str, Any]:
    """
    Call Veo 3.1 to generate a video clip at the specified duration.

    Args:
        model: Model info dict with 'id', 'name', 'endpoint'
        text_prompt: The detailed video prompt
        duration: Target duration in seconds
        aspect_ratio: Video aspect ratio (default 9:16 for vertical)
        scene_number: The scene number this video is for
        section_name: The section name (e.g., "HOOK", "BODY1")

    Returns:
        Dict with 'model_name', 'model_id', 'url', 'duration', 'error', etc.
    """
    if not FAL_AVAILABLE:
        return {
            "model_name": model["name"],
            "model_id": model["id"],
            "url": None,
            "error": "FAL_KEY not configured",
            "scene_number": scene_number,
            "section_name": section_name
        }

    try:
        # Get allowed durations for this model and clamp to valid value
        allowed_durations = model.get("allowed_durations", [4, 6, 8])
        clamped_duration = get_closest_allowed_duration(duration, allowed_durations)

        print(f"      [{section_name}] Calling {model['name']} (requested: {duration:.1f}s, using: {clamped_duration}s)...")

        headers = {
            "Authorization": f"Key {FAL_API_KEY}",
            "Content-Type": "application/json"
        }

        # Veo API requires duration as string with 's' suffix (e.g., "8s")
        payload = {
            "prompt": text_prompt,
            "aspect_ratio": aspect_ratio,
            "duration": f"{clamped_duration}s",  # Must be format like "8s", "6s", etc.
        }

        # Long timeout for video generation
        timeout = 600  # 10 minutes

        response = requests.post(
            model["endpoint"],
            headers=headers,
            json=payload,
            timeout=timeout
        )

        if response.status_code != 200:
            error_text = response.text[:300] if response.text else "Unknown error"
            print(f"      ✗ [{section_name}] {model['name']} failed: {response.status_code}")
            print(f"         Error: {error_text}")
            return {
                "model_name": model["name"],
                "model_id": model["id"],
                "url": None,
                "error": f"API error {response.status_code}: {error_text}",
                "scene_number": scene_number,
                "section_name": section_name,
                "requested_duration": duration
            }

        data = response.json()

        # Extract video URL from response
        video_url = None
        actual_duration = None

        # Try common response structures for Veo
        if "video" in data:
            video_data = data["video"]
            if isinstance(video_data, dict):
                video_url = video_data.get("url")
                actual_duration = video_data.get("duration")
            elif isinstance(video_data, str):
                video_url = video_data
        elif "url" in data:
            video_url = data["url"]
            actual_duration = data.get("duration")

        if video_url:
            duration_info = f", API returned: {actual_duration}s" if actual_duration else ""
            print(f"      ✓ [{section_name}] {model['name']}: Video generated (scene needs: {duration:.1f}s, generated: {clamped_duration}s{duration_info})")
            return {
                "model_name": model["name"],
                "model_id": model["id"],
                "url": video_url,
                "scene_number": scene_number,
                "section_name": section_name,
                "requested_duration": duration,
                "generated_duration": clamped_duration,  # The actual length of video generated
                "actual_duration": actual_duration  # What API reported (if any)
            }
        else:
            print(f"      ✗ [{section_name}] {model['name']}: No video URL in response")
            print(f"         Response: {json.dumps(data)[:200]}")
            return {
                "model_name": model["name"],
                "model_id": model["id"],
                "url": None,
                "error": "No video URL in response",
                "scene_number": scene_number,
                "section_name": section_name,
                "requested_duration": duration
            }

    except requests.Timeout:
        print(f"      ✗ [{section_name}] {model['name']} timed out after {timeout}s")
        return {
            "model_name": model["name"],
            "model_id": model["id"],
            "url": None,
            "error": "Request timed out",
            "scene_number": scene_number,
            "section_name": section_name,
            "requested_duration": duration
        }
    except Exception as e:
        print(f"      ✗ [{section_name}] {model['name']} error: {e}")
        return {
            "model_name": model["name"],
            "model_id": model["id"],
            "url": None,
            "error": str(e),
            "scene_number": scene_number,
            "section_name": section_name,
            "requested_duration": duration
        }


def generate_video_with_fallback_sync(
    scene_prompt: Dict[str, Any],
    scene_idx: int
) -> Dict[str, Any]:
    """
    Generate a video for a single scene using Veo 3.1, with fallbacks if it fails.

    Args:
        scene_prompt: Dict with 'section_name', 'prompt', 'duration'
        scene_idx: 0-based scene index

    Returns:
        Video result dict
    """
    section_name = scene_prompt["section_name"]
    text_prompt = scene_prompt["prompt"]
    duration = scene_prompt.get("duration", 5.0)

    # Try primary model (Veo 3.1) first
    result = call_veo_model(
        PRIMARY_MODEL,
        text_prompt,
        duration,
        "9:16",
        scene_idx + 1,
        section_name
    )

    if result.get("url"):
        result["prompt"] = text_prompt
        return result

    # Primary failed, try fallbacks
    print(f"      [{section_name}] Primary model failed, trying fallbacks...")

    for fallback_model in FALLBACK_MODELS:
        result = call_veo_model(
            fallback_model,
            text_prompt,
            duration,
            "9:16",
            scene_idx + 1,
            section_name
        )

        if result.get("url"):
            result["prompt"] = text_prompt
            return result

    # All models failed
    print(f"      [{section_name}] All models failed!")
    result["prompt"] = text_prompt
    return result


async def generate_videos_for_all_scenes(
    scene_prompts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate videos for all scenes using Veo 3.1.
    Runs all API calls in parallel using a thread pool.

    Args:
        scene_prompts: List of dicts with 'section_name', 'prompt', 'duration'

    Returns:
        List of video results (one per scene)
    """
    total_duration = sum(p.get("duration", 5) for p in scene_prompts)
    print(f"\n   🎬 Generating {len(scene_prompts)} videos in parallel (primary: {PRIMARY_MODEL['name']})")
    print(f"      Total requested duration: {total_duration:.1f}s")

    # Run all API calls in parallel using thread pool
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(
            None,
            generate_video_with_fallback_sync,
            scene_prompt,
            scene_idx
        )
        for scene_idx, scene_prompt in enumerate(scene_prompts)
    ]

    results = await asyncio.gather(*tasks)

    successful = sum(1 for r in results if r.get("url"))
    print(f"\n   ✅ Generated {successful}/{len(scene_prompts)} videos successfully")

    return list(results)


async def generate_video_for_scenes(
    topic: str,
    script: str,
    first_section_text: str,
    sections: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Main entry point: Generate AI videos for each scene using Veo 3.1.

    Videos are generated at the estimated scene duration based on word count,
    so they can be used directly without trimming.

    Args:
        topic: The overall video topic
        script: The full script text
        first_section_text: The text of the first section (HOOK) - used if sections not provided
        sections: Optional list of section dicts with 'name' and 'text' for each scene

    Returns:
        Dict with 'videos' (list of videos, one per scene), 'prompts' (list of prompts per scene)
    """
    print(f"\n🎬 Generating AI videos for: {topic}")
    print(f"   Using model: {PRIMARY_MODEL['name']} ({PRIMARY_MODEL['id']})")

    # If sections not provided, create a single section from first_section_text
    if not sections:
        sections = [{"name": "HOOK", "text": first_section_text}]

    # Step 1: Generate advanced prompts with duration estimates
    print(f"   Step 1: Generating cinematic video prompts for {len(sections)} scenes...")
    scene_prompts = generate_prompts_for_video_scenes(topic, script, sections)

    # Step 2: Generate videos for all scenes at their target durations
    print(f"   Step 2: Generating videos with {PRIMARY_MODEL['name']}...")
    videos = await generate_videos_for_all_scenes(scene_prompts)

    return {
        "videos": videos,
        "prompts": [
            {
                "section_name": p["section_name"],
                "prompt": p["prompt"],
                "duration": p["duration"]
            }
            for p in scene_prompts
        ]
    }


async def generate_video_for_scenes_with_durations(
    topic: str,
    script: str,
    sections: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate AI videos for each scene using pre-calculated exact durations.

    This function is called when TTS audio has already been generated and we have
    exact scene durations from the audio alignment data.

    Args:
        topic: The overall video topic
        script: The full script text
        sections: List of section dicts with 'name', 'text', and 'duration' for each scene

    Returns:
        Dict with 'videos' (list of videos, one per scene), 'prompts' (list of prompts per scene)
    """
    print(f"\n🎬 Generating AI videos with exact durations for: {topic}")
    print(f"   Using model: {PRIMARY_MODEL['name']} ({PRIMARY_MODEL['id']})")

    if not sections:
        print("⚠ No sections provided")
        return {"videos": [], "prompts": []}

    # Generate advanced prompts using the pre-calculated durations
    print(f"   Step 1: Generating cinematic video prompts for {len(sections)} scenes...")
    scene_prompts = []

    total_duration = sum(s.get("duration", 5.0) for s in sections)
    print(f"   Total exact duration from audio: {total_duration:.2f}s")

    for i, section in enumerate(sections):
        section_name = section.get("name", f"Scene {i+1}")
        section_text = section.get("text", "")
        duration = section.get("duration", 5.0)  # Use pre-calculated duration

        video_prompt = generate_advanced_video_prompt(
            topic=topic,
            script=script,
            section_text=section_text,
            section_name=section_name,
            duration=duration
        )

        scene_prompts.append({
            "section_name": section_name,
            "section_text": section_text,
            "prompt": video_prompt,
            "duration": duration
        })

    # Generate videos for all scenes at their exact durations
    print(f"   Step 2: Generating videos with {PRIMARY_MODEL['name']}...")
    videos = await generate_videos_for_all_scenes(scene_prompts)

    return {
        "videos": videos,
        "prompts": [
            {
                "section_name": p["section_name"],
                "prompt": p["prompt"],
                "duration": p["duration"]
            }
            for p in scene_prompts
        ]
    }


async def generate_single_video(
    prompt: str,
    topic: str,
    duration: float,
    aspect_ratio: str = "9:16"
) -> Optional[Dict[str, Any]]:
    """
    Generate a single AI video from a prompt.
    Used by the media router for individual clause visuals.

    Args:
        prompt: The video generation prompt/query
        topic: The overall video topic (for context)
        duration: Target duration in seconds
        aspect_ratio: Aspect ratio for the video (default 9:16 for vertical)

    Returns:
        Dict with 'url', 'duration' or None if failed
    """
    if not FAL_AVAILABLE:
        print("   ⚠ FAL not available for AI video generation")
        return None

    print(f"   🎬 Generating AI video ({duration:.1f}s): \"{prompt[:50]}...\"")

    # Generate an advanced cinematic prompt
    video_prompt = generate_advanced_video_prompt(
        topic=topic,
        script=prompt,  # Use prompt as context
        section_text=prompt,
        section_name="single",
        duration=duration
    )

    # Snap to allowed duration
    allowed_duration = get_closest_allowed_duration(duration, PRIMARY_MODEL.get("allowed_durations", [5, 8]))

    scene_prompt = {
        "section_name": "single",
        "section_text": prompt,
        "prompt": video_prompt,
        "duration": allowed_duration
    }

    # Generate video
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        generate_video_with_fallback_sync,
        scene_prompt,
        0  # scene_idx
    )

    if result.get("url"):
        return result

    print(f"   ⚠ Video generation failed")
    return None
