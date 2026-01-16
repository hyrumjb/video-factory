"""
AI Image generation module using Fal.ai.
Generates images using multiple models for comparison.
"""

import os
import json
import asyncio
import requests
from typing import Dict, Any, List
from config import openai_client


# Check if Fal.ai is configured
FAL_API_KEY = os.getenv("FAL_KEY")
FAL_AVAILABLE = FAL_API_KEY is not None

if FAL_AVAILABLE:
    print("✓ Fal.ai API key configured")
else:
    print("⚠ Warning: FAL_KEY not set - AI image generation will not work")

# Primary model for image generation
PRIMARY_MODEL = {
    "id": "fal-ai/nano-banana-pro",
    "name": "Nano Banana Pro",
    "endpoint": "https://fal.run/fal-ai/nano-banana-pro"
}

# Fallback models (used only if primary fails)
FALLBACK_MODELS = [
    {
        "id": "fal-ai/gpt-image-1.5",
        "name": "GPT Image 1.5",
        "endpoint": "https://fal.run/fal-ai/gpt-image-1.5"
    },
    {
        "id": "fal-ai/flux-2-max",
        "name": "FLUX 2 Max",
        "endpoint": "https://fal.run/fal-ai/flux-2-max"
    },
]


def generate_structured_image_prompt(
    topic: str,
    script: str,
    section_text: str,
    section_name: str = "HOOK"
) -> Dict[str, Any]:
    """
    Use OpenAI to generate a detailed JSON-structured image prompt for a single scene.

    Args:
        topic: The overall video topic
        script: The full script text (for context)
        section_text: The text of this section
        section_name: The name of the section (e.g., "HOOK", "BODY1")

    Returns:
        Structured JSON prompt dict for image generation
    """
    if not openai_client:
        return {
            "scene": f"A visually striking scene about {topic}",
            "subjects": [{"type": "main subject", "description": topic, "position": "center"}],
            "style": "cinematic, photorealistic",
            "lighting": "dramatic",
            "mood": "engaging",
            "composition": "rule of thirds",
            "camera": {"angle": "eye level", "distance": "medium shot", "lens": "35mm"}
        }

    try:
        prompt = f"""Generate a detailed JSON image prompt for an AI image generator.

TOPIC: {topic}

SECTION: {section_name}
SECTION TEXT (this is what the image should represent):
{section_text}

FULL SCRIPT (for context):
{script}

Create a visually striking, vertical (9:16 aspect ratio) image that captures the essence of this specific section. The image should be attention-grabbing for a TikTok-style short video.

Return ONLY valid JSON in this exact structure:
{{
  "scene": "Overall setting description - be specific and vivid",
  "subjects": [
    {{
      "type": "Subject category (person/object/element)",
      "description": "Detailed physical attributes - colors, textures, features",
      "pose": "Action, stance, or state",
      "position": "foreground/midground/background"
    }}
  ],
  "style": "Artistic style (e.g., cinematic photorealistic, dramatic illustration, etc.)",
  "color_palette": ["primary color", "secondary color", "accent color"],
  "lighting": "Specific lighting setup (e.g., golden hour side lighting, dramatic rim light)",
  "mood": "Emotional atmosphere (e.g., mysterious, inspiring, intense)",
  "composition": "rule of thirds/centered/dynamic diagonal",
  "camera": {{
    "angle": "eye level/low angle/high angle/bird's eye",
    "distance": "extreme close-up/close-up/medium shot/wide shot",
    "lens": "24mm/35mm/50mm/85mm/telephoto"
  }}
}}

Make the scene dramatic and visually compelling. Be specific with colors and details."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        structured_prompt = json.loads(response_text)

        print(f"   🎨 [{section_name}] Scene: {structured_prompt.get('scene', 'N/A')[:60]}...")

        return structured_prompt

    except json.JSONDecodeError as e:
        print(f"⚠ Failed to parse JSON prompt for {section_name}: {e}")
        return {
            "scene": f"A dramatic, visually striking scene representing {topic}",
            "subjects": [{"type": "main subject", "description": topic, "position": "center"}],
            "style": "cinematic, photorealistic, high detail",
            "color_palette": ["rich tones", "dramatic contrast"],
            "lighting": "dramatic cinematic lighting",
            "mood": "engaging and compelling",
            "composition": "rule of thirds",
            "camera": {"angle": "eye level", "distance": "medium shot", "lens": "35mm"}
        }
    except Exception as e:
        print(f"⚠ LLM prompt generation failed for {section_name}: {e}")
        return {
            "scene": f"A dramatic scene about {topic}",
            "subjects": [{"type": "main subject", "description": topic, "position": "center"}],
            "style": "cinematic photorealistic",
            "lighting": "dramatic",
            "mood": "engaging",
            "composition": "rule of thirds",
            "camera": {"angle": "eye level", "distance": "medium shot", "lens": "35mm"}
        }


def generate_prompts_for_scenes(
    topic: str,
    script: str,
    sections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate image prompts for all scenes/sections.

    Args:
        topic: The overall video topic
        script: The full script text
        sections: List of section dicts with 'name' and 'text'

    Returns:
        List of dicts with 'section_name', 'section_text', 'structured_prompt', 'text_prompt'
    """
    prompts = []

    print(f"\n🎨 Generating image prompts for {len(sections)} scenes...")

    for section in sections:
        section_name = section.get("name", "Scene")
        section_text = section.get("text", "")

        structured_prompt = generate_structured_image_prompt(
            topic=topic,
            script=script,
            section_text=section_text,
            section_name=section_name
        )

        text_prompt = structured_prompt_to_text(structured_prompt)

        prompts.append({
            "section_name": section_name,
            "section_text": section_text,
            "structured_prompt": structured_prompt,
            "text_prompt": text_prompt
        })

    return prompts


def structured_prompt_to_text(structured_prompt: Dict[str, Any]) -> str:
    """
    Convert a structured JSON prompt to a text prompt string.
    """
    parts = []

    if structured_prompt.get("scene"):
        parts.append(structured_prompt["scene"])

    subjects = structured_prompt.get("subjects", [])
    for subject in subjects:
        subject_desc = []
        if subject.get("description"):
            subject_desc.append(subject["description"])
        if subject.get("pose"):
            subject_desc.append(subject["pose"])
        if subject.get("position"):
            subject_desc.append(f"in the {subject['position']}")
        if subject_desc:
            parts.append(", ".join(subject_desc))

    if structured_prompt.get("style"):
        parts.append(structured_prompt["style"])

    colors = structured_prompt.get("color_palette", [])
    if colors:
        parts.append(f"color palette: {', '.join(colors)}")

    if structured_prompt.get("lighting"):
        parts.append(structured_prompt["lighting"])

    if structured_prompt.get("mood"):
        parts.append(f"{structured_prompt['mood']} mood")

    camera = structured_prompt.get("camera", {})
    camera_parts = []
    if camera.get("distance"):
        camera_parts.append(camera["distance"])
    if camera.get("angle"):
        camera_parts.append(camera["angle"])
    if camera.get("lens"):
        camera_parts.append(f"{camera['lens']} lens")
    if camera_parts:
        parts.append(", ".join(camera_parts))

    if structured_prompt.get("composition"):
        parts.append(f"{structured_prompt['composition']} composition")

    return ". ".join(parts)


def call_fal_model(
    model: Dict[str, str],
    text_prompt: str,
    aspect_ratio: str = "9:16",
    scene_number: int = 1,
    section_name: str = "Scene"
) -> Dict[str, Any]:
    """
    Call a single Fal.ai model to generate an image.

    Args:
        model: Model info dict with 'id', 'name', 'endpoint'
        text_prompt: The text prompt for image generation
        aspect_ratio: Image aspect ratio (default 9:16 for vertical)
        scene_number: The scene number this image is for
        section_name: The section name (e.g., "HOOK", "BODY1")

    Returns:
        Dict with 'model_name', 'model_id', 'url', 'width', 'height', 'error', 'scene_number', 'section_name'
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
        print(f"      [{section_name}] Calling {model['name']}...")

        headers = {
            "Authorization": f"Key {FAL_API_KEY}",
            "Content-Type": "application/json"
        }

        # Build request payload based on model
        payload: dict = {
            "prompt": text_prompt,
            "num_images": 1,
        }

        # Model-specific adjustments
        timeout = 120  # Default timeout
        if "flux" in model["id"].lower():
            payload["aspect_ratio"] = aspect_ratio
            payload["safety_tolerance"] = "5"
            payload["output_format"] = "jpeg"
        elif "gpt-image" in model["id"].lower():
            # GPT Image 1.5 requires specific sizes: 1024x1024, 1536x1024, 1024x1536
            payload["image_size"] = "1024x1536"  # Portrait size
            payload["output_format"] = "jpeg"
            timeout = 240  # GPT Image 1.5 needs more time
        elif "nano-banana" in model["id"].lower():
            payload["aspect_ratio"] = aspect_ratio
            payload["output_format"] = "jpeg"
        else:
            payload["image_size"] = "portrait_4_3"

        response = requests.post(
            model["endpoint"],
            headers=headers,
            json=payload,
            timeout=timeout
        )

        if response.status_code != 200:
            error_text = response.text[:200] if response.text else "Unknown error"
            print(f"      ✗ [{section_name}] {model['name']} failed: {response.status_code}")
            return {
                "model_name": model["name"],
                "model_id": model["id"],
                "url": None,
                "error": f"API error {response.status_code}: {error_text}",
                "scene_number": scene_number,
                "section_name": section_name
            }

        data = response.json()

        # Extract image URL from response (structure varies by model)
        image_url = None
        width = 0
        height = 0

        # Try common response structures
        if "images" in data and len(data["images"]) > 0:
            img = data["images"][0]
            if isinstance(img, dict):
                image_url = img.get("url")
                width = img.get("width", 0)
                height = img.get("height", 0)
            elif isinstance(img, str):
                image_url = img
        elif "image" in data:
            img = data["image"]
            if isinstance(img, dict):
                image_url = img.get("url")
                width = img.get("width", 0)
                height = img.get("height", 0)
            elif isinstance(img, str):
                image_url = img
        elif "url" in data:
            image_url = data["url"]

        if image_url:
            print(f"      ✓ [{section_name}] {model['name']}: {width}x{height}")
            return {
                "model_name": model["name"],
                "model_id": model["id"],
                "url": image_url,
                "width": width,
                "height": height,
                "scene_number": scene_number,
                "section_name": section_name
            }
        else:
            print(f"      ✗ [{section_name}] {model['name']}: No image URL")
            return {
                "model_name": model["name"],
                "model_id": model["id"],
                "url": None,
                "error": "No image URL in response",
                "scene_number": scene_number,
                "section_name": section_name
            }

    except requests.Timeout:
        print(f"      ✗ [{section_name}] {model['name']} timed out")
        return {
            "model_name": model["name"],
            "model_id": model["id"],
            "url": None,
            "error": "Request timed out",
            "scene_number": scene_number,
            "section_name": section_name
        }
    except Exception as e:
        print(f"      ✗ [{section_name}] {model['name']} error: {e}")
        return {
            "model_name": model["name"],
            "model_id": model["id"],
            "url": None,
            "error": str(e),
            "scene_number": scene_number,
            "section_name": section_name
        }


async def generate_image_with_fallback(
    scene_prompt: Dict[str, Any],
    scene_idx: int
) -> Dict[str, Any]:
    """
    Generate an image for a single scene using primary model, with fallbacks if it fails.

    Args:
        scene_prompt: Dict with 'section_name', 'text_prompt', 'structured_prompt'
        scene_idx: 0-based scene index

    Returns:
        Image result dict
    """
    section_name = scene_prompt["section_name"]
    text_prompt = scene_prompt["text_prompt"]

    # Try primary model first
    result = call_fal_model(
        PRIMARY_MODEL,
        text_prompt,
        "9:16",
        scene_idx + 1,
        section_name
    )

    if result.get("url"):
        result["prompt"] = text_prompt
        result["structured_prompt"] = scene_prompt["structured_prompt"]
        return result

    # Primary failed, try fallbacks
    print(f"      [{section_name}] Primary model failed, trying fallbacks...")

    for fallback_model in FALLBACK_MODELS:
        result = call_fal_model(
            fallback_model,
            text_prompt,
            "9:16",
            scene_idx + 1,
            section_name
        )

        if result.get("url"):
            result["prompt"] = text_prompt
            result["structured_prompt"] = scene_prompt["structured_prompt"]
            return result

    # All models failed
    print(f"      [{section_name}] All models failed!")
    result["prompt"] = text_prompt
    result["structured_prompt"] = scene_prompt["structured_prompt"]
    return result


async def generate_images_for_all_scenes(
    scene_prompts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate images for all scenes using primary model (Nano Banana Pro) with fallbacks.

    Args:
        scene_prompts: List of dicts with 'section_name', 'text_prompt', 'structured_prompt'

    Returns:
        List of image results (one per scene)
    """
    print(f"\n   🎨 Generating {len(scene_prompts)} images (primary: {PRIMARY_MODEL['name']})...")

    # Generate images for all scenes in parallel
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(
            None,
            lambda idx=i, prompt=scene_prompt: asyncio.run(
                generate_image_with_fallback(prompt, idx)
            )
        )
        for i, scene_prompt in enumerate(scene_prompts)
    ]

    # Actually, run_in_executor doesn't work well with async functions
    # Let's use a simpler synchronous approach
    results = []
    for scene_idx, scene_prompt in enumerate(scene_prompts):
        result = await generate_image_with_fallback(scene_prompt, scene_idx)
        results.append(result)

    successful = sum(1 for r in results if r.get("url"))
    print(f"\n   ✅ Generated {successful}/{len(scene_prompts)} images successfully")

    return results


async def generate_image_for_video(
    topic: str,
    script: str,
    first_section_text: str,
    sections: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main entry point: Generate AI images for each scene using Nano Banana Pro (with fallbacks).

    Args:
        topic: The overall video topic
        script: The full script text
        first_section_text: The text of the first section (HOOK) - used if sections not provided
        sections: Optional list of section dicts with 'name' and 'text' for each scene

    Returns:
        Dict with 'images' (list of images, one per scene), 'prompts' (list of prompts per scene)
    """
    print(f"\n🎨 Generating AI images for video about: {topic}")

    # If sections not provided, create a single section from first_section_text
    if not sections:
        sections = [{"name": "HOOK", "text": first_section_text}]

    # Step 1: Generate prompts for all scenes
    print(f"   Step 1: Generating image prompts for {len(sections)} scenes...")
    scene_prompts = generate_prompts_for_scenes(topic, script, sections)

    # Step 2: Generate images for all scenes (primary model with fallbacks)
    print(f"   Step 2: Generating images with {PRIMARY_MODEL['name']}...")
    images = await generate_images_for_all_scenes(scene_prompts)

    return {
        "images": images,  # List of images, one per scene
        "prompts": [
            {
                "section_name": p["section_name"],
                "prompt": p["text_prompt"]
            }
            for p in scene_prompts
        ]
    }
