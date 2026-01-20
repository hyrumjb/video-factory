"""
Media Router Module

Routes clauses to the appropriate media source based on LLM decisions.

Flow:
1. Clause + topic → LLM decides visual intent → media source → concrete retrieval call
2. Batches clauses (3-6) for efficiency and better pacing decisions
3. Routes to correct provider: ai_video, ai_image, youtube_video, stock_video, web_image
"""

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from config import xai_client, XAI_MODEL, get_media_duration


class MediaType(str, Enum):
    AI_VIDEO = "ai_video"
    AI_IMAGE = "ai_image"
    YOUTUBE_VIDEO = "youtube_video"
    STOCK_VIDEO = "stock_video"
    WEB_IMAGE = "web_image"


@dataclass
class MediaInstruction:
    """LLM-generated instruction for retrieving media for a clause."""
    clause_id: int
    media_type: MediaType
    query: str
    duration: float
    rationale: str


@dataclass
class MediaResult:
    """Result of media retrieval for a clause."""
    clause_id: int
    media_type: MediaType
    query: str
    duration: float
    media_url: Optional[str] = None
    error: Optional[str] = None


# Minimum duration thresholds
MIN_VIDEO_DURATION = 1.8  # Below this, force image
MAX_IMAGE_DURATION = 5.5  # Above this, prefer video
MIN_HOOK_DURATION = 2.5   # Hooks need at least this much motion


ROUTER_SYSTEM_PROMPT = """You are a visual media routing engine.

Your task is to decide the best media source and generate search queries for short-form video clauses.

You will be given:
- The overall video topic
- A batch of narration clauses with timing
- Each clause's idea_type and duration

MEDIA TYPE PRIORITY (CRITICAL - follow this order):
1. "youtube_video" - HIGHEST PRIORITY for any real-world topic, person, event, place, animal, or thing that exists
2. "ai_video" - For dramatic/cinematic shots, abstract concepts, or impossible-to-film visuals
3. "ai_image" - For artistic/abstract visuals when video isn't needed (duration < 3s)
4. "web_image" - For real photos of specific things/people when YouTube fails
5. "stock_video" - LAST RESORT ONLY - use ONLY for extremely generic visuals (clouds, water, typing hands)

WHEN TO USE EACH TYPE:
- youtube_video: People (celebrities, athletes, politicians), animals, places, events, products, anything REAL
- ai_video: Fantasy scenes, dramatic cinematic shots, visualizing abstract concepts, historical recreations
- ai_image: Short artistic moments, abstract ideas, when < 3s duration
- web_image: Specific real photos needed (products, logos, specific people) when YouTube unavailable
- stock_video: ONLY for ultra-generic B-roll (hands typing, clouds moving, water flowing) - NEVER for the main topic

CRITICAL - HOOK RULE:
- The FIRST clause (the hook) MUST use youtube_video with the exact topic as query
- Example: Topic "LeBron James" → hook query "LeBron James"
- Example: Topic "octopus intelligence" → hook query "octopus"
- The hook MUST show the actual topic to establish what the video is about

QUERY RULES:

For "youtube_video":
- Use the topic name directly, 2-4 words max
- Examples: "LeBron James", "octopus hunting", "Tesla factory", "ancient Rome"

For "ai_video" and "ai_image":
- Be EXTREMELY detailed and cinematic
- Include: subject, action, lighting, mood, camera angle, style
- Example: "dramatic close-up of basketball player dunking, sweat droplets frozen in air, arena lights flaring, cinematic slow motion"

For "web_image":
- Use specific name + context
- Examples: "LeBron James Lakers", "octopus closeup", "Tesla Model S"

For "stock_video" (LAST RESORT):
- Use ONLY generic 2-3 word descriptions
- Examples: "city skyline", "ocean waves", "person walking"
- NEVER use for the main topic - only for generic filler shots

VARIETY RULE:
- Don't use the same media type more than twice in a row
- Alternate between youtube_video, ai_video, ai_image for variety
- Example: youtube_video → ai_image → youtube_video → ai_video (GOOD)

DURATION RULES:
- If duration < 2s, prefer images (ai_image or web_image)
- If duration > 4s, prefer video (youtube_video or ai_video)
- For hooks, ALWAYS use youtube_video regardless of duration

Output JSON array only, no commentary:
[
  {
    "clause_id": 1,
    "media_type": "youtube_video",
    "query": "LeBron James dunk",
    "duration": 2.6,
    "rationale": "Real footage of the topic"
  }
]"""


async def route_clauses_to_media(
    topic: str,
    clauses: List[Dict[str, Any]],
    batch_size: int = 5
) -> List[MediaInstruction]:
    """
    Route clauses to appropriate media sources using LLM.

    Args:
        topic: The overall video topic
        clauses: List of clause dicts with clause_id, text, idea_type, start_time, next_start_time
        batch_size: Number of clauses to process in each LLM call (default 5)

    Returns:
        List of MediaInstruction objects with routing decisions
    """
    if not xai_client:
        raise RuntimeError("xAI client not available")

    all_instructions: List[MediaInstruction] = []

    # Process clauses in batches
    for i in range(0, len(clauses), batch_size):
        batch = clauses[i:i + batch_size]

        # Build batch input with previous media type for continuity
        previous_media_type = None
        if all_instructions:
            previous_media_type = all_instructions[-1].media_type.value

        batch_input = []
        for clause in batch:
            duration = clause.get("next_start_time", 0) - clause.get("start_time", 0)
            batch_input.append({
                "clause_id": clause["clause_id"],
                "text": clause["text"],
                "idea_type": clause.get("idea_type", "fact"),
                "duration": round(duration, 2),
                "start_time": clause.get("start_time", 0),
            })

        user_message = f"""Topic: {topic}

Previous clause media type: {previous_media_type or "none (this is the first batch)"}

Clauses to route:
{json.dumps(batch_input, indent=2)}

Return a JSON array with routing decisions for each clause."""

        print(f"🎯 Routing batch of {len(batch)} clauses...")

        response = xai_client.chat.completions.create(
            model=XAI_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,  # Low temp for consistent routing
            max_tokens=1500
        )

        response_content = response.choices[0].message.content
        if not response_content:
            print(f"⚠ Empty response for batch starting at clause {batch[0]['clause_id']}")
            continue

        # Parse JSON response
        try:
            json_match = re.search(r'\[[\s\S]*\]', response_content)
            if not json_match:
                raise ValueError("No JSON array found")

            decisions = json.loads(json_match.group())

            for decision in decisions:
                # Validate media_type
                media_type_str = decision.get("media_type", "stock_video")
                try:
                    media_type = MediaType(media_type_str)
                except ValueError:
                    print(f"⚠ Invalid media type '{media_type_str}', defaulting to stock_video")
                    media_type = MediaType.STOCK_VIDEO

                # Apply hard rules
                duration = decision.get("duration", 3.0)

                # Rule: Short durations force images
                if duration < MIN_VIDEO_DURATION and media_type in [MediaType.AI_VIDEO, MediaType.YOUTUBE_VIDEO, MediaType.STOCK_VIDEO]:
                    media_type = MediaType.WEB_IMAGE if "real" in decision.get("rationale", "").lower() else MediaType.AI_IMAGE
                    print(f"   → Clause {decision['clause_id']}: Forced image (duration {duration:.1f}s < {MIN_VIDEO_DURATION}s)")

                # Rule: Hooks need video if long enough
                clause_data = next((c for c in batch if c["clause_id"] == decision["clause_id"]), None)
                if clause_data and clause_data.get("idea_type") == "hook" and duration >= MIN_HOOK_DURATION:
                    if media_type in [MediaType.AI_IMAGE, MediaType.WEB_IMAGE]:
                        media_type = MediaType.STOCK_VIDEO
                        print(f"   → Clause {decision['clause_id']}: Forced video for hook")

                instruction = MediaInstruction(
                    clause_id=decision["clause_id"],
                    media_type=media_type,
                    query=decision.get("query", ""),
                    duration=duration,
                    rationale=decision.get("rationale", "")
                )
                all_instructions.append(instruction)

                print(f"   [{instruction.clause_id}] {instruction.media_type.value}: \"{instruction.query}\" ({instruction.duration:.1f}s)")

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠ Error parsing routing response: {e}")
            print(f"   Response was: {response_content[:300]}")

            # Fallback: create default instructions for this batch
            for clause in batch:
                duration = clause.get("next_start_time", 0) - clause.get("start_time", 0)
                all_instructions.append(MediaInstruction(
                    clause_id=clause["clause_id"],
                    media_type=MediaType.STOCK_VIDEO,
                    query=f"{topic} {clause.get('idea_type', 'scene')}",
                    duration=duration,
                    rationale="Fallback due to parsing error"
                ))

    print(f"✓ Routed {len(all_instructions)} clauses")

    # Enforce hook rule: first clause must contain the topic
    all_instructions = _enforce_hook_topic(all_instructions, topic)

    # Enforce variety: no back-to-back same media types
    all_instructions = _enforce_media_variety(all_instructions)

    return all_instructions


def _enforce_hook_topic(instructions: List[MediaInstruction], topic: str) -> List[MediaInstruction]:
    """
    Ensure the first clause (hook) always contains the topic in its query.
    Forces youtube_video or web_image for hooks to ensure topic visibility.
    """
    if not instructions:
        return instructions

    hook = instructions[0]

    # Check if topic (or main keywords from it) is in the query
    topic_words = set(topic.lower().split())
    query_words = set(hook.query.lower().split())

    # Allow if there's meaningful overlap (at least one main topic word)
    has_topic = bool(topic_words & query_words)

    # Also check if the query is just generic stock footage that doesn't relate to topic
    generic_stock_terms = {'person', 'people', 'crowd', 'city', 'nature', 'abstract', 'background', 'sky', 'water'}
    is_generic = query_words.issubset(generic_stock_terms | {'the', 'a', 'an', 'in', 'on', 'at'})

    if not has_topic or is_generic or hook.media_type == MediaType.STOCK_VIDEO:
        # Fix the hook - use topic directly and switch to youtube or web_image
        new_media_type = MediaType.YOUTUBE_VIDEO if hook.duration >= 2.5 else MediaType.WEB_IMAGE

        # Use simplified topic as query
        topic_query = topic.strip()
        if len(topic_query.split()) > 4:
            # Shorten very long topics to first 3-4 meaningful words
            topic_query = ' '.join(topic_query.split()[:4])

        instructions[0] = MediaInstruction(
            clause_id=hook.clause_id,
            media_type=new_media_type,
            query=topic_query,
            duration=hook.duration,
            rationale=f"Hook forced to show topic directly: {topic_query}"
        )
        print(f"   → Hook (clause 1): Forced to {new_media_type.value} with topic query \"{topic_query}\"")

    return instructions


def _enforce_media_variety(instructions: List[MediaInstruction]) -> List[MediaInstruction]:
    """
    Post-process routing decisions to ensure no back-to-back same media types.
    Changes the media type of clauses that repeat the previous clause's type.
    """
    if len(instructions) < 2:
        return instructions

    # Define alternative types for each media type (AI preferred over stock)
    alternatives = {
        MediaType.STOCK_VIDEO: [MediaType.AI_IMAGE, MediaType.WEB_IMAGE],
        MediaType.YOUTUBE_VIDEO: [MediaType.AI_IMAGE, MediaType.AI_VIDEO, MediaType.WEB_IMAGE],
        MediaType.WEB_IMAGE: [MediaType.AI_IMAGE, MediaType.YOUTUBE_VIDEO],
        MediaType.AI_IMAGE: [MediaType.YOUTUBE_VIDEO, MediaType.WEB_IMAGE],
        MediaType.AI_VIDEO: [MediaType.YOUTUBE_VIDEO, MediaType.AI_IMAGE],
    }

    changes_made = 0
    for i in range(1, len(instructions)):
        prev_type = instructions[i - 1].media_type
        curr_type = instructions[i].media_type

        if curr_type == prev_type:
            # Need to change this one
            alts = alternatives.get(curr_type, [MediaType.WEB_IMAGE])
            # Pick the first alternative that isn't also the same as i-2 (if exists)
            new_type = alts[0]
            if i >= 2 and instructions[i - 2].media_type == new_type and len(alts) > 1:
                new_type = alts[1]

            instructions[i] = MediaInstruction(
                clause_id=instructions[i].clause_id,
                media_type=new_type,
                query=instructions[i].query,
                duration=instructions[i].duration,
                rationale=f"Changed from {curr_type.value} to avoid repetition"
            )
            changes_made += 1
            print(f"   → Clause {instructions[i].clause_id}: Changed {curr_type.value} → {new_type.value} (variety)")

    if changes_made > 0:
        print(f"   Adjusted {changes_made} clauses to ensure variety")

    return instructions


async def retrieve_media_for_instruction(
    instruction: MediaInstruction,
    topic: str
) -> MediaResult:
    """
    Retrieve actual media based on a routing instruction.

    Args:
        instruction: MediaInstruction from the router
        topic: The video topic (for context)

    Returns:
        MediaResult with the retrieved media URL or error
    """
    result = MediaResult(
        clause_id=instruction.clause_id,
        media_type=instruction.media_type,
        query=instruction.query,
        duration=instruction.duration,
    )

    try:
        if instruction.media_type == MediaType.STOCK_VIDEO:
            result.media_url = await _retrieve_stock_video(instruction.query, min_duration=instruction.duration)

        elif instruction.media_type == MediaType.YOUTUBE_VIDEO:
            result.media_url = await _retrieve_youtube_video(instruction.query, topic, instruction.duration)

        elif instruction.media_type == MediaType.WEB_IMAGE:
            result.media_url = await _retrieve_web_image(instruction.query)

        elif instruction.media_type == MediaType.AI_IMAGE:
            result.media_url = await _retrieve_ai_image(instruction.query, topic)

        elif instruction.media_type == MediaType.AI_VIDEO:
            result.media_url = await _retrieve_ai_video(instruction.query, topic, instruction.duration)

    except Exception as e:
        result.error = str(e)
        print(f"⚠ Error retrieving media for clause {instruction.clause_id}: {e}")

    return result


async def _retrieve_stock_video(query: str, min_duration: float = 0) -> Optional[str]:
    """Retrieve stock video from Pexels/Pixabay that meets minimum duration. Downloads to local cache."""
    from media_video_search import search_pexels_videos, search_pixabay_videos
    import requests
    import tempfile
    import hashlib

    cache_dir = Path(tempfile.gettempdir()) / "video_factory_cache" / "stock_videos"
    cache_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    async def download_video(url: str, source: str) -> Optional[str]:
        """Download video and return local path."""
        try:
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            local_path = cache_dir / f"{source}_{url_hash}.mp4"

            # Check if already cached
            if local_path.exists() and local_path.stat().st_size > 10000:
                print(f"   ✓ Using cached {source} video: {local_path.name}")
                return str(local_path)

            resp = requests.get(url, timeout=60, headers=headers, stream=True)
            if resp.status_code != 200:
                print(f"   ⚠ Failed to download {source} video: HTTP {resp.status_code}")
                return None

            with open(local_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            if local_path.exists() and local_path.stat().st_size > 10000:
                print(f"   ✓ Downloaded {source} video: {local_path.name}")
                return str(local_path)
            else:
                local_path.unlink(missing_ok=True)
                return None
        except Exception as e:
            print(f"   ⚠ {source} video download failed: {e}")
            return None

    # Try Pexels first
    try:
        results = await asyncio.to_thread(search_pexels_videos, query, per_page=10, orientation="portrait")
        if results:
            # Filter by duration - need videos at least as long as required
            long_enough = [v for v in results if v.get("duration", 0) >= min_duration]
            if long_enough:
                # Prefer vertical videos
                vertical = next((v for v in long_enough if v.get("height", 0) > v.get("width", 0)), None)
                selected = vertical if vertical else long_enough[0]
                print(f"   Pexels found: {selected.get('duration', 0)}s (need {min_duration:.1f}s)")
                local_path = await download_video(selected["url"], "pexels")
                if local_path:
                    return local_path
            else:
                print(f"   ⚠ Pexels: {len(results)} results but none >= {min_duration:.1f}s")
    except Exception as e:
        print(f"   Pexels search failed: {e}")

    # Fallback to Pixabay
    try:
        results = await asyncio.to_thread(search_pixabay_videos, query, per_page=10)
        if results:
            # Filter by duration
            long_enough = [v for v in results if v.get("duration", 0) >= min_duration]
            if long_enough:
                # Prefer vertical videos
                vertical = next((v for v in long_enough if v.get("height", 0) > v.get("width", 0)), None)
                selected = vertical if vertical else long_enough[0]
                print(f"   Pixabay found: {selected.get('duration', 0)}s (need {min_duration:.1f}s)")
                local_path = await download_video(selected["url"], "pixabay")
                if local_path:
                    return local_path
            else:
                print(f"   ⚠ Pixabay: {len(results)} results but none >= {min_duration:.1f}s")
    except Exception as e:
        print(f"   Pixabay search failed: {e}")

    return None


async def _retrieve_youtube_video(query: str, topic: str, duration: float) -> Optional[str]:
    """Retrieve and clip YouTube video."""
    from media_youtube_download import download_single_youtube_clip

    try:
        result = await download_single_youtube_clip(
            query=query,
            topic=topic,
            target_duration=duration
        )
        return result.get("video_path") if result else None
    except Exception as e:
        print(f"   YouTube download failed: {e}")
        return None


async def _retrieve_web_image(query: str) -> Optional[str]:
    """Retrieve web image from Google. Actually downloads to verify and cache locally."""
    from media_image_search import search_google_images
    import requests
    import tempfile
    import hashlib

    # Cache directory for downloaded images
    cache_dir = Path(tempfile.gettempdir()) / "video_factory_cache" / "web_images"
    cache_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        # Get multiple results so we can try alternatives if first fails
        results = search_google_images(query, num_results=5)

        for img in results:
            url = img.get("url")
            if not url:
                continue

            # Actually download the image to verify it works
            try:
                resp = requests.get(url, timeout=10, headers=headers, stream=True)
                if resp.status_code != 200:
                    print(f"   ⚠ Web image blocked ({resp.status_code}): {url[:50]}...")
                    continue

                # Determine extension from content-type or URL
                content_type = resp.headers.get('content-type', '')
                if 'png' in content_type or url.lower().endswith('.png'):
                    ext = '.png'
                elif 'webp' in content_type or url.lower().endswith('.webp'):
                    ext = '.webp'
                else:
                    ext = '.jpg'

                # Save to cache with hash-based filename
                url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                local_path = cache_dir / f"web_{url_hash}{ext}"

                with open(local_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Verify file was saved and has content
                if local_path.exists() and local_path.stat().st_size > 1000:
                    print(f"   ✓ Downloaded web image: {local_path.name}")
                    return str(local_path)
                else:
                    print(f"   ⚠ Web image too small or empty: {url[:50]}...")
                    local_path.unlink(missing_ok=True)

            except Exception as e:
                print(f"   ⚠ Web image download failed: {url[:50]}... ({e})")
                continue

        # If all results failed, try simplified query
        words = query.split()
        if len(words) > 2:
            simplified = ' '.join(words[:2])
            print(f"   Trying simplified query: '{simplified}'")
            results = search_google_images(simplified, num_results=3)

            for img in results:
                url = img.get("url")
                if not url:
                    continue
                try:
                    resp = requests.get(url, timeout=10, headers=headers, stream=True)
                    if resp.status_code != 200:
                        continue

                    content_type = resp.headers.get('content-type', '')
                    ext = '.png' if 'png' in content_type else '.jpg'
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                    local_path = cache_dir / f"web_{url_hash}{ext}"

                    with open(local_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)

                    if local_path.exists() and local_path.stat().st_size > 1000:
                        print(f"   ✓ Downloaded web image (simplified): {local_path.name}")
                        return str(local_path)

                except Exception:
                    continue

        print(f"   ✗ No downloadable web images found for: {query}")
        return None

    except Exception as e:
        print(f"   Web image search failed: {e}")
        return None


async def _retrieve_ai_image(query: str, topic: str) -> Optional[str]:
    """Generate AI image and download to local cache."""
    from media_image_generate import generate_single_image
    import requests
    import tempfile
    import hashlib

    try:
        result = await generate_single_image(prompt=query, topic=topic)
        if not result or not result.get("url"):
            return None

        # Download the generated image to local cache
        url = result["url"]
        cache_dir = Path(tempfile.gettempdir()) / "video_factory_cache" / "ai_images"
        cache_dir.mkdir(parents=True, exist_ok=True)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        resp = requests.get(url, timeout=30, headers=headers, stream=True)
        if resp.status_code != 200:
            print(f"   ⚠ Failed to download AI image: HTTP {resp.status_code}")
            return None

        # Save to cache
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        local_path = cache_dir / f"ai_{url_hash}.jpg"

        with open(local_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        if local_path.exists() and local_path.stat().st_size > 1000:
            print(f"   ✓ Downloaded AI image: {local_path.name}")
            return str(local_path)
        else:
            print(f"   ⚠ AI image download too small or empty")
            local_path.unlink(missing_ok=True)
            return None

    except Exception as e:
        print(f"   AI image generation failed: {e}")
        return None


async def _retrieve_ai_video(query: str, topic: str, duration: float) -> Optional[str]:
    """Generate AI video and download to local cache."""
    from media_video_generate import generate_single_video
    import requests
    import tempfile
    import hashlib

    try:
        result = await generate_single_video(prompt=query, topic=topic, duration=duration)
        if not result or not result.get("url"):
            return None

        # Download the generated video to local cache
        url = result["url"]
        cache_dir = Path(tempfile.gettempdir()) / "video_factory_cache" / "ai_videos"
        cache_dir.mkdir(parents=True, exist_ok=True)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        resp = requests.get(url, timeout=120, headers=headers, stream=True)
        if resp.status_code != 200:
            print(f"   ⚠ Failed to download AI video: HTTP {resp.status_code}")
            return None

        # Save to cache
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        local_path = cache_dir / f"ai_video_{url_hash}.mp4"

        with open(local_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        if local_path.exists() and local_path.stat().st_size > 10000:
            print(f"   ✓ Downloaded AI video: {local_path.name}")
            return str(local_path)
        else:
            print(f"   ⚠ AI video download too small or empty")
            local_path.unlink(missing_ok=True)
            return None

    except Exception as e:
        print(f"   AI video generation failed: {e}")
        return None


def _get_video_duration(media_url: str) -> Optional[float]:
    """Get the duration of a video file. Returns None if not a video or can't determine."""
    if not media_url:
        return None

    # Skip images
    if any(media_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
        return None

    # Handle local file paths
    if media_url.startswith('/') or media_url.startswith('./'):
        path = Path(media_url)
        if not path.exists():
            return None
        try:
            return get_media_duration(str(path))
        except Exception:
            return None

    # For remote URLs, we can't easily check duration without downloading
    # So we'll trust that the source provided adequate content
    return None


# Fallback media types in priority order
# Fallback order: AI first, then web images, stock video as absolute last resort
FALLBACK_TYPES = [MediaType.AI_IMAGE, MediaType.WEB_IMAGE, MediaType.STOCK_VIDEO]


async def _get_replacement_media(
    instruction: MediaInstruction,
    topic: str,
    tried_types: set
) -> Optional[MediaResult]:
    """Try to get replacement media using a different media type."""
    for fallback_type in FALLBACK_TYPES:
        if fallback_type in tried_types:
            continue
        if fallback_type == instruction.media_type:
            continue

        # Mark as tried before attempting
        tried_types.add(fallback_type)

        print(f"   🔄 Trying fallback: {fallback_type.value} for clause {instruction.clause_id}")

        # Generate a simpler query for fallback
        query = instruction.query
        if len(query.split()) > 3 and fallback_type == MediaType.STOCK_VIDEO:
            # Simplify query for stock video
            words = query.split()[:2]
            query = ' '.join(words)

        result = MediaResult(
            clause_id=instruction.clause_id,
            media_type=fallback_type,
            query=query,
            duration=instruction.duration,
        )

        try:
            if fallback_type == MediaType.STOCK_VIDEO:
                result.media_url = await _retrieve_stock_video(query, min_duration=instruction.duration)
            elif fallback_type == MediaType.WEB_IMAGE:
                result.media_url = await _retrieve_web_image(query)
            elif fallback_type == MediaType.AI_IMAGE:
                # For AI image, use a more descriptive prompt
                result.media_url = await _retrieve_ai_image(instruction.query, topic)

            if result.media_url:
                print(f"   ✓ Got replacement {fallback_type.value} for clause {instruction.clause_id}")
                return result
            else:
                print(f"   ⚠ Fallback {fallback_type.value} returned no media")
        except Exception as e:
            print(f"   ⚠ Fallback {fallback_type.value} failed: {e}")

    return None


async def retrieve_all_media(
    instructions: List[MediaInstruction],
    topic: str,
    max_concurrent: int = 4
) -> List[MediaResult]:
    """
    Retrieve media for all instructions with concurrency limit.
    Automatically replaces failed or too-short media with alternatives.

    Args:
        instructions: List of MediaInstruction from router
        topic: The video topic
        max_concurrent: Maximum concurrent retrieval operations

    Returns:
        List of MediaResult objects
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def retrieve_with_limit(instruction: MediaInstruction) -> MediaResult:
        async with semaphore:
            return await retrieve_media_for_instruction(instruction, topic)

    print(f"\n📦 Retrieving media for {len(instructions)} clauses (max {max_concurrent} concurrent)...")

    tasks = [retrieve_with_limit(inst) for inst in instructions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    media_results: List[MediaResult] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            media_results.append(MediaResult(
                clause_id=instructions[i].clause_id,
                media_type=instructions[i].media_type,
                query=instructions[i].query,
                duration=instructions[i].duration,
                error=str(result)
            ))
        else:
            media_results.append(result)

    # Check for failed or too-short media and try replacements
    print(f"\n🔍 Checking for failed or insufficient media...")
    MIN_DURATION_TOLERANCE = 0.5  # Tolerate videos up to 0.5s shorter than needed

    for i, (result, instruction) in enumerate(zip(media_results, instructions)):
        needs_replacement = False
        reason = ""

        # Check if media retrieval failed
        if not result.media_url or result.error:
            needs_replacement = True
            reason = "retrieval failed"

        # Check if video duration is too short (only for local video files)
        elif result.media_type in [MediaType.STOCK_VIDEO, MediaType.YOUTUBE_VIDEO, MediaType.AI_VIDEO]:
            video_duration = _get_video_duration(result.media_url)
            if video_duration is not None and video_duration < (instruction.duration - MIN_DURATION_TOLERANCE):
                needs_replacement = True
                reason = f"video too short ({video_duration:.1f}s < {instruction.duration:.1f}s needed)"

        if needs_replacement:
            print(f"   ⚠ Clause {result.clause_id}: {reason}")
            tried_types = {instruction.media_type}
            if result.media_type != instruction.media_type:
                tried_types.add(result.media_type)

            replacement = await _get_replacement_media(instruction, topic, tried_types)
            if replacement:
                media_results[i] = replacement
            else:
                print(f"   ✗ No replacement found for clause {result.clause_id}")

    # Final summary
    successful = len([r for r in media_results if r.media_url])
    failed = len([r for r in media_results if r.error or not r.media_url])
    print(f"✓ Media retrieval complete: {successful} successful, {failed} failed")

    return media_results


async def route_and_retrieve_media(
    topic: str,
    clauses: List[Dict[str, Any]],
    batch_size: int = 5,
    max_concurrent: int = 4
) -> List[MediaResult]:
    """
    Full pipeline: route clauses to media types, then retrieve all media.

    Args:
        topic: The video topic
        clauses: List of clause dicts
        batch_size: Clauses per LLM routing call
        max_concurrent: Maximum concurrent retrievals

    Returns:
        List of MediaResult objects with media URLs
    """
    # Step 1: Route clauses to media types
    instructions = await route_clauses_to_media(topic, clauses, batch_size)

    # Step 2: Retrieve all media
    results = await retrieve_all_media(instructions, topic, max_concurrent)

    return results
