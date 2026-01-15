"""
Video search module for Pexels and Pixabay APIs.
Optimized for finding portrait/vertical stock videos for TikTok-style content.
"""

import os
import requests
from typing import List, Dict, Any
from fastapi import HTTPException
from models import VideoSearchRequest, VideoSearchResponse, VideoItem


# Pexels API documentation:
# - query: "Ocean, Tigers, Pears" or "Group of people working"
# - orientation: landscape, portrait, or square
# - size: large (4K), medium (Full HD), small (HD)


def search_pexels_videos(
    query: str,
    per_page: int = 5,
    orientation: str = "portrait"
) -> List[Dict[str, Any]]:
    """
    Search Pexels for videos matching the query.

    Args:
        query: Simple search terms like "ocean waves", "person typing", "city night"
        per_page: Number of results (1-80)
        orientation: "portrait" for vertical, "landscape" for horizontal, "square"

    Returns:
        List of video dicts with url, width, height, duration, thumbnail_url
    """
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        print("Warning: PEXELS_API_KEY not set")
        return []

    try:
        response = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": api_key},
            params={
                "query": query,
                "per_page": per_page,
                "orientation": orientation,  # portrait, landscape, or square
                "size": "medium",  # Full HD quality
            },
            timeout=10
        )

        if response.status_code != 200:
            print(f"Pexels API error: {response.status_code}")
            return []

        data = response.json()
        videos = data.get("videos", [])
        results = []

        for video in videos:
            video_files = video.get("video_files", [])
            if not video_files:
                continue

            # Find best quality video file (prefer HD/Full HD)
            # Sort by height descending, but cap at 1920 to avoid huge files
            suitable_files = [
                vf for vf in video_files
                if vf.get("height", 0) <= 1920 and vf.get("link")
            ]

            if not suitable_files:
                suitable_files = [vf for vf in video_files if vf.get("link")]

            if not suitable_files:
                continue

            # Sort by height descending to get best quality
            suitable_files.sort(key=lambda x: x.get("height", 0), reverse=True)
            best_file = suitable_files[0]

            results.append({
                "url": best_file["link"],
                "width": best_file.get("width", 0),
                "height": best_file.get("height", 0),
                "duration": video.get("duration", 0),
                "thumbnail_url": video.get("image", ""),
                "id": video.get("id"),
            })

        return results

    except Exception as e:
        print(f"Pexels search error: {e}")
        return []


def search_pixabay_videos(
    query: str,
    per_page: int = 5
) -> List[Dict[str, Any]]:
    """
    Search Pixabay for videos matching the query.

    API: https://pixabay.com/api/videos/?key=API_KEY&q=yellow+flowers

    Args:
        query: Simple search terms (will be URL encoded automatically)
        per_page: Number of results (3-200)

    Returns:
        List of video dicts with url, width, height, duration, thumbnail_url
    """
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key:
        print("Warning: PIXABAY_API_KEY not set")
        return []

    try:
        # requests library will URL-encode the 'q' parameter automatically
        response = requests.get(
            "https://pixabay.com/api/videos/",
            params={
                "key": api_key,
                "q": query,  # Don't pre-encode, requests handles it
                "per_page": per_page,
                "safesearch": "true",
                "order": "popular",
            },
            timeout=10
        )

        if response.status_code != 200:
            print(f"Pixabay API error: {response.status_code}")
            return []

        data = response.json()
        hits = data.get("hits", [])
        results = []

        for video in hits:
            videos_obj = video.get("videos", {})

            # Prefer medium quality (Full HD), fall back to others
            video_file = None
            for size_key in ["medium", "large", "small", "tiny"]:
                if size_key in videos_obj and videos_obj[size_key].get("url"):
                    video_file = videos_obj[size_key]
                    break

            if not video_file:
                continue

            results.append({
                "url": video_file["url"],
                "width": video_file.get("width", 0),
                "height": video_file.get("height", 0),
                "duration": video.get("duration", 0),
                "thumbnail_url": video_file.get("thumbnail", ""),
                "id": video.get("id"),
            })

        return results

    except Exception as e:
        print(f"Pixabay search error: {e}")
        return []


async def search_videos(request: VideoSearchRequest) -> VideoSearchResponse:
    """
    Search for videos from Pexels and Pixabay based on optimized search query.
    Returns videos from both sources combined.
    """
    if not request.search_query or not request.search_query.strip():
        raise HTTPException(status_code=400, detail="Search query is required")

    all_videos = []
    search_query = request.search_query.strip()

    # Get API keys from environment
    pexels_api_key = os.getenv("PEXELS_API_KEY")
    pixabay_api_key = os.getenv("PIXABAY_API_KEY")

    # Search Pexels (with portrait orientation for TikTok)
    if pexels_api_key:
        pexels_results = search_pexels_videos(search_query, per_page=3, orientation="portrait")
        for video in pexels_results:
            all_videos.append(VideoItem(
                id=f"pexels_{video['id']}",
                url=video["url"],
                thumbnail_url=video.get("thumbnail_url", ""),
                source="pexels",
                duration=video.get("duration"),
                width=video.get("width"),
                height=video.get("height")
            ))

    # Search Pixabay
    if pixabay_api_key:
        pixabay_results = search_pixabay_videos(search_query, per_page=3)
        # Filter for vertical videos if possible
        vertical_results = [v for v in pixabay_results if v.get("height", 0) > v.get("width", 0)]
        results_to_use = vertical_results if vertical_results else pixabay_results

        for video in results_to_use:
            all_videos.append(VideoItem(
                id=f"pixabay_{video['id']}",
                url=video["url"],
                thumbnail_url=video.get("thumbnail_url", ""),
                source="pixabay",
                duration=video.get("duration"),
                width=video.get("width"),
                height=video.get("height")
            ))

    if not all_videos and not pexels_api_key and not pixabay_api_key:
        raise HTTPException(
            status_code=503,
            detail="Video search APIs not configured. Please set PEXELS_API_KEY and/or PIXABAY_API_KEY."
        )

    return VideoSearchResponse(videos=all_videos)
