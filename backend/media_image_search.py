"""
Image search module for Google Programmable Search API.
Optimized for finding images for TikTok-style video backgrounds.
"""

import os
import requests
from typing import List, Dict, Any
from config import xai_client, XAI_MODEL


def search_google_images(
    query: str,
    num_results: int = 1
) -> List[Dict[str, Any]]:
    """
    Search Google for images using the Programmable Search API.

    Args:
        query: Search terms for the images
        num_results: Number of results to return (1-10)

    Returns:
        List of image dicts with url, width, height, thumbnail_url, title
    """
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cx = os.getenv("GOOGLE_SEARCH_CX")

    if not api_key or not cx:
        print("Warning: Google Search API not configured (missing GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_CX)")
        return []

    try:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": api_key,
                "cx": cx,
                "q": query,
                "searchType": "image",
                "num": min(num_results, 10),
                "imgSize": "xlarge",
                "safe": "active",
            },
            timeout=10
        )

        if response.status_code != 200:
            print(f"Google Search API error: {response.status_code} - {response.text[:200]}")
            return []

        data = response.json()
        items = data.get("items", [])
        results = []

        for item in items:
            image_info = item.get("image", {})
            results.append({
                "url": item.get("link", ""),
                "width": image_info.get("width", 0),
                "height": image_info.get("height", 0),
                "thumbnail_url": image_info.get("thumbnailLink", ""),
                "title": item.get("title", ""),
                "context_link": image_info.get("contextLink", ""),
            })

        return results

    except Exception as e:
        print(f"Google image search error: {e}")
        return []


def generate_image_search_queries(
    topic: str,
    script: str,
    scenes: List[Dict[str, Any]]
) -> List[str]:
    """
    Use LLM to generate specific Google Image search queries for ALL scenes at once.

    Args:
        topic: The overall video topic
        script: The full script text
        scenes: List of scene dicts with scene_number, section_name, description

    Returns:
        List of search queries, one per scene (in order)
    """
    if not xai_client:
        return [scene.get('description', '')[:50] for scene in scenes]

    try:
        scene_list = "\n".join([
            f"Scene {s.get('scene_number', i+1)} ({s.get('section_name', 'SECTION')}): {s.get('description', '')}"
            for i, s in enumerate(scenes)
        ])

        prompt = f"""Generate Google Image search queries for a video about: {topic}

SCRIPT:
{script}

SCENES:
{scene_list}

CRITICAL RULES:
1. EVERY query MUST start with the main topic "{topic}" (or the key name/subject from it)
2. Then ADD the specific detail from that section (event, year, person, concept)
3. Keep queries short: 3-5 words total
4. NO filler words (a, the, is, of, for, about, with)

FORMAT: [Main Topic] + [Specific Detail from Section]

EXAMPLE for topic "Woodrow Wilson":
- Section about early life → Woodrow Wilson young portrait
- Section about WWI → Woodrow Wilson World War 1
- Section about League of Nations → Woodrow Wilson League Nations speech
- Section about presidency → Woodrow Wilson president White House

EXAMPLE for topic "Spain basketball team":
- Section mentioning Ricky Rubio → Ricky Rubio Spain basketball
- Section about 2019 championship → Spain basketball 2019 World Cup
- Section about Olympics → Spain basketball Olympic medal

Return ONLY the queries, one per line, in scene order. Every query must be relevant to "{topic}"."""

        response = xai_client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )

        queries_text = response.choices[0].message.content.strip()
        queries = [q.strip().strip('"\'') for q in queries_text.split('\n') if q.strip()]

        # Ensure we have the right number of queries
        if len(queries) < len(scenes):
            for i in range(len(queries), len(scenes)):
                queries.append(scenes[i].get('description', '')[:50])
        elif len(queries) > len(scenes):
            queries = queries[:len(scenes)]

        print(f"🔍 LLM generated {len(queries)} image search queries:")
        for i, q in enumerate(queries):
            print(f"   Scene {i+1}: '{q}'")

        return queries

    except Exception as e:
        print(f"⚠ LLM image query generation failed: {e}, using fallback")
        return [scene.get('description', '')[:50] for scene in scenes]


async def search_images_with_query(
    search_query: str,
    fallback_query: str = ""
) -> Dict[str, Any]:
    """
    Search for images using a pre-generated query.
    Takes Google's top result (best relevance).

    Args:
        search_query: The primary search query (from LLM)
        fallback_query: Fallback query if primary fails

    Returns:
        Dict with 'query' (the query used) and 'images' (list of results)
    """
    print(f"    Searching Google images: '{search_query}'")

    results = search_google_images(search_query, num_results=1)

    if results:
        best = results[0]
        print(f"    ✓ Found image: {best.get('width')}x{best.get('height')} - {best.get('title', 'untitled')[:40]}")
        return {"query": search_query, "images": [best]}

    # Try fallback if provided and different
    if fallback_query and fallback_query != search_query:
        print(f"    Trying fallback: '{fallback_query}'")
        results = search_google_images(fallback_query, num_results=1)

        if results:
            best = results[0]
            print(f"    ✓ Found image: {best.get('width')}x{best.get('height')} - {best.get('title', 'untitled')[:40]}")
            return {"query": fallback_query, "images": [best]}

    # Final fallback: simplified query
    words = search_query.split()
    if len(words) > 2:
        simplified = ' '.join(words[:2])
        print(f"    Trying simplified: '{simplified}'")
        results = search_google_images(simplified, num_results=1)

        if results:
            best = results[0]
            print(f"    ✓ Found image: {best.get('width')}x{best.get('height')} - {best.get('title', 'untitled')[:40]}")
            return {"query": simplified, "images": [best]}

    print(f"    ✗ No images found")
    return {"query": search_query, "images": []}
