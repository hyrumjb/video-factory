import os
import requests
from urllib.parse import quote_plus
from fastapi import HTTPException
from models import VideoSearchRequest, VideoSearchResponse, VideoItem

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
    
    # Search Pexels
    if pexels_api_key:
        try:
            pexels_url = "https://api.pexels.com/videos/search"
            headers = {"Authorization": pexels_api_key}
            params = {
                "query": search_query,
                "per_page": 1,
                "orientation": "vertical"
            }
            
            pexels_response = requests.get(pexels_url, headers=headers, params=params, timeout=10)
            
            if pexels_response.status_code == 200:
                pexels_data = pexels_response.json()
                videos_list = pexels_data.get("videos", [])
                video_url = None
                
                # First, try to find videos that are vertical
                for video in videos_list:
                    video_files = video.get("video_files", [])
                    vertical_videos = [vf for vf in video_files if vf.get("height", 0) > vf.get("width", 0)]
                    if vertical_videos:
                        for vf in sorted(vertical_videos, key=lambda x: x.get("height", 0), reverse=True):
                            if vf.get("link"):
                                video_url = vf["link"]
                                break
                        if video_url:
                            break
                
                # If still no vertical video found, check video dimensions from API response
                if not video_url:
                    for video in videos_list:
                        video_width = video.get("width", 0)
                        video_height = video.get("height", 0)
                        if video_height > video_width:
                            video_files = video.get("video_files", [])
                            for vf in sorted(video_files, key=lambda x: x.get("height", 0), reverse=True):
                                if vf.get("link"):
                                    video_url = vf["link"]
                                    break
                            if video_url:
                                break
                
                # Final fallback: use first video if no vertical found
                if not video_url and videos_list:
                    video = videos_list[0]
                    video_files = video.get("video_files", [])
                    for vf in sorted(video_files, key=lambda x: x.get("height", 0), reverse=True):
                        if vf.get("link"):
                            video_url = vf["link"]
                            break
                    
                    if video_url:
                        all_videos.append(VideoItem(
                            id=f"pexels_{video.get('id')}",
                            url=video_url,
                            thumbnail_url=video.get("image", ""),
                            source="pexels",
                            duration=video.get("duration"),
                            width=video_files[0].get("width") if video_files else None,
                            height=video_files[0].get("height") if video_files else None
                        ))
        except Exception as e:
            print(f"Error fetching Pexels videos: {e}")
    
    # Search Pixabay
    if pixabay_api_key:
        try:
            # Try multiple search strategies
            search_queries = [search_query]
            words = search_query.split()
            if len(words) > 2:
                search_queries.append(' '.join(words[:2]))
            if len(words) > 1:
                search_queries.append(words[0])
            
            pixabay_url = "https://pixabay.com/api/videos/"
            
            for query_attempt in search_queries:
                encoded_query = quote_plus(query_attempt)
                
                params = {
                    "key": pixabay_api_key,
                    "q": encoded_query,
                    "video_type": "all",
                    "per_page": 3,
                    "safesearch": "true",
                    "order": "popular"
                }
                
                pixabay_response = requests.get(pixabay_url, params=params, timeout=10)
                
                if pixabay_response.status_code == 200:
                    pixabay_data = pixabay_response.json()
                    
                    if "error" in pixabay_data:
                        print(f"Pixabay API error in response: {pixabay_data.get('error')}")
                        continue
                    
                    hits = pixabay_data.get("hits", [])
                    total_hits = pixabay_data.get("totalHits", 0)
                    
                    if hits:
                        # First, try to find a vertical video
                        vertical_video = None
                        for video in hits:
                            videos_obj = video.get("videos", {})
                            is_vertical = False
                            for size_key in ["large", "medium", "small", "tiny"]:
                                if size_key in videos_obj:
                                    v_width = videos_obj[size_key].get("width", 0)
                                    v_height = videos_obj[size_key].get("height", 0)
                                    if v_height > v_width:
                                        is_vertical = True
                                        break
                            
                            if is_vertical:
                                vertical_video = video
                                break
                        
                        video = vertical_video if vertical_video else hits[0]
                        
                        video_url = None
                        thumbnail_url = ""
                        videos_obj = video.get("videos", {})
                        
                        # Prefer vertical video files if available
                        vertical_video_file = None
                        for size_key in ["medium", "small", "tiny", "large"]:
                            if size_key in videos_obj:
                                vf = videos_obj[size_key]
                                if vf.get("url") and vf.get("height", 0) > vf.get("width", 0):
                                    vertical_video_file = vf
                                    break
                        
                        if vertical_video_file:
                            video_url = vertical_video_file.get("url")
                            thumbnail_url = vertical_video_file.get("thumbnail", "")
                        elif "medium" in videos_obj and videos_obj["medium"].get("url"):
                            video_url = videos_obj["medium"].get("url")
                            thumbnail_url = videos_obj["medium"].get("thumbnail", "")
                        elif "small" in videos_obj and videos_obj["small"].get("url"):
                            video_url = videos_obj["small"].get("url")
                            thumbnail_url = videos_obj["small"].get("thumbnail", "")
                        elif "tiny" in videos_obj and videos_obj["tiny"].get("url"):
                            video_url = videos_obj["tiny"].get("url")
                            thumbnail_url = videos_obj["tiny"].get("thumbnail", "")
                        
                        if video_url:
                            all_videos.append(VideoItem(
                                id=f"pixabay_{video.get('id')}",
                                url=video_url,
                                thumbnail_url=thumbnail_url,
                                source="pixabay",
                                duration=video.get("duration"),
                                width=videos_obj.get("medium", {}).get("width") if "medium" in videos_obj else None,
                                height=videos_obj.get("medium", {}).get("height") if "medium" in videos_obj else None
                            ))
                            break
                    elif query_attempt == search_queries[-1]:
                        print(f"Pixabay: No hits found for any query variant. Total available: {total_hits}")
                elif pixabay_response.status_code == 429:
                    print(f"Pixabay: Rate limit exceeded")
                    break
                else:
                    print(f"Pixabay API error: {pixabay_response.status_code} - {pixabay_response.text[:200]}")
                    continue
        except Exception as e:
            print(f"Error fetching Pixabay videos: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_videos and not pexels_api_key and not pixabay_api_key:
        raise HTTPException(
            status_code=503, 
            detail="Video search APIs not configured. Please set PEXELS_API_KEY and/or PIXABAY_API_KEY in your environment variables."
        )
    
    return VideoSearchResponse(videos=all_videos)
