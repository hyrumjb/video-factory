'use client';

import { useRouter } from 'next/navigation';
import { Suspense, useEffect, useState, useRef } from 'react';
import Header from '../components/Header';

function BoxSkeleton() {
  return (
    <div className="w-full space-y-4 animate-pulse">
      <div className="h-6 bg-gray-300 rounded-lg w-3/4"></div>
      <div className="h-4 bg-gray-300 rounded-lg w-full"></div>
      <div className="h-4 bg-gray-300 rounded-lg w-5/6"></div>
      <div className="h-4 bg-gray-300 rounded-lg w-4/5"></div>
    </div>
  );
}

function ResultsContent() {
  const router = useRouter();
  const [script, setScript] = useState<string | null>(null);
  const [scenes, setScenes] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [videos, setVideos] = useState<any[]>([]);
  const [isSearchingVideos, setIsSearchingVideos] = useState(false);
  const [finalVideoUrl, setFinalVideoUrl] = useState<string | null>(null);
  const [isCompilingVideo, setIsCompilingVideo] = useState(false);
  const [isEntering, setIsEntering] = useState(true);
  const hasFetchedRef = useRef(false);

  useEffect(() => {
    // Trigger slide-in animation on mount
    setTimeout(() => {
      setIsEntering(false);
    }, 50); // Small delay to ensure initial state is rendered
    
    // Only run once on mount
    if (hasFetchedRef.current) return;
    
    // Get topic from sessionStorage
    if (typeof window === 'undefined') return;
    
    const topic = sessionStorage.getItem('pendingTopic');
    console.log('Topic from sessionStorage:', topic);
    
    if (!topic) {
      console.log('No topic found, redirecting to home');
      router.push('/');
      return;
    }

    // Mark as fetched immediately to prevent duplicates
    hasFetchedRef.current = true;
    
    // Clear sessionStorage immediately
    sessionStorage.removeItem('pendingTopic');

    const searchForVideos = async (scenesToSearch: any[]) => {
      console.log('🔍 searchForVideos called with:', scenesToSearch);
      if (!scenesToSearch) {
        console.log('❌ No scenes provided (null/undefined)');
        return;
      }
      if (!Array.isArray(scenesToSearch)) {
        console.log('❌ Scenes is not an array:', typeof scenesToSearch, scenesToSearch);
        return;
      }
      if (scenesToSearch.length === 0) {
        console.log('❌ Scenes array is empty');
        return;
      }
      
      console.log('✅ Starting video search for scenes:', scenesToSearch);
      setIsSearchingVideos(true);
      const allVideos: any[] = [];
      
      try {
        // Search for videos for each scene
        for (const scene of scenesToSearch) {
          console.log(`📋 Processing scene:`, JSON.stringify(scene, null, 2));
          const searchQuery = scene.search_query || scene.searchQuery || '';
          
          if (!searchQuery || searchQuery.trim().length === 0) {
            console.log(`⚠️ Skipping scene ${scene.scene_number || 'unknown'} - no search_query found. Scene object:`, scene);
            continue;
          }
          
          console.log(`🔍 Searching videos for scene ${scene.scene_number || 'unknown'} with search query:`, searchQuery);
          console.log(`📡 Making API call to http://localhost:8000/api/search-videos`);
          
          try {
            const requestBody = { search_query: searchQuery.trim() };
            console.log(`📤 Request body:`, JSON.stringify(requestBody));
            
            const response = await fetch('http://localhost:8000/api/search-videos', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(requestBody),
            });

            console.log(`📥 Video search response status for scene ${scene.scene_number || 'unknown'}:`, response.status);

            if (response.ok) {
              const data = await response.json();
              console.log(`Found ${data.videos?.length || 0} videos for scene ${scene.scene_number}`);
              // Add scene number to each video
              const videosWithScene = (data.videos || []).map((video: any) => ({
                ...video,
                scene_number: scene.scene_number,
                scene_search_query: searchQuery
              }));
              allVideos.push(...videosWithScene);
            } else {
              const errorData = await response.json().catch(() => ({}));
              console.error(`Video search failed for scene ${scene.scene_number}:`, errorData);
            }
          } catch (err) {
            console.error(`Error searching videos for scene ${scene.scene_number}:`, err);
          }
        }
        
        console.log(`Total videos found: ${allVideos.length}`);
        setVideos(allVideos);
      } catch (err) {
        console.error('Video search error:', err);
        // Don't show error to user, just log it
      } finally {
        setIsSearchingVideos(false);
      }
    };

    const generateTTS = async (scriptText: string, scenesForVideos: any[]) => {
      setIsGeneratingAudio(true);
      try {
        // Get selected voice from sessionStorage, default to en-US-Neural2-F
        const selectedVoice = typeof window !== 'undefined' 
          ? sessionStorage.getItem('selectedVoice') || 'en-US-Neural2-H'
          : 'en-US-Neural2-H';
        
        const response = await fetch('http://localhost:8000/api/generate-tts', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            text: scriptText,
            voice_name: selectedVoice
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Failed to generate audio');
        }

        const data = await response.json();
        setAudioUrl(data.audio_url);
        console.log('TTS generated successfully');
        
        // Note: Video search is already triggered after scenes are set, 
        // so we don't need to trigger it again here (avoids duplicate calls)
        console.log('TTS complete. Video search should already be in progress.');
      } catch (err) {
        console.error('TTS generation error:', err);
        // Don't show error to user, just log it
      } finally {
        setIsGeneratingAudio(false);
      }
    };

    const generateScript = async () => {
      setIsLoading(true);
      setError(null);
      console.log('Starting script generation for topic:', topic);

      try {
        const response = await fetch('http://localhost:8000/api/generate-script', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ topic: topic.trim() }),
        });

        console.log('Response status:', response.status);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          console.error('API error:', errorData);
          throw new Error(errorData.detail || 'Failed to generate script');
        }

        const data = await response.json();
        console.log('Script received - FULL DATA:', JSON.stringify(data, null, 2));
        
        if (!data.script) {
          throw new Error('Script is empty');
        }
        
        setScript(data.script);
        const scenesData = data.scenes || [];
        console.log('Scenes data type:', typeof scenesData, 'Length:', scenesData?.length);
        console.log('Scenes data content:', JSON.stringify(scenesData, null, 2));
        setScenes(scenesData);
        console.log('Script and scenes set in state. Scenes:', scenesData);
        
        // Search for videos immediately after scenes are received
        if (scenesData && Array.isArray(scenesData) && scenesData.length > 0) {
          console.log('✅ Scenes are valid, triggering video search immediately with scenes:', scenesData);
          // Use setTimeout to ensure state is set before calling
          setTimeout(() => {
            console.log('⏰ Timeout triggered, calling searchForVideos now');
            searchForVideos(scenesData);
          }, 100);
        } else {
          console.warn('❌ No valid scenes found. ScenesData:', scenesData, 'Type:', typeof scenesData, 'IsArray:', Array.isArray(scenesData));
        }
        
        // Automatically generate TTS after script is loaded (runs in parallel with video search)
        if (data.script) {
          generateTTS(data.script, scenesData);
        }
      } catch (err) {
        console.error('Script generation error:', err);
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setIsLoading(false);
        console.log('Loading set to false');
      }
    };

    generateScript();
  }, [router]);

  // Compile video when both audio and videos are ready
  useEffect(() => {
    const compileVideo = async () => {
      if (!audioUrl || !videos || videos.length === 0 || finalVideoUrl || isCompilingVideo) {
        return; // Not ready yet or already compiled/compiling
      }

      setIsCompilingVideo(true);
      try {
        // Use ALL videos available, sorted by scene number
        const sortedVideos = [...videos].sort((a, b) => {
          // First sort by scene number
          const sceneDiff = (a.scene_number || 0) - (b.scene_number || 0);
          if (sceneDiff !== 0) return sceneDiff;
          // Then by source (pexels first, then pixabay)
          if (a.source === 'pexels' && b.source !== 'pexels') return -1;
          if (a.source !== 'pexels' && b.source === 'pexels') return 1;
          return 0;
        });
        
        const videoUrls = sortedVideos.map(v => v.url);
        
        console.log(`Compiling video with ALL ${videoUrls.length} videos (${videos.length} total)`);

        // Get selected voice from sessionStorage
        const selectedVoice = typeof window !== 'undefined' 
          ? sessionStorage.getItem('selectedVoice') || 'en-US-Neural2-H'
          : 'en-US-Neural2-H';
        
        const response = await fetch('http://localhost:8000/api/compile-video', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            video_urls: videoUrls,
            audio_url: audioUrl,
            script: script || '',  // Include script for captions
            voice_name: selectedVoice
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Failed to compile video');
        }

        const data = await response.json();
        if (data.video_url) {
          setFinalVideoUrl(data.video_url);
        } else {
          throw new Error('No video URL in response');
        }
        console.log('Video compiled successfully');
      } catch (err) {
        console.error('Video compilation error:', err);
        // Don't show error to user, just log it
      } finally {
        setIsCompilingVideo(false);
      }
    };

    compileVideo();
  }, [audioUrl, videos, finalVideoUrl, isCompilingVideo]);

  const renderContentBox = (
    title: string,
    content: React.ReactNode,
    isLoading: boolean,
    isWaiting: boolean,
    fixedHeight: string = 'auto'
  ) => {
    return (
      <div 
        className="bg-gray-200 rounded-2xl p-6 border-none transition-all duration-300 hover:bg-gray-300 flex flex-col"
        style={{ height: fixedHeight }}
      >
        <h3 className="text-gray-900 text-lg mb-4 flex-shrink-0" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 500 }}>
          {title}
        </h3>
        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          {isWaiting ? (
            <div className="text-gray-500 text-center py-4 flex items-center justify-center flex-1" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 300 }}>
              Waiting for other content...
            </div>
          ) : isLoading ? (
            <BoxSkeleton />
          ) : (
            <div className="text-gray-900 flex-1 min-h-0 overflow-y-auto" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 300 }}>
              {content}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div 
      className={`min-h-screen bg-white flex flex-col items-center justify-center relative transition-transform duration-500 ease-in-out ${
        isEntering ? 'translate-y-full opacity-0' : 'translate-y-0 opacity-100'
      }`}
    >
      <Header />

      {/* Main content */}
      <div className="flex flex-col items-center justify-center w-full max-w-7xl px-6 py-20">
        <h2 className="text-gray-900 text-4xl md:text-5xl mb-12 text-center tracking-tight" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 500, letterSpacing: '-0.75px', lineHeight: '46px', fontSize: '56px' }}>
          Results
        </h2>

        {error ? (
          <div className="w-full mb-10">
            <div className="px-6 py-4 bg-red-100 text-red-900 rounded-2xl text-center border-none">
              {error}
            </div>
            <div className="flex justify-center mt-6">
              <button
                onClick={() => router.push('/')}
                className="px-5 py-2.5 bg-gray-900 text-white text-sm font-medium rounded-lg transition-colors duration-200 hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
                style={{ fontFamily: 'system-ui, -apple-system, sans-serif', boxShadow: '0px 2px 8px rgba(0, 0, 0, 0.15)' }}
              >
                Go Back
              </button>
            </div>
          </div>
        ) : (
          <div className="w-full grid grid-cols-1 md:grid-cols-3 md:grid-rows-4 gap-6">
            {/* Script Box - 1x1 (top left) - Height calculated: (500px - 24px gap) / 2 = 238px */}
            <div className="md:col-span-1 md:row-span-1">
              {renderContentBox(
                'Script',
                script ? (
                  <p className="text-base leading-relaxed whitespace-pre-wrap">{script}</p>
                ) : null,
                isLoading,
                !isLoading && !script,
                '238px'
              )}
            </div>

            {/* Audio Box - 1x1 (below script, still left column) - Height calculated: (500px - 24px gap) / 2 = 238px */}
            <div className="md:col-span-1 md:row-span-1">
              {renderContentBox(
                'Audio',
                audioUrl ? (
                  <div className="flex items-center justify-center py-2 h-full">
                    <audio 
                      controls 
                      className="w-full"
                      style={{ outline: 'none' }}
                    >
                      <source src={audioUrl} type="audio/mp3" />
                      Your browser does not support the audio element.
                    </audio>
                  </div>
                ) : null,
                isGeneratingAudio,
                !isLoading && !isGeneratingAudio && !audioUrl,
                '238px'
              )}
            </div>

            {/* Video Scenes Box - 2x2 (top right, spanning 2 columns and 2 rows) - Increased to 500px */}
            <div className="md:col-span-2 md:row-span-2 md:col-start-2 md:row-start-1">
              {renderContentBox(
                'Video Scenes',
                scenes && scenes.length > 0 ? (
                  <div className="space-y-3 overflow-y-auto h-full">
                    {scenes.map((scene, index) => (
                      <div key={index} className="bg-gray-100 rounded-xl p-4 border border-gray-300">
                        <div className="flex items-start gap-3 mb-2">
                          <span className="bg-gray-800 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-semibold flex-shrink-0" style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}>
                            {scene.scene_number || index + 1}
                          </span>
                          <p className="text-sm leading-relaxed flex-1">{scene.description}</p>
                        </div>
                        <div className="flex flex-wrap gap-1 mt-2">
                          {(scene.search_keywords || '').split(',').slice(0, 3).map((keyword: string, i: number) => (
                            <span key={i} className="bg-gray-800 text-white text-xs px-2 py-0.5 rounded-full" style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}>
                              {keyword.trim()}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : null,
                false,
                !isLoading && (!scenes || scenes.length === 0),
                '500px'
              )}
            </div>

            {/* Stock Videos Box - 2x2 (bottom left, spanning 2 columns and 2 rows, below script/audio) - Same height as Final Video */}
            <div className="md:col-span-2 md:row-span-2 md:col-start-1 md:row-start-3">
              {renderContentBox(
                'Stock Videos',
                videos && videos.length > 0 ? (
                  <div className="grid grid-cols-3 gap-2 overflow-y-auto h-full auto-rows-max">
                    {videos.map((video, index) => (
                      <div key={index} className="bg-gray-100 rounded-lg border border-gray-300 flex flex-col items-center gap-2 p-2">
                        <div className="relative bg-gray-300 flex-shrink-0 rounded overflow-hidden w-full" style={{ aspectRatio: '9/16' }}>
                          {video.thumbnail_url && (
                            <img 
                              src={video.thumbnail_url} 
                              alt={`Video from ${video.source}`}
                              className="w-full h-full object-cover"
                            />
                          )}
                          <div className="absolute top-0.5 right-0.5">
                            <span className="bg-gray-900 text-white text-xs px-1 py-0.5 rounded" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontSize: '9px' }}>
                              {video.source}
                            </span>
                          </div>
                          {video.scene_number && (
                            <div className="absolute top-0.5 left-0.5">
                              <span className="bg-blue-600 text-white text-xs px-1 py-0.5 rounded" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontSize: '9px' }}>
                                {video.scene_number}
                              </span>
                            </div>
                          )}
                        </div>
                        <div className="flex flex-col items-center w-full">
                          <a 
                            href={video.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:text-blue-800 text-xs underline text-center"
                            style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}
                          >
                            Download
                          </a>
                          {video.scene_number && (
                            <p className="text-xs text-gray-500 mt-0.5 text-center" style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}>
                              Scene {video.scene_number}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : null,
                isSearchingVideos,
                !isLoading && !isSearchingVideos && (!videos || videos.length === 0),
                '500px'
              )}
            </div>

            {/* Final Video Box - 1x2 (bottom right, spanning 1 column and 2 rows) - Vertical video */}
            <div className="md:col-span-1 md:row-span-2 md:col-start-3 md:row-start-3">
              {renderContentBox(
                'Final Video',
                finalVideoUrl ? (
                  <div className="flex flex-col items-center justify-center h-full">
                    <video 
                      controls 
                      className="rounded-xl max-w-full"
                      style={{ outline: 'none', maxHeight: '100%', aspectRatio: '9/16' }}
                    >
                      <source src={finalVideoUrl} type="video/mp4" />
                      Your browser does not support the video element.
                    </video>
                  </div>
                ) : isCompilingVideo ? (
                  <div className="flex flex-col items-center justify-center h-full">
                    <p className="text-sm text-gray-600 mb-4">Compiling video...</p>
                    <div className="bg-gray-300 rounded-xl flex items-center justify-center" style={{ width: '100%', maxWidth: '300px', aspectRatio: '9/16' }}>
                      <span className="text-gray-500 text-sm">Processing</span>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full">
                    <p className="text-sm text-gray-600 mb-4">Waiting for audio and videos...</p>
                    <div className="bg-gray-300 rounded-xl flex items-center justify-center" style={{ width: '100%', maxWidth: '300px', aspectRatio: '9/16' }}>
                      <span className="text-gray-500 text-sm">Preview placeholder</span>
                    </div>
                  </div>
                ),
                isCompilingVideo,
                !audioUrl || !videos || videos.length === 0,
                '500px'
              )}
            </div>
          </div>
        )}


        {/* Go Back Button */}
        {!error && (
          <div className="flex justify-center mt-10">
            <button
              onClick={() => router.push('/')}
              className="px-5 py-2.5 bg-gray-900 text-white text-sm font-medium rounded-lg disabled:opacity-60 disabled:cursor-not-allowed transition-colors duration-200 hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
              style={{ fontFamily: 'system-ui, -apple-system, sans-serif', boxShadow: '0px 2px 8px rgba(0, 0, 0, 0.15)' }}
            >
              Create Another
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-gray-900 text-xl" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 300 }}>
          Loading...
        </div>
      </div>
    }>
      <ResultsContent />
    </Suspense>
  );
}
