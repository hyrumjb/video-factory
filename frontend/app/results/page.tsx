'use client';

import { useRouter } from 'next/navigation';
import { Suspense, useEffect, useState, useRef } from 'react';
import Link from 'next/link';

type StepStatus = 'pending' | 'running' | 'done' | 'error' | 'retrying';

interface StepState {
  status: StepStatus;
  message?: string;
  progress?: number;
}

interface ProgressEvent {
  step: string;
  status: StepStatus;
  message?: string;
  progress?: number;
  data?: any;
}

interface Section {
  name: string;
  text: string;
  word_start: number;
  word_end: number;
}

function StatusIcon({ status }: { status: StepStatus }) {
  switch (status) {
    case 'pending':
      return <span className="w-5 h-5 rounded-full border-2 border-gray-300" />;
    case 'running':
      return (
        <span className="w-5 h-5 rounded-full border-2 border-foreground border-t-transparent animate-spin" />
      );
    case 'done':
      return (
        <span className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center text-white text-xs">
          ✓
        </span>
      );
    case 'error':
      return (
        <span className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center text-white text-xs">
          ✕
        </span>
      );
    case 'retrying':
      return (
        <span className="w-5 h-5 rounded-full border-2 border-yellow-500 border-t-transparent animate-spin" />
      );
    default:
      return null;
  }
}

function ProgressStep({ label, state }: { label: string; state: StepState }) {
  return (
    <div className="flex items-center gap-3 py-2">
      <StatusIcon status={state.status} />
      <div className="flex-1">
        <p className="text-sm font-medium">{label}</p>
        {state.message && (
          <p className="text-xs text-gray-500">{state.message}</p>
        )}
      </div>
    </div>
  );
}

function ResultsContent() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [isScrolled, setIsScrolled] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const topicRef = useRef<string | null>(null);
  const voiceIdRef = useRef<string>('nPczCjzI2devNBz1zQrb');
  const bgTypeRef = useRef<'videos' | 'images' | 'ai'>('videos');
  const [backgroundType, setBackgroundType] = useState<'videos' | 'images' | 'ai'>('videos');

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const [scriptStep, setScriptStep] = useState<StepState>({ status: 'pending' });
  const [ttsStep, setTtsStep] = useState<StepState>({ status: 'pending' });
  const [videosStep, setVideosStep] = useState<StepState>({ status: 'pending' });
  const [compileStep, setCompileStep] = useState<StepState>({ status: 'pending' });

  const [script, setScript] = useState<string | null>(null);
  const [sections, setSections] = useState<Section[]>([]);
  const [scenes, setScenes] = useState<any[]>([]);
  const [videos, setVideos] = useState<any[]>([]);
  const [aiImages, setAiImages] = useState<any[]>([]);
  const [aiPrompt, setAiPrompt] = useState<string | null>(null);
  const [ttsProvider, setTtsProvider] = useState<string>('elevenlabs');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [alignment, setAlignment] = useState<any>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  const [isPaused, setIsPaused] = useState(false);
  const [pausedData, setPausedData] = useState<any>(null);
  const [isGeneratingTts, setIsGeneratingTts] = useState(false);
  const [isCompiling, setIsCompiling] = useState(false);

  // On-demand media generation states
  const [isGeneratingVideos, setIsGeneratingVideos] = useState(false);
  const [isGeneratingImages, setIsGeneratingImages] = useState(false);
  const [isGeneratingAI, setIsGeneratingAI] = useState(false);
  const [topic, setTopic] = useState<string | null>(null);

  useEffect(() => {
    window.scrollTo(0, 0);

    if (typeof window === 'undefined') return;

    const pendingTopic = sessionStorage.getItem('pendingTopic');

    if (pendingTopic) {
      const voiceId = sessionStorage.getItem('selectedVoice') || 'nPczCjzI2devNBz1zQrb';
      const bgType = (sessionStorage.getItem('backgroundType') as 'videos' | 'images' | 'ai') || 'videos';

      topicRef.current = pendingTopic;
      voiceIdRef.current = voiceId;
      bgTypeRef.current = bgType;
      setBackgroundType(bgType);
      setTopic(pendingTopic);

      sessionStorage.removeItem('pendingTopic');
      sessionStorage.removeItem('selectedVoice');
      sessionStorage.removeItem('backgroundType');
    }

    if (eventSourceRef.current && eventSourceRef.current.readyState !== EventSource.CLOSED) {
      return;
    }

    const topic = topicRef.current;
    const voiceId = voiceIdRef.current;
    const bgTypeLocal = bgTypeRef.current;

    if (!topic) {
      router.push('/');
      return;
    }

    const eventSource = new EventSource(
      `http://localhost:8000/api/create-video?topic=${encodeURIComponent(topic.trim())}&voice_id=${encodeURIComponent(voiceId)}&staged=true&background_type=${encodeURIComponent(bgTypeLocal)}`
    );
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data: ProgressEvent = JSON.parse(event.data);

        const stepState: StepState = {
          status: data.status,
          message: data.message,
          progress: data.progress,
        };

        switch (data.step) {
          case 'script':
            setScriptStep(stepState);
            if (data.status === 'done' && data.data) {
              setScript(data.data.script);
              setScenes(data.data.scenes || []);
              setSections(data.data.sections || []);
            }
            break;

          case 'videos':
            setVideosStep(stepState);
            if (data.status === 'done' && data.data?.videos) {
              setVideos(data.data.videos);
            }
            if (data.status === 'done' && data.data?.ai_images) {
              setAiImages(data.data.ai_images);
              setAiPrompt(data.data.ai_prompt || null);
            }
            break;

          case 'paused':
            setIsPaused(true);
            setPausedData(data.data);
            if (data.data?.ai_images) {
              setAiImages(data.data.ai_images);
              setAiPrompt(data.data.ai_prompt || null);
            }
            eventSource.close();
            break;

          case 'tts':
            setTtsStep(stepState);
            if (data.status === 'done' && data.data?.provider) {
              setTtsProvider(data.data.provider);
            }
            break;

          case 'compile':
            setCompileStep(stepState);
            break;

          case 'complete':
            if (data.data?.video_url) {
              setVideoUrl(data.data.video_url);
            }
            eventSource.close();
            break;

          case 'error':
            setError(data.message || 'An error occurred');
            eventSource.close();
            break;
        }
      } catch (err) {
        console.error('Error parsing SSE event:', err);
      }
    };

    eventSource.onerror = () => {
      setError('Connection lost. Please try again.');
      eventSource.close();
    };

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [router]);

  const handleGenerateTts = async () => {
    if (!script || isGeneratingTts) return;

    setIsGeneratingTts(true);
    setTtsStep({ status: 'running', message: 'Generating audio...' });

    try {
      const response = await fetch('http://localhost:8000/api/generate-tts-elevenlabs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: script,
          voice_id: voiceIdRef.current,
        }),
      });

      if (!response.ok) throw new Error('TTS generation failed');

      const data = await response.json();
      setAudioUrl(data.audio_url);
      setAlignment(data.alignment);
      setTtsProvider('elevenlabs');
      setTtsStep({ status: 'done', message: 'Audio generated (ElevenLabs)' });
    } catch (err: any) {
      setTtsStep({ status: 'error', message: err.message });
    } finally {
      setIsGeneratingTts(false);
    }
  };

  const handleCompileVideo = async () => {
    if (!audioUrl || isCompiling) return;
    if (backgroundType === 'ai' && !aiImages.some(img => img.url)) return;
    if (backgroundType !== 'ai' && !videos.length) return;

    setIsCompiling(true);
    const useImages = backgroundType === 'images' || backgroundType === 'ai';
    const typeLabel = backgroundType === 'ai' ? 'AI image' : backgroundType === 'images' ? 'images' : 'videos';
    setCompileStep({ status: 'running', message: `Compiling with ${typeLabel}...` });

    try {
      let mediaUrls: string[];
      if (backgroundType === 'ai') {
        // Map each scene to its corresponding AI image by scene_number
        mediaUrls = [];
        for (let i = 0; i < scenes.length; i++) {
          const sceneNumber = scenes[i].scene_number;
          // Find AI image for this scene (by scene_number or by index)
          const matchingImage = aiImages.find(img => img.scene_number === sceneNumber) ||
                                aiImages[i];
          if (matchingImage?.url) {
            mediaUrls.push(matchingImage.url);
          } else {
            // Use first available image as fallback
            const fallback = aiImages.find(img => img.url);
            if (fallback?.url) {
              mediaUrls.push(fallback.url);
            }
          }
        }
        if (mediaUrls.length === 0) {
          throw new Error('No AI images available');
        }
      } else if (backgroundType === 'images') {
        mediaUrls = videos
          .filter(v => v.images && v.images.length > 0)
          .map(v => v.images[0].url);
      } else {
        mediaUrls = videos.filter(v => v.video_url).map(v => v.video_url);
      }

      const sceneTimings = scenes
        .filter(s => s.word_start !== undefined)
        .map(s => ({
          scene_number: s.scene_number,
          word_start: s.word_start,
          word_end: s.word_end,
        }));

      const response = await fetch('http://localhost:8000/api/compile-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_urls: mediaUrls,
          audio_url: audioUrl,
          script: script,
          alignment: alignment,
          tts_provider: ttsProvider,
          scenes: sceneTimings,
          use_images: useImages,
        }),
      });

      if (!response.ok) throw new Error('Video compilation failed');

      const data = await response.json();
      setVideoUrl(data.video_url);
      setCompileStep({ status: 'done', message: 'Video compiled!' });
    } catch (err: any) {
      setCompileStep({ status: 'error', message: err.message });
    } finally {
      setIsCompiling(false);
    }
  };

  const handleGenerateVideos = async () => {
    if (!script || !topic || isGeneratingVideos) return;

    setIsGeneratingVideos(true);

    try {
      const scenesPayload = scenes.map((s: any) => ({
        scene_number: s.scene_number,
        section_name: s.section_name,
        description: s.description,
        search_query: s.search_query,
      }));

      const response = await fetch('http://localhost:8000/api/search-stock-videos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          script,
          scenes: scenesPayload,
        }),
      });

      if (!response.ok) throw new Error('Video search failed');

      const data = await response.json();

      // Merge video results into existing videos state
      setVideos((prev) => {
        const updated = [...prev];
        for (const result of data.results) {
          const idx = updated.findIndex((v) => v.scene_number === result.scene_number);
          if (idx >= 0) {
            updated[idx] = {
              ...updated[idx],
              video_url: result.video_url,
              video_source: result.video_source,
              video_search_query: result.video_search_query,
            };
          } else {
            updated.push(result);
          }
        }
        return updated;
      });
    } catch (err: any) {
      console.error('Video generation error:', err);
    } finally {
      setIsGeneratingVideos(false);
    }
  };

  const handleGenerateImages = async () => {
    if (!script || !topic || isGeneratingImages) return;

    setIsGeneratingImages(true);

    try {
      const scenesPayload = scenes.map((s: any) => ({
        scene_number: s.scene_number,
        section_name: s.section_name,
        description: s.description,
        search_query: s.search_query,
      }));

      const response = await fetch('http://localhost:8000/api/search-stock-images', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          script,
          scenes: scenesPayload,
        }),
      });

      if (!response.ok) throw new Error('Image search failed');

      const data = await response.json();

      // Merge image results into existing videos state
      setVideos((prev) => {
        const updated = [...prev];
        for (const result of data.results) {
          const idx = updated.findIndex((v) => v.scene_number === result.scene_number);
          if (idx >= 0) {
            updated[idx] = {
              ...updated[idx],
              images: result.images,
              image_search_query: result.image_search_query,
            };
          } else {
            updated.push(result);
          }
        }
        return updated;
      });
    } catch (err: any) {
      console.error('Image generation error:', err);
    } finally {
      setIsGeneratingImages(false);
    }
  };

  const handleGenerateAI = async () => {
    if (!script || !topic || isGeneratingAI) return;

    setIsGeneratingAI(true);

    try {
      // Get first section text
      const firstSectionText = sections.length > 0
        ? sections[0].text
        : script.split(' ').slice(0, 100).join(' ');

      // Build sections for multi-scene generation
      const sectionsPayload = sections.length > 0
        ? sections.map((s: Section) => ({ name: s.name, text: s.text }))
        : [{ name: 'HOOK', text: firstSectionText }];

      const response = await fetch('http://localhost:8000/api/generate-ai-images', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          script,
          first_section_text: firstSectionText,
          sections: sectionsPayload,
        }),
      });

      if (!response.ok) throw new Error('AI image generation failed');

      const data = await response.json();
      setAiImages(data.images || []);
      setAiPrompt(data.prompts?.[0]?.prompt || null);
    } catch (err: any) {
      console.error('AI image generation error:', err);
    } finally {
      setIsGeneratingAI(false);
    }
  };

  const isComplete = videoUrl !== null;
  const isProcessing = !isPaused && !isComplete && !error;

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {/* Header */}
      <header
        className={`fixed top-0 left-0 right-0 z-50 px-6 py-4 bg-white transition-all duration-300 flex items-center justify-between ${
          isScrolled ? 'border-b border-gray-200' : ''
        }`}
      >
        <Link href="/" className="text-2xl font-semibold text-foreground">
          Lola
        </Link>
        <div className="flex items-center gap-3">
          <button className="px-4 py-2 text-sm font-medium text-foreground hover:bg-gray-100 rounded-full transition-colors">
            Log in
          </button>
          <button className="px-4 py-2 text-sm font-medium bg-foreground text-white rounded-full hover:bg-gray-800 transition-colors">
            Start creating
          </button>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 px-6 pt-24 pb-10">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-semibold text-foreground">
              {isComplete ? 'Video Complete' : isPaused ? 'Ready for Review' : isProcessing ? 'Generating...' : 'Results'}
            </h1>
            {isPaused && (
              <div className="flex gap-3">
                <button
                  onClick={handleGenerateTts}
                  disabled={isGeneratingTts || !!audioUrl}
                  className={`px-4 py-2 text-sm font-medium rounded-full transition-colors ${
                    audioUrl
                      ? 'bg-gray-100 text-gray-400'
                      : 'bg-gray-100 text-foreground hover:bg-gray-200'
                  }`}
                >
                  {audioUrl ? '✓ Audio Ready' : isGeneratingTts ? 'Generating...' : 'Generate TTS'}
                </button>
                <button
                  onClick={handleCompileVideo}
                  disabled={!audioUrl || isCompiling || !!videoUrl}
                  className={`px-4 py-2 text-sm font-medium rounded-full transition-colors ${
                    !audioUrl || videoUrl
                      ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                      : 'bg-foreground text-white hover:bg-gray-800'
                  }`}
                >
                  {videoUrl ? '✓ Video Ready' : isCompiling ? 'Compiling...' : 'Compile Video'}
                </button>
              </div>
            )}
          </div>

          {error ? (
            <div className="bg-white border border-gray-200 rounded-2xl p-6">
              <p className="text-red-500 mb-4">{error}</p>
              <button
                onClick={() => router.push('/')}
                className="px-4 py-2 text-sm font-medium bg-foreground text-white rounded-full hover:bg-gray-800 transition-colors"
              >
                Go Back
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
              {/* Column 1: Progress & Scripts */}
              <div className="space-y-4">
                <div className="bg-white border border-gray-200 rounded-2xl p-4">
                  <h3 className="text-sm font-semibold mb-3">Progress</h3>
                  <div className="space-y-1">
                    <ProgressStep label="Script" state={scriptStep} />
                    <ProgressStep label="Media Search" state={videosStep} />
                    <ProgressStep label="TTS Audio" state={ttsStep} />
                    <ProgressStep label="Compile" state={compileStep} />
                  </div>
                </div>

                <div className="bg-white border border-gray-200 rounded-2xl p-4">
                  <h3 className="text-sm font-semibold mb-3">Raw Script</h3>
                  {sections.length > 0 ? (
                    <div className="space-y-3 text-xs">
                      {sections.map((section, index) => (
                        <div key={index} className="border-l-2 border-gray-300 pl-3 py-1">
                          <p className="font-semibold text-gray-700 mb-1">[{section.name}]</p>
                          <p className="text-gray-500 leading-relaxed">{section.text}</p>
                        </div>
                      ))}
                    </div>
                  ) : script ? (
                    <p className="text-xs text-gray-500">No section data available</p>
                  ) : (
                    <div className="space-y-2">
                      <div className="h-4 bg-gray-100 rounded animate-pulse" />
                      <div className="h-4 bg-gray-100 rounded animate-pulse w-3/4" />
                    </div>
                  )}
                </div>

                <div className="bg-white border border-gray-200 rounded-2xl p-4">
                  <h3 className="text-sm font-semibold mb-3">Edited Script</h3>
                  {script ? (
                    <p className="text-xs leading-relaxed whitespace-pre-wrap text-gray-500">{script}</p>
                  ) : (
                    <div className="space-y-2">
                      <div className="h-4 bg-gray-100 rounded animate-pulse" />
                      <div className="h-4 bg-gray-100 rounded animate-pulse w-5/6" />
                    </div>
                  )}
                </div>
              </div>

              {/* Column 2: Stock Videos */}
              <div className="space-y-4">
                <div className="bg-white border border-gray-200 rounded-2xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold">Stock Videos</h3>
                    {backgroundType !== 'videos' && isPaused && !videos.some(v => v.video_url) && (
                      <button
                        onClick={handleGenerateVideos}
                        disabled={isGeneratingVideos || !script}
                        className={`px-2 py-1 text-xs font-medium rounded-full transition-colors ${
                          isGeneratingVideos
                            ? 'bg-gray-100 text-gray-400'
                            : 'bg-gray-100 text-foreground hover:bg-gray-200'
                        }`}
                      >
                        {isGeneratingVideos ? 'Searching...' : 'Generate'}
                      </button>
                    )}
                  </div>
                  {videos.some(v => v.video_url) ? (
                    <div className="space-y-3">
                      {videos.map((video, index) => (
                        <div key={index} className="p-3 bg-gray-50 rounded-xl">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="bg-foreground text-white rounded-full w-5 h-5 flex items-center justify-center text-xs font-medium">
                              {video.scene_number}
                            </span>
                            <span className={`text-xs ${video.video_url ? 'text-green-600' : 'text-red-500'}`}>
                              {video.video_url ? '✓' : '✕'}
                            </span>
                            {video.video_source && (
                              <span className="text-xs text-gray-500">{video.video_source}</span>
                            )}
                          </div>
                          <p className="text-xs text-gray-500 mb-1">
                            Query: "{video.video_search_query || video.search_query}"
                          </p>
                          {video.video_url && (
                            <a
                              href={video.video_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-xs text-blue-600 hover:underline break-all"
                            >
                              Preview video →
                            </a>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : isGeneratingVideos ? (
                    <div className="space-y-2">
                      <div className="h-16 bg-gray-100 rounded-xl animate-pulse" />
                      <div className="h-16 bg-gray-100 rounded-xl animate-pulse" />
                    </div>
                  ) : backgroundType === 'videos' && videosStep.status === 'running' ? (
                    <div className="space-y-2">
                      <div className="h-16 bg-gray-100 rounded-xl animate-pulse" />
                      <div className="h-16 bg-gray-100 rounded-xl animate-pulse" />
                    </div>
                  ) : backgroundType !== 'videos' ? (
                    <p className="text-xs text-gray-500">Click Generate to search for videos</p>
                  ) : (
                    <p className="text-xs text-gray-500">Waiting...</p>
                  )}
                </div>
              </div>

              {/* Column 3: Google Images */}
              <div className="space-y-4">
                <div className="bg-white border border-gray-200 rounded-2xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold">Google Images</h3>
                    {backgroundType !== 'images' && isPaused && !videos.some(v => v.images?.length > 0) && (
                      <button
                        onClick={handleGenerateImages}
                        disabled={isGeneratingImages || !script}
                        className={`px-2 py-1 text-xs font-medium rounded-full transition-colors ${
                          isGeneratingImages
                            ? 'bg-gray-100 text-gray-400'
                            : 'bg-gray-100 text-foreground hover:bg-gray-200'
                        }`}
                      >
                        {isGeneratingImages ? 'Searching...' : 'Generate'}
                      </button>
                    )}
                  </div>
                  {videos.some(v => v.images?.length > 0) ? (
                    <div className="space-y-3">
                      {videos.map((video, index) => (
                        <div key={index} className="p-3 bg-gray-50 rounded-xl">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="bg-foreground text-white rounded-full w-5 h-5 flex items-center justify-center text-xs font-medium">
                              {video.scene_number}
                            </span>
                            <span className={`text-xs ${video.images?.length > 0 ? 'text-green-600' : 'text-red-500'}`}>
                              {video.images?.length || 0} images
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 mb-2">
                            Query: "{video.image_search_query || video.search_query}"
                          </p>
                          {video.images && video.images.length > 0 && (
                            <a
                              href={video.images[0].url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="block"
                            >
                              <img
                                src={video.images[0].thumbnail_url || video.images[0].url}
                                alt={video.images[0].title || 'Scene image'}
                                className="w-full h-20 object-cover rounded-lg border border-gray-200 hover:border-gray-400 transition-colors"
                              />
                              <p className="text-xs text-gray-500 mt-1 truncate">
                                {video.images[0].title}
                              </p>
                            </a>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : isGeneratingImages ? (
                    <div className="space-y-2">
                      <div className="h-24 bg-gray-100 rounded-xl animate-pulse" />
                      <div className="h-24 bg-gray-100 rounded-xl animate-pulse" />
                    </div>
                  ) : backgroundType === 'images' && videosStep.status === 'running' ? (
                    <div className="space-y-2">
                      <div className="h-24 bg-gray-100 rounded-xl animate-pulse" />
                      <div className="h-24 bg-gray-100 rounded-xl animate-pulse" />
                    </div>
                  ) : backgroundType !== 'images' ? (
                    <p className="text-xs text-gray-500">Click Generate to search for images</p>
                  ) : (
                    <p className="text-xs text-gray-500">Waiting...</p>
                  )}
                </div>
              </div>

              {/* Column 4: AI Generated Images */}
              <div className="space-y-4">
                <div className="bg-white border border-gray-200 rounded-2xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold">AI Images</h3>
                    {backgroundType !== 'ai' && isPaused && aiImages.length === 0 && (
                      <button
                        onClick={handleGenerateAI}
                        disabled={isGeneratingAI || !script}
                        className={`px-2 py-1 text-xs font-medium rounded-full transition-colors ${
                          isGeneratingAI
                            ? 'bg-gray-100 text-gray-400'
                            : 'bg-gray-100 text-foreground hover:bg-gray-200'
                        }`}
                      >
                        {isGeneratingAI ? 'Generating...' : 'Generate'}
                      </button>
                    )}
                  </div>
                  {aiImages.length > 0 ? (
                    <div className="space-y-3">
                      {/* Display images by scene */}
                      {aiImages.map((img: any, index: number) => (
                        <div key={index} className="p-3 bg-gray-50 rounded-xl">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="bg-foreground text-white rounded-full w-5 h-5 flex items-center justify-center text-xs font-medium">
                              {img.scene_number || index + 1}
                            </span>
                            <span className="text-xs font-medium">{img.section_name || `Scene ${img.scene_number}`}</span>
                            <span className={`text-xs ${img.url ? 'text-green-600' : 'text-red-500'}`}>
                              {img.url ? '✓' : '✕'}
                            </span>
                          </div>
                          {img.url ? (
                            <a
                              href={img.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="block"
                            >
                              <img
                                src={img.url}
                                alt={`AI - ${img.section_name || `Scene ${img.scene_number}`}`}
                                className="w-full h-24 object-cover rounded-lg border border-gray-200 hover:border-gray-400 transition-colors"
                              />
                              <p className="text-[10px] text-gray-400 mt-1">
                                {img.model_name} {img.width > 0 && `• ${img.width}x${img.height}`}
                              </p>
                            </a>
                          ) : (
                            <p className="text-xs text-red-500">{img.error || 'Failed'}</p>
                          )}
                        </div>
                      ))}
                      <p className="text-[10px] text-gray-400">
                        {aiImages.filter((i: any) => i.url).length}/{aiImages.length} generated
                      </p>
                    </div>
                  ) : isGeneratingAI ? (
                    <div className="space-y-2">
                      <p className="text-xs text-gray-500 mb-2">Generating images for all scenes...</p>
                      <div className="h-28 bg-gray-100 rounded-xl animate-pulse" />
                      <div className="h-28 bg-gray-100 rounded-xl animate-pulse" />
                      <div className="h-28 bg-gray-100 rounded-xl animate-pulse" />
                    </div>
                  ) : backgroundType === 'ai' && videosStep.status === 'running' ? (
                    <div className="space-y-2">
                      <p className="text-xs text-gray-500 mb-2">Generating images for all scenes...</p>
                      <div className="h-28 bg-gray-100 rounded-xl animate-pulse" />
                      <div className="h-28 bg-gray-100 rounded-xl animate-pulse" />
                      <div className="h-28 bg-gray-100 rounded-xl animate-pulse" />
                    </div>
                  ) : backgroundType !== 'ai' ? (
                    <p className="text-xs text-gray-500">Click Generate to create AI images</p>
                  ) : (
                    <p className="text-xs text-gray-500">Waiting...</p>
                  )}
                </div>
              </div>

              {/* Column 5: Final Video & Audio */}
              <div className="space-y-4">
                {audioUrl && (
                  <div className="bg-white border border-gray-200 rounded-2xl p-4">
                    <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      Audio
                      <span className="text-xs bg-gray-100 px-2 py-0.5 rounded-full text-gray-600">
                        {ttsProvider}
                      </span>
                    </h3>
                    <audio controls className="w-full" src={audioUrl}>
                      Your browser does not support audio.
                    </audio>
                  </div>
                )}

                <div className="bg-white border border-gray-200 rounded-2xl p-4">
                  <h3 className="text-sm font-semibold mb-3">Final Video</h3>
                  {videoUrl ? (
                    <video
                      controls
                      autoPlay
                      className="w-full rounded-xl"
                      style={{ aspectRatio: '9/16', maxHeight: '400px' }}
                    >
                      <source src={videoUrl} type="video/mp4" />
                    </video>
                  ) : (
                    <div
                      className="w-full bg-gray-50 rounded-xl flex items-center justify-center"
                      style={{ aspectRatio: '9/16', maxHeight: '300px' }}
                    >
                      {compileStep.status === 'running' ? (
                        <div className="text-center">
                          <span className="block w-8 h-8 mx-auto mb-2 rounded-full border-2 border-foreground border-t-transparent animate-spin" />
                          <p className="text-xs text-gray-500">Compiling...</p>
                        </div>
                      ) : isPaused && !audioUrl ? (
                        <p className="text-xs text-gray-500 text-center px-4">
                          Click "Generate TTS" then "Compile Video"
                        </p>
                      ) : isPaused && audioUrl ? (
                        <p className="text-xs text-gray-500 text-center px-4">
                          Click "Compile Video" to create final video
                        </p>
                      ) : (
                        <p className="text-xs text-gray-500">Waiting...</p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {!error && (
            <div className="flex justify-center pt-6">
              <button
                onClick={() => router.push('/')}
                className="px-4 py-2 text-sm font-medium bg-gray-100 text-foreground rounded-full hover:bg-gray-200 transition-colors"
              >
                Create Another
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-white flex items-center justify-center">
          <div className="text-center">
            <div className="w-12 h-12 mx-auto mb-4 rounded-full border-2 border-foreground border-t-transparent animate-spin" />
            <p className="text-gray-500">Loading...</p>
          </div>
        </div>
      }
    >
      <ResultsContent />
    </Suspense>
  );
}
