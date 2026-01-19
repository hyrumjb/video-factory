'use client';

import { useRouter } from 'next/navigation';
import { Suspense, useEffect, useState, useRef } from 'react';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';
import { logOut } from '@/lib/firebase';

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

interface Clause {
  clause_id: number;
  text: string;
  idea_type: string;
  start_time: number;
  next_start_time: number;
}

interface MediaItem {
  clause_id: number;
  media_type: string;
  media_url: string | null;
  duration: number;
  error?: string;
}

type TabType = 'script' | 'audio' | 'clauses' | 'media';

const STEPS = [
  { id: 'script', label: 'Script' },
  { id: 'tts', label: 'Audio' },
  { id: 'clauses', label: 'Clauses' },
  { id: 'routing', label: 'Planning' },
  { id: 'media', label: 'Media' },
  { id: 'compile', label: 'Compile' },
];

function ResultsContent() {
  const router = useRouter();
  const { user } = useAuth();
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const topicRef = useRef<string | null>(null);
  const voiceIdRef = useRef<string>('nPczCjzI2devNBz1zQrb');
  const inputModeRef = useRef<'idea' | 'script'>('idea');

  const handleLogout = async () => {
    try {
      await logOut();
      router.push('/');
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  // Active tab for middle section
  const [activeTab, setActiveTab] = useState<TabType>('script');

  // Step states
  const [scriptStep, setScriptStep] = useState<StepState>({ status: 'pending' });
  const [ttsStep, setTtsStep] = useState<StepState>({ status: 'pending' });
  const [clausesStep, setClausesStep] = useState<StepState>({ status: 'pending' });
  const [routingStep, setRoutingStep] = useState<StepState>({ status: 'pending' });
  const [mediaStep, setMediaStep] = useState<StepState>({ status: 'pending' });
  const [compileStep, setCompileStep] = useState<StepState>({ status: 'pending' });

  // Data from each step
  const [script, setScript] = useState<string | null>(null);
  const [ttsProvider, setTtsProvider] = useState<string>('elevenlabs');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [clauses, setClauses] = useState<Clause[]>([]);
  const [mediaItems, setMediaItems] = useState<MediaItem[]>([]);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [totalDuration, setTotalDuration] = useState<number>(0);

  const stepStates: Record<string, StepState> = {
    script: scriptStep,
    tts: ttsStep,
    clauses: clausesStep,
    routing: routingStep,
    media: mediaStep,
    compile: compileStep,
  };

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const pendingTopic = sessionStorage.getItem('pendingTopic');

    if (pendingTopic) {
      const voiceId = sessionStorage.getItem('selectedVoice') || 'nPczCjzI2devNBz1zQrb';
      const inputMode = sessionStorage.getItem('inputMode') as 'idea' | 'script' || 'idea';
      topicRef.current = pendingTopic;
      voiceIdRef.current = voiceId;
      inputModeRef.current = inputMode;
      sessionStorage.removeItem('pendingTopic');
      sessionStorage.removeItem('selectedVoice');
      sessionStorage.removeItem('inputMode');
    }

    if (eventSourceRef.current && eventSourceRef.current.readyState !== EventSource.CLOSED) {
      return;
    }

    const currentTopic = topicRef.current;
    const voiceId = voiceIdRef.current;
    const inputMode = inputModeRef.current;

    if (!currentTopic) {
      router.push('/');
      return;
    }

    const existingJobId = sessionStorage.getItem('currentJobId');
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    let url = `${apiUrl}/api/create-video?topic=${encodeURIComponent(currentTopic.trim())}&voice_id=${encodeURIComponent(voiceId)}&input_mode=${encodeURIComponent(inputMode)}`;
    if (existingJobId) {
      url += `&job_id=${encodeURIComponent(existingJobId)}`;
    }

    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    const handleEvent = (event: MessageEvent, stepName: string) => {
      try {
        const data: ProgressEvent = JSON.parse(event.data);
        const stepState: StepState = {
          status: data.status,
          message: data.message,
          progress: data.progress,
        };
        return { data, stepState };
      } catch (err) {
        console.error(`Error parsing ${stepName} event:`, err);
        return null;
      }
    };

    eventSource.addEventListener('job', (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        sessionStorage.setItem('currentJobId', data.job_id);
      } catch (err) {
        console.error('Error parsing job event:', err);
      }
    });

    eventSource.addEventListener('script', (event: MessageEvent) => {
      const result = handleEvent(event, 'script');
      if (!result) return;
      const { data, stepState } = result;
      setScriptStep(stepState);
      if (data.status === 'done' && data.data) {
        setScript(data.data.script);
      }
    });

    eventSource.addEventListener('tts', (event: MessageEvent) => {
      const result = handleEvent(event, 'tts');
      if (!result) return;
      const { data, stepState } = result;
      setTtsStep(stepState);
      if (data.status === 'done' && data.data) {
        if (data.data.provider) setTtsProvider(data.data.provider);
        if (data.data.audio_url) setAudioUrl(data.data.audio_url);
        setActiveTab('audio');
      }
    });

    eventSource.addEventListener('clauses', (event: MessageEvent) => {
      const result = handleEvent(event, 'clauses');
      if (!result) return;
      const { data, stepState } = result;
      setClausesStep(stepState);
      if (data.status === 'done' && data.data) {
        setClauses(data.data.clauses || []);
        if (data.data.total_duration) setTotalDuration(data.data.total_duration);
        setActiveTab('clauses');
      }
    });

    eventSource.addEventListener('routing', (event: MessageEvent) => {
      const result = handleEvent(event, 'routing');
      if (!result) return;
      const { stepState } = result;
      setRoutingStep(stepState);
    });

    eventSource.addEventListener('media', (event: MessageEvent) => {
      const result = handleEvent(event, 'media');
      if (!result) return;
      const { data, stepState } = result;
      setMediaStep(stepState);
      if (data.status === 'done' && data.data) {
        setMediaItems(data.data.media || []);
        setActiveTab('media');
      }
    });

    eventSource.addEventListener('compile', (event: MessageEvent) => {
      const result = handleEvent(event, 'compile');
      if (!result) return;
      const { data, stepState } = result;
      setCompileStep(stepState);
      if (data.status === 'done' && data.data?.video_url) {
        setVideoUrl(data.data.video_url);
      }
    });

    eventSource.addEventListener('complete', (event: MessageEvent) => {
      const result = handleEvent(event, 'complete');
      if (!result) return;
      const { data } = result;
      if (data.data?.video_url) setVideoUrl(data.data.video_url);
      if (data.data?.total_duration) setTotalDuration(data.data.total_duration);
      setCompileStep({ status: 'done', message: 'Complete!' });
      sessionStorage.removeItem('currentJobId');
      eventSource.close();
    });

    eventSource.addEventListener('error', (event: MessageEvent) => {
      if (event.data) {
        const result = handleEvent(event, 'error');
        if (result) setError(result.data.message || 'An error occurred');
      }
      sessionStorage.removeItem('currentJobId');
      eventSource.close();
    });

    eventSource.onerror = () => {
      setError('Connection lost. Refresh to reconnect.');
      eventSource.close();
    };

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [router]);

  const validMedia = mediaItems.filter(m => m.media_url);

  return (
    <div className="h-screen bg-[#27272a] flex overflow-hidden">
      {/* LEFT SIDEBAR - Navigation */}
      <div className="w-56 bg-[#111113] border-r border-[#1f1f23] flex flex-col">
        {/* Logo */}
        <div className="p-4 border-b border-[#1f1f23]">
          <Link href="/" className="text-xl font-semibold text-white logo-text">
            Lightfall
          </Link>
        </div>

        {/* Nav Items */}
        <nav className="flex-1 p-3 space-y-1">
          <Link
            href="/"
            className="flex items-center gap-3 px-3 py-2.5 text-sm text-gray-400 hover:text-white hover:bg-[#1f1f23] rounded-lg transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
            Home
          </Link>
          <Link
            href="/"
            className="flex items-center gap-3 px-3 py-2.5 text-sm text-gray-400 hover:text-white hover:bg-[#1f1f23] rounded-lg transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Video
          </Link>
        </nav>

        {/* User info at bottom */}
        {user && (
          <div className="p-3 border-t border-[#1f1f23]">
            <div className="flex items-center gap-2 p-2 rounded-lg bg-[#1f1f23]">
              <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center flex-shrink-0">
                <span className="text-xs font-semibold text-[#27272a]">
                  {user.email?.charAt(0).toUpperCase() || 'U'}
                </span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-gray-300 truncate">{user.email}</p>
              </div>
              <button
                onClick={handleLogout}
                className="p-1.5 text-gray-500 hover:text-white hover:bg-[#2a2a2e] rounded transition-colors"
                title="Log out"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
              </button>
            </div>
          </div>
        )}
      </div>

      {/* MAIN CONTENT AREA */}
      <div className="flex-1 flex flex-col">
        {/* PROGRESS BAR - spans middle and right sections */}
        <div className="h-14 bg-[#111113] border-b border-[#1f1f23] flex items-center px-6">
          <div className="flex items-center gap-1 flex-1">
            {STEPS.map((step, index) => {
              const state = stepStates[step.id];
              const isActive = state.status === 'running';
              const isDone = state.status === 'done';
              const isError = state.status === 'error';

              return (
                <div key={step.id} className="flex items-center">
                  {/* Step indicator */}
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-md">
                    {isActive ? (
                      <span className="w-4 h-4 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
                    ) : isDone ? (
                      <span className="w-4 h-4 rounded-full bg-green-500 flex items-center justify-center">
                        <svg className="w-2.5 h-2.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      </span>
                    ) : isError ? (
                      <span className="w-4 h-4 rounded-full bg-red-500 flex items-center justify-center text-white text-[10px]">✕</span>
                    ) : (
                      <span className="w-4 h-4 rounded-full border-2 border-[#2a2a2e]" />
                    )}
                    <span className={`text-xs font-medium ${isActive ? 'text-blue-400' : isDone ? 'text-green-400' : isError ? 'text-red-400' : 'text-gray-500'}`}>
                      {step.label}
                    </span>
                  </div>
                  {/* Connector line */}
                  {index < STEPS.length - 1 && (
                    <div className={`w-8 h-0.5 ${isDone ? 'bg-green-500/50' : 'bg-[#2a2a2e]'}`} />
                  )}
                </div>
              );
            })}
          </div>

          {/* Video ready badge */}
          {videoUrl && (
            <span className="px-3 py-1 text-xs font-medium bg-green-500/20 text-green-400 rounded-full border border-green-500/30">
              Video Ready
            </span>
          )}
        </div>

        {/* CONTENT AREA - Middle and Right sections */}
        <div className="flex-1 flex overflow-hidden">
          {/* MIDDLE SECTION - Tabs */}
          <div className="flex-1 flex flex-col border-r border-[#1f1f23]">
            {/* Tabs */}
            <div className="flex border-b border-[#1f1f23]">
              {(['script', 'audio', 'clauses', 'media'] as TabType[]).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-5 py-3 text-sm font-medium transition-colors ${
                    activeTab === tab
                      ? 'text-white border-b-2 border-blue-500 bg-[#1f1f23]/50'
                      : 'text-gray-500 hover:text-gray-300 hover:bg-[#1f1f23]/30'
                  }`}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto p-5">
              {error && (
                <div className="mb-4 p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                  <p className="text-red-400 text-sm">{error}</p>
                </div>
              )}

              {/* Script Tab */}
              {activeTab === 'script' && (
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-3">Generated Script</h3>
                  {script ? (
                    <p className="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">{script}</p>
                  ) : scriptStep.status === 'running' ? (
                    <div className="flex items-center gap-2 text-gray-500">
                      <span className="w-4 h-4 rounded-full border-2 border-gray-500 border-t-transparent animate-spin" />
                      <span className="text-sm">Generating script...</span>
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">Script will appear here...</p>
                  )}
                </div>
              )}

              {/* Audio Tab */}
              {activeTab === 'audio' && (
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                    Audio
                    {audioUrl && (
                      <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full">
                        {ttsProvider}
                      </span>
                    )}
                  </h3>
                  {audioUrl ? (
                    <audio controls className="w-full max-w-md" src={audioUrl}>
                      Your browser does not support audio.
                    </audio>
                  ) : ttsStep.status === 'running' ? (
                    <div className="flex items-center gap-2 text-gray-500">
                      <span className="w-4 h-4 rounded-full border-2 border-gray-500 border-t-transparent animate-spin" />
                      <span className="text-sm">Generating audio...</span>
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">Audio will appear here...</p>
                  )}
                </div>
              )}

              {/* Clauses Tab */}
              {activeTab === 'clauses' && (
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-3">
                    Clauses {clauses.length > 0 && <span className="text-gray-600">({clauses.length})</span>}
                  </h3>
                  {clauses.length > 0 ? (
                    <div className="space-y-3">
                      {clauses.map((clause) => (
                        <div key={clause.clause_id} className="p-3 bg-[#1f1f23] rounded-lg border border-[#2a2a2e]">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-xs font-medium text-blue-400">#{clause.clause_id}</span>
                            <span className="text-xs px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded">{clause.idea_type}</span>
                            <span className="text-xs text-gray-500">
                              {clause.start_time.toFixed(1)}s - {clause.next_start_time.toFixed(1)}s
                            </span>
                          </div>
                          <p className="text-sm text-gray-300">{clause.text}</p>
                        </div>
                      ))}
                    </div>
                  ) : clausesStep.status === 'running' ? (
                    <div className="flex items-center gap-2 text-gray-500">
                      <span className="w-4 h-4 rounded-full border-2 border-gray-500 border-t-transparent animate-spin" />
                      <span className="text-sm">Segmenting clauses...</span>
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">Clauses will appear here...</p>
                  )}
                </div>
              )}

              {/* Media Tab */}
              {activeTab === 'media' && (
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-3">
                    Media {mediaItems.length > 0 && <span className="text-gray-600">({validMedia.length}/{mediaItems.length} loaded)</span>}
                  </h3>
                  {mediaItems.length > 0 ? (
                    <div className="grid grid-cols-2 gap-3">
                      {mediaItems.map((item) => {
                        const clause = clauses.find(c => c.clause_id === item.clause_id);
                        const isVideo = item.media_type.includes('video');
                        const isImage = item.media_type.includes('image');

                        return (
                          <div key={item.clause_id} className="bg-[#1f1f23] rounded-lg border border-[#2a2a2e] overflow-hidden">
                            <div className="p-2 border-b border-[#2a2a2e]">
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-medium text-gray-300">#{item.clause_id}</span>
                                <span className={`text-xs px-1.5 py-0.5 rounded ${
                                  item.media_type === 'stock_video' ? 'bg-blue-500/20 text-blue-400' :
                                  item.media_type === 'youtube_video' ? 'bg-red-500/20 text-red-400' :
                                  item.media_type === 'ai_video' ? 'bg-purple-500/20 text-purple-400' :
                                  item.media_type === 'ai_image' ? 'bg-pink-500/20 text-pink-400' :
                                  item.media_type === 'web_image' ? 'bg-green-500/20 text-green-400' :
                                  'bg-gray-500/20 text-gray-400'
                                }`}>
                                  {item.media_type.replace('_', ' ')}
                                </span>
                                <span className="text-xs text-gray-500 ml-auto">{item.duration.toFixed(1)}s</span>
                              </div>
                              {clause && (
                                <p className="text-xs text-gray-500 mt-1 line-clamp-1">{clause.text}</p>
                              )}
                            </div>
                            <div className="aspect-video bg-black">
                              {item.media_url ? (
                                isVideo ? (
                                  <video
                                    src={item.media_url}
                                    className="w-full h-full object-cover"
                                    muted
                                    playsInline
                                    onMouseEnter={(e) => (e.target as HTMLVideoElement).play()}
                                    onMouseLeave={(e) => {
                                      (e.target as HTMLVideoElement).pause();
                                      (e.target as HTMLVideoElement).currentTime = 0;
                                    }}
                                  />
                                ) : isImage ? (
                                  <img src={item.media_url} alt={`Clause ${item.clause_id}`} className="w-full h-full object-cover" />
                                ) : null
                              ) : item.error ? (
                                <div className="w-full h-full flex items-center justify-center bg-red-500/10">
                                  <p className="text-xs text-red-400 text-center px-2">{item.error}</p>
                                </div>
                              ) : (
                                <div className="w-full h-full flex items-center justify-center">
                                  <span className="w-4 h-4 rounded-full border-2 border-gray-600 border-t-transparent animate-spin" />
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : mediaStep.status === 'running' || routingStep.status === 'running' ? (
                    <div className="flex items-center gap-2 text-gray-500">
                      <span className="w-4 h-4 rounded-full border-2 border-gray-500 border-t-transparent animate-spin" />
                      <span className="text-sm">{routingStep.status === 'running' ? 'Planning media...' : 'Retrieving media...'}</span>
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">Media will appear here...</p>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* RIGHT SECTION - Final Video */}
          <div className="w-80 bg-[#111113] flex flex-col">
            <div className="p-4 border-b border-[#1f1f23] flex items-center justify-between">
              <h3 className="text-sm font-medium text-gray-400">Final Video</h3>
              {totalDuration > 0 && (
                <span className="text-xs text-gray-500">
                  {totalDuration.toFixed(1)}s
                </span>
              )}
            </div>
            <div className="flex-1 flex flex-col items-center justify-center p-4 gap-4">
              {videoUrl ? (
                <>
                  <video
                    controls
                    autoPlay
                    className="w-full rounded-lg"
                    style={{ aspectRatio: '9/16', maxHeight: 'calc(100vh - 14rem)' }}
                  >
                    <source src={videoUrl} type="video/mp4" />
                  </video>
                  <a
                    href={videoUrl}
                    download="lightfall-video.mp4"
                    className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-white text-[#27272a] rounded-full font-medium text-sm hover:bg-gray-200 transition-colors"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    Download
                  </a>
                  <p className="text-xs text-gray-500 text-center">
                    Video will be deleted in 30 min or when you create a new video
                  </p>
                </>
              ) : (
                <div
                  className="w-full bg-[#1f1f23] rounded-lg flex items-center justify-center border border-[#2a2a2e]"
                  style={{ aspectRatio: '9/16', maxHeight: 'calc(100vh - 10rem)' }}
                >
                  {compileStep.status === 'running' ? (
                    <div className="text-center">
                      <span className="block w-8 h-8 mx-auto mb-3 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
                      <p className="text-sm text-gray-400">Compiling...</p>
                      <p className="text-xs text-gray-500 mt-1">{compileStep.message}</p>
                    </div>
                  ) : mediaStep.status === 'done' ? (
                    <div className="text-center">
                      <span className="block w-6 h-6 mx-auto mb-2 rounded-full border-2 border-gray-500 border-t-transparent animate-spin" />
                      <p className="text-xs text-gray-500">Preparing...</p>
                    </div>
                  ) : (
                    <p className="text-xs text-gray-500">Video preview</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense
      fallback={
        <div className="h-screen bg-[#27272a] flex items-center justify-center">
          <div className="text-center">
            <div className="w-10 h-10 mx-auto mb-4 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
            <p className="text-gray-500 text-sm">Loading...</p>
          </div>
        </div>
      }
    >
      <ResultsContent />
    </Suspense>
  );
}
