'use client';

import { useRouter } from 'next/navigation';
import { Suspense, useEffect, useState, useRef } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';

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

function StatusIcon({ status }: { status: StepStatus }) {
  switch (status) {
    case 'pending':
      return <span className="w-5 h-5 rounded-full border-2 border-muted-foreground/30" />;
    case 'running':
      return (
        <span className="w-5 h-5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
      );
    case 'done':
      return (
        <span className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center text-white text-xs">
          ✓
        </span>
      );
    case 'error':
      return (
        <span className="w-5 h-5 rounded-full bg-destructive flex items-center justify-center text-white text-xs">
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
          <p className="text-xs text-muted-foreground">{state.message}</p>
        )}
      </div>
    </div>
  );
}

function ResultsContent() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const hasStartedRef = useRef(false);

  // Step states
  const [scriptStep, setScriptStep] = useState<StepState>({ status: 'pending' });
  const [ttsStep, setTtsStep] = useState<StepState>({ status: 'pending' });
  const [videosStep, setVideosStep] = useState<StepState>({ status: 'pending' });
  const [compileStep, setCompileStep] = useState<StepState>({ status: 'pending' });

  // Data from steps
  const [script, setScript] = useState<string | null>(null);
  const [scenes, setScenes] = useState<any[]>([]);
  const [ttsProvider, setTtsProvider] = useState<string>('elevenlabs');
  const [videos, setVideos] = useState<any[]>([]);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  useEffect(() => {
    window.scrollTo(0, 0);

    if (hasStartedRef.current) return;
    if (typeof window === 'undefined') return;

    const topic = sessionStorage.getItem('pendingTopic');

    if (!topic) {
      router.push('/');
      return;
    }

    hasStartedRef.current = true;
    sessionStorage.removeItem('pendingTopic');

    // Create SSE connection
    const eventSource = new EventSource(
      `http://localhost:8000/api/create-video?topic=${encodeURIComponent(topic.trim())}`
    );
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data: ProgressEvent = JSON.parse(event.data);
        console.log('SSE Event:', data);

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
            }
            break;

          case 'tts':
            setTtsStep(stepState);
            if (data.status === 'done' && data.data?.provider) {
              setTtsProvider(data.data.provider);
            }
            break;

          case 'videos':
            setVideosStep(stepState);
            if (data.status === 'done' && data.data?.videos) {
              setVideos(data.data.videos);
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

    eventSource.onerror = (err) => {
      console.error('SSE Error:', err);
      setError('Connection lost. Please try again.');
      eventSource.close();
    };

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [router]);

  const isComplete = videoUrl !== null;
  const isProcessing = !isComplete && !error;

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 px-6 py-4 bg-background/80 backdrop-blur-sm border-b">
        <Link href="/" className="text-lg font-medium hover:opacity-80 transition-opacity">
          Video Factory
        </Link>
      </header>

      {/* Main content */}
      <main className="flex-1 px-6 pt-20 pb-10">
        <div className="max-w-5xl mx-auto space-y-6">
          <h1 className="text-2xl font-medium">
            {isComplete ? 'Video Complete' : isProcessing ? 'Creating Video...' : 'Results'}
          </h1>

          {error ? (
            <Card>
              <CardContent className="pt-6">
                <p className="text-destructive mb-4">{error}</p>
                <Button onClick={() => router.push('/')}>Go Back</Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left column - Progress & Info */}
              <div className="lg:col-span-1 space-y-6">
                {/* Progress Steps */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">Progress</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-1">
                    <ProgressStep label="Generate Script" state={scriptStep} />
                    <ProgressStep label="Generate Audio" state={ttsStep} />
                    <ProgressStep label="Find Stock Videos" state={videosStep} />
                    <ProgressStep label="Compile Video" state={compileStep} />
                  </CardContent>
                </Card>

                {/* Script */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">Script</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {script ? (
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">{script}</p>
                    ) : scriptStep.status === 'running' ? (
                      <div className="space-y-3">
                        <Skeleton className="h-4 w-3/4" />
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-5/6" />
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">Waiting...</p>
                    )}
                  </CardContent>
                </Card>

                {/* Scenes */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">Scenes</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {scenes.length > 0 ? (
                      <div className="space-y-3">
                        {scenes.map((scene, index) => (
                          <div key={index} className="p-3 bg-muted rounded-lg">
                            <div className="flex items-start gap-2 mb-2">
                              <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs flex-shrink-0">
                                {scene.scene_number || index + 1}
                              </span>
                              <p className="text-sm">{scene.description}</p>
                            </div>
                            {scene.search_query && (
                              <p className="text-xs text-muted-foreground ml-7">
                                Search: "{scene.search_query}"
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : scriptStep.status === 'running' || scriptStep.status === 'pending' ? (
                      <div className="space-y-3">
                        <Skeleton className="h-16 w-full" />
                        <Skeleton className="h-16 w-full" />
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">Waiting...</p>
                    )}
                  </CardContent>
                </Card>

                {/* Stock Videos Found */}
                {videos.length > 0 && (
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base">Stock Videos</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {videos.map((video, index) => (
                          <div
                            key={index}
                            className="flex items-center gap-2 text-sm"
                          >
                            <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs flex-shrink-0">
                              {video.scene_number}
                            </span>
                            <span
                              className={
                                video.video_url
                                  ? 'text-green-600'
                                  : 'text-muted-foreground'
                              }
                            >
                              {video.video_url ? '✓' : '✕'} {video.search_query}
                            </span>
                            {video.video_source && (
                              <span className="text-xs text-muted-foreground">
                                ({video.video_source})
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>

              {/* Center/Right column - Final Video */}
              <div className="lg:col-span-2">
                <Card className="h-full">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                      Final Video
                      {ttsProvider === 'elevenlabs' && ttsStep.status === 'done' && (
                        <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded">
                          ElevenLabs
                        </span>
                      )}
                      {ttsProvider === 'google' && ttsStep.status === 'done' && (
                        <span className="text-xs bg-muted text-muted-foreground px-2 py-0.5 rounded">
                          Google TTS
                        </span>
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Video */}
                    <div className="flex justify-center">
                      {videoUrl ? (
                        <video
                          controls
                          autoPlay
                          className="w-full max-w-sm rounded-lg"
                          style={{ aspectRatio: '9/16' }}
                        >
                          <source src={videoUrl} type="video/mp4" />
                        </video>
                      ) : (
                        <div className="text-center space-y-4 w-full max-w-sm">
                          <div className="w-full aspect-[9/16] bg-muted rounded-lg flex items-center justify-center">
                            {compileStep.status === 'running' ? (
                              <div className="text-center">
                                <span className="block w-8 h-8 mx-auto mb-2 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                                <p className="text-sm text-muted-foreground">Compiling...</p>
                              </div>
                            ) : (
                              <p className="text-sm text-muted-foreground">Preview</p>
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {scriptStep.status === 'running'
                              ? 'Generating script...'
                              : ttsStep.status === 'running'
                              ? 'Generating audio...'
                              : videosStep.status === 'running'
                              ? 'Finding videos...'
                              : compileStep.status === 'running'
                              ? 'Compiling video...'
                              : 'Waiting...'}
                          </p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

          {/* Back button */}
          {!error && (
            <div className="flex justify-center pt-4">
              <Button variant="outline" onClick={() => router.push('/')}>
                Create Another
              </Button>
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
        <div className="min-h-screen bg-background flex items-center justify-center">
          <p className="text-muted-foreground">Loading...</p>
        </div>
      }
    >
      <ResultsContent />
    </Suspense>
  );
}
