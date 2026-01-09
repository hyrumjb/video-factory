'use client';

import { useRouter } from 'next/navigation';
import { Suspense, useEffect, useState, useRef } from 'react';
import Header from '../components/Header';

function ScriptSkeleton() {
  return (
    <div className="w-full space-y-4 animate-pulse">
      <div className="h-6 bg-gray-400 rounded-lg w-3/4"></div>
      <div className="h-4 bg-gray-400 rounded-lg w-full"></div>
      <div className="h-4 bg-gray-400 rounded-lg w-5/6"></div>
      <div className="h-4 bg-gray-400 rounded-lg w-4/5"></div>
      <div className="h-6 bg-gray-400 rounded-lg w-2/3 mt-6"></div>
      <div className="h-4 bg-gray-400 rounded-lg w-full"></div>
      <div className="h-4 bg-gray-400 rounded-lg w-5/6"></div>
    </div>
  );
}

function ResultsContent() {
  const router = useRouter();
  const [script, setScript] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const hasFetchedRef = useRef(false);

  useEffect(() => {
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

    const generateTTS = async (scriptText: string) => {
      setIsGeneratingAudio(true);
      try {
        const response = await fetch('http://localhost:8000/api/generate-tts', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: scriptText }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Failed to generate audio');
        }

        const data = await response.json();
        setAudioUrl(data.audio_url);
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
        console.log('Script received:', data);
        
        if (!data.script) {
          throw new Error('Script is empty');
        }
        
        setScript(data.script);
        console.log('Script set in state');
        
        // Automatically generate TTS after script is loaded
        if (data.script) {
          generateTTS(data.script);
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


  return (
    <div className="min-h-screen bg-gray-600 flex flex-col items-center justify-center relative">
      <Header />

      {/* Main content */}
      <div className="flex flex-col items-center justify-center w-full max-w-3xl px-6 py-20">
        <h2 className="text-white text-4xl md:text-5xl mb-12 text-center tracking-tight" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 300 }}>
          Results:
        </h2>

        {isLoading ? (
          <div className="w-full bg-gray-200 rounded-2xl p-8 md:p-10 mb-10 border-none">
            <ScriptSkeleton />
          </div>
        ) : error ? (
          <div className="w-full mb-10">
            <div className="px-6 py-4 bg-red-800/90 text-white rounded-2xl text-center border-none">
              {error}
            </div>
            <div className="flex justify-center mt-6">
              <button
                onClick={() => router.push('/')}
                className="px-8 py-5 bg-gray-800 text-white text-lg rounded-full transition-all duration-300 hover:bg-gray-700 hover:scale-105 hover:shadow-lg active:scale-95 focus:outline-none focus:ring-2 focus:ring-white/20"
                style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 400, boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.15)' }}
              >
                Go Back
              </button>
            </div>
          </div>
        ) : script && script.trim() ? (
          <>
            <div className="w-full bg-gray-200 rounded-2xl p-8 md:p-10 mb-10 border-none transition-all duration-300 hover:bg-gray-300">
              <p className="text-gray-900 text-lg leading-relaxed whitespace-pre-wrap" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 300 }}>
                {script}
              </p>
            </div>

            {/* Audio Section */}
            {isGeneratingAudio && (
              <div className="w-full bg-gray-700 rounded-2xl p-6 mb-6">
                <p className="text-white text-center mb-4" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 300 }}>
                  Generating audio...
                </p>
                <div className="w-full h-2 bg-gray-600 rounded-full overflow-hidden">
                  <div className="h-full bg-gray-400 rounded-full animate-pulse" style={{ width: '100%' }}></div>
                </div>
              </div>
            )}

            {audioUrl && (
              <div className="w-full bg-gray-800 rounded-2xl p-6 mb-10">
                <p className="text-white text-center mb-4 text-lg" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 400 }}>
                  Audio Preview:
                </p>
                <audio 
                  controls 
                  className="w-full"
                  style={{ outline: 'none' }}
                >
                  <source src={audioUrl} type="audio/mp3" />
                  Your browser does not support the audio element.
                </audio>
              </div>
            )}

            <div className="flex gap-4">
              <button
                onClick={() => router.push('/')}
                className="px-8 py-5 bg-gray-800 text-white text-lg rounded-full transition-all duration-300 hover:bg-gray-700 hover:scale-105 hover:shadow-lg active:scale-95 focus:outline-none focus:ring-2 focus:ring-white/20"
                style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 400, boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.15)' }}
              >
                Create Another
              </button>
            </div>
          </>
        ) : !isLoading && !error ? (
          <div className="w-full mb-10">
            <div className="px-6 py-4 bg-yellow-800/90 text-white rounded-2xl text-center border-none">
              No script received. Please try again.
            </div>
            <div className="flex justify-center mt-6">
              <button
                onClick={() => router.push('/')}
                className="px-8 py-5 bg-gray-800 text-white text-lg rounded-full transition-all duration-300 hover:bg-gray-700 hover:scale-105 hover:shadow-lg active:scale-95 focus:outline-none focus:ring-2 focus:ring-white/20"
                style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 400, boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.15)' }}
              >
                Go Back
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-gray-600 flex items-center justify-center">
        <div className="text-white text-xl" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 300 }}>
          Loading...
        </div>
      </div>
    }>
      <ResultsContent />
    </Suspense>
  );
}
