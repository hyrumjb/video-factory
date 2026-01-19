'use client';

import { useState, useEffect, useRef, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';
import { logOut } from '@/lib/firebase';

function DashboardContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const showNewVideo = searchParams.get('new') === 'true';

  const { user, loading: authLoading } = useAuth();
  const [topic, setTopic] = useState('');
  const [placeholder, setPlaceholder] = useState('');
  const [selectedVoice, setSelectedVoice] = useState('nPczCjzI2devNBz1zQrb');
  const [isSelectOpen, setIsSelectOpen] = useState(false);
  const [inputMode, setInputMode] = useState<'idea' | 'script'>('idea');
  const [showInputBox, setShowInputBox] = useState(showNewVideo);
  const selectRef = useRef<HTMLDivElement>(null);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login?mode=signin');
    }
  }, [user, authLoading, router]);

  // Update showInputBox when URL changes
  useEffect(() => {
    setShowInputBox(showNewVideo);
  }, [showNewVideo]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (selectRef.current && !selectRef.current.contains(event.target as Node)) {
        setIsSelectOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = async () => {
    try {
      await logOut();
      router.push('/');
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const voices = [
    { id: 'nPczCjzI2devNBz1zQrb', name: 'Brian', description: 'Male, American, narration' },
    { id: '21m00Tcm4TlvDq8ikWAM', name: 'Rachel', description: 'Female, American, calm' },
    { id: 'pNInz6obpgDQGcFmaJgB', name: 'Adam', description: 'Male, American, deep' },
    { id: 'ThT5KcBeYPX3keUQqHPh', name: 'Dorothy', description: 'Female, British, soft' },
    { id: 'N2lVS1w4EtoT3dr4eOWO', name: 'Callum', description: 'Male, British, hoarse' },
    { id: 'XB0fDUnXU5powFXDhCwa', name: 'Charlotte', description: 'Female, Swedish, seductive' },
  ];

  const ideaPlaceholders = [
    'Your prompt here...',
    'Tell us a story...',
    'I want to see...'
  ];

  const scriptPlaceholders = [
    'Paste your script here...',
    'Write your narration...',
    'Enter your script...'
  ];

  const placeholders = inputMode === 'idea' ? ideaPlaceholders : scriptPlaceholders;

  const currentIndexRef = useRef(0);
  const currentTextRef = useRef('');
  const isDeletingRef = useRef(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    // Reset animation state
    currentIndexRef.current = 0;
    currentTextRef.current = '';
    isDeletingRef.current = false;
    setPlaceholder('');

    // Don't animate if there's text or input box is hidden
    if (topic.trim().length > 0 || !showInputBox) {
      return;
    }

    const type = () => {
      const currentPlaceholder = placeholders[currentIndexRef.current];

      if (isDeletingRef.current) {
        currentTextRef.current = currentPlaceholder.substring(0, currentTextRef.current.length - 1);
        setPlaceholder(currentTextRef.current);

        if (currentTextRef.current === '') {
          isDeletingRef.current = false;
          currentIndexRef.current = (currentIndexRef.current + 1) % placeholders.length;
          timeoutRef.current = setTimeout(type, 300);
        } else {
          timeoutRef.current = setTimeout(type, 30);
        }
      } else {
        const nextChar = currentPlaceholder[currentTextRef.current.length];
        currentTextRef.current = currentPlaceholder.substring(0, currentTextRef.current.length + 1);
        setPlaceholder(currentTextRef.current);

        if (currentTextRef.current === currentPlaceholder) {
          timeoutRef.current = setTimeout(() => {
            isDeletingRef.current = true;
            type();
          }, 6000);
        } else {
          const isWordBoundary = nextChar === ' ' || nextChar === '.';
          const delay = isWordBoundary ? 200 : 60;
          timeoutRef.current = setTimeout(type, delay);
        }
      }
    };

    // Start typing immediately
    type();

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [inputMode, topic, showInputBox, placeholders]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!topic.trim()) return;

    if (typeof window !== 'undefined') {
      sessionStorage.setItem('pendingTopic', topic.trim());
      sessionStorage.setItem('selectedVoice', selectedVoice);
      sessionStorage.setItem('inputMode', inputMode);
    }

    router.push('/results');
  };

  const selectedVoiceData = voices.find(v => v.id === selectedVoice);

  // Placeholder videos (will be replaced with real data from Firebase)
  const userVideos: any[] = [];

  if (authLoading) {
    return (
      <div className="h-screen bg-[#27272a] flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="h-screen bg-[#27272a] flex overflow-hidden">
      {/* LEFT SIDEBAR */}
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
            href="/dashboard"
            className="flex items-center gap-3 px-3 py-2.5 text-sm text-white bg-[#1f1f23] rounded-lg transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
            </svg>
            Dashboard
          </Link>
          <button
            onClick={() => {
              setShowInputBox(true);
              router.push('/dashboard?new=true');
            }}
            className="w-full flex items-center gap-3 px-3 py-2.5 text-sm text-gray-400 hover:text-white hover:bg-[#1f1f23] rounded-lg transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Video
          </button>
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

      {/* MAIN CONTENT */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Input box area - centered when shown */}
        {showInputBox ? (
          <div className="flex-1 flex items-center justify-center p-6">
            <form onSubmit={handleSubmit} className="w-full max-w-2xl">
              <div className="gradient-border-wrapper">
                <div className="bg-[#27272a] rounded-[calc(1.5rem-4px)] p-4">
                  <textarea
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    placeholder={topic ? '' : placeholder}
                    rows={3}
                    className="w-full bg-transparent text-base text-white resize-none outline-none placeholder:text-gray-500"
                    autoFocus
                  />

                  {/* Controls row */}
                  <div className="flex items-center justify-between pt-2 border-t border-[#1f1f23]">
                    {/* Mode Toggle */}
                    <div className="flex items-center bg-[#1f1f23] rounded-full p-0.5">
                      <button
                        type="button"
                        onClick={() => setInputMode('idea')}
                        className={`px-3 py-1 text-xs font-medium rounded-full transition-all duration-200 ${
                          inputMode === 'idea'
                            ? 'bg-[#2a2a2e] text-white'
                            : 'text-gray-500 hover:text-gray-300'
                        }`}
                      >
                        Provide Idea
                      </button>
                      <button
                        type="button"
                        onClick={() => setInputMode('script')}
                        className={`px-3 py-1 text-xs font-medium rounded-full transition-all duration-200 ${
                          inputMode === 'script'
                            ? 'bg-[#2a2a2e] text-white'
                            : 'text-gray-500 hover:text-gray-300'
                        }`}
                      >
                        Provide Script
                      </button>
                    </div>

                    <div className="flex items-center gap-2">
                      {/* Voice Select */}
                      <div className="relative" ref={selectRef}>
                        <button
                          type="button"
                          onClick={() => setIsSelectOpen(!isSelectOpen)}
                          className="bg-[#1f1f23] text-gray-300 rounded-full px-3 py-1.5 text-xs font-medium flex items-center gap-1.5 hover:bg-[#2a2a2e] transition-colors"
                        >
                          <span>{selectedVoiceData?.name || 'Voice'}</span>
                          <svg
                            className={`w-3 h-3 transition-transform duration-200 ${isSelectOpen ? 'rotate-180' : ''}`}
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                          </svg>
                        </button>

                        {/* Dropdown */}
                        <div
                          className={`absolute bottom-full right-0 mb-2 w-52 bg-[#1f1f23] rounded-xl overflow-hidden shadow-lg border border-[#2a2a2e] transition-all duration-200 origin-bottom ${
                            isSelectOpen
                              ? 'opacity-100 scale-100'
                              : 'opacity-0 scale-95 pointer-events-none'
                          }`}
                        >
                          <div className="py-1">
                            {voices.map((voice) => (
                              <button
                                key={voice.id}
                                type="button"
                                onClick={() => {
                                  setSelectedVoice(voice.id);
                                  setIsSelectOpen(false);
                                }}
                                className={`w-full px-3 py-2 text-left transition-colors hover:bg-[#2a2a2e] ${
                                  selectedVoice === voice.id ? 'bg-[#2a2a2e]' : ''
                                }`}
                              >
                                <div className="font-medium text-xs text-gray-200">{voice.name}</div>
                                <div className="text-xs text-gray-500">{voice.description}</div>
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>

                      {/* Cancel button */}
                      <button
                        type="button"
                        onClick={() => {
                          setShowInputBox(false);
                          setTopic('');
                          router.push('/dashboard');
                        }}
                        className="px-3 py-1.5 text-xs font-medium text-gray-400 hover:text-white rounded-full transition-colors"
                      >
                        Cancel
                      </button>

                      {/* Generate Button */}
                      <button
                        type="submit"
                        disabled={!topic.trim()}
                        className={`px-4 py-1.5 text-xs font-medium rounded-full transition-all duration-200 ${
                          topic.trim()
                            ? 'bg-white text-[#27272a] hover:bg-gray-200'
                            : 'bg-[#2a2a2e] text-gray-600 cursor-not-allowed'
                        }`}
                      >
                        Generate
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </form>
          </div>
        ) : (
          /* Videos grid - only shown when input box is not visible */
          <div className="flex-1 overflow-y-auto p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-white">Your Videos</h2>
            <button
              onClick={() => {
                setShowInputBox(true);
                router.push('/dashboard?new=true');
              }}
              className="px-4 py-2 text-sm font-medium bg-white text-[#27272a] rounded-full hover:bg-gray-200 transition-colors"
            >
              Create New
            </button>
          </div>

          {userVideos.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20">
              <div className="w-16 h-16 rounded-full bg-[#1f1f23] flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-white mb-2">No videos yet</h3>
              <p className="text-gray-500 text-sm mb-6">Create your first video to get started</p>
              <button
                onClick={() => {
                  setShowInputBox(true);
                  router.push('/dashboard?new=true');
                }}
                className="px-6 py-2.5 text-sm font-medium bg-white text-[#27272a] rounded-full hover:bg-gray-200 transition-colors"
              >
                Create Video
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {userVideos.map((video, index) => (
                <div key={index} className="bg-[#1f1f23] rounded-lg overflow-hidden border border-[#2a2a2e] hover:border-[#3f3f46] transition-colors cursor-pointer">
                  <div className="aspect-[9/16] bg-[#111113]">
                    {/* Video thumbnail will go here */}
                  </div>
                  <div className="p-3">
                    <p className="text-sm text-white truncate">{video.title}</p>
                    <p className="text-xs text-gray-500">{video.duration}s</p>
                  </div>
                </div>
              ))}
            </div>
          )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <Suspense
      fallback={
        <div className="h-screen bg-[#27272a] flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin" />
        </div>
      }
    >
      <DashboardContent />
    </Suspense>
  );
}
