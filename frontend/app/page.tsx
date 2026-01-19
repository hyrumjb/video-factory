'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';
import { logOut } from '@/lib/firebase';

export default function Home() {
  const [topic, setTopic] = useState('');
  const [placeholder, setPlaceholder] = useState('');
  const [selectedVoice, setSelectedVoice] = useState('nPczCjzI2devNBz1zQrb');
  const [isSelectOpen, setIsSelectOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [inputMode, setInputMode] = useState<'idea' | 'script'>('idea');
  const selectRef = useRef<HTMLDivElement>(null);
  const helpRef = useRef<HTMLDivElement>(null);
  const router = useRouter();
  const { user, loading } = useAuth();

  const handleLogout = async () => {
    try {
      await logOut();
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (selectRef.current && !selectRef.current.contains(event.target as Node)) {
        setIsSelectOpen(false);
      }
      if (helpRef.current && !helpRef.current.contains(event.target as Node)) {
        setIsHelpOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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

  // Placeholder typing animation - restarts when inputMode changes
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

    // Don't animate if user has typed something
    if (topic.trim().length > 0) {
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
  }, [inputMode, topic, placeholders]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!topic.trim()) return;

    if (typeof window !== 'undefined') {
      sessionStorage.setItem('pendingTopic', topic.trim());
      sessionStorage.setItem('selectedVoice', selectedVoice);
      sessionStorage.setItem('inputMode', inputMode);
    }

    // If not logged in, redirect to login first
    if (!user) {
      router.push('/login?mode=signup&redirect=generate');
      return;
    }

    router.push('/results');
  };

  const selectedVoiceData = voices.find(v => v.id === selectedVoice);

  return (
    <div className="h-screen bg-[#27272a] flex flex-col overflow-hidden">
      {/* Header */}
      <header className="px-6 py-4 flex items-center justify-between">
        <Link href="/" className="text-2xl font-semibold text-white logo-text">
          Lightfall
        </Link>
        <div className="flex items-center gap-3">
          {loading ? (
            <div className="w-8 h-8 rounded-full bg-[#3f3f46] animate-pulse" />
          ) : user ? (
            <div className="flex items-center gap-2 bg-[#1f1f23] rounded-full pl-1 pr-1 py-1">
              {/* User avatar - clickable to go to dashboard */}
              <button
                onClick={() => router.push('/dashboard')}
                className="w-7 h-7 rounded-full bg-white flex items-center justify-center hover:bg-gray-200 transition-colors"
              >
                <span className="text-xs font-semibold text-[#27272a]">
                  {user.email?.charAt(0).toUpperCase() || 'U'}
                </span>
              </button>
              {/* Email - hidden on mobile, clickable to go to dashboard */}
              <button
                onClick={() => router.push('/dashboard')}
                className="text-sm text-gray-300 hidden sm:block max-w-[150px] truncate hover:text-white transition-colors"
              >
                {user.email}
              </button>
              {/* Logout button */}
              <button
                onClick={handleLogout}
                className="px-3 py-1.5 text-xs font-medium text-gray-400 hover:text-white hover:bg-[#3f3f46] rounded-full transition-colors"
              >
                Log out
              </button>
            </div>
          ) : (
            <>
              <button
                onClick={() => router.push('/login?mode=signin')}
                className="px-4 py-2 text-sm font-medium text-gray-300 hover:text-white hover:bg-[#1f1f23] rounded-full transition-colors"
              >
                Log in
              </button>
              <button
                onClick={() => router.push('/login?mode=signup')}
                className="px-4 py-2 text-sm font-medium bg-white text-[#27272a] rounded-full hover:bg-gray-200 transition-colors"
              >
                Start creating
              </button>
            </>
          )}
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 flex flex-col items-center justify-start pt-40 px-6">
        <div className="w-full max-w-xl space-y-8">
          <h1 className="text-4xl md:text-5xl font-semibold text-center tracking-tight text-white">
            Create beautiful videos with zero effort
          </h1>

          <form onSubmit={handleSubmit}>
            {/* Input card with animated gradient border */}
            <div className="gradient-border-wrapper">
              <div className="bg-[#27272a] rounded-[calc(1.5rem-4px)] p-4">
                <textarea
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder={topic ? '' : placeholder}
                  rows={3}
                  className="w-full bg-transparent text-base text-white resize-none outline-none placeholder:text-gray-500"
                />

                {/* Controls row inside the card */}
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

                    {/* Generate Button */}
                    <button
                      type="submit"
                      disabled={!topic.trim()}
                      className={`px-4 py-1.5 text-xs font-medium rounded-full transition-all duration-200 ${
                        topic.trim()
                          ? 'bg-white text-[#18181b] hover:bg-gray-200'
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
      </main>

      {/* Help button */}
      <div className="fixed bottom-5 left-5" ref={helpRef}>
        <button
          onClick={() => setIsHelpOpen(!isHelpOpen)}
          className="w-7 h-7 rounded-full bg-[#3f3f46] hover:bg-[#52525b] flex items-center justify-center text-gray-300 hover:text-white transition-colors text-sm font-medium shadow-lg"
        >
          ?
        </button>

        {/* Help popup */}
        <div
          className={`absolute bottom-full left-0 mb-2 bg-[#1f1f23] rounded-2xl shadow-md border border-[#2a2a2e] px-4 py-3 transition-all duration-200 origin-bottom-left ${
            isHelpOpen
              ? 'opacity-100 scale-100'
              : 'opacity-0 scale-95 pointer-events-none'
          }`}
        >
          <p className="text-xs text-gray-500 mb-1">Need help?</p>
          <a
            href="mailto:bradshaw.hyrum@gmail.com"
            className="text-xs text-gray-400 underline inline-flex items-center gap-1 hover:text-gray-300 transition-colors"
          >
            bradshaw.hyrum@gmail.com
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
        </div>
      </div>
    </div>
  );
}
