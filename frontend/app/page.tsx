'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export default function Home() {
  const [topic, setTopic] = useState('');
  const [placeholder, setPlaceholder] = useState('');
  const [selectedVoice, setSelectedVoice] = useState('nPczCjzI2devNBz1zQrb');
  const [backgroundType, setBackgroundType] = useState<'videos' | 'images' | 'ai'>('videos');
  const [isSelectOpen, setIsSelectOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const selectRef = useRef<HTMLDivElement>(null);
  const helpRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

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

  const placeholders = [
    'Your prompt here...',
    'Tell us a story...',
    'I want to see...'
  ];

  const currentIndexRef = useRef(0);
  const currentTextRef = useRef('');
  const isDeletingRef = useRef(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (topic.trim().length > 0) {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
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

    type();

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [topic]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!topic.trim()) return;

    if (typeof window !== 'undefined') {
      sessionStorage.setItem('pendingTopic', topic.trim());
      sessionStorage.setItem('selectedVoice', selectedVoice);
      sessionStorage.setItem('backgroundType', backgroundType);
    }

    router.push('/results');
  };

  const selectedVoiceData = voices.find(v => v.id === selectedVoice);

  return (
    <div className="h-screen bg-white flex flex-col overflow-hidden">
      {/* Header */}
      <header className="px-6 py-4 flex items-center justify-between">
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
      <main className="flex-1 flex flex-col items-center justify-start pt-40 px-6">
        <div className="w-full max-w-xl space-y-8">
          <h1 className="text-4xl md:text-5xl font-semibold text-center tracking-tight text-foreground">
            Create beautiful videos with zero effort
          </h1>

          <form onSubmit={handleSubmit}>
            {/* Input card with animated gradient border */}
            <div className="gradient-border-wrapper">
              <div className="bg-white rounded-[calc(1.5rem-4px)] p-4">
                <textarea
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder={topic ? '' : placeholder}
                  rows={3}
                  className="w-full bg-transparent text-base resize-none outline-none placeholder:text-gray-400"
                />

                {/* Controls row inside the card */}
                <div className="flex items-center gap-3 justify-between pt-2 border-t border-gray-100">
                  {/* Background type toggle */}
                  <div className="flex items-center gap-1 bg-gray-100 rounded-full p-1">
                    {(['videos', 'images', 'ai'] as const).map((type) => (
                      <button
                        key={type}
                        type="button"
                        onClick={() => setBackgroundType(type)}
                        className={`px-3 py-1.5 text-xs font-medium rounded-full transition-all duration-200 ${
                          backgroundType === type
                            ? 'bg-white text-foreground shadow-sm'
                            : 'text-gray-500 hover:text-foreground'
                        }`}
                      >
                        {type === 'ai' ? 'AI' : type.charAt(0).toUpperCase() + type.slice(1)}
                      </button>
                    ))}
                  </div>

                  <div className="flex items-center gap-2">
                    {/* Voice Select */}
                    <div className="relative" ref={selectRef}>
                      <button
                        type="button"
                        onClick={() => setIsSelectOpen(!isSelectOpen)}
                        className="bg-gray-100 rounded-full px-3 py-1.5 text-xs font-medium flex items-center gap-1.5 hover:bg-gray-200 transition-colors"
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
                        className={`absolute bottom-full right-0 mb-2 w-52 bg-white rounded-xl overflow-hidden shadow-lg border border-gray-200 transition-all duration-200 origin-bottom ${
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
                              className={`w-full px-3 py-2 text-left transition-colors hover:bg-gray-50 ${
                                selectedVoice === voice.id ? 'bg-gray-100' : ''
                              }`}
                            >
                              <div className="font-medium text-xs">{voice.name}</div>
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
                          ? 'bg-foreground text-white hover:bg-gray-800'
                          : 'bg-gray-200 text-gray-400 cursor-not-allowed'
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
          className="w-6 h-6 rounded-full bg-gray-200 hover:bg-gray-300 flex items-center justify-center text-gray-500 hover:text-gray-600 transition-colors text-xs"
        >
          ?
        </button>

        {/* Help popup */}
        <div
          className={`absolute bottom-full left-0 mb-2 bg-white rounded-2xl shadow-md border border-gray-100 px-4 py-3 transition-all duration-200 origin-bottom-left ${
            isHelpOpen
              ? 'opacity-100 scale-100'
              : 'opacity-0 scale-95 pointer-events-none'
          }`}
        >
          <p className="text-xs text-gray-500 mb-1">Need help?</p>
          <a
            href="mailto:bradshaw.hyrum@gmail.com"
            className="text-xs text-gray-500 underline inline-flex items-center gap-1 hover:text-gray-700 transition-colors"
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
