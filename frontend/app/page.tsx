'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Header from './components/Header';

export default function Home() {
  const [topic, setTopic] = useState('');
  const [placeholder, setPlaceholder] = useState('');
  const [selectedVoice, setSelectedVoice] = useState('en-US-Neural2-H');
  const [isTransitioning, setIsTransitioning] = useState(false);
  const router = useRouter();

  const voices = [
    { name: 'en-US-Neural2-H', label: 'Neural2-H (Female)', starred: false },
    { name: 'en-US-Neural2-A', label: 'Neural2-A (Male)', starred: false },
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
    // Don't animate if user is typing
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
        // Delete one character
        currentTextRef.current = currentPlaceholder.substring(0, currentTextRef.current.length - 1);
        setPlaceholder(currentTextRef.current);
        
        if (currentTextRef.current === '') {
          isDeletingRef.current = false;
          currentIndexRef.current = (currentIndexRef.current + 1) % placeholders.length;
          timeoutRef.current = setTimeout(type, 300); // Pause before typing next
        } else {
          timeoutRef.current = setTimeout(type, 30); // Faster delete speed
        }
      } else {
        // Type one character
        const nextChar = currentPlaceholder[currentTextRef.current.length];
        currentTextRef.current = currentPlaceholder.substring(0, currentTextRef.current.length + 1);
        setPlaceholder(currentTextRef.current);
        
        if (currentTextRef.current === currentPlaceholder) {
          // Finished typing, wait 6 seconds then start deleting
          timeoutRef.current = setTimeout(() => {
            isDeletingRef.current = true;
            type();
          }, 6000);
        } else {
          // Check if next character is a space (word boundary)
          const isWordBoundary = nextChar === ' ' || nextChar === '.';
          // Fast typing for letters, pause after words
          const delay = isWordBoundary ? 200 : 60; // 200ms pause after word, 60ms per letter
          timeoutRef.current = setTimeout(type, delay);
        }
      }
    };

    // Start typing
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
    if (!topic.trim() || isTransitioning) return;
    
    // Start transition animation
    setIsTransitioning(true);
    
    // Store topic and selected voice in sessionStorage
    if (typeof window !== 'undefined') {
      sessionStorage.setItem('pendingTopic', topic.trim());
      sessionStorage.setItem('selectedVoice', selectedVoice);
    }
    
    // Wait for slide-up animation to complete before navigating
    setTimeout(() => {
      router.push('/results');
    }, 400); // Match animation duration
  };

  return (
    <div 
      className={`min-h-screen bg-white flex flex-col items-center justify-center relative transition-transform duration-500 ease-in-out ${
        isTransitioning ? '-translate-y-full opacity-0' : 'translate-y-0 opacity-100'
      }`}
    >
      <Header />

      {/* Main content */}
      <div className="flex flex-col items-center justify-center w-full max-w-2xl px-6 py-20">
        <h2 className="text-gray-900 text-4xl md:text-5xl mb-12 text-center tracking-tight" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 500, letterSpacing: '-0.75px', lineHeight: '46px', fontSize: '56px' }}>
          Create beautiful videos with zero effort
        </h2>

        <form onSubmit={handleSubmit} className="w-full">
          <div className="relative">
            <textarea
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder={topic ? '' : placeholder}
              rows={4}
              className="w-full px-6 py-5 pr-24 bg-gray-200 text-gray-900 text-lg rounded-2xl border-none outline-none placeholder-gray-500 resize-none transition-all duration-300 hover:bg-gray-300 focus:bg-gray-300 focus:ring-0 focus:outline-none"
              style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 300, boxShadow: 'none' }}
            />
            <div className="absolute bottom-4 right-4 flex items-center gap-3">
              <select
                value={selectedVoice}
                onChange={(e) => setSelectedVoice(e.target.value)}
                className="px-3 py-2.5 bg-white text-gray-900 text-sm font-medium rounded-lg border border-gray-300 outline-none focus:ring-2 focus:ring-gray-400 focus:border-gray-400 transition-colors"
                style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}
                onClick={(e) => e.stopPropagation()}
              >
                {voices.map((voice) => (
                  <option key={voice.name} value={voice.name}>
                    {voice.starred ? '⭐ ' : ''}{voice.label}
                  </option>
                ))}
              </select>
              <button
                type="submit"
                disabled={!topic.trim()}
                className="px-5 py-2.5 bg-gray-900 text-white text-sm font-medium rounded-lg disabled:opacity-60 disabled:cursor-not-allowed transition-colors duration-200 hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
                style={{ fontFamily: 'system-ui, -apple-system, sans-serif', boxShadow: '0px 2px 8px rgba(0, 0, 0, 0.15)' }}
              >
                Generate
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
