'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Header from './components/Header';

export default function Home() {
  const [topic, setTopic] = useState('');
  const router = useRouter();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!topic.trim()) return;
    
    // Store topic in sessionStorage and navigate to clean URL
    if (typeof window !== 'undefined') {
      sessionStorage.setItem('pendingTopic', topic.trim());
    }
    
    router.push('/results');
  };

  return (
    <div className="min-h-screen bg-gray-600 flex flex-col items-center justify-center relative">
      <Header />

      {/* Main content */}
      <div className="flex flex-col items-center justify-center w-full max-w-2xl px-6 py-20">
        <h2 className="text-white text-4xl md:text-5xl mb-12 text-center tracking-tight" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 500, letterSpacing: '-0.75px', lineHeight: '46px', fontSize: '56px' }}>
          Create beautiful videos with zero effort
        </h2>

        <form onSubmit={handleSubmit} className="w-full space-y-6">
          <div className="flex flex-col gap-6">
            <textarea
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Your prompt here..."
              rows={4}
              className="w-full px-6 py-5 bg-gray-200 text-gray-900 text-lg rounded-2xl border-none outline-none placeholder-gray-500 resize-none transition-all duration-300 hover:bg-gray-300 focus:bg-gray-300 focus:ring-0 focus:outline-none"
              style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 300, boxShadow: 'none' }}
            />
            <button
              type="submit"
              disabled={!topic.trim()}
              className="w-full px-6 py-5 bg-gray-800 text-white text-lg rounded-full disabled:opacity-60 disabled:cursor-not-allowed transition-all duration-300 hover:bg-gray-700 hover:scale-[1.02] hover:shadow-lg active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-white/20"
              style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 400, boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.15)' }}
            >
              Generate Script
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
