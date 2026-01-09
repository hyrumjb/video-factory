'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const [topic, setTopic] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!topic.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/generate-script', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic: topic.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to generate script');
      }

      const data = await response.json();
      router.push(`/results?script=${encodeURIComponent(data.script)}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-800 flex flex-col items-center justify-center relative">
      {/* Top right: video factory */}
      <div className="absolute top-8 right-8">
        <h1 className="text-white text-2xl font-bold" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 700 }}>
          video factory
        </h1>
      </div>

      {/* Main content */}
      <div className="flex flex-col items-center justify-center w-full max-w-2xl px-4">
        <h2 className="text-white text-4xl md:text-5xl font-bold mb-12 text-center" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 700 }}>
          Create beautiful videos with zero effort
        </h2>

        <form onSubmit={handleSubmit} className="w-full">
          <div className="flex flex-col gap-4">
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Enter your topic..."
              className="w-full px-6 py-4 bg-gray-700 text-white text-lg rounded-lg border-2 border-gray-600 focus:border-white focus:outline-none placeholder-gray-400"
              style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 400 }}
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !topic.trim()}
              className="w-full px-6 py-4 bg-white text-gray-800 text-lg font-bold rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 700 }}
            >
              {isLoading ? 'Generating...' : 'Generate Script'}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-6 px-6 py-4 bg-red-900 text-white rounded-lg text-center">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
