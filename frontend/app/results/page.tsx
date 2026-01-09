'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { Suspense } from 'react';

function ResultsContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const script = searchParams.get('script');

  if (!script) {
    return (
      <div className="min-h-screen bg-gray-800 flex flex-col items-center justify-center">
        <div className="text-white text-xl mb-8">No script found</div>
        <button
          onClick={() => router.push('/')}
          className="px-6 py-3 bg-white text-gray-800 font-bold rounded-lg hover:bg-gray-100"
          style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 700 }}
        >
          Go Back
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-800 flex flex-col items-center justify-center relative">
      {/* Top right: video factory */}
      <div className="absolute top-8 right-8">
        <h1 className="text-white text-2xl font-bold" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 700 }}>
          video factory
        </h1>
      </div>

      {/* Main content */}
      <div className="flex flex-col items-center justify-center w-full max-w-3xl px-4">
        <h2 className="text-white text-3xl md:text-4xl font-bold mb-8 text-center" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 700 }}>
          Your 30-Second Script
        </h2>

        <div className="w-full bg-gray-700 rounded-lg p-8 mb-8">
          <p className="text-white text-lg leading-relaxed whitespace-pre-wrap" style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 400 }}>
            {decodeURIComponent(script)}
          </p>
        </div>

        <div className="flex gap-4">
          <button
            onClick={() => router.push('/')}
            className="px-6 py-3 bg-white text-gray-800 font-bold rounded-lg hover:bg-gray-100"
            style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 700 }}
          >
            Create Another
          </button>
        </div>
      </div>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-gray-800 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    }>
      <ResultsContent />
    </Suspense>
  );
}
