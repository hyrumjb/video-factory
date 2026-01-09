'use client';

import Link from 'next/link';

export default function Header() {
  return (
    <header className="absolute top-0 left-0 right-0 z-50 px-8 py-6">
      <Link 
        href="/"
        className="text-white text-xl inline-block transition-all duration-300 hover:opacity-80 hover:underline"
        style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 500 }}
      >
        Video Factory
      </Link>
    </header>
  );
}
