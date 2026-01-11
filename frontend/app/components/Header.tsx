'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

export default function Header() {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header className={`fixed top-0 left-0 right-0 z-50 px-8 py-6 transition-all duration-300 ${
      isScrolled ? 'bg-white/80 backdrop-blur-md' : 'bg-transparent'
    }`}>
      <Link 
        href="/"
        className="text-gray-900 text-xl inline-block transition-all duration-300 hover:opacity-80 hover:underline"
        style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontWeight: 500 }}
      >
        Video Factory
      </Link>
    </header>
  );
}
