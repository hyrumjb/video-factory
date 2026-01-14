'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

export default function Home() {
  const [topic, setTopic] = useState('');
  const [placeholder, setPlaceholder] = useState('');
  const [selectedVoice, setSelectedVoice] = useState('en-AU-Neural2-A');
  const router = useRouter();

  const voices = [
    { name: 'en-AU-Neural2-A', label: 'Female' },
    { name: 'en-AU-Neural2-B', label: 'Male' },
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
    }

    router.push('/results');
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 px-6 py-4 bg-background/80 backdrop-blur-sm border-b">
        <Link href="/" className="text-lg font-medium hover:opacity-80 transition-opacity">
          Studio Nine
        </Link>
      </header>

      {/* Main content */}
      <main className="flex-1 flex flex-col items-center justify-center px-6 pt-20 pb-10">
        <div className="w-full max-w-xl space-y-8">
          <h1 className="text-3xl md:text-4xl font-medium text-center tracking-tight">
            Create beautiful videos with zero effort
          </h1>

          <form onSubmit={handleSubmit} className="space-y-4">
            <Textarea
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder={topic ? '' : placeholder}
              rows={4}
              className="resize-none text-base"
            />

            <div className="flex items-center gap-3 justify-end">
              <Select value={selectedVoice} onValueChange={setSelectedVoice}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {voices.map((voice) => (
                    <SelectItem key={voice.name} value={voice.name}>
                      {voice.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Button type="submit" disabled={!topic.trim()}>
                Generate
              </Button>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
