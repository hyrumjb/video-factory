'use client';

import { useEffect } from 'react';
import { app } from '@/lib/firebase';
import { AuthProvider } from '@/contexts/AuthContext';

export default function FirebaseProvider({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    // Firebase is initialized when the module is imported
    // This component ensures it happens on the client side
    if (process.env.NODE_ENV === 'development') {
      console.log('Firebase initialized:', app.name);
    }
  }, []);

  return <AuthProvider>{children}</AuthProvider>;
}
