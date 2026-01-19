import { initializeApp, getApps } from "firebase/app";
import { getAnalytics, isSupported } from "firebase/analytics";
import {
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  OAuthProvider,
  signOut,
  onAuthStateChanged,
  User
} from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyDVOlk3RWlg5a-3iXzbTvp98peo1jm18lA",
  authDomain: "lightfall-ai.firebaseapp.com",
  projectId: "lightfall-ai",
  storageBucket: "lightfall-ai.firebasestorage.app",
  messagingSenderId: "4344064279",
  appId: "1:4344064279:web:d21947000dec047b38af52",
  measurementId: "G-KZ5ZLR9EQC"
};

// Initialize Firebase (only once)
const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];

// Initialize Auth
const auth = getAuth(app);

// Auth providers
const googleProvider = new GoogleAuthProvider();
const appleProvider = new OAuthProvider('apple.com');
appleProvider.addScope('email');
appleProvider.addScope('name');

// Initialize Analytics (only in browser)
let analytics: ReturnType<typeof getAnalytics> | null = null;

if (typeof window !== 'undefined') {
  isSupported().then((supported) => {
    if (supported) {
      analytics = getAnalytics(app);
    }
  });
}

// Auth functions
export const signInWithEmail = (email: string, password: string) => {
  return signInWithEmailAndPassword(auth, email, password);
};

export const signUpWithEmail = (email: string, password: string) => {
  return createUserWithEmailAndPassword(auth, email, password);
};

export const signInWithGoogle = () => {
  return signInWithPopup(auth, googleProvider);
};

export const signInWithApple = () => {
  return signInWithPopup(auth, appleProvider);
};

export const logOut = () => {
  return signOut(auth);
};

export const onAuthChange = (callback: (user: User | null) => void) => {
  return onAuthStateChanged(auth, callback);
};

export { app, auth, analytics };
export type { User };
