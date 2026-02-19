import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";

function normalize_storage_bucket(bucket: string): string {
  if (!bucket) {
    return "quinn-app-a28ef.appspot.com";
  }
  if (bucket.endsWith(".firebasestorage.app")) {
    return bucket.replace(/\.firebasestorage\.app$/, ".appspot.com");
  }
  return bucket;
}

const firebaseConfig = {
  apiKey: "AIzaSyDGctL5PImQoVk6LFvac28kqaJF6Y1rq54",
  authDomain: "quinn-app-a28ef.firebaseapp.com",
  projectId: "quinn-app-a28ef",
  storageBucket: normalize_storage_bucket(
    import.meta.env.VITE_FIREBASE_STORAGE_BUCKET ||
      "quinn-app-a28ef.appspot.com",
  ),
  messagingSenderId: "592937500286",
  appId: "1:592937500286:web:2125a21406d0e0b692222f",
  measurementId: "G-NP39FFVWTG",
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export const storage = getStorage(app);
export const googleProvider = new GoogleAuthProvider();
