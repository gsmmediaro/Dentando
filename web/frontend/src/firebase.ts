import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyDGctL5PImQoVk6LFvac28kqaJF6Y1rq54",
  authDomain: "quinn-app-a28ef.firebaseapp.com",
  projectId: "quinn-app-a28ef",
  storageBucket: "quinn-app-a28ef.firebasestorage.app",
  messagingSenderId: "592937500286",
  appId: "1:592937500286:web:2125a21406d0e0b692222f",
  measurementId: "G-NP39FFVWTG",
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export const googleProvider = new GoogleAuthProvider();
