// Firebase Configuration
import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getAnalytics } from 'firebase/analytics';

const firebaseConfig = {
  apiKey: "AIzaSyDEAOr3iqU6TJZ6uztMvC5mquZECPcBkkE",
  authDomain: "fir-config-d3c36.firebaseapp.com",
  projectId: "fir-config-d3c36",
  storageBucket: "fir-config-d3c36.firebasestorage.app",
  messagingSenderId: "477435579926",
  appId: "1:477435579926:web:d370e9fb5a3c5a05316f37",
  measurementId: "G-PXZBD6HN1X"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication
export const auth = getAuth(app);

// Initialize Analytics (optional)
let analytics;
if (typeof window !== 'undefined') {
  analytics = getAnalytics(app);
}

export { analytics };
export default app;
