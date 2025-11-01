// Firebase Authentication Service
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  updateProfile,
  User,
  GoogleAuthProvider,
  signInWithPopup,
  sendPasswordResetEmail,
  sendEmailVerification,
  UserCredential
} from 'firebase/auth';
import { auth } from '../config/firebase';

export interface FirebaseUser {
  uid: string;
  email: string | null;
  displayName: string | null;
  photoURL: string | null;
  emailVerified: boolean;
}

class FirebaseAuthService {
  // Register new user with email and password
  async register(email: string, password: string, displayName?: string): Promise<FirebaseUser> {
    try {
      const userCredential: UserCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      // Update display name if provided
      if (displayName) {
        await updateProfile(user, { displayName });
      }

      // Send email verification
      await sendEmailVerification(user);

      return this.mapFirebaseUser(user);
    } catch (error: any) {
      throw this.handleAuthError(error);
    }
  }

  // Login with email and password
  async login(email: string, password: string): Promise<FirebaseUser> {
    try {
      const userCredential: UserCredential = await signInWithEmailAndPassword(auth, email, password);
      return this.mapFirebaseUser(userCredential.user);
    } catch (error: any) {
      throw this.handleAuthError(error);
    }
  }

  // Login with Google
  async loginWithGoogle(): Promise<FirebaseUser> {
    try {
      const provider = new GoogleAuthProvider();
      const userCredential: UserCredential = await signInWithPopup(auth, provider);
      return this.mapFirebaseUser(userCredential.user);
    } catch (error: any) {
      throw this.handleAuthError(error);
    }
  }

  // Logout
  async logout(): Promise<void> {
    try {
      await signOut(auth);
    } catch (error: any) {
      throw this.handleAuthError(error);
    }
  }

  // Reset password
  async resetPassword(email: string): Promise<void> {
    try {
      await sendPasswordResetEmail(auth, email);
    } catch (error: any) {
      throw this.handleAuthError(error);
    }
  }

  // Get current user
  getCurrentUser(): FirebaseUser | null {
    const user = auth.currentUser;
    return user ? this.mapFirebaseUser(user) : null;
  }

  // Check if user is authenticated
  isAuthenticated(): boolean {
    return auth.currentUser !== null;
  }

  // Listen to auth state changes
  onAuthStateChange(callback: (user: FirebaseUser | null) => void): () => void {
    return onAuthStateChanged(auth, (user) => {
      callback(user ? this.mapFirebaseUser(user) : null);
    });
  }

  // Get Firebase ID token for backend API calls
  async getIdToken(): Promise<string | null> {
    const user = auth.currentUser;
    if (user) {
      return await user.getIdToken();
    }
    return null;
  }

  // Map Firebase User to our custom user type
  private mapFirebaseUser(user: User): FirebaseUser {
    return {
      uid: user.uid,
      email: user.email,
      displayName: user.displayName,
      photoURL: user.photoURL,
      emailVerified: user.emailVerified
    };
  }

  // Handle Firebase auth errors
  private handleAuthError(error: any): Error {
    switch (error.code) {
      case 'auth/email-already-in-use':
        return new Error('This email is already registered');
      case 'auth/invalid-email':
        return new Error('Invalid email address');
      case 'auth/operation-not-allowed':
        return new Error('Operation not allowed');
      case 'auth/weak-password':
        return new Error('Password is too weak (minimum 6 characters)');
      case 'auth/user-disabled':
        return new Error('This account has been disabled');
      case 'auth/user-not-found':
        return new Error('No account found with this email');
      case 'auth/wrong-password':
        return new Error('Incorrect password');
      case 'auth/invalid-credential':
        return new Error('Invalid email or password');
      case 'auth/too-many-requests':
        return new Error('Too many attempts. Please try again later');
      case 'auth/network-request-failed':
        return new Error('Network error. Please check your connection');
      case 'auth/popup-closed-by-user':
        return new Error('Sign-in popup was closed');
      default:
        return new Error(error.message || 'Authentication failed');
    }
  }
}

export const firebaseAuthService = new FirebaseAuthService();
