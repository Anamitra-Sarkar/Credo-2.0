// Firebase Auth Context for React
import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { firebaseAuthService, FirebaseUser } from '../services/firebaseAuth';

interface AuthContextType {
  user: FirebaseUser | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<FirebaseUser>;
  register: (email: string, password: string, displayName?: string) => Promise<FirebaseUser>;
  loginWithGoogle: () => Promise<FirebaseUser>;
  logout: () => Promise<void>;
  resetPassword: (email: string) => Promise<void>;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<FirebaseUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Listen to auth state changes
    const unsubscribe = firebaseAuthService.onAuthStateChange((user) => {
      setUser(user);
      setLoading(false);
    });

    // Cleanup subscription
    return () => unsubscribe();
  }, []);

  const login = async (email: string, password: string) => {
    const user = await firebaseAuthService.login(email, password);
    setUser(user);
    return user;
  };

  const register = async (email: string, password: string, displayName?: string) => {
    const user = await firebaseAuthService.register(email, password, displayName);
    setUser(user);
    return user;
  };

  const loginWithGoogle = async () => {
    const user = await firebaseAuthService.loginWithGoogle();
    setUser(user);
    return user;
  };

  const logout = async () => {
    await firebaseAuthService.logout();
    setUser(null);
  };

  const resetPassword = async (email: string) => {
    await firebaseAuthService.resetPassword(email);
  };

  const value: AuthContextType = {
    user,
    loading,
    login,
    register,
    loginWithGoogle,
    logout,
    resetPassword,
    isAuthenticated: user !== null
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
}

// Custom hook to use auth context
export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
