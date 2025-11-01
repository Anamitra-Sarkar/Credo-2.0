/**
 * Authentication Service
 * Handles user authentication, token management, and session
 */

import { ApiClient } from './api';

export interface User {
  id: number;
  email: string;
  username: string;
  full_name?: string;
  role: 'user' | 'moderator' | 'admin';
  preferred_language: string;
}

class AuthService {
  private static TOKEN_KEY = 'access_token';
  private static REFRESH_KEY = 'refresh_token';
  private static USER_KEY = 'user';

  /**
   * Register a new user
   */
  async register(data: {
    email: string;
    username: string;
    password: string;
    full_name?: string;
  }): Promise<void> {
    await ApiClient.register(data);
  }

  /**
   * Login user
   */
  async login(email: string, password: string): Promise<User> {
    const response = await ApiClient.login(email, password);
    
    // Store tokens
    localStorage.setItem(AuthService.TOKEN_KEY, response.access_token);
    localStorage.setItem(AuthService.REFRESH_KEY, response.refresh_token);
    
    // Get user info
    const user = await ApiClient.getCurrentUser();
    localStorage.setItem(AuthService.USER_KEY, JSON.stringify(user));
    
    return user;
  }

  /**
   * Logout user
   */
  logout(): void {
    localStorage.removeItem(AuthService.TOKEN_KEY);
    localStorage.removeItem(AuthService.REFRESH_KEY);
    localStorage.removeItem(AuthService.USER_KEY);
  }

  /**
   * Get current user from localStorage
   */
  getCurrentUser(): User | null {
    const userStr = localStorage.getItem(AuthService.USER_KEY);
    if (!userStr) return null;
    
    try {
      return JSON.parse(userStr);
    } catch {
      return null;
    }
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return !!localStorage.getItem(AuthService.TOKEN_KEY);
  }

  /**
   * Refresh access token
   */
  async refreshToken(): Promise<void> {
    const refreshToken = localStorage.getItem(AuthService.REFRESH_KEY);
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await ApiClient.refreshToken(refreshToken);
    localStorage.setItem(AuthService.TOKEN_KEY, response.access_token);
  }

  /**
   * Get access token
   */
  getAccessToken(): string | null {
    return localStorage.getItem(AuthService.TOKEN_KEY);
  }
}

export const authService = new AuthService();
