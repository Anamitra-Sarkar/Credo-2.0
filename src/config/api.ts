/**
 * API Configuration for VeriFy AI
 * Backend integration settings
 */

// API Base URL - change this based on environment
export const API_CONFIG = {
  // Local development
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  
  // Production (uncomment when deployed)
  // BASE_URL: 'https://api.verify-ai.com',
  
  VERSION: 'v1',
  TIMEOUT: 30000, // 30 seconds
};

export const API_ENDPOINTS = {
  // Authentication
  AUTH: {
    REGISTER: '/auth/register',
    LOGIN: '/auth/login',
    REFRESH: '/auth/refresh',
    LOGOUT: '/auth/logout',
    ME: '/auth/me',
  },
  
  // Detection
  DETECTION: {
    CHECK_TEXT: '/check-text',
    CHECK_IMAGE: '/check-image',
    CHECK_VIDEO: '/check-video',
    CHECK_VOICE: '/check-voice',
    VIDEO_RESULT: '/check-video/result',
    EXPLANATION: '/explanation',
  },
  
  // Reports
  REPORT: {
    SUBMIT: '/report',
    LIST: '/reports',
  },
  
  // Trending
  TRENDING: {
    GET: '/trending',
    MAP: '/trending/map',
  },
  
  // Health
  HEALTH: '/health',
};

/**
 * Get full API URL
 */
export function getApiUrl(endpoint: string): string {
  return `${API_CONFIG.BASE_URL}/api/${API_CONFIG.VERSION}${endpoint}`;
}

/**
 * Get authorization headers
 */
export function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem('access_token');
  return {
    'Content-Type': 'application/json',
    ...(token && { 'Authorization': `Bearer ${token}` }),
  };
}
