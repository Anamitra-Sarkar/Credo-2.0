/**
 * API Client for VeriFy AI Backend
 * Handles all HTTP requests to the backend API
 */

import { getApiUrl, getAuthHeaders, API_CONFIG } from '../config/api';

export interface ApiError {
  error: string;
  detail: string | any;
  status: number;
}

export class ApiClient {
  private static async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = getApiUrl(endpoint);
    
    const config: RequestInit = {
      ...options,
      headers: {
        ...getAuthHeaders(),
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({
          error: 'Request failed',
          detail: response.statusText,
        }));
        
        throw {
          ...error,
          status: response.status,
        } as ApiError;
      }

      return await response.json();
    } catch (error) {
      if ((error as ApiError).status) {
        throw error;
      }
      throw {
        error: 'Network Error',
        detail: 'Failed to connect to server',
        status: 0,
      } as ApiError;
    }
  }

  // Authentication
  static async register(data: {
    email: string;
    username: string;
    password: string;
    full_name?: string;
  }) {
    return this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  static async login(email: string, password: string) {
    return this.request<{
      access_token: string;
      refresh_token: string;
      token_type: string;
    }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  }

  static async refreshToken(refreshToken: string) {
    return this.request<{ access_token: string }>('/auth/refresh', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  }

  static async getCurrentUser() {
    return this.request('/auth/me', {
      method: 'GET',
    });
  }

  // Text Detection
  static async checkText(text: string, language?: string) {
    return this.request<{
      detection_id: number;
      verdict: 'real' | 'fake' | 'unverified';
      confidence: number;
      explanation: string | null;
      model_used: string;
      processing_time_ms: number;
      original_language: string | null;
      translated_to_english: boolean;
    }>('/check-text', {
      method: 'POST',
      body: JSON.stringify({ text, language }),
    });
  }

  // Image Detection
  static async checkImage(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const url = getApiUrl('/check-image');
    const token = localStorage.getItem('access_token');

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        ...(token && { 'Authorization': `Bearer ${token}` }),
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Request failed',
        detail: response.statusText,
      }));
      throw { ...error, status: response.status } as ApiError;
    }

    return await response.json();
  }

  // Video Detection
  static async checkVideo(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const url = getApiUrl('/check-video');
    const token = localStorage.getItem('access_token');

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        ...(token && { 'Authorization': `Bearer ${token}` }),
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Request failed',
        detail: response.statusText,
      }));
      throw { ...error, status: response.status } as ApiError;
    }

    return await response.json() as {
      job_id: string;
      status: string;
      progress: number;
      message: string;
    };
  }

  // Get Video Result
  static async getVideoResult(jobId: string) {
    return this.request<{
      job_id: string;
      status: 'pending' | 'processing' | 'completed' | 'failed';
      progress: number;
      verdict: 'real' | 'fake' | 'unverified' | null;
      confidence: number | null;
      explanation: string | null;
      error_message: string | null;
    }>(`/check-video/result/${jobId}`, {
      method: 'GET',
    });
  }

  // Voice Detection
  static async checkVoice(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const url = getApiUrl('/check-voice');
    const token = localStorage.getItem('access_token');

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        ...(token && { 'Authorization': `Bearer ${token}` }),
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Request failed',
        detail: response.statusText,
      }));
      throw { ...error, status: response.status } as ApiError;
    }

    return await response.json();
  }

  // Get Explanation
  static async getExplanation(detectionId: number) {
    return this.request(`/explanation/${detectionId}`, {
      method: 'GET',
    });
  }

  // Submit Report
  static async submitReport(data: {
    content_url?: string;
    content_text?: string;
    report_reason: string;
    report_category?: string;
    user_location?: string;
  }) {
    return this.request('/report', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Get Trending
  static async getTrending(params?: {
    time_window?: number;
    country?: string;
    category?: string;
  }) {
    const queryParams = new URLSearchParams(params as any);
    return this.request(`/trending?${queryParams}`, {
      method: 'GET',
    });
  }

  // Get Trending Map
  static async getTrendingMap() {
    return this.request('/trending/map', {
      method: 'GET',
    });
  }

  // Health Check
  static async healthCheck() {
    return this.request('/health', {
      method: 'GET',
    });
  }
}
