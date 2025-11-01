/**
 * Custom React Hook for Detection API Integration
 * Handles all detection operations with the backend
 */

import { useState } from 'react';
import { ApiClient, ApiError } from '../services/api';
import { toast } from 'sonner';

export interface DetectionResult {
  detection_id: number;
  verdict: 'real' | 'fake' | 'unverified';
  confidence: number;
  explanation: string | null;
  model_used: string;
  processing_time_ms: number;
  translated_to_english?: boolean;
  original_language?: string;
}

export interface VideoJobStatus {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  verdict?: 'real' | 'fake' | 'unverified' | null;
  confidence?: number | null;
  explanation?: string | null;
  error_message?: string | null;
}

export function useDetection() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  /**
   * Check text for fake news
   */
  const checkText = async (text: string, language?: string): Promise<DetectionResult | null> => {
    if (!text.trim()) {
      toast.error('Please enter some text to analyze');
      return null;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await ApiClient.checkText(text, language);
      setResult(response);
      
      // Show success toast
      const verdictColor = response.verdict === 'fake' ? '🚨' : response.verdict === 'real' ? '✅' : '⚠️';
      toast.success(`${verdictColor} Analysis Complete`, {
        description: `Verdict: ${response.verdict.toUpperCase()} (${(response.confidence * 100).toFixed(1)}% confidence)`,
      });
      
      return response;
    } catch (err) {
      const apiError = err as ApiError;
      const errorMsg = apiError.detail || 'Failed to analyze text';
      setError(errorMsg);
      toast.error('Analysis Failed', {
        description: errorMsg,
      });
      return null;
    } finally {
      setLoading(false);
    }
  };

  /**
   * Check image for deepfakes
   */
  const checkImage = async (file: File): Promise<DetectionResult | null> => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await ApiClient.checkImage(file);
      setResult(response);
      
      const verdictColor = response.verdict === 'fake' ? '🚨' : response.verdict === 'real' ? '✅' : '⚠️';
      toast.success(`${verdictColor} Image Analysis Complete`, {
        description: `Verdict: ${response.verdict.toUpperCase()} (${(response.confidence * 100).toFixed(1)}% confidence)`,
      });
      
      return response;
    } catch (err) {
      const apiError = err as ApiError;
      const errorMsg = apiError.detail || 'Failed to analyze image';
      setError(errorMsg);
      toast.error('Image Analysis Failed', {
        description: errorMsg,
      });
      return null;
    } finally {
      setLoading(false);
    }
  };

  /**
   * Check video for deepfakes (starts async job)
   */
  const checkVideo = async (file: File): Promise<string | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await ApiClient.checkVideo(file);
      
      toast.success('🎬 Video Uploaded', {
        description: 'Processing in background. Check status below.',
      });
      
      return response.job_id;
    } catch (err) {
      const apiError = err as ApiError;
      const errorMsg = apiError.detail || 'Failed to upload video';
      setError(errorMsg);
      toast.error('Video Upload Failed', {
        description: errorMsg,
      });
      return null;
    } finally {
      setLoading(false);
    }
  };

  /**
   * Check voice/audio for deepfakes
   */
  const checkVoice = async (file: File): Promise<DetectionResult | null> => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await ApiClient.checkVoice(file);
      setResult(response);
      
      const verdictColor = response.verdict === 'fake' ? '🚨' : response.verdict === 'real' ? '✅' : '⚠️';
      toast.success(`${verdictColor} Voice Analysis Complete`, {
        description: `Verdict: ${response.verdict.toUpperCase()} (${(response.confidence * 100).toFixed(1)}% confidence)`,
      });
      
      return response;
    } catch (err) {
      const apiError = err as ApiError;
      const errorMsg = apiError.detail || 'Failed to analyze voice';
      setError(errorMsg);
      toast.error('Voice Analysis Failed', {
        description: errorMsg,
      });
      return null;
    } finally {
      setLoading(false);
    }
  };

  /**
   * Get video processing status
   */
  const getVideoStatus = async (jobId: string): Promise<VideoJobStatus | null> => {
    try {
      const response = await ApiClient.getVideoResult(jobId);
      return response;
    } catch (err) {
      const apiError = err as ApiError;
      console.error('Failed to get video status:', apiError);
      return null;
    }
  };

  /**
   * Get detailed explanation
   */
  const getExplanation = async (detectionId: number): Promise<any> => {
    try {
      const response = await ApiClient.getExplanation(detectionId);
      return response;
    } catch (err) {
      const apiError = err as ApiError;
      toast.error('Failed to load explanation', {
        description: apiError.detail || 'Please try again',
      });
      return null;
    }
  };

  /**
   * Reset detection state
   */
  const reset = () => {
    setLoading(false);
    setResult(null);
    setError(null);
  };

  return {
    loading,
    result,
    error,
    checkText,
    checkImage,
    checkVideo,
    checkVoice,
    getVideoStatus,
    getExplanation,
    reset,
  };
}
