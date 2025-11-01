/**
 * EXAMPLE: Backend-Integrated Analyze Component
 * 
 * This is a working example showing how to integrate the backend API
 * into your existing AnalyzePage.tsx component.
 * 
 * Copy the relevant parts into your actual AnalyzePage.tsx
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Upload, FileText, Image as ImageIcon, Video, Mic, CheckCircle, XCircle, AlertTriangle, Loader2 } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Card } from './ui/card';
import { useDetection } from '../hooks/useDetection';

export function AnalyzePageIntegrated() {
  const [inputText, setInputText] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  // Backend integration hook
  const { loading, result, error, checkText, checkImage, checkVideo, checkVoice } = useDetection();

  // Text Analysis
  const handleTextAnalysis = async () => {
    if (!inputText.trim()) {
      return;
    }
    await checkText(inputText);
  };

  // Image Analysis
  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      await checkImage(file);
    }
  };

  // Video Analysis
  const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const jobId = await checkVideo(file);
      
      if (jobId) {
        // Video is being processed asynchronously
        // You can poll for status or show a message
        console.log('Video job started:', jobId);
      }
    }
  };

  // Voice Analysis
  const handleVoiceUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      await checkVoice(file);
    }
  };

  // Get verdict icon and color
  const getVerdictDisplay = () => {
    if (!result) return null;
    
    switch (result.verdict) {
      case 'real':
        return {
          icon: CheckCircle,
          color: 'text-green-600',
          bgColor: 'bg-green-50 dark:bg-green-950',
          label: 'Likely Real',
        };
      case 'fake':
        return {
          icon: XCircle,
          color: 'text-red-600',
          bgColor: 'bg-red-50 dark:bg-red-950',
          label: 'Fake Detected',
        };
      case 'unverified':
        return {
          icon: AlertTriangle,
          color: 'text-yellow-600',
          bgColor: 'bg-yellow-50 dark:bg-yellow-950',
          label: 'Unverified',
        };
    }
  };

  const verdictDisplay = getVerdictDisplay();

  return (
    <div className="min-h-screen pt-28 pb-16">
      <div className="container mx-auto px-4 max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12 text-center"
        >
          <h2 className="mb-4">Analyze Content</h2>
          <p className="text-xl text-gray-600 dark:text-gray-400">
            Upload or paste content to verify its authenticity
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Panel */}
          <Card className="glass-card p-6">
            <Tabs defaultValue="text" className="w-full">
              <TabsList className="grid w-full grid-cols-4 mb-6">
                <TabsTrigger value="text" className="gap-2">
                  <FileText className="w-4 h-4" />
                  Text
                </TabsTrigger>
                <TabsTrigger value="image" className="gap-2">
                  <ImageIcon className="w-4 h-4" />
                  Image
                </TabsTrigger>
                <TabsTrigger value="video" className="gap-2">
                  <Video className="w-4 h-4" />
                  Video
                </TabsTrigger>
                <TabsTrigger value="voice" className="gap-2">
                  <Mic className="w-4 h-4" />
                  Voice
                </TabsTrigger>
              </TabsList>

              {/* Text Tab */}
              <TabsContent value="text" className="space-y-4">
                <Textarea
                  placeholder="Paste text, article, or claim to verify..."
                  className="min-h-[300px] text-base rounded-2xl resize-none"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                />
                <Button
                  onClick={handleTextAnalysis}
                  disabled={loading || !inputText.trim()}
                  className="w-full h-12 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    'Analyze Text'
                  )}
                </Button>
              </TabsContent>

              {/* Image Tab */}
              <TabsContent value="image" className="space-y-4">
                <label className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-2xl p-16 text-center hover:border-blue-500 transition-colors cursor-pointer block">
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    onChange={handleImageUpload}
                    className="hidden"
                    disabled={loading}
                  />
                  <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                  <p className="mb-2">Click to upload an image</p>
                  <p className="text-xs text-gray-400 mt-4">
                    JPG, PNG, WebP • Max 10MB
                  </p>
                </label>
              </TabsContent>

              {/* Video Tab */}
              <TabsContent value="video" className="space-y-4">
                <label className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-2xl p-16 text-center hover:border-blue-500 transition-colors cursor-pointer block">
                  <input
                    type="file"
                    accept="video/mp4,video/webm,video/quicktime"
                    onChange={handleVideoUpload}
                    className="hidden"
                    disabled={loading}
                  />
                  <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                  <p className="mb-2">Click to upload a video</p>
                  <p className="text-xs text-gray-400 mt-4">
                    MP4, WebM, MOV • Max 50MB
                  </p>
                </label>
              </TabsContent>

              {/* Voice Tab */}
              <TabsContent value="voice" className="space-y-4">
                <label className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-2xl p-16 text-center hover:border-blue-500 transition-colors cursor-pointer block">
                  <input
                    type="file"
                    accept="audio/mpeg,audio/wav,audio/mp4,audio/ogg"
                    onChange={handleVoiceUpload}
                    className="hidden"
                    disabled={loading}
                  />
                  <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                  <p className="mb-2">Click to upload audio</p>
                  <p className="text-xs text-gray-400 mt-4">
                    MP3, WAV, M4A, OGG • Max 20MB
                  </p>
                </label>
              </TabsContent>
            </Tabs>
          </Card>

          {/* Results Panel */}
          <Card className="glass-card p-6 min-h-[500px]">
            <h3 className="mb-6">Analysis Results</h3>

            {/* Loading State */}
            {loading && (
              <div className="flex flex-col items-center justify-center h-96">
                <Loader2 className="w-24 h-24 text-blue-600 animate-spin mb-6" />
                <p className="text-lg mb-2">Analyzing content...</p>
                <p className="text-sm text-gray-500">This may take a few seconds</p>
              </div>
            )}

            {/* Error State */}
            {error && (
              <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg p-6">
                <XCircle className="w-12 h-12 text-red-600 mx-auto mb-4" />
                <p className="text-center text-red-800 dark:text-red-200">{error}</p>
              </div>
            )}

            {/* Result */}
            {result && verdictDisplay && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-6"
              >
                {/* Verdict Card */}
                <div className={`${verdictDisplay.bgColor} border rounded-2xl p-6`}>
                  <verdictDisplay.icon className={`w-16 h-16 ${verdictDisplay.color} mx-auto mb-4`} />
                  <h4 className={`text-2xl font-bold text-center ${verdictDisplay.color} mb-2`}>
                    {verdictDisplay.label}
                  </h4>
                  <p className="text-center text-gray-600 dark:text-gray-400">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </p>
                </div>

                {/* Details */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Model:</span>
                    <span className="font-medium">{result.model_used}</span>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Processing Time:</span>
                    <span className="font-medium">{result.processing_time_ms}ms</span>
                  </div>

                  {result.translated_to_english && (
                    <div className="bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded-lg p-3 text-sm">
                      <p className="text-blue-800 dark:text-blue-200">
                        ℹ️ Content was translated from {result.original_language?.toUpperCase()}
                      </p>
                    </div>
                  )}

                  {/* Explanation */}
                  {result.explanation && (
                    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                      <h5 className="font-semibold mb-2">Explanation:</h5>
                      <p className="text-sm text-gray-700 dark:text-gray-300">
                        {result.explanation}
                      </p>
                    </div>
                  )}
                </div>
              </motion.div>
            )}

            {/* Empty State */}
            {!loading && !result && !error && (
              <div className="flex flex-col items-center justify-center h-96 text-gray-400">
                <AlertTriangle className="w-16 h-16 mb-4" />
                <p>No content analyzed yet</p>
                <p className="text-sm mt-2">Submit content to see results</p>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
