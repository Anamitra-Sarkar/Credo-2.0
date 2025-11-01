import { useState, useRef, useEffect } from 'react';
import { motion } from 'motion/react';
import { Upload, Image as ImageIcon, Video, Mic, FileText, X, Clock, Trash2 } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { useDetection } from '../hooks/useDetection';
import { useAuth } from '../context/AuthContext';
import { toast } from 'sonner';

interface AnalysisHistory {
  id: string;
  type: 'text' | 'image' | 'video' | 'audio';
  content: string;
  result?: {
    verdict: string;
    confidence: number;
  };
  timestamp: number;
}

export function AnalyzePage({ language }: { language: string }) {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState<'text' | 'image' | 'video' | 'audio'>('text');
  const [textInput, setTextInput] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [history, setHistory] = useState<AnalysisHistory[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  
  const { loading, result, error, checkText, checkImage, checkVideo, checkVoice, reset } = useDetection();
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load history from localStorage
  useEffect(() => {
    if (user) {
      const savedHistory = localStorage.getItem(`analysis_history_${user.uid}`);
      if (savedHistory) {
        setHistory(JSON.parse(savedHistory));
      }
    }
  }, [user]);

  // Save history to localStorage
  const saveToHistory = (item: Omit<AnalysisHistory, 'id' | 'timestamp'>) => {
    if (!user) return;
    
    const newItem: AnalysisHistory = {
      ...item,
      id: Date.now().toString(),
      timestamp: Date.now()
    };
    
    const updatedHistory = [newItem, ...history].slice(0, 50); // Keep last 50 items
    setHistory(updatedHistory);
    localStorage.setItem(`analysis_history_${user.uid}`, JSON.stringify(updatedHistory));
  };

  // Delete history item
  const deleteHistoryItem = (id: string) => {
    if (!user) return;
    
    if (window.confirm('Are you sure you want to delete this item from your history?')) {
      const updatedHistory = history.filter(item => item.id !== id);
      setHistory(updatedHistory);
      localStorage.setItem(`analysis_history_${user.uid}`, JSON.stringify(updatedHistory));
      toast.success('History item deleted');
    }
  };

  // Clear all history
  const clearAllHistory = () => {
    if (!user) return;
    
    if (window.confirm('Are you sure you want to clear all history? This action cannot be undone.')) {
      setHistory([]);
      localStorage.removeItem(`analysis_history_${user.uid}`);
      toast.success('History cleared');
    }
  };

  // Handle drag events
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  // Handle drop
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      handleFileSelection(file);
    }
  };

  // Handle file selection
  const handleFileSelection = (file: File) => {
    // Validate file type based on active tab
    const validTypes: Record<string, string[]> = {
      image: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
      video: ['video/mp4', 'video/webm', 'video/quicktime'],
      audio: ['audio/mpeg', 'audio/wav', 'audio/mp3', 'audio/ogg', 'audio/x-m4a']
    };

    if (activeTab !== 'text') {
      const allowed = validTypes[activeTab];
      if (!allowed.includes(file.type)) {
        toast.error(`Invalid file type. Please select a valid ${activeTab} file.`);
        return;
      }

      // Check file size
      const maxSize = activeTab === 'video' ? 100 : activeTab === 'audio' ? 20 : 10;
      if (file.size > maxSize * 1024 * 1024) {
        toast.error(`File too large. Maximum size: ${maxSize}MB`);
        return;
      }

      setSelectedFile(file);
      toast.success(`File selected: ${file.name}`);
    }
  };

  // Handle file input change
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  // Trigger file input click
  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  // Handle analysis
  const handleAnalyze = async () => {
    if (!user) {
      toast.error('Please log in to analyze content');
      return;
    }

    try {
      let analysisResult;

      if (activeTab === 'text') {
        if (!textInput.trim()) {
          toast.error('Please enter some text to analyze');
          return;
        }
        analysisResult = await checkText(textInput, language);
        saveToHistory({
          type: 'text',
          content: textInput.substring(0, 100) + (textInput.length > 100 ? '...' : ''),
          result: analysisResult ? {
            verdict: analysisResult.verdict,
            confidence: analysisResult.confidence
          } : undefined
        });
      } else if (activeTab === 'image') {
        if (!selectedFile) {
          toast.error('Please select an image file');
          return;
        }
        analysisResult = await checkImage(selectedFile);
        saveToHistory({
          type: 'image',
          content: selectedFile.name,
          result: analysisResult ? {
            verdict: analysisResult.verdict,
            confidence: analysisResult.confidence
          } : undefined
        });
      } else if (activeTab === 'video') {
        if (!selectedFile) {
          toast.error('Please select a video file');
          return;
        }
        analysisResult = await checkVideo(selectedFile);
        saveToHistory({
          type: 'video',
          content: selectedFile.name,
          result: undefined // Video results come async
        });
      } else if (activeTab === 'audio') {
        if (!selectedFile) {
          toast.error('Please select an audio file');
          return;
        }
        analysisResult = await checkVoice(selectedFile);
        saveToHistory({
          type: 'audio',
          content: selectedFile.name,
          result: analysisResult ? {
            verdict: analysisResult.verdict,
            confidence: analysisResult.confidence
          } : undefined
        });
      }
    } catch (err: any) {
      toast.error(err.message || 'Analysis failed');
    }
  };

  // Reset for new analysis
  const handleReset = () => {
    reset();
    setTextInput('');
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getFileIcon = () => {
    switch (activeTab) {
      case 'image': return <ImageIcon className="w-8 h-8 mb-2" />;
      case 'video': return <Video className="w-8 h-8 mb-2" />;
      case 'audio': return <Mic className="w-8 h-8 mb-2" />;
      default: return <FileText className="w-8 h-8 mb-2" />;
    }
  };

  const getAcceptTypes = () => {
    switch (activeTab) {
      case 'image': return 'image/jpeg,image/jpg,image/png,image/webp';
      case 'video': return 'video/mp4,video/webm,video/quicktime';
      case 'audio': return 'audio/mpeg,audio/wav,audio/mp3,audio/ogg,audio/x-m4a';
      default: return '';
    }
  };

  return (
    <div className="min-h-screen pt-28 pb-16 px-4">
      <div className="container mx-auto max-w-6xl">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-5xl font-display font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 text-transparent bg-clip-text">
            Analyze Content
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400">
            Upload or paste content to check for fake news and deepfakes
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Main Analysis Area */}
          <div className="lg:col-span-2">
            {/* Tabs */}
            <div className="glass-card p-2 mb-6">
              <div className="grid grid-cols-4 gap-2">
                {(['text', 'image', 'video', 'audio'] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => {
                      setActiveTab(tab);
                      handleReset();
                    }}
                    className={`px-4 py-3 rounded-lg font-medium transition-all ${
                      activeTab === tab
                        ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                        : 'hover:bg-gray-100 dark:hover:bg-gray-800'
                    }`}
                  >
                    {tab === 'text' && <FileText className="w-5 h-5 inline mr-2" />}
                    {tab === 'image' && <ImageIcon className="w-5 h-5 inline mr-2" />}
                    {tab === 'video' && <Video className="w-5 h-5 inline mr-2" />}
                    {tab === 'audio' && <Mic className="w-5 h-5 inline mr-2" />}
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Area */}
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="glass-card p-6"
            >
              {activeTab === 'text' ? (
                <div>
                  <label className="block mb-2 font-medium">Enter text to analyze:</label>
                  <Textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="Paste news article, claim, or any text content here..."
                    className="min-h-[300px] mb-4"
                    disabled={loading}
                  />
                </div>
              ) : (
                <div>
                  {/* Hidden file input */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept={getAcceptTypes()}
                    onChange={handleFileInputChange}
                    className="hidden"
                  />

                  {/* Drop zone */}
                  <div
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={triggerFileInput}
                    className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
                      dragActive
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-300 dark:border-gray-700 hover:border-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800'
                    }`}
                  >
                    {selectedFile ? (
                      <div className="space-y-4">
                        {getFileIcon()}
                        <p className="text-lg font-medium text-green-600">
                          ✓ {selectedFile.name}
                        </p>
                        <p className="text-sm text-gray-500">
                          {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                        <Button
                          variant="outline"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedFile(null);
                            if (fileInputRef.current) {
                              fileInputRef.current.value = '';
                            }
                          }}
                        >
                          <X className="w-4 h-4 mr-2" />
                          Remove File
                        </Button>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <Upload className="w-16 h-16 mx-auto text-gray-400" />
                        <div>
                          <p className="text-lg font-medium mb-2">
                            Drag & drop {activeTab} file here
                          </p>
                          <p className="text-sm text-gray-500 mb-4">
                            or click to browse
                          </p>
                          <p className="text-xs text-gray-400">
                            {activeTab === 'image' && 'Supported: JPG, PNG, WebP (Max 10MB)'}
                            {activeTab === 'video' && 'Supported: MP4, WebM, MOV (Max 100MB)'}
                            {activeTab === 'audio' && 'Supported: MP3, WAV, OGG (Max 20MB)'}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-4 mt-6">
                <Button
                  onClick={handleAnalyze}
                  disabled={loading || (activeTab === 'text' ? !textInput.trim() : !selectedFile)}
                  className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                >
                  {loading ? 'Analyzing...' : 'Analyze Content'}
                </Button>
                {(result || error) && (
                  <Button variant="outline" onClick={handleReset}>
                    New Analysis
                  </Button>
                )}
              </div>

              {/* Results */}
              {result && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="mt-6 p-6 rounded-xl border-2 border-gray-200 dark:border-gray-700"
                >
                  <div className={`text-2xl font-bold mb-4 ${
                    result.verdict === 'fake' ? 'text-red-600' :
                    result.verdict === 'real' ? 'text-green-600' :
                    'text-yellow-600'
                  }`}>
                    {result.verdict === 'fake' ? '🚨 FAKE DETECTED' :
                     result.verdict === 'real' ? '✅ LIKELY REAL' :
                     '⚠️ UNVERIFIED'}
                  </div>
                  <div className="space-y-2 text-sm">
                    <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%</p>
                    <p><strong>Model:</strong> {result.model_used}</p>
                    <p><strong>Processing Time:</strong> {result.processing_time_ms}ms</p>
                    {result.explanation && (
                      <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded">
                        <p className="font-semibold mb-2">Explanation:</p>
                        <p className="text-gray-700 dark:text-gray-300">{result.explanation}</p>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </motion.div>
          </div>

          {/* History Sidebar */}
          <div className="lg:col-span-1">
            <div className="glass-card p-6 sticky top-32">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold flex items-center">
                  <Clock className="w-5 h-5 mr-2" />
                  Recent Analysis
                </h3>
                {history.length > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearAllHistory}
                    className="text-red-600 hover:text-red-700"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                )}
              </div>

              {!user ? (
                <p className="text-sm text-gray-500 text-center py-8">
                  Log in to see your analysis history
                </p>
              ) : history.length === 0 ? (
                <p className="text-sm text-gray-500 text-center py-8">
                  No analysis history yet
                </p>
              ) : (
                <div className="space-y-3 max-h-[600px] overflow-y-auto">
                  {history.map((item) => (
                    <motion.div
                      key={item.id}
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="p-3 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors group"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            {item.type === 'text' && <FileText className="w-4 h-4 flex-shrink-0" />}
                            {item.type === 'image' && <ImageIcon className="w-4 h-4 flex-shrink-0" />}
                            {item.type === 'video' && <Video className="w-4 h-4 flex-shrink-0" />}
                            {item.type === 'audio' && <Mic className="w-4 h-4 flex-shrink-0" />}
                            <span className="text-xs font-medium text-gray-500 uppercase">
                              {item.type}
                            </span>
                          </div>
                          <p className="text-sm truncate mb-2">{item.content}</p>
                          {item.result && (
                            <div className="flex items-center gap-2">
                              <span className={`text-xs font-semibold ${
                                item.result.verdict === 'fake' ? 'text-red-600' :
                                item.result.verdict === 'real' ? 'text-green-600' :
                                'text-yellow-600'
                              }`}>
                                {item.result.verdict.toUpperCase()}
                              </span>
                              <span className="text-xs text-gray-500">
                                {(item.result.confidence * 100).toFixed(0)}%
                              </span>
                            </div>
                          )}
                          <p className="text-xs text-gray-400 mt-1">
                            {new Date(item.timestamp).toLocaleString()}
                          </p>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => deleteHistoryItem(item.id)}
                          className="opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <Trash2 className="w-4 h-4 text-red-600" />
                        </Button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
