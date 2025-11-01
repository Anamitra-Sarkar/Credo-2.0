import { useState, useRef, DragEvent } from 'react';
import { motion } from 'motion/react';
import { FileText, Image, Video, Mic, Upload, Loader2, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { Button } from './ui/button';
import { toast } from 'sonner';

type ContentType = 'text' | 'image' | 'video' | 'voice';

interface AnalysisResult {
  status: 'real' | 'fake' | 'uncertain';
  confidence: number;
  details: string;
  source?: string;
}

const contentTypes = [
  { id: 'text' as ContentType, label: 'Text', icon: FileText },
  { id: 'image' as ContentType, label: 'Image', icon: Image },
  { id: 'video' as ContentType, label: 'Video', icon: Video },
  { id: 'voice' as ContentType, label: 'Voice', icon: Mic },
];

export function AnalyzePageWithDragDrop({ language }: { language: string }) {
  const [activeType, setActiveType] = useState<ContentType>('image');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [textInput, setTextInput] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // File type validation
  const acceptedFiles = {
    image: { types: ['image/jpeg', 'image/png', 'image/webp'], maxSize: 10 * 1024 * 1024, extensions: 'JPG, PNG, WebP' },
    video: { types: ['video/mp4', 'video/webm', 'video/quicktime'], maxSize: 50 * 1024 * 1024, extensions: 'MP4, WebM, MOV' },
    voice: { types: ['audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/ogg'], maxSize: 20 * 1024 * 1024, extensions: 'MP3, WAV, M4A, OGG' },
  };

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const validateFile = (file: File): boolean => {
    if (activeType === 'text') return false;

    const config = acceptedFiles[activeType];
    
    // Check file type
    if (!config.types.includes(file.type)) {
      toast.error(`Invalid file type. Please upload ${config.extensions}`);
      return false;
    }

    // Check file size
    if (file.size > config.maxSize) {
      toast.error(`File too large. Max size: ${config.maxSize / (1024 * 1024)}MB`);
      return false;
    }

    return true;
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
        toast.success(`File "${file.name}" ready for analysis`);
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
        toast.success(`File "${file.name}" ready for analysis`);
      }
    }
  };

  const handleAnalyze = async () => {
    if (activeType === 'text' && !textInput.trim()) {
      toast.error('Please enter some text to analyze');
      return;
    }

    if (activeType !== 'text' && !selectedFile) {
      toast.error('Please select a file to analyze');
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      const formData = new FormData();
      
      if (activeType === 'text') {
        formData.append('content', textInput);
        const response = await fetch('http://localhost:8000/api/v1/check-text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: textInput }),
        });
        const data = await response.json();
        setResult({
          status: data.is_fake ? 'fake' : 'real',
          confidence: data.confidence,
          details: data.analysis || 'Analysis complete',
        });
      } else if (selectedFile) {
        formData.append('file', selectedFile);
        
        const endpoints = {
          image: '/api/v1/check-image',
          video: '/api/v1/check-video',
          voice: '/api/v1/check-voice',
        };

        const response = await fetch(`http://localhost:8000${endpoints[activeType]}`, {
          method: 'POST',
          body: formData,
        });
        
        const data = await response.json();
        setResult({
          status: data.is_fake ? 'fake' : 'real',
          confidence: data.confidence,
          details: data.analysis || 'Analysis complete',
        });
      }

      toast.success('Analysis complete!');
    } catch (error) {
      console.error('Analysis error:', error);
      toast.error('Analysis failed. Please try again.');
      setResult({
        status: 'uncertain',
        confidence: 0,
        details: 'Unable to complete analysis. Please try again.',
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTypeChange = (type: ContentType) => {
    setActiveType(type);
    setSelectedFile(null);
    setTextInput('');
    setResult(null);
    setIsDragging(false);
  };

  const getResultIcon = () => {
    if (!result) return null;
    switch (result.status) {
      case 'real':
        return <CheckCircle className="w-16 h-16 text-green-500" />;
      case 'fake':
        return <XCircle className="w-16 h-16 text-red-500" />;
      case 'uncertain':
        return <AlertCircle className="w-16 h-16 text-yellow-500" />;
    }
  };

  const getResultColor = () => {
    if (!result) return '';
    switch (result.status) {
      case 'real':
        return 'text-green-600 dark:text-green-400';
      case 'fake':
        return 'text-red-600 dark:text-red-400';
      case 'uncertain':
        return 'text-yellow-600 dark:text-yellow-400';
    }
  };

  return (
    <div className="min-h-screen pt-28 pb-16">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-4xl mx-auto"
        >
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-display mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Analyze Content
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Select content type and upload your file for AI-powered verification
            </p>
          </div>

          {/* Content Type Tabs */}
          <div className="glass-card rounded-2xl p-6 mb-8">
            <div className="grid grid-cols-4 gap-4">
              {contentTypes.map(({ id, label, icon: Icon }) => (
                <motion.button
                  key={id}
                  onClick={() => handleTypeChange(id)}
                  className={`flex flex-col items-center gap-3 p-4 rounded-xl transition-all ${
                    activeType === id
                      ? 'bg-gradient-to-br from-blue-600 to-purple-600 text-white shadow-lg'
                      : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700'
                  }`}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Icon className="w-6 h-6" />
                  <span className="text-sm font-medium">{label}</span>
                </motion.button>
              ))}
            </div>
          </div>

          {/* Upload Area */}
          <div className="glass-card rounded-2xl p-8 mb-8">
            {activeType === 'text' ? (
              <div>
                <label className="block mb-3 text-sm font-medium">
                  Enter text to analyze
                </label>
                <textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Paste or type the text you want to verify..."
                  className="w-full h-48 px-4 py-3 rounded-xl border-2 border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 focus:border-blue-500 focus:outline-none resize-none"
                />
              </div>
            ) : (
              <div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={acceptedFiles[activeType].types.join(',')}
                  onChange={handleFileSelect}
                  className="hidden"
                />
                
                <div
                  onDragEnter={handleDragEnter}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
                    isDragging
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-300 dark:border-gray-700 hover:border-blue-400'
                  }`}
                >
                  <Upload className={`w-16 h-16 mx-auto mb-4 ${isDragging ? 'text-blue-500' : 'text-gray-400'}`} />
                  
                  {selectedFile ? (
                    <div>
                      <p className="text-lg font-medium text-green-600 dark:text-green-400 mb-2">
                        ✓ {selectedFile.name}
                      </p>
                      <p className="text-sm text-gray-500">
                        {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                      <p className="text-xs text-gray-400 mt-2">
                        Click to change file
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p className="text-lg font-medium mb-2">
                        {isDragging ? `Drop your ${activeType} here` : `Drag & drop a${activeType === 'image' ? 'n' : ''} ${activeType}`}
                      </p>
                      <p className="text-gray-500 mb-1">or click to browse</p>
                      <p className="text-sm text-gray-400">
                        Supports {acceptedFiles[activeType].extensions} • Max {acceptedFiles[activeType].maxSize / (1024 * 1024)}MB
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Analyze Button */}
          <Button
            onClick={handleAnalyze}
            disabled={isAnalyzing || (activeType === 'text' ? !textInput.trim() : !selectedFile)}
            className="w-full h-14 text-lg font-semibold bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              'Analyze Content'
            )}
          </Button>

          {/* Results */}
          {result && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="glass-card rounded-2xl p-8 mt-8"
            >
              <div className="text-center">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 200 }}
                >
                  {getResultIcon()}
                </motion.div>
                
                <h3 className={`text-3xl font-bold mt-4 mb-2 ${getResultColor()}`}>
                  {result.status.toUpperCase()}
                </h3>
                
                <p className="text-xl mb-4">
                  Confidence: {Math.round(result.confidence * 100)}%
                </p>
                
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-6">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${result.confidence * 100}%` }}
                    transition={{ duration: 1, ease: 'easeOut' }}
                    className={`h-3 rounded-full ${
                      result.status === 'real'
                        ? 'bg-green-500'
                        : result.status === 'fake'
                        ? 'bg-red-500'
                        : 'bg-yellow-500'
                    }`}
                  />
                </div>
                
                <p className="text-gray-600 dark:text-gray-400">
                  {result.details}
                </p>
              </div>
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
