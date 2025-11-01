import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Upload, FileText, Image as ImageIcon, Video, Mic, CheckCircle, XCircle, AlertTriangle, Info } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Card } from './ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './ui/dialog';

interface AnalyzePageProps {
  language: string;
}

type Verdict = 'true' | 'fake' | 'deepfake' | 'unverified' | null;

export function AnalyzePage({ language }: AnalyzePageProps) {
  const [analyzing, setAnalyzing] = useState(false);
  const [verdict, setVerdict] = useState<Verdict>(null);
  const [confidence, setConfidence] = useState(0);
  const [showExplainer, setShowExplainer] = useState(false);
  const [inputText, setInputText] = useState('');

  const handleAnalyze = () => {
    setAnalyzing(true);
    setVerdict(null);

    // Simulate analysis
    setTimeout(() => {
      const verdicts: Verdict[] = ['true', 'fake', 'deepfake', 'unverified'];
      const randomVerdict = verdicts[Math.floor(Math.random() * verdicts.length)];
      setVerdict(randomVerdict);
      setConfidence(Math.floor(Math.random() * 30) + 70);
      setAnalyzing(false);
    }, 2500);
  };

  const getVerdictConfig = (v: Verdict) => {
    switch (v) {
      case 'true':
        return {
          icon: CheckCircle,
          color: 'text-green-600',
          bgColor: 'bg-green-50 dark:bg-green-950',
          borderColor: 'border-green-300',
          label: 'Likely True',
          gradient: 'from-green-500 to-emerald-500',
        };
      case 'fake':
        return {
          icon: XCircle,
          color: 'text-red-600',
          bgColor: 'bg-red-50 dark:bg-red-950',
          borderColor: 'border-red-300',
          label: 'Fake News Detected',
          gradient: 'from-red-500 to-rose-500',
        };
      case 'deepfake':
        return {
          icon: AlertTriangle,
          color: 'text-orange-600',
          bgColor: 'bg-orange-50 dark:bg-orange-950',
          borderColor: 'border-orange-300',
          label: 'Deepfake Detected',
          gradient: 'from-orange-500 to-amber-500',
        };
      case 'unverified':
        return {
          icon: Info,
          color: 'text-yellow-600',
          bgColor: 'bg-yellow-50 dark:bg-yellow-950',
          borderColor: 'border-yellow-300',
          label: 'Unverified',
          gradient: 'from-yellow-500 to-amber-500',
        };
      default:
        return null;
    }
  };

  const verdictConfig = verdict ? getVerdictConfig(verdict) : null;

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
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
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

                <TabsContent value="text" className="space-y-4">
                  <div className="space-y-2">
                    <Textarea
                      placeholder="Paste text, article, or claim to verify..."
                      className="min-h-[300px] text-base rounded-2xl resize-none"
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                    />
                    <div className="flex items-center gap-2">
                      <Button variant="outline" size="sm" className="gap-2">
                        <Mic className="w-4 h-4" />
                        Voice Input
                      </Button>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="image" className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-2xl p-16 text-center hover:border-blue-500 transition-colors cursor-pointer">
                    <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                    <p className="mb-2">Drag & drop an image</p>
                    <p className="text-sm text-gray-500">or click to browse</p>
                    <p className="text-xs text-gray-400 mt-4">
                      Supports JPG, PNG, WebP • Max 10MB
                    </p>
                  </div>
                </TabsContent>

                <TabsContent value="video" className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-2xl p-16 text-center hover:border-blue-500 transition-colors cursor-pointer">
                    <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                    <p className="mb-2">Drag & drop a video</p>
                    <p className="text-sm text-gray-500">or click to browse</p>
                    <p className="text-xs text-gray-400 mt-4">
                      Supports MP4, WebM, MOV • Max 50MB
                    </p>
                  </div>
                </TabsContent>

                <TabsContent value="voice" className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-2xl p-16 text-center hover:border-blue-500 transition-colors cursor-pointer">
                    <Mic className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                    <p className="mb-2">Drag & drop an audio file</p>
                    <p className="text-sm text-gray-500">or click to browse</p>
                    <p className="text-xs text-gray-400 mt-4">
                      Supports MP3, WAV, M4A, OGG • Max 20MB
                    </p>
                  </div>
                  <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                      <div className="w-full border-t border-gray-300 dark:border-gray-700"></div>
                    </div>
                    <div className="relative flex justify-center text-xs uppercase">
                      <span className="bg-white dark:bg-slate-900 px-2 text-gray-500">Or</span>
                    </div>
                  </div>
                  <Button variant="outline" className="w-full gap-2 rounded-xl h-12">
                    <Mic className="w-5 h-5" />
                    Record Audio
                  </Button>
                </TabsContent>
              </Tabs>

              <Button
                onClick={handleAnalyze}
                disabled={analyzing}
                className="w-full h-12 mt-6 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
              >
                {analyzing ? (
                  <>
                    <motion.div
                      className="w-5 h-5 border-2 border-white border-t-transparent rounded-full mr-2"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    />
                    Analyzing...
                  </>
                ) : (
                  'Analyze Content'
                )}
              </Button>
            </Card>
          </motion.div>

          {/* Results Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card className="glass-card p-6 min-h-[500px]">
              <h3 className="mb-6">Analysis Results</h3>

              <AnimatePresence mode="wait">
                {analyzing && (
                  <motion.div
                    key="analyzing"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center justify-center h-96"
                  >
                    <motion.div
                      className="w-24 h-24 border-4 border-blue-600 border-t-transparent rounded-full mb-6"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    />
                    <p className="text-lg mb-2">Analyzing content...</p>
                    <p className="text-sm text-gray-500">This may take a few seconds</p>
                  </motion.div>
                )}

                {!analyzing && !verdict && (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex flex-col items-center justify-center h-96 text-gray-400"
                  >
                    <Info className="w-16 h-16 mb-4" />
                    <p>No content analyzed yet</p>
                    <p className="text-sm mt-2">Submit content to see results</p>
                  </motion.div>
                )}

                {!analyzing && verdict && verdictConfig && (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="space-y-6"
                  >
                    {/* Verdict Badge */}
                    <div className={`${verdictConfig.bgColor} border-2 ${verdictConfig.borderColor} rounded-2xl p-6`}>
                      <div className="flex items-center gap-4 mb-4">
                        <verdictConfig.icon className={`w-12 h-12 ${verdictConfig.color}`} />
                        <div>
                          <h4 className={verdictConfig.color}>{verdictConfig.label}</h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Confidence: {confidence}%
                          </p>
                        </div>
                      </div>
                      
                      <Progress 
                        value={confidence} 
                        className="h-3 mb-3"
                      />
                      
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${confidence}%` }}
                        transition={{ duration: 1, ease: 'easeOut' }}
                        className={`h-1 bg-gradient-to-r ${verdictConfig.gradient} rounded-full`}
                      />
                    </div>

                    {/* Explanation Preview */}
                    <div className="glass-card rounded-xl p-4">
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                        Our analysis detected several indicators suggesting this content may {verdict === 'true' ? 'be authentic' : 'contain misleading information'}. 
                        Key factors include source reliability, metadata consistency, and cross-reference verification.
                      </p>
                      <Button
                        variant="outline"
                        onClick={() => setShowExplainer(true)}
                        className="w-full gap-2"
                      >
                        <Info className="w-4 h-4" />
                        View Detailed Explanation
                      </Button>
                    </div>

                    {/* Related Sources */}
                    <div>
                      <h4 className="mb-3">Related Sources</h4>
                      <div className="space-y-2">
                        {['Source 1: Fact-check article', 'Source 2: Similar claim debunked', 'Source 3: Expert opinion'].map((source, idx) => (
                          <motion.div
                            key={idx}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className="glass-card rounded-lg p-3 text-sm hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-colors"
                          >
                            {source}
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </Card>
          </motion.div>
        </div>
      </div>

      {/* Explainability Modal */}
      <Dialog open={showExplainer} onOpenChange={setShowExplainer}>
        <DialogContent className="max-w-2xl glass-card">
          <DialogHeader>
            <DialogTitle>Detailed Analysis Explanation</DialogTitle>
            <DialogDescription>
              Understanding how we reached this verdict
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-6 mt-4">
            <div>
              <h4 className="mb-3">Key Indicators</h4>
              <div className="space-y-3">
                <div className="flex items-start gap-3">
                  <Badge variant="outline" className="mt-1">1</Badge>
                  <div>
                    <p className="mb-1">Source Reliability Score: {Math.floor(Math.random() * 30) + 70}%</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Cross-referenced with known reliable and unreliable sources
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Badge variant="outline" className="mt-1">2</Badge>
                  <div>
                    <p className="mb-1">Content Pattern Analysis</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Linguistic patterns match typical characteristics of {verdict === 'fake' ? 'misleading' : 'authentic'} content
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Badge variant="outline" className="mt-1">3</Badge>
                  <div>
                    <p className="mb-1">Historical Verification</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Similar claims have been {verdict === 'fake' ? 'debunked' : 'verified'} by fact-checking organizations
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="glass-card rounded-xl p-4">
              <h4 className="mb-2">Verification Timeline</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Initial content scan</span>
                  <span>0.3s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Source cross-reference</span>
                  <span>1.2s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Pattern analysis</span>
                  <span>0.8s</span>
                </div>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
