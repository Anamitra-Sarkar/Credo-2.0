import { motion, AnimatePresence } from 'motion/react';
import { Search, Image, Video, Sparkles, Mic } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { useState, useEffect } from 'react';

interface HeroSectionProps {
  onAnalyzeClick: () => void;
  language: string;
}

const translations: Record<string, any> = {
  en: {
    tagline: 'Fight misinformation.',
    taglineHighlight: 'Instantly.',
    subtitle: 'Verify content authenticity in seconds. Detect deepfakes, fake news, fake voice, and misleading information with advanced analysis.',
    placeholder: 'Paste text, link, or upload media to verify...',
    analyzeButton: 'Analyze Now',
    trySample: 'Try Sample',
    stats: [
      { value: '2M+', label: 'Verifications' },
      { value: '500K+', label: 'Users Trust Us' },
      { value: '99.2%', label: 'Accuracy' },
    ],
  },
  hi: {
    tagline: 'गलत सूचना से लड़ें।',
    taglineHighlight: 'तुरंत।',
    subtitle: 'सेकंडों में सामग्री की प्रामाणिकता सत्यापित करें। उन्नत विश्लेषण के साथ डीपफेक, फर्जी खबरें, नकली आवाज़ और भ्रामक जानकारी का पता लगाएं।',
    placeholder: 'सत्यापन के लिए टेक्स्ट, लिंक पेस्ट करें या मीडिया अपलोड करें...',
    analyzeButton: 'अभी विश्लेषण करें',
    trySample: 'नमूना आज़माएं',
    stats: [
      { value: '20 लाख+', label: 'सत्यापन' },
      { value: '5 लाख+', label: 'उपयोगकर्ता विश्वास' },
      { value: '99.2%', label: 'सटीकता' },
    ],
  },
  bn: {
    tagline: 'ভুল তথ্যের বিরুদ্ধে লড়াই করুন।',
    taglineHighlight: 'তাৎক্ষণিকভাবে।',
    subtitle: 'সেকেন্ডের মধ্যে বিষয়বস্তুর সত্যতা যাচাই করুন। উন্নত বিশ্লেষণের মাধ্যমে ডিপফেক, নকল খবর, নকল ভয়েস এবং বিভ্রান্তিকর তথ্য সনাক্ত করুন।',
    placeholder: 'যাচাই করার জন্য টেক্সট, লিংক পেস্ট করুন বা মিডিয়া আপলোড করুন...',
    analyzeButton: 'এখনই বিশ্লেষণ করুন',
    trySample: 'নমুনা চেষ্টা করুন',
    stats: [
      { value: '২০ লক্ষ+', label: 'যাচাইকরণ' },
      { value: '৫ লক্ষ+', label: 'ব্যবহারকারী বিশ্বাস' },
      { value: '৯৯.২%', label: 'নির্ভুলতা' },
    ],
  },
};

// Animated tagline cycling through languages
const animatedTaglines = [
  { lang: 'en', text: 'FIGHT MISINFORMATION', highlight: 'INSTANTLY.' },
  { lang: 'hi', text: 'गलत सूचना से लड़ें', highlight: 'तुरंत।' },
  { lang: 'bn', text: 'ভুল তথ্যের বিরুদ্ধে লড়াই', highlight: 'তাৎক্ষণিকভাবে।' },
];

export function HeroSection({ onAnalyzeClick, language }: HeroSectionProps) {
  const t = translations[language] || translations.en;
  const [currentTaglineIndex, setCurrentTaglineIndex] = useState(0);

  // Cycle through languages every 3 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTaglineIndex((prev: number) => (prev + 1) % animatedTaglines.length);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const currentTagline = animatedTaglines[currentTaglineIndex];

  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden pt-24 pb-16">
      {/* Animated Background */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-float" />
        <div className="absolute top-40 right-10 w-72 h-72 bg-purple-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-float" style={{ animationDelay: '2s' }} />
        <div className="absolute -bottom-8 left-1/2 w-72 h-72 bg-pink-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-float" style={{ animationDelay: '4s' }} />
      </div>

      <div className="container mx-auto px-4">
        <div className="max-w-5xl mx-auto">
          {/* Hero Text */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-6"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
            >
              <Sparkles className="w-4 h-4 text-yellow-500" />
              <span className="text-sm font-mono">Powered by Advanced Verification</span>
            </motion.div>

            <div className="mb-6 min-h-[280px] sm:min-h-[240px] flex flex-col items-center justify-center overflow-hidden px-4">
              <h1 className="text-center w-full">
                <AnimatePresence mode="wait">
                  <motion.span
                    key={currentTaglineIndex}
                    initial={{ opacity: 0, y: 20, rotateX: -90 }}
                    animate={{ opacity: 1, y: 0, rotateX: 0 }}
                    exit={{ opacity: 0, y: -20, rotateX: 90 }}
                    transition={{ duration: 0.5 }}
                    className="block mb-4"
                  >
                    {currentTagline.text}
                  </motion.span>
                </AnimatePresence>
                <AnimatePresence mode="wait">
                  <motion.span
                    key={`highlight-${currentTaglineIndex}`}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                    className="gradient-text block"
                  >
                    {currentTagline.highlight}
                  </motion.span>
                </AnimatePresence>
              </h1>
            </div>

            <motion.p
              className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto font-bitcount"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              {t.subtitle}
            </motion.p>
          </motion.div>

          {/* Search/Analyze Card */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.6 }}
            className="relative backdrop-blur-xl bg-white/80 dark:bg-slate-900/60 border border-gray-200 dark:border-slate-700/50 rounded-3xl p-6 sm:p-8 mb-12 shadow-2xl overflow-hidden"
          >
            {/* Gradient overlay */}
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-purple-500/10 to-pink-500/10 dark:from-blue-500/5 dark:via-purple-500/5 dark:to-pink-500/5 pointer-events-none" />
            
            {/* Glow effect */}
            <div className="absolute -top-24 -right-24 w-48 h-48 bg-blue-400/30 dark:bg-blue-500/20 rounded-full blur-3xl" />
            <div className="absolute -bottom-24 -left-24 w-48 h-48 bg-purple-400/30 dark:bg-purple-500/20 rounded-full blur-3xl" />

            <div className="relative z-10">
              <Tabs defaultValue="text" className="w-full">
                <TabsList className="grid w-full grid-cols-4 mb-6 bg-gray-100/80 dark:bg-slate-800/50 backdrop-blur-sm border border-gray-200 dark:border-slate-700/50 p-1 rounded-2xl">
                  <TabsTrigger 
                    value="text" 
                    className="gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-blue-500 data-[state=active]:text-white data-[state=active]:shadow-md text-gray-700 dark:text-gray-300 rounded-xl transition-all duration-300 font-medium"
                  >
                    <Search className="w-4 h-4" />
                    <span className="hidden sm:inline">Text</span>
                  </TabsTrigger>
                  <TabsTrigger 
                    value="image" 
                    className="gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-blue-500 data-[state=active]:text-white data-[state=active]:shadow-md text-gray-700 dark:text-gray-300 rounded-xl transition-all duration-300 font-medium"
                  >
                    <Image className="w-4 h-4" />
                    <span className="hidden sm:inline">Image</span>
                  </TabsTrigger>
                  <TabsTrigger 
                    value="video" 
                    className="gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-blue-500 data-[state=active]:text-white data-[state=active]:shadow-md text-gray-700 dark:text-gray-300 rounded-xl transition-all duration-300 font-medium"
                  >
                    <Video className="w-4 h-4" />
                    <span className="hidden sm:inline">Video</span>
                  </TabsTrigger>
                  <TabsTrigger 
                    value="voice" 
                    className="gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-blue-500 data-[state=active]:text-white data-[state=active]:shadow-md text-gray-700 dark:text-gray-300 rounded-xl transition-all duration-300 font-medium"
                  >
                    <Mic className="w-4 h-4" />
                    <span className="hidden sm:inline">Voice</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="text" className="space-y-4 mt-6">
                  <div className="relative">
                    <Input
                      placeholder={t.placeholder}
                      className="h-16 text-base sm:text-lg rounded-2xl bg-gray-50 dark:bg-slate-800/50 border-gray-300 dark:border-slate-700/50 text-gray-900 dark:text-white placeholder:text-gray-500 dark:placeholder:text-slate-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-300 pl-5 pr-5"
                    />
                  </div>
                  <div className="flex flex-col sm:flex-row gap-3">
                    <Button
                      onClick={onAnalyzeClick}
                      className="flex-1 h-14 rounded-2xl bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 hover:from-blue-500 hover:via-purple-500 hover:to-blue-500 text-white font-semibold text-lg shadow-lg hover:shadow-xl hover:shadow-blue-500/25 transition-all duration-300 hover:scale-[1.02]"
                    >
                      {t.analyzeButton}
                    </Button>
                    <Button 
                      variant="outline" 
                      className="sm:w-auto h-14 rounded-2xl border-2 border-gray-300 dark:border-slate-700/50 bg-white/50 dark:bg-slate-800/30 hover:bg-gray-100 dark:hover:bg-slate-800/60 hover:border-gray-400 dark:hover:border-slate-600 text-gray-700 dark:text-white font-medium transition-all duration-300"
                    >
                      {t.trySample}
                    </Button>
                  </div>
                </TabsContent>

                <TabsContent value="image" className="space-y-4 mt-6">
                  <div className="relative group">
                    <div className="border-2 border-dashed border-gray-300 dark:border-slate-700/50 rounded-2xl p-12 text-center hover:border-blue-500/70 dark:hover:border-blue-500/50 transition-all duration-300 cursor-pointer bg-gray-50 dark:bg-slate-800/20 hover:bg-gray-100 dark:hover:bg-slate-800/40 backdrop-blur-sm">
                      <div className="relative inline-block">
                        <Image className="w-14 h-14 mx-auto mb-4 text-gray-400 dark:text-slate-400 group-hover:text-blue-500 dark:group-hover:text-blue-400 transition-colors duration-300" />
                        <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                      </div>
                      <p className="text-gray-700 dark:text-slate-300 font-medium mb-2">
                        Drag & drop an image or click to upload
                      </p>
                      <p className="text-sm text-gray-500 dark:text-slate-500">
                        Supports JPG, PNG, WebP up to 10MB
                      </p>
                    </div>
                  </div>
                  <Button
                    onClick={onAnalyzeClick}
                    className="w-full h-14 rounded-2xl bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 hover:from-blue-500 hover:via-purple-500 hover:to-blue-500 text-white font-semibold text-lg shadow-lg hover:shadow-xl hover:shadow-blue-500/25 transition-all duration-300 hover:scale-[1.02]"
                  >
                    {t.analyzeButton}
                  </Button>
                </TabsContent>

                <TabsContent value="video" className="space-y-4 mt-6">
                  <div className="relative group">
                    <div className="border-2 border-dashed border-gray-300 dark:border-slate-700/50 rounded-2xl p-12 text-center hover:border-purple-500/70 dark:hover:border-purple-500/50 transition-all duration-300 cursor-pointer bg-gray-50 dark:bg-slate-800/20 hover:bg-gray-100 dark:hover:bg-slate-800/40 backdrop-blur-sm">
                      <div className="relative inline-block">
                        <Video className="w-14 h-14 mx-auto mb-4 text-gray-400 dark:text-slate-400 group-hover:text-purple-500 dark:group-hover:text-purple-400 transition-colors duration-300" />
                        <div className="absolute inset-0 bg-purple-500/20 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                      </div>
                      <p className="text-gray-700 dark:text-slate-300 font-medium mb-2">
                        Drag & drop a video or click to upload
                      </p>
                      <p className="text-sm text-gray-500 dark:text-slate-500">
                        Supports MP4, WebM, MOV up to 50MB
                      </p>
                    </div>
                  </div>
                  <Button
                    onClick={onAnalyzeClick}
                    className="w-full h-14 rounded-2xl bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 hover:from-blue-500 hover:via-purple-500 hover:to-blue-500 text-white font-semibold text-lg shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300 hover:scale-[1.02]"
                  >
                    {t.analyzeButton}
                  </Button>
                </TabsContent>

                <TabsContent value="voice" className="space-y-4 mt-6">
                  <div className="relative group">
                    <div className="border-2 border-dashed border-gray-300 dark:border-slate-700/50 rounded-2xl p-12 text-center hover:border-green-500/70 dark:hover:border-green-500/50 transition-all duration-300 cursor-pointer bg-gray-50 dark:bg-slate-800/20 hover:bg-gray-100 dark:hover:bg-slate-800/40 backdrop-blur-sm">
                      <div className="relative inline-block">
                        <Mic className="w-14 h-14 mx-auto mb-4 text-gray-400 dark:text-slate-400 group-hover:text-green-500 dark:group-hover:text-green-400 transition-colors duration-300" />
                        <div className="absolute inset-0 bg-green-500/20 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                      </div>
                      <p className="text-gray-700 dark:text-slate-300 font-medium mb-2">
                        Drag & drop an audio file or click to upload
                      </p>
                      <p className="text-sm text-gray-500 dark:text-slate-500">
                        Supports MP3, WAV, M4A up to 25MB
                      </p>
                    </div>
                  </div>
                  <Button
                    onClick={onAnalyzeClick}
                    className="w-full h-14 rounded-2xl bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 hover:from-blue-500 hover:via-purple-500 hover:to-blue-500 text-white font-semibold text-lg shadow-lg hover:shadow-xl hover:shadow-green-500/25 transition-all duration-300 hover:scale-[1.02]"
                  >
                    {t.analyzeButton}
                  </Button>
                </TabsContent>
              </Tabs>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="grid grid-cols-1 sm:grid-cols-3 gap-6"
          >
            {t.stats.map((stat: any, index: number) => (
              <motion.div
                key={index}
                className="glass-card rounded-2xl p-6 text-center"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 + index * 0.1 }}
                whileHover={{ scale: 1.05, y: -5 }}
              >
                <div className="font-display text-4xl mb-2 gradient-text">{stat.value}</div>
                <div className="text-gray-600 dark:text-gray-400">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
