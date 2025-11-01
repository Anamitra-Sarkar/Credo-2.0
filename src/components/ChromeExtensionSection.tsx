import { motion } from 'motion/react';
import { Chrome, Download, CheckCircle, Zap, Shield, Eye } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Badge } from './ui/badge';

export function ChromeExtensionSection() {
  const features = [
    {
      icon: Zap,
      title: 'Instant Verification',
      description: 'Right-click any content to verify instantly',
    },
    {
      icon: Shield,
      title: 'Real-time Protection',
      description: 'Automatic alerts on suspicious content',
    },
    {
      icon: Eye,
      title: 'Non-intrusive',
      description: 'Minimal UI that stays out of your way',
    },
  ];

  const steps = [
    'Click "Add to Chrome" button',
    'Confirm the installation',
    'Pin the extension to toolbar',
    'Start verifying content instantly',
  ];

  return (
    <div className="py-24 bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 dark:from-slate-900 dark:via-blue-950/50 dark:to-purple-950/50">
      <div className="container mx-auto px-4 max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <Badge className="mb-4 bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 border-blue-200 dark:border-blue-800">
            NEW RELEASE
          </Badge>
          <h2 className="mb-6">
            Verify Content Right in Your Browser
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Install our Chrome extension and get instant fact-checking while you browse the web
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 items-center mb-16">
          {/* Extension Preview */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <div className="relative">
              {/* Browser Chrome */}
              <div className="glass-card rounded-2xl overflow-hidden shadow-2xl">
                {/* Browser Header */}
                <div className="bg-gray-200 dark:bg-gray-800 p-3 flex items-center gap-2 border-b border-gray-300 dark:border-gray-700">
                  <div className="flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-red-500" />
                    <div className="w-3 h-3 rounded-full bg-yellow-500" />
                    <div className="w-3 h-3 rounded-full bg-green-500" />
                  </div>
                  <div className="flex-1 bg-white dark:bg-gray-900 rounded-lg px-3 py-1 text-sm text-gray-500 dark:text-gray-400">
                    example.com
                  </div>
                  <motion.div
                    className="bg-blue-600 text-white rounded-lg px-3 py-1 flex items-center gap-2"
                    animate={{ scale: [1, 1.05, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <Shield className="w-4 h-4" />
                    <span className="text-xs">VeriFy AI</span>
                  </motion.div>
                </div>

                {/* Content Area */}
                <div className="bg-white dark:bg-gray-900 p-6">
                  <div className="mb-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      "Breaking: New study reveals shocking health discovery..."
                    </p>
                  </div>

                  {/* Extension Popup */}
                  <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    whileInView={{ scale: 1, opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.5 }}
                    className="glass-card border-2 border-blue-500 rounded-xl p-4 shadow-lg"
                  >
                    <div className="flex items-start gap-3 mb-3">
                      <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-900 flex items-center justify-center flex-shrink-0">
                        <CheckCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                      </div>
                      <div className="flex-1">
                        <h4 className="text-sm mb-1">Likely False</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          This claim has been debunked by multiple fact-checkers
                        </p>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <button className="text-xs px-3 py-1 rounded-lg bg-blue-600 text-white">
                        View Details
                      </button>
                      <button className="text-xs px-3 py-1 rounded-lg bg-gray-200 dark:bg-gray-700">
                        Dismiss
                      </button>
                    </div>
                  </motion.div>
                </div>
              </div>

              {/* Floating Badge */}
              <motion.div
                className="absolute -right-4 -top-4 glass-card rounded-2xl p-4 shadow-xl"
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 3, repeat: Infinity }}
              >
                <Chrome className="w-12 h-12 text-blue-600" />
              </motion.div>
            </div>
          </motion.div>

          {/* Features & CTA */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="space-y-8"
          >
            {/* Features */}
            <div className="space-y-4">
              {features.map((feature, idx) => {
                const Icon = feature.icon;
                return (
                  <motion.div
                    key={feature.title}
                    initial={{ opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.4 + idx * 0.1 }}
                    className="flex items-start gap-4"
                  >
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <h4 className="mb-1">{feature.title}</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {feature.description}
                      </p>
                    </div>
                  </motion.div>
                );
              })}
            </div>

            {/* Download CTA */}
            <div className="glass-card rounded-2xl p-6 border-2 border-blue-200 dark:border-blue-800">
              <div className="flex items-center gap-3 mb-4">
                <Chrome className="w-8 h-8 text-blue-600" />
                <div>
                  <h4>Chrome Extension</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Free • 50K+ users</p>
                </div>
              </div>
              <Button className="w-full h-12 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 gap-2">
                <Download className="w-5 h-5" />
                Add to Chrome - It's Free
              </Button>
              <p className="text-xs text-gray-500 text-center mt-3">
                Works with Chrome, Edge, Brave, and other Chromium browsers
              </p>
            </div>
          </motion.div>
        </div>

        {/* How to Install */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
        >
          <h3 className="text-center mb-8">How to Install</h3>
          <div className="grid md:grid-cols-4 gap-6">
            {steps.map((step, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.6 + idx * 0.1 }}
                className="relative"
              >
                <Card className="glass-card p-6 text-center h-full">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center mx-auto mb-4">
                    <span className="font-display text-xl text-white">{idx + 1}</span>
                  </div>
                  <p className="text-sm">{step}</p>
                </Card>
                {idx < steps.length - 1 && (
                  <div className="hidden md:block absolute top-1/2 -right-3 w-6 h-0.5 bg-gradient-to-r from-blue-300 to-purple-300 dark:from-blue-700 dark:to-purple-700" />
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
