import { motion } from 'motion/react';
import { Shield, Target, Users, Globe, Zap, Lock } from 'lucide-react';
import { Card } from './ui/card';

export function AboutPage() {
  const features = [
    {
      icon: Shield,
      title: 'Advanced Verification',
      description: 'Multi-layered analysis combining pattern recognition, source verification, and cross-referencing with trusted databases.',
    },
    {
      icon: Zap,
      title: 'Instant Results',
      description: 'Get verification results in seconds with detailed explanations and confidence scores.',
    },
    {
      icon: Globe,
      title: 'Multilingual Support',
      description: 'Analyze content in multiple Indian languages including Hindi, Bengali, Tamil, Marathi, and more.',
    },
    {
      icon: Users,
      title: 'Community-Driven',
      description: 'Join thousands of truth guardians helping to combat misinformation across the nation.',
    },
    {
      icon: Target,
      title: 'Precision Accuracy',
      description: 'Our platform maintains 99%+ accuracy through continuous learning and expert validation.',
    },
    {
      icon: Lock,
      title: 'Privacy First',
      description: 'Your data is encrypted and never shared. We prioritize your security and privacy.',
    },
  ];

  const stats = [
    { value: '2M+', label: 'Content Verified' },
    { value: '500K+', label: 'Active Users' },
    { value: '99.2%', label: 'Accuracy Rate' },
    { value: '15+', label: 'Languages' },
  ];

  return (
    <div className="min-h-screen pt-28 pb-16">
      <div className="container mx-auto px-4 max-w-6xl">
        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-3 mb-6">
            <Shield className="w-12 h-12 text-blue-600" />
            <h2>VeriFy AI</h2>
          </div>
          <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto mb-8">
            Our mission is to empower everyone with the tools to identify and combat misinformation, 
            creating a more informed and truthful digital society.
          </p>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16"
        >
          {stats.map((stat, idx) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 + idx * 0.1 }}
              className="glass-card rounded-2xl p-6 text-center"
            >
              <div className="font-display text-4xl mb-2 gradient-text">{stat.value}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>

        {/* Features */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mb-16"
        >
          <h3 className="text-center mb-12">Why Choose VeriFy AI?</h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, idx) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 + idx * 0.1 }}
                  whileHover={{ y: -5 }}
                >
                  <Card className="glass-card p-6 h-full">
                    <Icon className="w-10 h-10 text-blue-600 mb-4" />
                    <h4 className="mb-3">{feature.title}</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {feature.description}
                    </p>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* How It Works */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mb-16"
        >
          <h3 className="text-center mb-12">How It Works</h3>
          <div className="max-w-4xl mx-auto">
            <div className="space-y-8">
              {[
                {
                  step: '01',
                  title: 'Submit Content',
                  description: 'Upload or paste any text, image, or video that you want to verify.',
                },
                {
                  step: '02',
                  title: 'Advanced Analysis',
                  description: 'Our platform analyzes the content using multiple verification layers including source checking, pattern recognition, and cross-referencing.',
                },
                {
                  step: '03',
                  title: 'Instant Verdict',
                  description: 'Receive a clear verdict with confidence score and detailed explanation of how we reached the conclusion.',
                },
                {
                  step: '04',
                  title: 'Learn & Share',
                  description: 'Access related sources, educate others, and contribute to our community-driven fact-checking effort.',
                },
              ].map((item, idx) => (
                <motion.div
                  key={item.step}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.7 + idx * 0.1 }}
                  className="flex gap-6"
                >
                  <div className="flex-shrink-0">
                    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                      <span className="font-display text-xl text-white">{item.step}</span>
                    </div>
                  </div>
                  <div className="flex-1 glass-card rounded-xl p-6">
                    <h4 className="mb-2">{item.title}</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {item.description}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1 }}
          className="glass-card rounded-3xl p-12 text-center bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950"
        >
          <h3 className="mb-4">Join the Fight Against Misinformation</h3>
          <p className="text-lg text-gray-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto">
            Every verification helps make the digital world more truthful. Start analyzing content today 
            and become part of our community of truth guardians.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <button className="px-8 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700 transition-colors">
              Get Started
            </button>
            <button className="px-8 py-3 rounded-xl glass-card hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
              Learn More
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
