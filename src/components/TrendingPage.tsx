import { motion } from 'motion/react';
import { TrendingUp, MapPin, Clock, Users, AlertCircle, Eye } from 'lucide-react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface TrendingPageProps {
  language: string;
}

const trendingStories = [
  {
    id: 1,
    headline: 'Viral video claims government policy change',
    region: 'Maharashtra',
    reports: 1247,
    status: 'fake',
    timeAgo: '2 hours ago',
    engagement: '45K views',
  },
  {
    id: 2,
    headline: 'Celebrity endorsement screenshot circulating',
    region: 'Karnataka',
    reports: 892,
    status: 'deepfake',
    timeAgo: '5 hours ago',
    engagement: '32K views',
  },
  {
    id: 3,
    headline: 'Health remedy claims on WhatsApp',
    region: 'West Bengal',
    reports: 2156,
    status: 'fake',
    timeAgo: '1 day ago',
    engagement: '78K views',
  },
  {
    id: 4,
    headline: 'Political statement attributed to leader',
    region: 'Tamil Nadu',
    reports: 654,
    status: 'unverified',
    timeAgo: '3 hours ago',
    engagement: '23K views',
  },
  {
    id: 5,
    headline: 'Natural disaster warning message',
    region: 'Delhi',
    reports: 3421,
    status: 'fake',
    timeAgo: '30 minutes ago',
    engagement: '102K views',
  },
  {
    id: 6,
    headline: 'Government benefit scheme announcement',
    region: 'Gujarat',
    reports: 567,
    status: 'verified',
    timeAgo: '6 hours ago',
    engagement: '18K views',
  },
];

const regionalHotspots = [
  { region: 'Maharashtra', count: 3421, intensity: 95 },
  { region: 'West Bengal', count: 2890, intensity: 82 },
  { region: 'Karnataka', count: 2156, intensity: 75 },
  { region: 'Tamil Nadu', count: 1847, intensity: 68 },
  { region: 'Delhi', count: 1523, intensity: 62 },
  { region: 'Gujarat', count: 1204, intensity: 55 },
];

export function TrendingPage({ language }: TrendingPageProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'fake':
        return 'bg-red-100 text-red-700 border-red-300';
      case 'deepfake':
        return 'bg-orange-100 text-orange-700 border-orange-300';
      case 'unverified':
        return 'bg-yellow-100 text-yellow-700 border-yellow-300';
      case 'verified':
        return 'bg-green-100 text-green-700 border-green-300';
      default:
        return 'bg-gray-100 text-gray-700 border-gray-300';
    }
  };

  return (
    <div className="min-h-screen pt-28 pb-16">
      <div className="container mx-auto px-4 max-w-7xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <div className="flex items-center gap-3 mb-4">
            <TrendingUp className="w-8 h-8 text-blue-600" />
            <h2>Trending Misinformation</h2>
          </div>
          <p className="text-xl text-gray-600 dark:text-gray-400">
            Real-time tracking of viral misinformation across regions
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8 mb-12">
          {/* Heatmap Visualization */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2"
          >
            <Card className="glass-card p-6">
              <h3 className="mb-6">Regional Activity Heatmap</h3>
              
              {/* Map Background */}
              <div className="relative rounded-2xl overflow-hidden mb-6 h-96 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950">
                <ImageWithFallback
                  src="https://images.unsplash.com/photo-1570106413982-7f2897b8d0c5?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx3b3JsZCUyMG1hcCUyMG5ldHdvcmt8ZW58MXx8fHwxNzYxODA1NDg2fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
                  alt="Map background"
                  className="w-full h-full object-cover opacity-30"
                />
                
                {/* Animated Hotspots */}
                <div className="absolute inset-0 flex items-center justify-center">
                  {[
                    { top: '30%', left: '45%', size: 80 },
                    { top: '50%', left: '55%', size: 60 },
                    { top: '40%', left: '35%', size: 70 },
                    { top: '60%', left: '50%', size: 50 },
                  ].map((spot, idx) => (
                    <motion.div
                      key={idx}
                      className="absolute"
                      style={{ top: spot.top, left: spot.left }}
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 0.6 }}
                      transition={{ delay: idx * 0.2, duration: 0.5 }}
                    >
                      <motion.div
                        className="rounded-full bg-red-500"
                        style={{ width: spot.size, height: spot.size }}
                        animate={{
                          scale: [1, 1.2, 1],
                          opacity: [0.6, 0.3, 0.6],
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          delay: idx * 0.3,
                        }}
                      />
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Regional Stats */}
              <div className="grid grid-cols-2 gap-3">
                {regionalHotspots.map((hotspot, idx) => (
                  <motion.div
                    key={hotspot.region}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 + idx * 0.1 }}
                    className="glass-card rounded-xl p-3 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors cursor-pointer"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <MapPin className="w-4 h-4 text-red-500" />
                        <span className="text-sm">{hotspot.region}</span>
                      </div>
                      <span className="text-sm text-gray-500">{hotspot.count}</span>
                    </div>
                    <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-red-500 to-orange-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${hotspot.intensity}%` }}
                        transition={{ delay: 0.8 + idx * 0.1, duration: 0.8 }}
                      />
                    </div>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>

          {/* Live Stats */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="space-y-6"
          >
            <Card className="glass-card p-6">
              <h4 className="mb-4">Live Statistics</h4>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Active Reports</span>
                    <motion.span
                      className="font-display text-2xl"
                      initial={{ scale: 1 }}
                      animate={{ scale: [1, 1.1, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      12,487
                    </motion.span>
                  </div>
                  <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-blue-600"
                      animate={{ width: ['70%', '75%', '70%'] }}
                      transition={{ duration: 3, repeat: Infinity }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Verified Today</span>
                    <span className="font-display text-2xl">8,234</span>
                  </div>
                  <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div className="h-full bg-green-600" style={{ width: '66%' }} />
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Flagged as Fake</span>
                    <span className="font-display text-2xl">4,253</span>
                  </div>
                  <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div className="h-full bg-red-600" style={{ width: '34%' }} />
                  </div>
                </div>
              </div>
            </Card>

            <Card className="glass-card p-6">
              <h4 className="mb-4">Top Categories</h4>
              <div className="space-y-3">
                {['Politics', 'Health', 'Technology', 'Entertainment'].map((category, idx) => (
                  <div key={category} className="flex items-center justify-between">
                    <span className="text-sm">{category}</span>
                    <Badge variant="outline">{Math.floor(Math.random() * 3000) + 500}</Badge>
                  </div>
                ))}
              </div>
            </Card>
          </motion.div>
        </div>

        {/* Trending Stories Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h3 className="mb-6">Top Trending Stories</h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {trendingStories.map((story, idx) => (
              <motion.div
                key={story.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 + idx * 0.1 }}
                whileHover={{ y: -5 }}
              >
                <Card className="glass-card p-6 h-full hover:shadow-xl transition-shadow cursor-pointer">
                  <div className="flex items-start justify-between mb-3">
                    <Badge className={`${getStatusColor(story.status)} border`}>
                      {story.status.toUpperCase()}
                    </Badge>
                    <div className="flex items-center gap-1 text-sm text-gray-500">
                      <AlertCircle className="w-4 h-4" />
                      <span>{story.reports}</span>
                    </div>
                  </div>

                  <h4 className="mb-4 line-clamp-2">{story.headline}</h4>

                  <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                    <div className="flex items-center gap-2">
                      <MapPin className="w-4 h-4" />
                      <span>{story.region}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      <span>{story.timeAgo}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Eye className="w-4 h-4" />
                      <span>{story.engagement}</span>
                    </div>
                  </div>

                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <button className="text-sm text-blue-600 hover:text-blue-700">
                      View Details →
                    </button>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
