import { motion } from 'motion/react';
import { User, Mail, Calendar, Award, TrendingUp, CheckCircle, Settings } from 'lucide-react';
import { Card } from './ui/card';
import { Avatar, AvatarFallback } from './ui/avatar';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

interface ProfilePageProps {
  user: any;
}

export function ProfilePage({ user }: ProfilePageProps) {
  const userStats = {
    totalVerifications: 127,
    accuracy: 94,
    rank: 247,
    level: 5,
    xp: 2340,
    xpToNextLevel: 5000,
    streak: 14,
  };

  const recentActivity = [
    { type: 'verified', content: 'Fact-checked political claim', date: '2 hours ago', status: 'fake' },
    { type: 'verified', content: 'Health remedy verification', date: '5 hours ago', status: 'true' },
    { type: 'badge', content: 'Earned "Rising Star" badge', date: '1 day ago', status: 'achievement' },
    { type: 'verified', content: 'Video deepfake detection', date: '2 days ago', status: 'deepfake' },
  ];

  const achievements = [
    { name: 'Rising Star', earned: true, description: 'Complete 100 verifications' },
    { name: 'Accuracy Expert', earned: true, description: 'Maintain 90%+ accuracy' },
    { name: 'Community Helper', earned: false, description: 'Help 50 users in discussions' },
    { name: 'Streak Master', earned: false, description: 'Maintain a 30-day streak' },
  ];

  return (
    <div className="min-h-screen pt-28 pb-16">
      <div className="container mx-auto px-4 max-w-6xl">
        {/* Profile Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card rounded-3xl p-8 mb-8"
        >
          <div className="flex flex-col md:flex-row items-center md:items-start gap-6">
            <Avatar className="w-24 h-24">
              <AvatarFallback className="bg-gradient-to-br from-blue-500 to-purple-600 text-white text-3xl">
                {user?.avatar || 'U'}
              </AvatarFallback>
            </Avatar>

            <div className="flex-1 text-center md:text-left">
              <h2 className="mb-2">{user?.name || 'User'}</h2>
              <p className="text-gray-600 dark:text-gray-400 mb-4">{user?.email || 'user@example.com'}</p>
              
              <div className="flex flex-wrap gap-2 justify-center md:justify-start mb-4">
                <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300">
                  Level {userStats.level}
                </Badge>
                <Badge className="bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300">
                  Rank #{userStats.rank}
                </Badge>
                <Badge className="bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300">
                  {userStats.streak} Day Streak 🔥
                </Badge>
              </div>

              <div className="max-w-md mx-auto md:mx-0">
                <div className="flex justify-between text-sm mb-2">
                  <span>Level {userStats.level}</span>
                  <span className="text-gray-500">{userStats.xp} / {userStats.xpToNextLevel} XP</span>
                </div>
                <Progress value={(userStats.xp / userStats.xpToNextLevel) * 100} className="h-2" />
              </div>
            </div>

            <Button variant="outline" className="gap-2">
              <Settings className="w-4 h-4" />
              Edit Profile
            </Button>
          </div>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
        >
          {[
            { icon: CheckCircle, label: 'Verifications', value: userStats.totalVerifications, color: 'text-blue-600' },
            { icon: TrendingUp, label: 'Accuracy', value: `${userStats.accuracy}%`, color: 'text-green-600' },
            { icon: Award, label: 'Achievements', value: '2/4', color: 'text-purple-600' },
            { icon: Calendar, label: 'Member Since', value: 'Jan 2025', color: 'text-orange-600' },
          ].map((stat, idx) => {
            const Icon = stat.icon;
            return (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 + idx * 0.05 }}
              >
                <Card className="glass-card p-6 text-center">
                  <Icon className={`w-8 h-8 mx-auto mb-3 ${stat.color}`} />
                  <div className="font-display text-2xl mb-1">{stat.value}</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">{stat.label}</div>
                </Card>
              </motion.div>
            );
          })}
        </motion.div>

        {/* Tabs */}
        <Tabs defaultValue="activity" className="mb-8">
          <TabsList className="grid w-full max-w-md mx-auto grid-cols-2">
            <TabsTrigger value="activity">Recent Activity</TabsTrigger>
            <TabsTrigger value="achievements">Achievements</TabsTrigger>
          </TabsList>

          <TabsContent value="activity" className="mt-6">
            <Card className="glass-card p-6">
              <h3 className="mb-6">Recent Activity</h3>
              <div className="space-y-4">
                {recentActivity.map((activity, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className="flex items-start gap-4 p-4 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
                  >
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
                      activity.status === 'fake' ? 'bg-red-100 dark:bg-red-900/30' :
                      activity.status === 'true' ? 'bg-green-100 dark:bg-green-900/30' :
                      activity.status === 'deepfake' ? 'bg-orange-100 dark:bg-orange-900/30' :
                      'bg-purple-100 dark:bg-purple-900/30'
                    }`}>
                      <CheckCircle className={`w-5 h-5 ${
                        activity.status === 'fake' ? 'text-red-600 dark:text-red-400' :
                        activity.status === 'true' ? 'text-green-600 dark:text-green-400' :
                        activity.status === 'deepfake' ? 'text-orange-600 dark:text-orange-400' :
                        'text-purple-600 dark:text-purple-400'
                      }`} />
                    </div>
                    <div className="flex-1">
                      <p className="mb-1">{activity.content}</p>
                      <p className="text-sm text-gray-500">{activity.date}</p>
                    </div>
                    {activity.status !== 'achievement' && (
                      <Badge className={
                        activity.status === 'fake' ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300' :
                        activity.status === 'true' ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300' :
                        'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300'
                      }>
                        {activity.status}
                      </Badge>
                    )}
                  </motion.div>
                ))}
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="achievements" className="mt-6">
            <div className="grid md:grid-cols-2 gap-4">
              {achievements.map((achievement, idx) => (
                <motion.div
                  key={achievement.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                >
                  <Card className={`glass-card p-6 ${
                    achievement.earned 
                      ? 'border-2 border-green-300 dark:border-green-700' 
                      : 'opacity-60'
                  }`}>
                    <div className="flex items-start gap-4">
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 ${
                        achievement.earned 
                          ? 'bg-gradient-to-br from-yellow-400 to-yellow-600' 
                          : 'bg-gray-200 dark:bg-gray-700'
                      }`}>
                        <Award className={`w-6 h-6 ${achievement.earned ? 'text-white' : 'text-gray-400'}`} />
                      </div>
                      <div className="flex-1">
                        <h4 className="mb-1">{achievement.name}</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {achievement.description}
                        </p>
                        {achievement.earned && (
                          <Badge className="mt-2 bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300">
                            Unlocked
                          </Badge>
                        )}
                      </div>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
