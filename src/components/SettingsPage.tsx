import { useState, useRef } from 'react';
import { motion } from 'motion/react';
import { Camera, Save, Upload, Trash2, Shield, Mail, Calendar } from 'lucide-react';
import { Button } from './ui/button';
import { useAuth } from '../context/AuthContext';
import { toast } from 'sonner';

const AVATAR_PRESETS = [
  { id: 1, emoji: '😊', label: 'Happy' },
  { id: 2, emoji: '😎', label: 'Cool' },
  { id: 3, emoji: '🤓', label: 'Nerdy' },
  { id: 4, emoji: '🥳', label: 'Party' },
  { id: 5, emoji: '🤖', label: 'Robot' },
  { id: 6, emoji: '👨‍💻', label: 'Developer' },
  { id: 7, emoji: '👩‍🔬', label: 'Scientist' },
  { id: 8, emoji: '🦸', label: 'Hero' },
  { id: 9, emoji: '🧙', label: 'Wizard' },
  { id: 10, emoji: '🐱', label: 'Cat' },
  { id: 11, emoji: '🐶', label: 'Dog' },
  { id: 12, emoji: '🦊', label: 'Fox' },
  { id: 13, emoji: '🐼', label: 'Panda' },
  { id: 14, emoji: '🦁', label: 'Lion' },
  { id: 15, emoji: '🐺', label: 'Wolf' },
  { id: 16, emoji: '🦄', label: 'Unicorn' },
];

const GRADIENT_COLORS = [
  { id: 1, name: 'Ocean', colors: ['#6366f1', '#8b5cf6'] },
  { id: 2, name: 'Sunset', colors: ['#f97316', '#ec4899'] },
  { id: 3, name: 'Forest', colors: ['#10b981', '#14b8a6'] },
  { id: 4, name: 'Fire', colors: ['#ef4444', '#f59e0b'] },
  { id: 5, name: 'Sky', colors: ['#3b82f6', '#06b6d4'] },
  { id: 6, name: 'Purple', colors: ['#8b5cf6', '#d946ef'] },
  { id: 7, name: 'Rose', colors: ['#e11d48', '#f472b6'] },
  { id: 8, name: 'Emerald', colors: ['#059669', '#10b981'] },
];

interface SettingsPageProps {
  user: any;
}

export function SettingsPage({ user: propUser }: SettingsPageProps) {
  const { user: authUser } = useAuth();
  const user = propUser || authUser;
  const [customImage, setCustomImage] = useState<string | null>(
    user ? localStorage.getItem(`user_avatar_${user.uid}`) : null
  );
  const [selectedEmoji, setSelectedEmoji] = useState<string | null>(
    user ? localStorage.getItem(`user_avatar_emoji_${user.uid}`) : null
  );
  const [selectedGradient, setSelectedGradient] = useState<number>(
    user ? parseInt(localStorage.getItem(`user_avatar_gradient_${user.uid}`) || '1') : 1
  );
  const [avatarType, setAvatarType] = useState<'image' | 'emoji' | 'gradient'>(
    user ? (localStorage.getItem(`user_avatar_type_${user.uid}`) as any) || 'gradient' : 'gradient'
  );

  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!user) {
    return (
      <div className="min-h-screen pt-28 pb-16 px-4 flex items-center justify-center">
        <div className="text-center">
          <Shield className="w-16 h-16 mx-auto mb-4 text-gray-400" />
          <h2 className="text-2xl font-bold mb-2">Please Log In</h2>
          <p className="text-gray-600 dark:text-gray-400">
            You need to be logged in to access settings
          </p>
        </div>
      </div>
    );
  }

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast.error('Please select a valid image file');
      return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      toast.error('Image size must be less than 5MB');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const imageData = event.target?.result as string;
      setCustomImage(imageData);
      setAvatarType('image');
      toast.success('Image uploaded successfully');
    };
    reader.readAsDataURL(file);
  };

  const handleEmojiSelect = (emoji: string) => {
    setSelectedEmoji(emoji);
    setAvatarType('emoji');
  };

  const handleGradientSelect = (id: number) => {
    setSelectedGradient(id);
    setAvatarType('gradient');
  };

  const handleSave = () => {
    if (!user) return;

    localStorage.setItem(`user_avatar_type_${user.uid}`, avatarType);

    if (avatarType === 'image' && customImage) {
      localStorage.setItem(`user_avatar_${user.uid}`, customImage);
    } else {
      localStorage.removeItem(`user_avatar_${user.uid}`);
    }

    if (avatarType === 'emoji' && selectedEmoji) {
      localStorage.setItem(`user_avatar_emoji_${user.uid}`, selectedEmoji);
    } else {
      localStorage.removeItem(`user_avatar_emoji_${user.uid}`);
    }

    if (avatarType === 'gradient') {
      localStorage.setItem(`user_avatar_gradient_${user.uid}`, selectedGradient.toString());
    }

    toast.success('Avatar settings saved! Refresh to see changes.');
    
    // Trigger a custom event to update the header
    window.dispatchEvent(new Event('avatarUpdated'));
  };

  const handleRemoveImage = () => {
    setCustomImage(null);
    setAvatarType('gradient');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getPreviewAvatar = () => {
    if (avatarType === 'image' && customImage) {
      return (
        <img
          src={customImage}
          alt="Avatar Preview"
          className="w-full h-full object-cover"
        />
      );
    }

    if (avatarType === 'emoji' && selectedEmoji) {
      return (
        <div className="w-full h-full flex items-center justify-center text-6xl">
          {selectedEmoji}
        </div>
      );
    }

    // Gradient avatar
    const gradient = GRADIENT_COLORS.find(g => g.id === selectedGradient) || GRADIENT_COLORS[0];
    const initials = user.displayName
      ? user.displayName.split(' ').map((n: string) => n[0]).join('').toUpperCase().slice(0, 2)
      : user.email?.slice(0, 2).toUpperCase();

    return (
      <div
        className="w-full h-full flex items-center justify-center text-4xl font-bold text-white"
        style={{
          background: `linear-gradient(135deg, ${gradient.colors[0]}, ${gradient.colors[1]})`
        }}
      >
        {initials}
      </div>
    );
  };

  return (
    <div className="min-h-screen pt-28 pb-16 px-4">
      <div className="container mx-auto max-w-4xl">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-5xl font-display font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 text-transparent bg-clip-text">
            Profile Settings
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400">
            Customize your avatar and manage your account
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Profile Info Sidebar */}
          <div className="lg:col-span-1">
            <div className="glass-card p-6 sticky top-32">
              <div className="text-center">
                {/* Avatar Preview */}
                <div className="w-32 h-32 rounded-full overflow-hidden mx-auto mb-4 shadow-2xl ring-4 ring-gray-200 dark:ring-gray-700">
                  {getPreviewAvatar()}
                </div>

                <h3 className="text-xl font-bold mb-1">
                  {user.displayName || 'User'}
                </h3>
                <p className="text-sm text-gray-500 mb-4 break-all">
                  {user.email}
                </p>

                {/* Account Info */}
                <div className="space-y-3 text-sm text-left border-t border-gray-200 dark:border-gray-700 pt-4">
                  <div className="flex items-center gap-2">
                    <Mail className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-600 dark:text-gray-400">Email:</span>
                    <span className={user.emailVerified ? 'text-green-600' : 'text-yellow-600'}>
                      {user.emailVerified ? 'Verified' : 'Not Verified'}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-600 dark:text-gray-400">Member since:</span>
                    <span>{new Date().toLocaleDateString()}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Shield className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-600 dark:text-gray-400">User ID:</span>
                    <span className="text-xs truncate">{user.uid.slice(0, 8)}...</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Avatar Customization */}
          <div className="lg:col-span-2 space-y-6">
            {/* Upload Custom Image */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="glass-card p-6"
            >
              <h2 className="text-2xl font-bold mb-4 flex items-center">
                <Camera className="w-6 h-6 mr-2" />
                Custom Image
              </h2>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />

              <div className="space-y-4">
                {customImage ? (
                  <div className="relative">
                    <img
                      src={customImage}
                      alt="Custom Avatar"
                      className="w-32 h-32 rounded-full object-cover mx-auto shadow-lg"
                    />
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleRemoveImage}
                      className="absolute top-0 right-1/2 translate-x-16"
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      Remove
                    </Button>
                  </div>
                ) : (
                  <Button
                    onClick={() => fileInputRef.current?.click()}
                    variant="outline"
                    className="w-full py-8"
                  >
                    <Upload className="w-6 h-6 mr-2" />
                    Upload Custom Image (Max 5MB)
                  </Button>
                )}
              </div>
            </motion.div>

            {/* Emoji Avatars */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="glass-card p-6"
            >
              <h2 className="text-2xl font-bold mb-4">Emoji Avatars</h2>
              <div className="grid grid-cols-8 gap-2">
                {AVATAR_PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => handleEmojiSelect(preset.emoji)}
                    className={`aspect-square rounded-lg text-3xl hover:scale-110 transition-transform ${
                      avatarType === 'emoji' && selectedEmoji === preset.emoji
                        ? 'ring-4 ring-blue-600 bg-blue-50 dark:bg-blue-900/20'
                        : 'hover:bg-gray-100 dark:hover:bg-gray-800'
                    }`}
                    title={preset.label}
                  >
                    {preset.emoji}
                  </button>
                ))}
              </div>
            </motion.div>

            {/* Gradient Avatars */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-card p-6"
            >
              <h2 className="text-2xl font-bold mb-4">Gradient Backgrounds</h2>
              <div className="grid grid-cols-4 gap-4">
                {GRADIENT_COLORS.map((gradient) => (
                  <button
                    key={gradient.id}
                    onClick={() => handleGradientSelect(gradient.id)}
                    className={`aspect-square rounded-xl transition-transform hover:scale-105 ${
                      avatarType === 'gradient' && selectedGradient === gradient.id
                        ? 'ring-4 ring-blue-600'
                        : ''
                    }`}
                    style={{
                      background: `linear-gradient(135deg, ${gradient.colors[0]}, ${gradient.colors[1]})`
                    }}
                    title={gradient.name}
                  >
                    <span className="text-white font-bold text-sm">{gradient.name}</span>
                  </button>
                ))}
              </div>
            </motion.div>

            {/* Save Button */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <Button
                onClick={handleSave}
                className="w-full py-6 text-lg bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
              >
                <Save className="w-5 h-5 mr-2" />
                Save Avatar Settings
              </Button>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
