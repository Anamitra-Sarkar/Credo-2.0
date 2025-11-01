import { useState } from 'react';
import { motion } from 'motion/react';
import { Moon, Sun, Menu, X, Globe, User, LogOut, Settings } from 'lucide-react';
import { Button } from './ui/button';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from './ui/dropdown-menu';

interface HeaderProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  language: string;
  onLanguageChange: (lang: string) => void;
  isDark: boolean;
  onThemeToggle: () => void;
  user: any;
  onLoginClick: () => void;
  onLogout: () => void;
}

const languages = [
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'हिंदी' },
  { code: 'bn', name: 'বাংলা' },
  { code: 'ta', name: 'தமிழ்' },
  { code: 'mr', name: 'मराठी' },
  { code: 'te', name: 'తెలుగు' },
];

// Generate gradient avatar based on user ID
const generateAvatarGradient = (userId: string) => {
  const colors = [
    'from-blue-500 to-purple-600',
    'from-green-500 to-teal-600',
    'from-orange-500 to-red-600',
    'from-pink-500 to-rose-600',
    'from-indigo-500 to-blue-600',
    'from-yellow-500 to-orange-600',
  ];
  const index = userId ? userId.charCodeAt(0) % colors.length : 0;
  return colors[index];
};

// Get user initials
const getUserInitials = (user: any) => {
  if (!user) return '?';
  const name = user.displayName || user.name || user.email || 'User';
  const parts = name.split(' ');
  if (parts.length >= 2) {
    return (parts[0][0] + parts[1][0]).toUpperCase();
  }
  return name.substring(0, 2).toUpperCase();
};

export function HeaderWithSettings({ currentPage, onNavigate, language, onLanguageChange, isDark, onThemeToggle, user, onLoginClick, onLogout }: HeaderProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = ['Home', 'Analyze', 'Trending', 'Community', 'Settings', 'About'];

  // Get user avatar from localStorage or generate default
  const getUserAvatar = () => {
    if (user) {
      const savedAvatar = localStorage.getItem(`avatar_${user.uid || user.id || user.email}`);
      if (savedAvatar) {
        const avatarData = JSON.parse(savedAvatar);
        if (avatarData.type === 'image' && avatarData.url) {
          return <AvatarImage src={avatarData.url} alt="User avatar" />;
        } else if (avatarData.type === 'emoji') {
          return <div className="text-2xl">{avatarData.emoji}</div>;
        }
      }
      // Fallback to initials with gradient
      const gradient = generateAvatarGradient(user.uid || user.id || user.email);
      return (
        <AvatarFallback className={`bg-gradient-to-br ${gradient} text-white font-semibold`}>
          {getUserInitials(user)}
        </AvatarFallback>
      );
    }
    return null;
  };

  return (
    <motion.header
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="fixed top-0 left-0 right-0 z-50 glass-card border-b"
    >
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <motion.div
            className="flex items-center gap-3 cursor-pointer"
            onClick={() => onNavigate('Home')}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <div className="relative">
              <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L4 6V11C4 16.55 7.84 21.74 13 23C18.16 21.74 22 16.55 22 11V6L12 2Z"      
                      stroke="#3B82F6"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      fill="none"/>
              </svg>
              <motion.div
                className="absolute inset-0 bg-blue-600 rounded-full opacity-20 blur-xl"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            </div>
            <span className="font-display text-xl md:text-2xl">VeriFy AI</span>
          </motion.div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => (
              <motion.button
                key={item}
                onClick={() => onNavigate(item)}
                className={`px-4 py-2 rounded-lg transition-colors font-bitcount ${
                  currentPage === item
                    ? 'bg-blue-600 text-white'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {item}
              </motion.button>
            ))}
          </nav>

          {/* Right Actions */}
          <div className="flex items-center gap-3">
            {/* Language Selector */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon" className="relative hidden sm:flex">
                  <Globe className="h-5 w-5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="glass-card">
                {languages.map((lang) => (
                  <DropdownMenuItem
                    key={lang.code}
                    onClick={() => onLanguageChange(lang.code)}
                    className={language === lang.code ? 'bg-blue-100 dark:bg-blue-900' : ''}
                  >
                    {lang.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Theme Toggle */}
            <Button
              variant="outline"
              size="icon"
              onClick={onThemeToggle}
              className="relative overflow-hidden hidden sm:flex"
            >
              <motion.div
                initial={false}
                animate={{ rotate: isDark ? 180 : 0 }}
                transition={{ duration: 0.3 }}
              >
                {isDark ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
              </motion.div>
            </Button>

            {/* User Menu / Login */}
            {user ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="icon" className="relative">
                    <Avatar className="w-8 h-8">
                      {getUserAvatar()}
                    </Avatar>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="glass-card w-56">
                  <div className="px-2 py-2">
                    <p className="font-medium">{user.displayName || user.name || 'User'}</p>
                    <p className="text-sm text-gray-500 truncate">{user.email}</p>
                  </div>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => onNavigate('Profile')}>
                    <User className="mr-2 h-4 w-4" />
                    Profile
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => onNavigate('Settings')}>
                    <Settings className="mr-2 h-4 w-4" />
                    Settings
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={onLogout} className="text-red-600 dark:text-red-400">
                    <LogOut className="mr-2 h-4 w-4" />
                    Logout
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <Button onClick={onLoginClick} className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
                <User className="mr-2 h-4 w-4" />
                Login
              </Button>
            )}

            {/* Mobile Menu Button */}
            <Button
              variant="outline"
              size="icon"
              className="md:hidden"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <motion.nav
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden mt-4 pb-4 border-t border-gray-200 dark:border-gray-800"
          >
            <div className="flex flex-col gap-2 mt-4">
              {navItems.map((item) => (
                <button
                  key={item}
                  onClick={() => {
                    onNavigate(item);
                    setMobileMenuOpen(false);
                  }}
                  className={`px-4 py-2 rounded-lg text-left transition-colors ${
                    currentPage === item
                      ? 'bg-blue-600 text-white'
                      : 'hover:bg-gray-100 dark:hover:bg-gray-800'
                  }`}
                >
                  {item}
                </button>
              ))}
            </div>
          </motion.nav>
        )}
      </div>
    </motion.header>
  );
}
