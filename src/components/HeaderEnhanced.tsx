import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Menu, X, Moon, Sun, Globe, Shield, LogOut, Settings, User, ChevronDown } from 'lucide-react';
import { Button } from './ui/button';
import { useAuth } from '../context/AuthContext';

interface HeaderEnhancedProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  language: string;
  onLanguageChange: (lang: string) => void;
  isDark: boolean;
  onThemeToggle: () => void;
  onLoginClick: () => void;
}

// Avatar color palettes
const AVATAR_COLORS = [
  ['#6366f1', '#8b5cf6'], // Indigo to Purple
  ['#3b82f6', '#06b6d4'], // Blue to Cyan
  ['#10b981', '#14b8a6'], // Green to Teal
  ['#f59e0b', '#f97316'], // Amber to Orange
  ['#ef4444', '#ec4899'], // Red to Pink
  ['#8b5cf6', '#d946ef'], // Purple to Fuchsia
];

function UserAvatar({ user, size = 'md' }: { user: any; size?: 'sm' | 'md' | 'lg' }) {
  const sizeClasses = {
    sm: 'w-8 h-8 text-sm',
    md: 'w-10 h-10 text-base',
    lg: 'w-16 h-16 text-2xl'
  };

  // Generate consistent color based on user ID
  const colorIndex = user?.uid ? parseInt(user.uid.slice(0, 8), 16) % AVATAR_COLORS.length : 0;
  const [startColor, endColor] = AVATAR_COLORS[colorIndex];

  // Get initials from email or display name
  const getInitials = () => {
    if (user?.displayName) {
      const names = user.displayName.split(' ');
      return names.length > 1
        ? `${names[0][0]}${names[names.length - 1][0]}`.toUpperCase()
        : names[0].slice(0, 2).toUpperCase();
    }
    if (user?.email) {
      return user.email.slice(0, 2).toUpperCase();
    }
    return 'U';
  };

  // Check if user has custom photo from localStorage
  const [customPhoto, setCustomPhoto] = useState<string | null>(null);
  useEffect(() => {
    if (user) {
      const saved = localStorage.getItem(`user_avatar_${user.uid}`);
      setCustomPhoto(saved);
    }
  }, [user]);

  if (customPhoto || user?.photoURL) {
    return (
      <img
        src={customPhoto || user.photoURL}
        alt="User Avatar"
        className={`${sizeClasses[size]} rounded-full object-cover border-2 border-white dark:border-gray-800 shadow-lg`}
      />
    );
  }

  return (
    <div
      className={`${sizeClasses[size]} rounded-full flex items-center justify-center font-bold text-white shadow-lg`}
      style={{
        background: `linear-gradient(135deg, ${startColor}, ${endColor})`
      }}
    >
      {getInitials()}
    </div>
  );
}

export function HeaderEnhanced({
  currentPage,
  onNavigate,
  language,
  onLanguageChange,
  isDark,
  onThemeToggle,
  onLoginClick
}: HeaderEnhancedProps) {
  const { user, logout } = useAuth();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const userMenuRef = useRef<HTMLDivElement>(null);

  // Handle scroll effect
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close user menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) {
        setIsUserMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = async () => {
    try {
      await logout();
      setIsUserMenuOpen(false);
      onNavigate('home');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const navItems = [
    { id: 'home', label: 'Home' },
    { id: 'analyze', label: 'Analyze' },
    { id: 'chrome', label: 'Chrome Extension' },
    { id: 'about', label: 'About' },
  ];

  return (
    <motion.header
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled
          ? 'bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl shadow-lg'
          : 'bg-transparent'
      }`}
    >
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-20">
          {/* Logo */}
          <motion.div
            className="flex items-center space-x-2 cursor-pointer"
            whileHover={{ scale: 1.05 }}
            onClick={() => onNavigate('home')}
          >
            <Shield className="w-8 h-8 text-blue-600" />
            <span className="text-2xl font-display font-bold bg-gradient-to-r from-blue-600 to-purple-600 text-transparent bg-clip-text">
              TruthGuard AI
            </span>
          </motion.div>

          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center space-x-1">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  currentPage === item.id
                    ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20'
                    : 'text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400'
                }`}
              >
                {item.label}
              </button>
            ))}
          </nav>

          {/* Right Side Actions */}
          <div className="flex items-center space-x-3">
            {/* Language Selector */}
            <div className="hidden md:flex items-center space-x-2 px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-800">
              <Globe className="w-4 h-4" />
              <select
                value={language}
                onChange={(e) => onLanguageChange(e.target.value)}
                className="bg-transparent text-sm font-medium outline-none cursor-pointer"
              >
                <option value="en">EN</option>
                <option value="es">ES</option>
                <option value="fr">FR</option>
                <option value="de">DE</option>
              </select>
            </div>

            {/* Theme Toggle */}
            <Button
              variant="ghost"
              size="icon"
              onClick={onThemeToggle}
              className="hidden md:flex"
            >
              <AnimatePresence mode="wait">
                {isDark ? (
                  <motion.div
                    key="sun"
                    initial={{ rotate: -90, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    exit={{ rotate: 90, opacity: 0 }}
                  >
                    <Sun className="w-5 h-5" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="moon"
                    initial={{ rotate: 90, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    exit={{ rotate: -90, opacity: 0 }}
                  >
                    <Moon className="w-5 h-5" />
                  </motion.div>
                )}
              </AnimatePresence>
            </Button>

            {/* User Menu or Login Button */}
            {user ? (
              <div className="relative" ref={userMenuRef}>
                <button
                  onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
                  className="flex items-center space-x-2 px-3 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                >
                  <UserAvatar user={user} size="sm" />
                  <span className="hidden md:block font-medium text-sm max-w-[150px] truncate">
                    {user.displayName || user.email?.split('@')[0]}
                  </span>
                  <ChevronDown
                    className={`w-4 h-4 transition-transform ${
                      isUserMenuOpen ? 'rotate-180' : ''
                    }`}
                  />
                </button>

                {/* Dropdown Menu */}
                <AnimatePresence>
                  {isUserMenuOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="absolute right-0 mt-2 w-64 bg-white dark:bg-gray-800 rounded-xl shadow-2xl border border-gray-200 dark:border-gray-700 overflow-hidden"
                    >
                      {/* User Info */}
                      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                        <div className="flex items-center space-x-3">
                          <UserAvatar user={user} size="md" />
                          <div className="flex-1 min-w-0">
                            <p className="font-semibold truncate">
                              {user.displayName || 'User'}
                            </p>
                            <p className="text-sm text-gray-500 truncate">
                              {user.email}
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Menu Items */}
                      <div className="py-2">
                        <button
                          onClick={() => {
                            setIsUserMenuOpen(false);
                            onNavigate('profile');
                          }}
                          className="w-full flex items-center space-x-3 px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                        >
                          <User className="w-5 h-5" />
                          <span>Profile</span>
                        </button>
                        <button
                          onClick={() => {
                            setIsUserMenuOpen(false);
                            onNavigate('settings');
                          }}
                          className="w-full flex items-center space-x-3 px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                        >
                          <Settings className="w-5 h-5" />
                          <span>Settings</span>
                        </button>
                      </div>

                      {/* Logout */}
                      <div className="border-t border-gray-200 dark:border-gray-700 p-2">
                        <button
                          onClick={handleLogout}
                          className="w-full flex items-center space-x-3 px-4 py-2 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
                        >
                          <LogOut className="w-5 h-5" />
                          <span>Logout</span>
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ) : (
              <Button
                onClick={onLoginClick}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
              >
                <User className="w-4 h-4 mr-2" />
                Login
              </Button>
            )}

            {/* Mobile Menu Button */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="lg:hidden"
            >
              {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="lg:hidden border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900"
          >
            <div className="container mx-auto px-4 py-4 space-y-2">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    onNavigate(item.id);
                    setIsMenuOpen(false);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg font-medium transition-colors ${
                    currentPage === item.id
                      ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                  }`}
                >
                  {item.label}
                </button>
              ))}

              {/* Mobile Theme & Language */}
              <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center space-x-2">
                  <Globe className="w-4 h-4" />
                  <select
                    value={language}
                    onChange={(e) => onLanguageChange(e.target.value)}
                    className="bg-gray-100 dark:bg-gray-800 px-3 py-2 rounded-lg text-sm font-medium outline-none"
                  >
                    <option value="en">English</option>
                    <option value="es">Español</option>
                    <option value="fr">Français</option>
                    <option value="de">Deutsch</option>
                  </select>
                </div>
                <Button variant="ghost" size="icon" onClick={onThemeToggle}>
                  {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.header>
  );
}
