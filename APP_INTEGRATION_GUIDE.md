# Quick Integration Guide for App.tsx

## Step 1: Add Settings Route

Find this code in your `App.tsx` (around line 185):

```typescript
        {currentPage === 'Profile' && (
          <motion.div
            key="profile"
            initial="initial"
            animate="in"
            exit="out"
            variants={pageVariants}
            transition={pageTransition}
          >
            {user ? (
              <ProfilePage user={user} />
            ) : (
              <div className="min-h-screen flex items-center justify-center">
                <div className="text-center">
                  <h3 className="mb-4">Please log in to view your profile</h3>
                  <button onClick={() => setShowLoginDialog(true)}>Login</button>
                </div>
              </div>
            )}
          </motion.div>
        )}
```

**Add this RIGHT AFTER the Profile section:**

```typescript
        {currentPage === 'Settings' && (
          <motion.div
            key="settings"
            initial="initial"
            animate="in"
            exit="out"
            variants={pageVariants}
            transition={pageTransition}
          >
            <SettingsPage />
          </motion.div>
        )}
```

## Step 2: Update HeaderEnhanced for Chrome Page

The HeaderEnhanced component expects lowercase page names ('home', 'analyze', 'about', 'chrome'). 

Find this line in App.tsx (around line 77):

```typescript
        currentPage={currentPage.toLowerCase()}
```

This should already handle the conversion. However, you may need to add 'chrome' to the navigation mapping.

In your `handleNavigate` function, ensure it handles 'chrome':

```typescript
  const handleNavigate = (page: Page) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
```

And update the HeaderEnhanced component call to properly map Chrome to the Chrome Extension page:

```typescript
        onNavigate={(page) => {
          // Map lowercase page names to your Page type
          const pageMap: Record<string, Page> = {
            'home': 'Home',
            'analyze': 'Analyze',
            'chrome': 'Home', // Map chrome to the page that has ChromeExtensionSection
            'about': 'About',
            'profile': 'Profile',
            'settings': 'Settings',
          };
          handleNavigate(pageMap[page] || 'Home');
        }}
```

## Step 3: Update Chrome Extension Page Route

If you have a separate Chrome page, update the page type:

```typescript
type Page = 'Home' | 'Analyze' | 'Trending' | 'Community' | 'About' | 'Profile' | 'Settings' | 'Chrome';
```

And add the route:

```typescript
        {currentPage === 'Chrome' && (
          <motion.div
            key="chrome"
            initial="initial"
            animate="in"
            exit="out"
            variants={pageVariants}
            transition={pageTransition}
          >
            <ChromeExtensionSection />
          </motion.div>
        )}
```

## Complete App.tsx Structure

Here's how your main App component should look:

```typescript
function App() {
  const { user, loading } = useAuth();
  const [currentPage, setCurrentPage] = useState<Page>('Home');
  const [language, setLanguage] = useState('en');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [showLoginDialog, setShowLoginDialog] = useState(false);

  const handleNavigate = (page: Page) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div className={isDarkMode ? 'dark' : ''}>
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 transition-colors">
        <HeaderEnhanced
          currentPage={currentPage.toLowerCase()}
          onNavigate={(page) => {
            const pageMap: Record<string, Page> = {
              'home': 'Home',
              'analyze': 'Analyze',
              'chrome': 'Home', // Or 'Chrome' if you have a separate page
              'about': 'About',
              'profile': 'Profile',
              'settings': 'Settings',
            };
            handleNavigate(pageMap[page] || 'Home');
          }}
          language={language}
          onLanguageChange={setLanguage}
          isDark={isDarkMode}
          onThemeToggle={toggleDarkMode}
          onLoginClick={() => setShowLoginDialog(true)}
        />

        <AnimatePresence mode="wait">
          {/* All your page routes here */}
          {currentPage === 'Home' && (
            <motion.div key="home" {...animationProps}>
              <HeroSection onNavigate={handleNavigate} />
            </motion.div>
          )}

          {currentPage === 'Analyze' && (
            <motion.div key="analyze" {...animationProps}>
              <AnalyzePage language={language} />
            </motion.div>
          )}

          {/* ... other routes ... */}

          {currentPage === 'Settings' && (
            <motion.div key="settings" {...animationProps}>
              <SettingsPage />
            </motion.div>
          )}
        </AnimatePresence>

        <LoginDialog
          isOpen={showLoginDialog}
          onClose={() => setShowLoginDialog(false)}
        />

        <Toaster />
      </div>
    </div>
  );
}
```

## Step 4: Test Everything

1. **Save all files**
2. **Restart the frontend server** (Ctrl+C then `npm run dev`)
3. **Test the features**:
   - Login and check if avatar appears
   - Click avatar to see dropdown
   - Click "Settings" to open settings page
   - Upload a file in Analyze page
   - Check history sidebar
   - Delete a history item

## Quick Fix for Common Issues

### Issue: "Cannot find module HeaderEnhanced"
**Fix**: Make sure the import path is correct:
```typescript
import { HeaderEnhanced } from './components/HeaderEnhanced';
```

### Issue: "Page not found when clicking Settings"
**Fix**: Ensure 'Settings' is added to the Page type:
```typescript
type Page = 'Home' | 'Analyze' | 'Trending' | 'Community' | 'About' | 'Profile' | 'Settings';
```

### Issue: Avatar doesn't show after login
**Fix**: The HeaderEnhanced component gets user from AuthContext automatically. Just make sure AuthProvider wraps your App.

### Issue: File upload not working
**Fix**: Check browser console for errors. Ensure the file type matches the selected tab.

---

That's it! All features should now be working. Check `FEATURES_IMPLEMENTED.md` for detailed feature documentation.
