# Component Architecture & Data Flow

## 📊 Component Hierarchy

```
App.tsx
├── HeaderEnhanced
│   ├── UserAvatar (shows after login)
│   ├── Navigation Menu
│   ├── Theme Toggle
│   ├── Language Selector
│   └── User Dropdown Menu
│       ├── User Info
│       ├── Profile Button → ProfilePage
│       ├── Settings Button → SettingsPage
│       └── Logout Button
│
├── AnalyzePageEnhanced
│   ├── Tab Navigation (Text/Image/Video/Audio)
│   ├── Input Area
│   │   ├── Text Input (for text tab)
│   │   └── File Upload Zone (for media tabs)
│   │       ├── Drag & Drop Handler
│   │       ├── Click to Browse
│   │       ├── File Validation
│   │       └── File Preview
│   ├── Analyze Button
│   ├── Results Display
│   └── History Sidebar
│       ├── History List
│       ├── Delete Item Button (per item)
│       └── Clear All Button
│
├── SettingsPage
│   ├── Profile Info Sidebar
│   │   ├── Avatar Preview
│   │   ├── User Info
│   │   └── Account Details
│   └── Avatar Customization
│       ├── Custom Image Upload
│       ├── Emoji Selector (16 options)
│       ├── Gradient Selector (8 options)
│       └── Save Button
│
└── LoginDialog (Firebase)
    ├── Email/Password Form
    ├── Google Sign-In
    └── Register/Forgot Password
```

---

## 🔄 Data Flow Diagrams

### 1. User Authentication Flow

```
User Clicks Login
    ↓
LoginDialog Opens
    ↓
User Enters Credentials / Google OAuth
    ↓
Firebase Authentication
    ↓
AuthContext Updates (user object)
    ↓
Header Rerenders
    ↓
Login Button → User Avatar (with menu)
    ↓
localStorage keys created for user:
- analysis_history_{uid}
- user_avatar_{uid}
- user_avatar_type_{uid}
- user_avatar_emoji_{uid}
- user_avatar_gradient_{uid}
```

### 2. File Upload Flow

```
User on AnalyzePage
    ↓
Selects Tab (Image/Video/Audio)
    ↓
User Action:
├─→ Drags File Over Upload Zone
│   ├─→ handleDrag() - Visual feedback
│   └─→ handleDrop() - Get file
│
└─→ Clicks Upload Zone
    └─→ triggerFileInput() - Opens file picker

    ↓
File Selected
    ↓
handleFileSelection()
├─→ Validate file type (matches tab?)
├─→ Validate file size (within limits?)
├─→ If invalid: Show toast error
└─→ If valid: Set selectedFile, show preview

    ↓
User Clicks "Analyze"
    ↓
handleAnalyze()
├─→ Send file to backend API
├─→ Get result (verdict, confidence)
├─→ Save to history (saveToHistory)
│   ├─→ Create history item
│   ├─→ Add to array
│   ├─→ Save to localStorage
│   └─→ Update UI
└─→ Display result
```

### 3. Avatar Customization Flow

```
User Clicks Avatar → Settings
    ↓
SettingsPage Loads
    ↓
localStorage Check:
├─→ user_avatar_type_{uid}
├─→ user_avatar_{uid} (custom image)
├─→ user_avatar_emoji_{uid}
└─→ user_avatar_gradient_{uid}

    ↓
User Selects Avatar Type:
├─→ Upload Image
│   ├─→ File validation (type, size)
│   ├─→ Read as base64
│   ├─→ Show preview
│   └─→ Store in state
│
├─→ Choose Emoji
│   ├─→ Click emoji button
│   ├─→ Update selectedEmoji
│   └─→ Show preview
│
└─→ Select Gradient
    ├─→ Click color button
    ├─→ Update selectedGradient
    └─→ Show preview

    ↓
User Clicks "Save"
    ↓
handleSave()
├─→ Store type in localStorage
├─→ Store selected option in localStorage
├─→ Show success toast
└─→ Dispatch 'avatarUpdated' event

    ↓
Header Listens to Event
    ↓
Avatar Updates Globally
```

### 4. History Management Flow

```
Analysis Completed
    ↓
saveToHistory() Called
├─→ Create history item:
│   ├─→ id: timestamp
│   ├─→ type: text/image/video/audio
│   ├─→ content: first 100 chars or filename
│   ├─→ result: verdict + confidence
│   └─→ timestamp: current time
│
├─→ Add to existing history array
├─→ Keep only last 50 items
├─→ Save to localStorage:
│   └─→ analysis_history_{uid}
└─→ Update UI (history sidebar)

    ↓
User Hovers Over History Item
    ↓
Delete Button Appears
    ↓
User Clicks Delete
    ↓
deleteHistoryItem(id)
├─→ Show confirmation dialog
│   └─→ "Are you sure you want to delete?"
├─→ If confirmed:
│   ├─→ Filter item from array
│   ├─→ Update localStorage
│   ├─→ Update UI
│   └─→ Show success toast
└─→ If cancelled: Do nothing

    ↓
User Clicks "Clear All"
    ↓
clearAllHistory()
├─→ Show double confirmation
│   └─→ "Are you sure? This cannot be undone."
├─→ If confirmed:
│   ├─→ Clear history array
│   ├─→ Remove from localStorage
│   ├─→ Update UI
│   └─→ Show success toast
└─→ If cancelled: Do nothing
```

---

## 💾 Data Storage Structure

### localStorage Keys (Per User)

```javascript
// Analysis History
`analysis_history_${uid}`: [
  {
    id: "1234567890",
    type: "text" | "image" | "video" | "audio",
    content: "First 100 chars or filename",
    result: {
      verdict: "fake" | "real" | "unverified",
      confidence: 0.0 - 1.0
    },
    timestamp: 1234567890000
  },
  // ... up to 50 items
]

// Avatar Settings
`user_avatar_type_${uid}`: "image" | "emoji" | "gradient"
`user_avatar_${uid}`: "data:image/png;base64,..." // base64 image
`user_avatar_emoji_${uid}`: "😊" // selected emoji
`user_avatar_gradient_${uid}`: "1-8" // gradient ID
```

### AuthContext User Object

```javascript
{
  uid: "firebase_user_id",
  email: "user@example.com",
  displayName: "User Name",
  photoURL: "https://...",
  emailVerified: true/false
}
```

---

## 🎨 UI State Management

### HeaderEnhanced States

```javascript
const [isMenuOpen, setIsMenuOpen] = useState(false);        // Mobile menu
const [isUserMenuOpen, setIsUserMenuOpen] = useState(false); // Avatar dropdown
const [isScrolled, setIsScrolled] = useState(false);         // Scroll effect
```

### AnalyzePageEnhanced States

```javascript
const [activeTab, setActiveTab] = useState('text');          // Current tab
const [textInput, setTextInput] = useState('');              // Text content
const [selectedFile, setSelectedFile] = useState(null);      // Uploaded file
const [dragActive, setDragActive] = useState(false);         // Drag feedback
const [history, setHistory] = useState([]);                  // Analysis history
const [showHistory, setShowHistory] = useState(false);       // History visibility
const { loading, result, error } = useDetection();           // API states
```

### SettingsPage States

```javascript
const [customImage, setCustomImage] = useState(null);        // Uploaded image
const [selectedEmoji, setSelectedEmoji] = useState(null);    // Chosen emoji
const [selectedGradient, setSelectedGradient] = useState(1); // Gradient ID
const [avatarType, setAvatarType] = useState('gradient');    // Current type
```

---

## 🔌 API Integration Points

### useDetection Hook (from hooks/useDetection.ts)

```javascript
const {
  loading,              // Boolean: API call in progress
  result,              // Object: Detection result
  error,               // String: Error message
  checkText,           // Function: (text, language) => Promise
  checkImage,          // Function: (file) => Promise
  checkVideo,          // Function: (file) => Promise
  checkVoice,          // Function: (file) => Promise
  reset                // Function: () => void
} = useDetection();
```

### API Endpoints (backend/simple_server.py)

```
POST /api/v1/check-text
- Body: { text, language }
- Response: { verdict, confidence, model_used, ... }

POST /api/v1/check-image
- Body: FormData with 'file'
- Response: { verdict, confidence, model_used, ... }

POST /api/v1/check-video
- Body: FormData with 'file'
- Response: { video_id, status, message }

POST /api/v1/check-voice
- Body: FormData with 'file'
- Response: { verdict, confidence, model_used, ... }
```

---

## 🎯 Event Flow

### Avatar Update Event

```javascript
// Settings Page (after save)
window.dispatchEvent(new Event('avatarUpdated'));

// Header Component (listening)
useEffect(() => {
  const handleAvatarUpdate = () => {
    // Reload avatar from localStorage
    const saved = localStorage.getItem(`user_avatar_${user.uid}`);
    setCustomPhoto(saved);
  };
  
  window.addEventListener('avatarUpdated', handleAvatarUpdate);
  return () => window.removeEventListener('avatarUpdated', handleAvatarUpdate);
}, [user]);
```

### Click Outside to Close

```javascript
// User Menu Dropdown
const userMenuRef = useRef(null);

useEffect(() => {
  const handleClickOutside = (event) => {
    if (userMenuRef.current && !userMenuRef.current.contains(event.target)) {
      setIsUserMenuOpen(false);
    }
  };
  
  document.addEventListener('mousedown', handleClickOutside);
  return () => document.removeEventListener('mousedown', handleClickOutside);
}, []);
```

---

## 📱 Responsive Breakpoints

```css
Mobile:   < 640px   (sm)
Tablet:   640-1024px (md, lg)
Desktop:  > 1024px   (xl, 2xl)
```

### Component Behavior by Screen Size

```javascript
// HeaderEnhanced
- Desktop: Full navigation + avatar menu
- Mobile: Hamburger menu + avatar menu

// AnalyzePageEnhanced
- Desktop: 2/3 content + 1/3 history sidebar
- Mobile: Stacked layout, history below

// SettingsPage
- Desktop: Sidebar + main content (2 columns)
- Mobile: Stacked vertically (1 column)
```

---

## 🎭 Animation Timeline

### Page Transitions

```javascript
variants = {
  initial: { opacity: 0, y: 20 },      // Start
  in: { opacity: 1, y: 0 },            // Enter
  out: { opacity: 0, y: -20 }          // Exit
}

transition = { duration: 0.3 }
```

### Dropdown Menu

```javascript
// Avatar menu dropdown
initial: { opacity: 0, y: -10 }    // Closed
animate: { opacity: 1, y: 0 }      // Open
exit: { opacity: 0, y: -10 }       // Closing
```

### File Drag Feedback

```javascript
dragActive ? 
  'border-blue-500 bg-blue-50' :   // During drag
  'border-gray-300 hover:border-blue-400' // Default
```

---

## 🔐 Security Considerations

### File Upload Security

```javascript
// Type validation
const validTypes = {
  image: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
  video: ['video/mp4', 'video/webm', 'video/quicktime'],
  audio: ['audio/mpeg', 'audio/wav', 'audio/mp3', 'audio/ogg']
};

// Size validation
const maxSize = {
  image: 10 * 1024 * 1024,    // 10MB
  video: 100 * 1024 * 1024,   // 100MB
  audio: 20 * 1024 * 1024     // 20MB
};
```

### Data Isolation

```javascript
// Each user's data is isolated by Firebase UID
const historyKey = `analysis_history_${user.uid}`;
const avatarKey = `user_avatar_${user.uid}`;

// No cross-user data access possible
```

---

## 🎪 Component Props Interface

### HeaderEnhanced

```typescript
interface HeaderEnhancedProps {
  currentPage: string;                    // Current page (lowercase)
  onNavigate: (page: string) => void;     // Navigation handler
  language: string;                       // Current language
  onLanguageChange: (lang: string) => void; // Language change handler
  isDark: boolean;                        // Dark mode state
  onThemeToggle: () => void;              // Theme toggle handler
  onLoginClick: () => void;               // Login dialog opener
}
```

### AnalyzePage (Enhanced)

```typescript
interface AnalyzePageProps {
  language: string;                       // Language for text analysis
}
```

### SettingsPage

```typescript
// No props required - gets user from AuthContext
```

---

This architecture ensures:
- ✅ Clean separation of concerns
- ✅ Reusable components
- ✅ Type-safe interfaces
- ✅ Efficient state management
- ✅ Scalable codebase
- ✅ Easy to maintain and extend

