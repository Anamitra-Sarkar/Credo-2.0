# 🔑 API Keys Setup Guide

## Required API Keys for Real AI Backend

### 1. Tavily API Key (Required for Real-time Fact Checking)

**What is Tavily?**
- Real-time search API specifically designed for AI applications
- Provides fact-checking and context for news verification
- Essential for real-time claim verification

**How to get it:**
1. Go to: https://tavily.com/
2. Sign up for a free account
3. Get your API key from the dashboard
4. Free tier includes: 1,000 searches/month

**Add to `.env`:**
```bash
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxx
```

---

### 2. Hugging Face Token (Optional - for faster model downloads)

**What is it?**
- Access token for Hugging Face model hub
- Allows downloading models and using gated models
- Not strictly required for public models

**How to get it:**
1. Go to: https://huggingface.co/
2. Create an account (free)
3. Go to Settings → Access Tokens
4. Create a new token with "read" permission

**Add to `.env`:**
```bash
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx
```

---

### 3. Google Gemini API Key (Optional - for enhanced explanations)

**What is it?**
- Google's Gemini 2.0 Flash API
- Can be used for generating detailed explanations
- Not currently used in base implementation

**How to get it:**
1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Create API key

**Add to `.env`:**
```bash
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxx
```

---

## Current `.env` File Location

File: `e:\OneDrive\Desktop\Gen Ai Project Final\backend\.env`

## How to Add API Keys

### Option 1: Edit in VS Code
1. Open `.env` file in VS Code
2. Replace placeholder values:
   ```bash
   # Before:
   TAVILY_API_KEY=your_tavily_api_key_here
   
   # After:
   TAVILY_API_KEY=tvly-your-actual-key
   ```

### Option 2: Using PowerShell
```powershell
# Navigate to backend folder
cd "e:\OneDrive\Desktop\Gen Ai Project Final\backend"

# Edit .env file
notepad .env
```

---

## What Happens Without API Keys?

### ✅ Will Still Work (with limitations):
- **Text Analysis**: Uses Hugging Face models offline (no internet needed)
- **Image Analysis**: Uses pre-trained models
- **Basic fake news detection**: Works with downloaded models

### ❌ Won't Work:
- **Real-time fact checking**: Needs Tavily API
- **Source verification**: Needs Tavily API
- **Latest news context**: Needs Tavily API

---

## Checking Current Setup

Run this to check what's configured:
```powershell
cd "e:\OneDrive\Desktop\Gen Ai Project Final\backend"
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Tavily:', 'Configured' if os.getenv('TAVILY_API_KEY', '').startswith('tvly-') else 'Not configured')"
```

---

## Models Being Used

### Text Analysis (Fake News Detection):
- **Model**: `hamzab/roberta-fake-news-classification`
- **Source**: Hugging Face
- **Size**: ~500MB
- **First run**: Downloads automatically
- **Requirements**: No API key needed

### Image Analysis:
- **Model**: `microsoft/resnet-50`
- **Source**: Hugging Face
- **Size**: ~100MB
- **First run**: Downloads automatically
- **Note**: For production, use dedicated deepfake detection model

### Real-time Fact Checking:
- **Service**: Tavily API
- **Requires**: API key
- **Benefits**: Real-time search, source verification, context

---

## Next Steps

1. **Get Tavily API Key** (Most Important):
   - Sign up at https://tavily.com/
   - Copy your API key
   - Add to `.env` file

2. **Update `.env` file**:
   ```bash
   TAVILY_API_KEY=tvly-your-actual-key-here
   ```

3. **Restart backend**:
   ```powershell
   cd "e:\OneDrive\Desktop\Gen Ai Project Final\backend"
   python ai_server.py
   ```

4. **Verify** models are loading:
   - Check terminal output for "✅ Tavily API initialized successfully"
   - Check terminal output for "✅ Fake news detector loaded successfully"

---

## Testing Without API Keys

You can still test the system without API keys:
- Text analysis will work with downloaded models
- Image analysis will work
- Just won't have real-time fact checking from Tavily

The backend will work in "offline mode" and show which features are available in the health check endpoint.

---

## Quick Start (Without API Keys)

If you want to test immediately without setting up API keys:

```powershell
# Kill any running backend
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Start new AI backend
cd "e:\OneDrive\Desktop\Gen Ai Project Final\backend"
python ai_server.py
```

First run will download models (~600MB total). After that, it runs offline!
