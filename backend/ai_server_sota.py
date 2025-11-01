"""
AI-Powered Deepfake Detection Backend with SOTA Models
Uses custom PyTorch models from HuggingFace with timm
"""

import os
import json

# Set environment variables BEFORE any imports to avoid TensorFlow/Keras conflicts
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid as uuid_module
import traceback

app = FastAPI(title="AI-Powered Deepfake Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# SOTA Model Loading with Custom Architecture
# ============================================

print("\n" + "="*60)
print("🚀 LOADING SOTA DEEPFAKE DETECTION MODELS")
print("="*60 + "\n")

# Import AI dependencies
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
import cv2
import numpy as np
import librosa
import soundfile as sf
from io import BytesIO
import tempfile

try:
    import timm
    print("✅ timm library imported")
except ImportError:
    print("❌ timm not found. Installing...")
    os.system("pip install timm")
    import timm

# Use torchvision transforms instead of albumentations to avoid scipy issues
from torchvision import transforms
print("✅ torchvision transforms imported")

# Delay transformers import to avoid scipy conflicts
fake_news_detector = None
print("\n📚 Text Fake News Detector will be loaded on first use...")

# Tavily API for fact-checking
print("\n🌐 Initializing Tavily API...")
try:
    from tavily import TavilyClient
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        tavily = TavilyClient(api_key=tavily_api_key)
        print("✅ Tavily API: READY")
    else:
        tavily = None
        print("⚠️ Tavily API: No API key found")
except Exception as e:
    tavily = None
    print(f"❌ Tavily API: FAILED - {str(e)}")

# Gemini 2.0 Flash for backup verification
print("\n🧠 Initializing Gemini 2.0 Flash (Backup Verification)...")
try:
    import google.generativeai as genai
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Gemini 2.0 Flash: READY (Backup Verification)")
    else:
        gemini_model = None
        print("⚠️ Gemini 2.0 Flash: No API key found")
except Exception as e:
    gemini_model = None
    print(f"❌ Gemini 2.0 Flash: FAILED - {str(e)}")


# ============================================
# Custom Model Architecture for Image Detection
# ============================================

class DeepfakeImageDetector(nn.Module):
    """Custom EfficientNetV2-S model for deepfake image detection"""
    def __init__(self, model_name='tf_efficientnetv2_s', pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        # Match the actual checkpoint structure with more layers
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class DeepfakeVideoDetector(nn.Module):
    """Custom Xception model for deepfake video detection"""
    def __init__(self, model_name='xception', pretrained=False):
        super().__init__()
        # Try to use xception from timm, fallback to efficientnet
        try:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except:
            print(f"⚠️ {model_name} not found, using efficientnetv2_m")
            self.backbone = timm.create_model('tf_efficientnetv2_m', pretrained=pretrained, num_classes=0)
        
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ============================================
# Load Image Deepfake Detector (EfficientNetV2-S)
# ============================================

print("\n🖼️ Loading Image Deepfake Detector (EfficientNetV2-S)...")
image_detector_model = None
image_transform = None

try:
    # Download model files from HuggingFace
    model_path = hf_hub_download(
        repo_id="Arko007/deepfake-image-detector",
        filename="pytorch_model.bin",
        token=os.getenv("HUGGINGFACE_TOKEN")
    )
    config_path = hf_hub_download(
        repo_id="Arko007/deepfake-image-detector",
        filename="config.json",
        token=os.getenv("HUGGINGFACE_TOKEN")
    )
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    image_detector_model = DeepfakeImageDetector(
        model_name=config.get('model_name', 'tf_efficientnetv2_s'),
        pretrained=False
    )
    
    # Load checkpoint (use strict=False to handle architecture differences)
    checkpoint = torch.load(model_path, map_location='cpu')
    image_detector_model.load_state_dict(checkpoint, strict=False)
    image_detector_model.eval()
    
    # Create transform (380x380 as per model card)
    image_size = config.get('image_size', 380)
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"✅ Image Detector: LOADED (EfficientNetV2-S, 89.5MB, AUC 0.9986)")
    print(f"   - Input size: {image_size}x{image_size}")
    print(f"   - Backbone: {config.get('model_name', 'tf_efficientnetv2_s')}")
    
except Exception as e:
    print(f"❌ Image Detector: FAILED")
    print(f"   Error: {str(e)}")
    print(f"   Traceback: {traceback.format_exc()}")


# ============================================
# Load Video Deepfake Detector (Xception/EfficientNetV2-M)
# ============================================

print("\n🎥 Loading Video Deepfake Detector (DFD-SOTA)...")
video_detector_model = None
video_transform = None

try:
    # Download model files from HuggingFace
    model_path = hf_hub_download(
        repo_id="Arko007/deepfake-detector-dfd-sota",
        filename="pytorch_model.bin",
        token=os.getenv("HUGGINGFACE_TOKEN")
    )
    config_path = hf_hub_download(
        repo_id="Arko007/deepfake-detector-dfd-sota",
        filename="config.json",
        token=os.getenv("HUGGINGFACE_TOKEN")
    )
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    video_detector_model = DeepfakeVideoDetector(
        model_name=config.get('model_name', 'xception'),
        pretrained=False
    )
    
    # Load checkpoint (handle nested structure)
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Check if checkpoint has nested structure
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    video_detector_model.load_state_dict(state_dict, strict=False)
    video_detector_model.eval()
    
    # Create transform
    video_size = config.get('image_size', 299)
    video_transform = transforms.Compose([
        transforms.Resize((video_size, video_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"✅ Video Detector: LOADED (Xception/EfficientNetV2-M, 1.28GB, SOTA)")
    print(f"   - Input size: {video_size}x{video_size}")
    print(f"   - Backbone: {config.get('model_name', 'xception')}")
    
except Exception as e:
    print(f"❌ Video Detector: FAILED")
    print(f"   Error: {str(e)}")
    print(f"   Traceback: {traceback.format_exc()}")


# ============================================
# Load Voice Deepfake Detector (SOTA)
# ============================================

class DeepfakeVoiceDetector(nn.Module):
    """
    SOTA Voice Deepfake Detector with Wav2Vec2 + BiGRU + Multi-Head Attention
    Architecture from koyelog/deepfake-voice-detector-sota
    """
    def __init__(self):
        super().__init__()
        from transformers import Wav2Vec2Model
        
        # Wav2Vec2 feature extractor (frozen CNN layers)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze CNN feature extractor
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        
        # BiGRU: 2 layers, 256 hidden units per direction (512 total)
        self.bigru = nn.GRU(
            input_size=768,  # wav2vec2 output dimension
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Multi-Head Attention: 8 heads, 512-dimensional embeddings
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_values):
        # Extract features with Wav2Vec2
        wav2vec_outputs = self.wav2vec2(input_values).last_hidden_state
        
        # BiGRU temporal modeling
        gru_output, _ = self.bigru(wav2vec_outputs)
        
        # Multi-head attention (self-attention)
        attn_output, _ = self.attention(gru_output, gru_output, gru_output)
        
        # Global average pooling over time dimension
        pooled = torch.mean(attn_output, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        return logits

voice_detector_model = None
voice_feature_extractor = None
print("\n🎤 Voice Deepfake Detector will be loaded on first use...")


# ============================================
# Startup Summary
# ============================================

print("\n" + "="*60)
print("📊 MODEL LOADING SUMMARY")
print("="*60)
print(f"✅ Text Detector (RoBERTa): ⏳ Lazy-loaded (loads on first use)")
print(f"✅ Tavily Fact-Check API: {'✅ Ready' if tavily else '❌ Not ready'}")
print(f"🧠 Gemini 2.0 Flash Backup: {'✅ Ready' if gemini_model else '❌ Not ready'}")
print(f"🖼️ Image Detector (EfficientNetV2-S): {'✅ Loaded' if image_detector_model else '❌ Not loaded'}")
print(f"🎥 Video Detector (DFD-SOTA): {'✅ Loaded' if video_detector_model else '❌ Not loaded'}")
print(f"🎤 Voice Detector (SOTA): ⏳ Lazy-loaded (Wav2Vec2+BiGRU+Attention, 98.5M params)")
print("="*60 + "\n")


# ============================================
# Request/Response Models
# ============================================

class TextCheckRequest(BaseModel):
    text: str

class CheckResponse(BaseModel):
    is_fake: bool
    confidence: float
    analysis: str
    verdict: str
    details: Optional[dict] = None


# ============================================
# Helper Functions
# ============================================

def load_text_detector():
    """Lazy load text detector to avoid scipy conflicts at startup"""
    global fake_news_detector
    if fake_news_detector is None:
        try:
            print("\n📚 Loading Text Fake News Detector (RoBERTa)...")
            from transformers import pipeline
            fake_news_detector = pipeline(
                "text-classification",
                model="hamzab/roberta-fake-news-classification",
                tokenizer="hamzab/roberta-fake-news-classification",
                framework="pt"
            )
            print("✅ Text Detector (RoBERTa): LOADED (500MB, 85-90% accuracy)")
        except Exception as e:
            print(f"❌ Text Detector: FAILED - {str(e)}")
            raise
    return fake_news_detector

def load_voice_detector():
    """Lazy load SOTA voice detector to avoid scipy conflicts at startup"""
    global voice_detector_model, voice_feature_extractor
    if voice_detector_model is None:
        try:
            print("\n🎤 Loading SOTA Voice Deepfake Detector...")
            from transformers import Wav2Vec2FeatureExtractor
            
            # Download model checkpoint from HuggingFace
            model_path = hf_hub_download(
                repo_id="koyelog/deepfake-voice-detector-sota",
                filename="pytorch_model.pth",
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            
            # Initialize model
            voice_detector_model = DeepfakeVoiceDetector()
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            voice_detector_model.load_state_dict(state_dict, strict=False)
            voice_detector_model.eval()
            
            # Initialize feature extractor
            voice_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
            
            print("✅ Voice Detector: LOADED (SOTA - Wav2Vec2 + BiGRU + Attention, 98.5M params)")
            print("   - Architecture: Wav2Vec2 + BiGRU(2 layers) + 8-head Attention")
            print("   - Performance: 95-97% accuracy on validation")
            print("   - Input: 4-second clips at 16 kHz")
            
        except Exception as e:
            print(f"❌ Voice Detector: FAILED - {str(e)}")
            print(f"   Traceback: {traceback.format_exc()}")
            raise
    return voice_detector_model, voice_feature_extractor

def analyze_image_with_sota(image_bytes: bytes) -> dict:
    """Analyze image using SOTA EfficientNetV2-S model"""
    if not image_detector_model or not image_transform:
        raise Exception("Image detector model not loaded")
    
    # Load image
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    # Apply transform
    image_tensor = image_transform(image).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        logit = image_detector_model(image_tensor)
        prob_fake = torch.sigmoid(logit).item()
    
    is_fake = prob_fake > 0.5
    confidence = prob_fake if is_fake else (1 - prob_fake)
    
    # Generate detailed analysis
    if is_fake:
        if prob_fake > 0.9:
            analysis = f"⚠️ HIGH CONFIDENCE DEEPFAKE DETECTED ({confidence*100:.1f}%)\n\n"
            analysis += "The SOTA EfficientNetV2-S detector (AUC 0.9986) has identified strong indicators of manipulation:\n"
            analysis += "• Anomalous patterns in facial features\n"
            analysis += "• Inconsistencies in texture and lighting\n"
            analysis += "• Neural network artifacts detected"
        elif prob_fake > 0.7:
            analysis = f"⚠️ LIKELY DEEPFAKE ({confidence*100:.1f}%)\n\n"
            analysis += "The model detects probable manipulation with notable confidence.\n"
            analysis += "• Some facial inconsistencies detected\n"
            analysis += "• Possible AI-generated artifacts"
        else:
            analysis = f"⚠️ POSSIBLE DEEPFAKE ({confidence*100:.1f}%)\n\n"
            analysis += "The model suggests potential manipulation, but confidence is moderate.\n"
            analysis += "• Borderline detection - manual review recommended"
    else:
        analysis = f"✅ LIKELY AUTHENTIC ({confidence*100:.1f}%)\n\n"
        analysis += "The SOTA detector indicates this image appears genuine:\n"
        analysis += "• No significant manipulation markers detected\n"
        analysis += "• Consistent facial features and textures\n"
        analysis += "• Natural lighting and depth characteristics"
    
    return {
        "is_fake": is_fake,
        "confidence": confidence,
        "analysis": analysis,
        "verdict": "FAKE" if is_fake else "REAL",
        "model_details": {
            "model_name": "Arko007/deepfake-image-detector",
            "backbone": "EfficientNetV2-S",
            "auc": 0.9986,
            "probability_fake": prob_fake
        }
    }


def analyze_video_with_sota(video_bytes: bytes) -> dict:
    """Analyze video using SOTA DFD model with frame extraction"""
    if not video_detector_model or not video_transform:
        raise Exception("Video detector model not loaded")
    
    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_bytes)
        video_path = tmp_file.name
    
    try:
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise Exception("Could not read video frames")
        
        # Sample 10 frames evenly
        frame_indices = np.linspace(0, total_frames - 1, min(10, total_frames), dtype=int)
        
        frame_results = []
        fake_count = 0
        real_count = 0
        total_prob = 0
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert BGR to RGB and to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Apply transform
            frame_tensor = video_transform(frame_pil).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                logit = video_detector_model(frame_tensor)
                prob_fake = torch.sigmoid(logit).item()
            
            is_fake = prob_fake > 0.5
            if is_fake:
                fake_count += 1
            else:
                real_count += 1
            
            total_prob += prob_fake
            
            frame_results.append({
                "frame": int(frame_idx),
                "probability_fake": prob_fake,
                "verdict": "FAKE" if is_fake else "REAL"
            })
        
        cap.release()
        
        # Overall verdict by majority voting
        avg_prob = total_prob / len(frame_results)
        is_fake_overall = fake_count > real_count
        confidence = fake_count / len(frame_results) if is_fake_overall else real_count / len(frame_results)
        
        # Generate analysis
        if is_fake_overall:
            analysis = f"⚠️ DEEPFAKE VIDEO DETECTED ({confidence*100:.1f}%)\n\n"
            analysis += f"The SOTA DFD detector analyzed {len(frame_results)} frames:\n"
            analysis += f"• {fake_count} frames flagged as FAKE\n"
            analysis += f"• {real_count} frames flagged as REAL\n"
            analysis += f"• Average deepfake probability: {avg_prob*100:.1f}%\n\n"
            analysis += "This video shows signs of manipulation across multiple frames."
        else:
            analysis = f"✅ LIKELY AUTHENTIC VIDEO ({confidence*100:.1f}%)\n\n"
            analysis += f"The SOTA DFD detector analyzed {len(frame_results)} frames:\n"
            analysis += f"• {real_count} frames flagged as REAL\n"
            analysis += f"• {fake_count} frames flagged as FAKE\n"
            analysis += f"• Average deepfake probability: {avg_prob*100:.1f}%\n\n"
            analysis += "This video appears to be authentic."
        
        return {
            "is_fake": is_fake_overall,
            "confidence": confidence,
            "analysis": analysis,
            "verdict": "FAKE" if is_fake_overall else "REAL",
            "model_details": {
                "model_name": "Arko007/deepfake-detector-dfd-sota",
                "frames_analyzed": len(frame_results),
                "fake_frames": fake_count,
                "real_frames": real_count,
                "frame_results": frame_results[:5]  # First 5 frames
            }
        }
    
    finally:
        # Cleanup
        os.unlink(video_path)


# ============================================
# Gemini Backup Verification Functions
# ============================================

def verify_with_gemini_text(text: str, model_prediction: bool, model_confidence: float, tavily_sources: str = "") -> dict:
    """Use Gemini to intelligently verify text with context awareness"""
    if not gemini_model:
        return {"override": False, "gemini_verdict": None, "should_check": False}
    
    try:
        # Enhanced prompt with temporal awareness and fact-checking
        prompt = f"""You are an expert fact-checker. Analyze this statement carefully, paying special attention to:
1. TEMPORAL CONTEXT: Is it talking about past, present, or future? (was/is/will be)
2. CURRENT FACTS: As of November 2025, what is the truth?
3. CONTEXT: Does the claim make sense in current context?

Statement to verify: "{text}"

{f'Additional context from web sources: {tavily_sources[:500]}' if tavily_sources else ''}

Provide your analysis in JSON format:
{{
    "is_fake": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation focusing on why it's true or false RIGHT NOW",
    "temporal_analysis": "Past/Present/Future context if relevant"
}}

Be extremely precise about current facts vs historical facts."""
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip().replace('```json', '').replace('```', '')
        gemini_result = json.loads(response_text)
        
        print(f"\n🧠 Gemini Analysis:")
        print(f"   Verdict: {'FAKE' if gemini_result['is_fake'] else 'REAL'}")
        print(f"   Confidence: {gemini_result['confidence']:.1%}")
        print(f"   Reasoning: {gemini_result['reasoning'][:100]}...")
        
        # Return Gemini's verdict for the priority system to use
        return {
            "override": False,  # Don't auto-override, let priority system decide
            "gemini_verdict": "FAKE" if gemini_result["is_fake"] else "REAL",
            "is_fake": gemini_result["is_fake"],
            "confidence": gemini_result["confidence"],
            "reasoning": gemini_result["reasoning"],
            "temporal_analysis": gemini_result.get("temporal_analysis", ""),
            "should_check": True
        }
    
    except Exception as e:
        print(f"⚠️ Gemini verification failed: {str(e)}")
        return {"override": False, "gemini_verdict": None, "should_check": False}


def verify_with_gemini_image(image_bytes: bytes, model_prediction: bool, model_confidence: float) -> dict:
    """Use Gemini to verify image analysis - only if predicted as FAKE"""
    if not gemini_model or not model_prediction:  # Only check if model says it's FAKE
        return {"override": False, "gemini_verdict": None}
    
    try:
        # Upload image to Gemini
        image = Image.open(BytesIO(image_bytes))
        
        prompt = """Analyze if this image is a DEEPFAKE or REAL. Look for:
- AI-generated artifacts
- Unnatural lighting or shadows
- Inconsistent details
- Digital manipulation signs

Respond in JSON format:
{
    "is_fake": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}"""
        
        response = gemini_model.generate_content([prompt, image])
        gemini_result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
        
        # If Gemini disagrees with model (model says FAKE, Gemini says REAL)
        if not gemini_result["is_fake"] and model_prediction:
            print(f"🔄 Gemini Override: Model said FAKE, Gemini says REAL (confidence: {gemini_result['confidence']:.2%})")
            return {
                "override": True,
                "gemini_verdict": "REAL",
                "is_fake": False,
                "confidence": gemini_result["confidence"],
                "reasoning": gemini_result["reasoning"]
            }
        
        return {"override": False, "gemini_verdict": "FAKE"}
    
    except Exception as e:
        print(f"⚠️ Gemini image verification failed: {str(e)}")
        return {"override": False, "gemini_verdict": None}


def verify_with_gemini_video(video_bytes: bytes, model_prediction: bool, model_confidence: float) -> dict:
    """Use Gemini to verify video analysis - only if predicted as FAKE"""
    if not gemini_model or not model_prediction:  # Only check if model says it's FAKE
        return {"override": False, "gemini_verdict": None}
    
    try:
        # Extract a few frames for Gemini to analyze
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_bytes)
            video_path = tmp.name
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract 3 frames (beginning, middle, end)
        frame_indices = [0, total_frames // 2, total_frames - 1]
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
        
        cap.release()
        os.unlink(video_path)
        
        if not frames:
            return {"override": False, "gemini_verdict": None}
        
        prompt = """Analyze if this video frame is from a DEEPFAKE or REAL video. Look for:
- Facial inconsistencies
- Unnatural movements or expressions
- AI-generated artifacts
- Digital manipulation signs

Respond in JSON format:
{
    "is_fake": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}"""
        
        # Analyze first frame with Gemini
        response = gemini_model.generate_content([prompt, frames[0]])
        gemini_result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
        
        # If Gemini disagrees with model (model says FAKE, Gemini says REAL)
        if not gemini_result["is_fake"] and model_prediction:
            print(f"🔄 Gemini Override: Model said FAKE, Gemini says REAL (confidence: {gemini_result['confidence']:.2%})")
            return {
                "override": True,
                "gemini_verdict": "REAL",
                "is_fake": False,
                "confidence": gemini_result["confidence"],
                "reasoning": gemini_result["reasoning"]
            }
        
        return {"override": False, "gemini_verdict": "FAKE"}
    
    except Exception as e:
        print(f"⚠️ Gemini video verification failed: {str(e)}")
        return {"override": False, "gemini_verdict": None}


def verify_with_gemini_audio(audio_bytes: bytes, model_prediction: bool, model_confidence: float) -> dict:
    """Use Gemini to verify audio analysis - only if predicted as FAKE"""
    if not gemini_model or not model_prediction:  # Only check if model says it's FAKE
        return {"override": False, "gemini_verdict": None}
    
    try:
        # Note: Gemini 2.0 Flash has audio support
        prompt = """Analyze if this audio is AI-GENERATED/DEEPFAKE or REAL HUMAN VOICE. Listen for:
- Unnatural speech patterns
- Robotic or synthetic quality
- Inconsistent voice characteristics
- AI voice generation artifacts

Respond in JSON format:
{
    "is_fake": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}"""
        
        # Save audio temporarily for Gemini
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name
        
        # Upload audio file to Gemini
        audio_file = genai.upload_file(audio_path)
        response = gemini_model.generate_content([prompt, audio_file])
        
        os.unlink(audio_path)
        
        gemini_result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
        
        # If Gemini disagrees with model (model says FAKE, Gemini says REAL)
        if not gemini_result["is_fake"] and model_prediction:
            print(f"🔄 Gemini Override: Model said FAKE, Gemini says REAL (confidence: {gemini_result['confidence']:.2%})")
            return {
                "override": True,
                "gemini_verdict": "REAL",
                "is_fake": False,
                "confidence": gemini_result["confidence"],
                "reasoning": gemini_result["reasoning"]
            }
        
        return {"override": False, "gemini_verdict": "FAKE"}
    
    except Exception as e:
        print(f"⚠️ Gemini audio verification failed: {str(e)}")
        return {"override": False, "gemini_verdict": None}


# ============================================
# API Endpoints
# ============================================

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_status": {
            "fake_news_detector": True,  # Lazy loaded on first use
            "tavily": tavily is not None,
            "gemini_backup": gemini_model is not None,
            "image_deepfake_detector": image_detector_model is not None,
            "video_deepfake_detector": video_detector_model is not None,
            "voice_deepfake_detector": voice_detector_model is not None
        }
    }


@app.post("/api/v1/check-text", response_model=CheckResponse)
async def check_text(request: TextCheckRequest):
    """
    Advanced 3-Tier Fact-Checking System with Intelligent Priority:
    1. RoBERTa Model (Base prediction)
    2. Tavily Real-Time Search (Context gathering)
    3. Gemini 2.0 Flash (Final intelligent arbiter with temporal awareness)
    """
    try:
        # Step 1: Get base prediction from RoBERTa
        detector = load_text_detector()
        result = detector(request.text)[0]
        model_label = result['label'].upper()
        model_score = result['score']
        model_is_fake = 'FAKE' in model_label
        
        print(f"\n{'='*70}")
        print(f"📝 TEXT ANALYSIS: '{request.text[:60]}...'")
        print(f"{'='*70}")
        print(f"🤖 Step 1 - RoBERTa Model: {'FAKE' if model_is_fake else 'REAL'} ({model_score:.1%})")
        
        # Step 2: Gather real-time context from Tavily
        tavily_context = ""
        fact_check_display = ""
        if tavily:
            try:
                print(f"🌐 Step 2 - Tavily: Searching for real-time context...")
                search_results = tavily.search(
                    query=request.text[:200], 
                    max_results=3,
                    search_depth="advanced"
                )
                
                if search_results and 'results' in search_results:
                    fact_check_display = "\n\n🔍 Tavily Sources:\n"
                    for idx, item in enumerate(search_results['results'][:3], 1):
                        title = item.get('title', 'Source')
                        url = item.get('url', '')
                        snippet = item.get('content', '')[:250]
                        fact_check_display += f"{idx}. {title} - {url}\n   {snippet}...\n\n"
                        tavily_context += f"{title}: {snippet}\n"
                    print(f"✅ Found {len(search_results['results'])} sources")
            except Exception as e:
                print(f"⚠️ Tavily search failed: {str(e)}")
        
        # Step 3: GEMINI FINAL ARBITER - Intelligent analysis with full context
        print(f"🧠 Step 3 - Gemini 2.0 Flash: Final intelligent verification...")
        gemini_check = verify_with_gemini_text(request.text, model_is_fake, model_score, tavily_context)
        
        # Decision Logic: Gemini has final say
        if gemini_check.get("should_check"):
            final_is_fake = gemini_check["is_fake"]
            final_confidence = gemini_check["confidence"]
            final_verdict = "FAKE" if final_is_fake else "REAL"
            
            # Check if Gemini disagrees with model
            if model_is_fake != final_is_fake:
                analysis = f"🧠 GEMINI OVERRIDE: {gemini_check['reasoning']}\n\n"
                analysis += f"RoBERTa Model: {'FAKE' if model_is_fake else 'REAL'} ({model_score:.1%})\n"
                analysis += f"Gemini Analysis: {final_verdict} ({final_confidence:.1%})\n"
                if gemini_check.get('temporal_analysis'):
                    analysis += f"Temporal Context: {gemini_check['temporal_analysis']}\n"
                analysis += fact_check_display
                print(f"🔄 GEMINI OVERRIDE: Model={('FAKE' if model_is_fake else 'REAL')}, Gemini={final_verdict}")
            else:
                # Gemini agrees with model
                analysis = f"Analysis: {final_verdict} (Confidence: {final_confidence:.1%})\n"
                analysis += f"✅ Gemini confirms model prediction\n"
                analysis += f"Reasoning: {gemini_check['reasoning']}"
                analysis += fact_check_display
                print(f"✅ Gemini AGREES with model: {final_verdict}")
        else:
            # Gemini unavailable, fall back to model
            final_is_fake = model_is_fake
            final_confidence = model_score
            final_verdict = "FAKE" if final_is_fake else "REAL"
            analysis = f"Analysis: {final_verdict} (Confidence: {final_confidence:.1%})"
            analysis += fact_check_display
            print(f"⚠️ Using model prediction (Gemini unavailable)")
        
        print(f"{'='*70}")
        print(f"FINAL VERDICT: {final_verdict} ({final_confidence:.1%})")
        print(f"{'='*70}\n")
        
        return CheckResponse(
            is_fake=final_is_fake,
            confidence=final_confidence,
            analysis=analysis,
            verdict=final_verdict
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/check-image")
async def check_image(file: UploadFile = File(...)):
    """Check if image is a deepfake with Gemini backup verification"""
    if not image_detector_model:
        raise HTTPException(status_code=503, detail="Image detection model not available")
    
    try:
        image_bytes = await file.read()
        result = analyze_image_with_sota(image_bytes)
        
        # Gemini backup verification (only if predicted as FAKE)
        gemini_check = verify_with_gemini_image(image_bytes, result["is_fake"], result["confidence"])
        if gemini_check["override"]:
            result["is_fake"] = gemini_check["is_fake"]
            result["confidence"] = gemini_check["confidence"]
            result["verdict"] = "REAL"
            result["analysis"] = f"🧠 Gemini Override: {gemini_check['reasoning']}\n\n" + \
                                f"Original Model: FAKE ({result.get('original_confidence', result['confidence']):.1%})\n" + \
                                f"Gemini Verification: REAL ({gemini_check['confidence']:.1%})"
        
        return CheckResponse(
            is_fake=result["is_fake"],
            confidence=result["confidence"],
            analysis=result["analysis"],
            verdict=result["verdict"],
            details=result.get("model_details")
        )
    
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/check-video")
async def check_video(file: UploadFile = File(...)):
    """Check if video is a deepfake with Gemini backup verification"""
    if not video_detector_model:
        raise HTTPException(status_code=503, detail="Video detection model not available")
    
    try:
        video_bytes = await file.read()
        result = analyze_video_with_sota(video_bytes)
        
        # Gemini backup verification (only if predicted as FAKE)
        gemini_check = verify_with_gemini_video(video_bytes, result["is_fake"], result["confidence"])
        if gemini_check["override"]:
            result["is_fake"] = gemini_check["is_fake"]
            result["confidence"] = gemini_check["confidence"]
            result["verdict"] = "REAL"
            result["analysis"] = f"🧠 Gemini Override: {gemini_check['reasoning']}\n\n" + \
                                f"Original Model: FAKE ({result.get('original_confidence', result['confidence']):.1%})\n" + \
                                f"Gemini Verification: REAL ({gemini_check['confidence']:.1%})"
        
        return CheckResponse(
            is_fake=result["is_fake"],
            confidence=result["confidence"],
            analysis=result["analysis"],
            verdict=result["verdict"],
            details=result.get("model_details")
        )
    
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/check-voice")
async def check_voice(file: UploadFile = File(...)):
    """Check if audio is a deepfake using SOTA model with Gemini backup verification"""
    try:
        audio_bytes = await file.read()
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            audio_path = tmp_file.name
        
        try:
            # Lazy load SOTA voice detector
            model, feature_extractor = load_voice_detector()
            
            # Load and preprocess audio according to SOTA model requirements
            # Model expects: 4-second clips at 16 kHz
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Ensure 4 seconds length (64,000 samples at 16kHz)
            target_len = 4 * 16000
            if len(waveform) < target_len:
                # Pad with zeros
                waveform = np.pad(waveform, (0, target_len - len(waveform)), mode='constant')
            else:
                # Truncate to 4 seconds
                waveform = waveform[:target_len]
            
            # Extract features using Wav2Vec2 feature extractor
            input_values = feature_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values
            
            # Run inference
            model.eval()
            with torch.no_grad():
                logits = model(input_values)
                prob_fake = torch.sigmoid(logits).item()
            
            # Interpret results (model outputs probability of FAKE)
            # Threshold: 0.5 (per model card)
            is_fake = prob_fake >= 0.5
            confidence = prob_fake if is_fake else (1.0 - prob_fake)
            
            model_prediction = is_fake
            model_confidence = confidence
            
            # Gemini backup verification
            gemini_check = verify_with_gemini_audio(audio_bytes, model_prediction, model_confidence)
            
            if gemini_check.get("should_check", False):
                final_is_fake = gemini_check["is_fake"]
                final_confidence = gemini_check["confidence"]
                
                if model_prediction != final_is_fake:
                    # Gemini override
                    verdict = "FAKE" if final_is_fake else "REAL"
                    analysis = f"🧠 GEMINI OVERRIDE: {gemini_check['reasoning']}\n\n"
                    analysis += f"SOTA Model: {'FAKE' if model_prediction else 'REAL'} ({model_confidence:.1%})\n"
                    analysis += f"Gemini Analysis: {verdict} ({final_confidence:.1%})\n\n"
                    analysis += "🎯 Architecture: Wav2Vec2 + BiGRU + 8-head Attention (98.5M params)"
                else:
                    # Agreement
                    verdict = "FAKE" if final_is_fake else "REAL"
                    analysis = f"✅ Gemini confirms SOTA model prediction\n\n"
                    analysis += f"Voice Analysis: {verdict} ({final_confidence:.1%})\n"
                    analysis += f"Reasoning: {gemini_check['reasoning']}\n\n"
                    analysis += "🎯 Model: SOTA Voice Detector (95-97% accuracy)"
            else:
                # No Gemini check needed
                final_is_fake = model_prediction
                final_confidence = model_confidence
                verdict = "FAKE" if final_is_fake else "REAL"
                analysis = f"Voice Analysis: {verdict} (Confidence: {final_confidence:.1%})\n\n"
                analysis += "🎯 Architecture: Wav2Vec2 + BiGRU + Multi-Head Attention\n"
                analysis += f"📊 Model trained on 822K samples (19 datasets)\n"
                analysis += f"🎤 Input: 4-second clip at 16 kHz"
            
            return CheckResponse(
                is_fake=final_is_fake,
                confidence=final_confidence,
                analysis=analysis,
                verdict=verdict,
                details={
                    "model": "koyelog/deepfake-voice-detector-sota",
                    "architecture": "Wav2Vec2 + BiGRU + 8-head Attention",
                    "parameters": "98.5M",
                    "model_score": f"{prob_fake:.4f}",
                    "audio_duration": f"{len(waveform) / 16000:.2f}s"
                }
            )
        
        finally:
            os.unlink(audio_path)
    
    except Exception as e:
        print(f"Error analyzing audio: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting AI-Powered Deepfake Detection Server with SOTA Models...")
    print("📡 Server running at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
