# 🎤 SOTA Voice Deepfake Detector Integration

## Model Information

**Repository**: `koyelog/deepfake-voice-detector-sota`  
**HuggingFace URL**: https://huggingface.co/koyelog/deepfake-voice-detector-sota  
**License**: Apache-2.0  
**Last Updated**: 2025-10-31

---

## Architecture

### Model Components

```
Input (4s @ 16kHz raw audio)
    ↓
Wav2Vec2 Feature Extractor (facebook/wav2vec2-base)
    ↓ [768-dim features per frame]
Bidirectional GRU (2 layers, 256 units/direction → 512 total)
    ↓
Multi-Head Attention (8 heads, 512-dim embeddings)
    ↓ [Global Average Pooling]
Classification Head:
    ├─ Linear(512 → 512) + ReLU + BatchNorm + Dropout(0.4)
    ├─ Linear(512 → 128) + ReLU + BatchNorm + Dropout(0.3)
    └─ Linear(128 → 1) + Sigmoid
    ↓
Output: Probability [0..1] (0=Real, 1=Fake)
```

### Technical Specifications

- **Total Parameters**: ~98.5M
- **Trainable Parameters**: ~98.5M
- **Framework**: PyTorch + Transformers
- **Input Requirements**:
  - Duration: 4 seconds (fixed)
  - Sample Rate: 16 kHz
  - Channels: Mono (single-channel)
  - Format: Raw waveform (no mel-spectrogram)
- **Output**: Single probability value (threshold: 0.5)

---

## Training Details

### Dataset Composition

- **Total Samples**: 822,166 audio clips
- **Real/Bonafide**: 387,422 samples (47.1%)
- **Fake/Deepfake**: 434,744 samples (52.9%)
- **Datasets Used**: 19 combined datasets including:
  - ASVspoof 2021
  - WaveFake
  - Audio-Deepfake
  - Fake-Real-Audio
  - Deepfake-Audio
  - Combined-Real-Voices
  - Scenefake
  - Gender-Balanced-Audio-Deepfake
  - Synthetic-Speech-Commands
  - 10+ additional Kaggle/academic datasets

### Training Configuration

```python
Training Split: 657,732 samples (80%)
Validation Split: 164,434 samples (20%)

Optimizer: AdamW
Learning Rate: 5e-5
Weight Decay: 0.01
Batch Size: 24
Gradient Accumulation: 2 (effective batch size: 48)
Epochs: 20
Scheduler: Cosine Annealing with Warm Restarts (T_0=5, T_mult=2)
Loss Function: Binary Cross-Entropy (BCE)

Hardware: Tesla P100-PCIE-16GB
Training Time: ~16 hours
```

---

## Performance Metrics

### Validation Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 95-97% |
| **Precision** | ~0.95 |
| **Recall** | ~0.94 |
| **F1-Score** | ~0.94 |
| **AUC-ROC** | ~0.96 |

⚠️ **Note**: These are aggregated validation metrics. Performance varies by:
- Dataset source
- Language and accent
- Recording conditions
- Manipulation techniques

---

## Implementation in Our System

### PyTorch Model Class

```python
class DeepfakeVoiceDetector(nn.Module):
    """
    SOTA Voice Deepfake Detector with Wav2Vec2 + BiGRU + Multi-Head Attention
    """
    def __init__(self):
        super().__init__()
        
        # Wav2Vec2 feature extractor (frozen CNN)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze CNN layers
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        
        # BiGRU: 2 layers, 256 hidden units/direction
        self.bigru = nn.GRU(
            input_size=768,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Multi-Head Attention: 8 heads
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(...)
```

### Preprocessing Pipeline

```python
# 1. Load audio at 16 kHz
waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

# 2. Ensure exactly 4 seconds (64,000 samples)
target_len = 4 * 16000
if len(waveform) < target_len:
    # Pad with zeros
    waveform = np.pad(waveform, (0, target_len - len(waveform)))
else:
    # Truncate
    waveform = waveform[:target_len]

# 3. Extract features with Wav2Vec2
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
input_values = feature_extractor(
    waveform,
    sampling_rate=16000,
    return_tensors="pt"
).input_values

# 4. Run inference
model.eval()
with torch.no_grad():
    logits = model(input_values)
    prob_fake = torch.sigmoid(logits).item()

# 5. Classify
is_fake = prob_fake >= 0.5
confidence = prob_fake if is_fake else (1.0 - prob_fake)
```

### API Endpoint

**POST** `/api/v1/check-voice`

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/check-voice \
  -F "file=@audio_sample.wav"
```

**Response**:
```json
{
  "is_fake": false,
  "confidence": 0.87,
  "analysis": "Voice Analysis: REAL (Confidence: 87.0%)\n\n🎯 Architecture: Wav2Vec2 + BiGRU + Multi-Head Attention\n📊 Model trained on 822K samples (19 datasets)\n🎤 Input: 4-second clip at 16 kHz",
  "verdict": "REAL",
  "details": {
    "model": "koyelog/deepfake-voice-detector-sota",
    "architecture": "Wav2Vec2 + BiGRU + 8-head Attention",
    "parameters": "98.5M",
    "model_score": "0.1234",
    "audio_duration": "4.00s"
  }
}
```

---

## Integration with Gemini Backup

The SOTA voice detector is integrated with our **3-tier intelligent system**:

### Verification Flow

1. **Tier 1**: SOTA Voice Model
   - Processes 4-second audio clips
   - Outputs probability of deepfake (0-1)
   - Threshold: 0.5 for classification

2. **Tier 2**: Gemini 2.0 Flash (Backup Arbiter)
   - Activated when model confidence is low or result seems suspicious
   - Can analyze audio metadata and patterns
   - Provides reasoning for override decisions

3. **Final Decision**:
   - If Gemini agrees: Return model prediction with confirmation
   - If Gemini overrides: Return Gemini's decision with detailed reasoning

### Example Override Scenario

```python
# Model predicts FAKE with 85% confidence
model_prediction = True
model_confidence = 0.85

# Gemini backup verification
gemini_check = verify_with_gemini_audio(audio_bytes, model_prediction, model_confidence)

if gemini_check["override"]:
    # Gemini overrides to REAL
    final_verdict = "REAL"
    analysis = "🧠 GEMINI OVERRIDE: {gemini reasoning}"
else:
    # Gemini confirms
    final_verdict = "FAKE"
    analysis = "✅ Gemini confirms model prediction"
```

---

## Limitations

### Known Constraints

1. **Clip Duration**: Optimized for 4-second clips only
   - Shorter clips: May degrade performance
   - Longer clips: Automatically truncated

2. **Language Coverage**: 
   - Trained on multi-language datasets
   - Underrepresented languages may show reduced accuracy

3. **Dataset Bias**:
   - Slight skew towards fake samples (52.9% vs 47.1%)
   - May increase false positives in some scenarios

4. **Novel Attacks**:
   - Not evaluated on post-2025 deepfake techniques
   - Unknown performance on zero-shot generator families

5. **Environmental Factors**:
   - Recording quality affects accuracy
   - Background noise can impact results
   - Codec artifacts may influence predictions

---

## Ethical Considerations

⚠️ **Important Guidelines**:

1. **Human Oversight Required**: Do not use as sole evidence for high-stakes decisions
2. **No Accusations**: Avoid using model output to definitively accuse individuals
3. **Privacy Compliance**: Respect legal restrictions when processing audio
4. **Transparency**: Always disclose limitations and potential biases
5. **Demographic Fairness**: Be aware of potential biases across demographics

---

## Hardware Requirements

### Minimum Specifications

- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: ~500 MB for model files

### Recommended Specifications

- **GPU**: NVIDIA GPU with CUDA support (e.g., GTX 1060 or better)
- **VRAM**: ~2 GB for single-sample inference
- **RAM**: 16 GB for batch processing

### Performance

- **CPU Inference**: ~5-10 seconds per 4-second clip
- **GPU Inference**: ~0.5-1 second per 4-second clip
- **Batch Processing**: Recommended for throughput optimization

---

## File Structure

```
backend/
├── ai_server_sota.py          # Main backend with SOTA voice detector
├── .env                        # API keys (HUGGINGFACE_TOKEN, GEMINI_API_KEY)
└── VOICE_DETECTOR_SOTA.md     # This documentation

HuggingFace Cache (~/.cache/huggingface/):
└── hub/
    └── models--koyelog--deepfake-voice-detector-sota/
        ├── pytorch_model.pth        # 73.7 MB
        ├── config.json              # 245 Bytes
        └── training_history.csv     # 4.48 kB
```

---

## Testing

### Test Audio Samples

Use these characteristics for testing:

**Real Voice Samples**:
- Clear human speech
- Natural prosody and intonation
- Consistent breathing patterns
- Authentic emotional expression

**Fake Voice Samples**:
- Robotic or monotone delivery
- Unnatural pauses or transitions
- Inconsistent audio quality
- Missing or artificial background noise

### Example Test

```python
# Test with a 4-second audio file
import requests

response = requests.post(
    "http://localhost:8000/api/v1/check-voice",
    files={"file": open("test_audio.wav", "rb")}
)

result = response.json()
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Analysis: {result['analysis']}")
```

---

## Citation

```bibtex
@misc{deepfake-voice-detector-sota,
  author = {koyelog},
  title = {Deepfake Voice Detector - Transfer Learning with Wav2Vec2},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/koyelog/deepfake-voice-detector-sota}
}
```

---

## Support & Contact

- **Model Owner**: koyelog
- **Model Hub**: https://huggingface.co/koyelog/deepfake-voice-detector-sota
- **Issues**: Report via HuggingFace model page

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-01 | Initial integration with SOTA model |
| - | 2025-10-31 | Model published to HuggingFace |

---

**Status**: ✅ **Production Ready**

The SOTA voice detector is fully integrated and lazy-loaded on first audio analysis. It provides state-of-the-art performance with intelligent Gemini backup verification for maximum accuracy.
