# VeriFy AI Backend - Production Deployment

Complete production-grade backend for multilingual deepfake and fake news detection platform.

## 🏗️ Architecture Overview

### Services
- **API Gateway**: FastAPI with JWT auth, rate limiting, request validation
- **Translation Router**: Gemini 2.0 Flash integration for multilingual support + intelligent routing
- **Model Servers**: 5 specialized ML models (LIAR, Brain2, Image, Video, Voice deepfake detectors)
- **Video Worker**: Async processing for video deepfakes via Cloud Pub/Sub
- **Trending Service**: Crowdsourced reports, geographic trending, heatmaps
- **Database**: PostgreSQL for users, detections, reports, trending data
- **Cache**: Redis for frequent queries and rate limiting
- **Storage**: Google Cloud Storage for media uploads

### Tech Stack
- **Framework**: FastAPI (Python 3.11)
- **ML Serving**: TorchServe / FastAPI model endpoints
- **Database**: PostgreSQL 15 + SQLAlchemy
- **Cache**: Redis 7
- **Message Queue**: Google Cloud Pub/Sub
- **Orchestration**: Kubernetes (GKE)
- **Monitoring**: Prometheus + Grafana + Cloud Logging
- **CI/CD**: Google Cloud Build
- **IaC**: Terraform

## 📁 Project Structure

```
backend/
├── services/
│   ├── gateway/              # API Gateway service
│   ├── translation_router/   # Translation + routing service
│   ├── models/               # ML model servers
│   │   ├── text_liar/       # US politics fact-checker
│   │   ├── text_brain2/     # General fact-checker
│   │   ├── image_detector/  # Image deepfake detector
│   │   ├── video_detector/  # Video deepfake detector
│   │   └── voice_detector/  # Voice deepfake detector
│   ├── video_worker/        # Async video processing
│   └── trending/            # Trending & reports service
├── shared/                   # Shared utilities
│   ├── auth/                # JWT authentication
│   ├── database/            # Database models
│   ├── cache/               # Redis utilities
│   └── monitoring/          # Logging & metrics
├── infrastructure/           # IaC and K8s configs
│   ├── terraform/           # GCP infrastructure
│   ├── kubernetes/          # K8s manifests
│   └── helm/                # Helm charts
├── scripts/                  # Deployment scripts
└── tests/                    # Integration tests
```

## 🚀 Quick Start

### Prerequisites
- Google Cloud SDK installed and authenticated
- Docker installed
- kubectl configured for GKE
- Python 3.11+
- Terraform 1.5+

### Local Development

1. **Install dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. **Start local services**:
```bash
docker-compose up -d
```

4. **Run gateway**:
```bash
cd services/gateway
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for complete deployment guide.

Quick deploy:
```bash
./scripts/deploy.sh production
```

## 🔑 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get JWT token
- `POST /api/v1/auth/refresh` - Refresh access token

### Detection
- `POST /api/v1/check-text` - Verify text/news article
- `POST /api/v1/check-image` - Detect image deepfakes
- `POST /api/v1/check-video` - Detect video deepfakes (async)
- `POST /api/v1/check-voice` - Detect voice deepfakes
- `GET /api/v1/check-video/result/{job_id}` - Get video detection result

### Reporting & Trending
- `POST /api/v1/report` - Submit user report
- `GET /api/v1/trending` - Get trending misinformation
- `GET /api/v1/trending/map` - Geographic heatmap data
- `GET /api/v1/explanation/{detection_id}` - Get detailed explanation

### Admin
- `GET /api/v1/health` - Health check
- `GET /api/v1/metrics` - Prometheus metrics

## 🔒 Security Features

- JWT-based authentication with refresh tokens
- Rate limiting per user and IP
- API key validation for service-to-service calls
- CORS configuration for trusted origins
- Input validation and sanitization
- Media file malware scanning
- Encrypted secrets via Google Secret Manager
- TLS/HTTPS everywhere

## 📊 Monitoring

- **Metrics**: Prometheus metrics exposed at `/metrics`
- **Logging**: Structured JSON logs to Cloud Logging
- **Tracing**: OpenTelemetry integration
- **Alerts**: Configured via Alertmanager
- **Dashboards**: Grafana dashboards included

## 🧪 Testing

```bash
# Unit tests
pytest tests/unit

# Integration tests
pytest tests/integration

# Load tests
locust -f tests/load/locustfile.py
```

## 📝 Environment Variables

Key environment variables (see `.env.example` for complete list):

- `GEMINI_API_KEY` - Google Gemini API key
- `TAVILY_API_KEY` - Tavily API for real-time info
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `GCS_BUCKET` - Google Cloud Storage bucket name
- `JWT_SECRET_KEY` - Secret for JWT signing
- `ENVIRONMENT` - production/staging/development

## 🤝 Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## 📄 License

Apache 2.0 - See [LICENSE](./LICENSE)

## 📞 Support

- Documentation: [docs/](./docs/)
- Issues: GitHub Issues
- Email: support@verify-ai.com
