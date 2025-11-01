# VeriFy AI Backend - Complete Deployment Guide

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Google Cloud Platform Setup](#google-cloud-platform-setup)
4. [Production Deployment](#production-deployment)
5. [Environment Variables](#environment-variables)
6. [Model Deployment](#model-deployment)
7. [Monitoring & Logging](#monitoring--logging)
8. [Scaling & Performance](#scaling--performance)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Docker** 20.10+ and Docker Compose 2.0+
- **Google Cloud SDK** (gcloud CLI)
- **kubectl** 1.27+
- **Terraform** 1.5+ (for infrastructure as code)
- **Python** 3.11+
- **Git**

### Google Cloud Requirements
- Active GCP account with billing enabled
- Project with the following APIs enabled:
  - Kubernetes Engine API
  - Cloud Storage API
  - Cloud Pub/Sub API
  - Secret Manager API
  - Cloud Logging API
  - Cloud Monitoring API

### API Keys Required
1. **Gemini API Key** - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Tavily API Key** - Get from [Tavily.com](https://tavily.com)

---

## Local Development Setup

### 1. Clone and Setup

```bash
cd backend
cp .env.example .env
# Edit .env with your API keys
```

### 2. Install Python Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start Services with Docker Compose

```bash
# Start all services (without GPU models for local dev)
docker-compose up -d postgres redis gateway trending-service

# Check logs
docker-compose logs -f gateway
```

### 4. Initialize Database

```bash
# Run migrations
python -m alembic upgrade head

# Create initial admin user (optional)
python scripts/create_admin.py
```

### 5. Test API

```bash
# Health check
curl http://localhost:8000/api/v1/health

# API Documentation
open http://localhost:8000/docs
```

---

## Google Cloud Platform Setup

### 1. Create GCP Project

```bash
# Set your project ID
export PROJECT_ID="verify-ai-production"
export REGION="us-central1"
export CLUSTER_NAME="verify-ai-cluster"

# Create project
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID

# Enable billing (replace with your billing account ID)
gcloud beta billing projects link $PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT_ID
```

### 2. Enable Required APIs

```bash
gcloud services enable \
  container.googleapis.com \
  storage.googleapis.com \
  pubsub.googleapis.com \
  secretmanager.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  cloudbuild.googleapis.com
```

### 3. Create GKE Cluster

```bash
# Create cluster with GPU node pool
gcloud container clusters create $CLUSTER_NAME \
  --region=$REGION \
  --num-nodes=2 \
  --machine-type=n1-standard-4 \
  --enable-autoscaling \
  --min-nodes=2 \
  --max-nodes=10 \
  --enable-stackdriver-kubernetes \
  --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver

# Add GPU node pool for model inference
gcloud container node-pools create gpu-pool \
  --cluster=$CLUSTER_NAME \
  --region=$REGION \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --machine-type=n1-standard-4 \
  --num-nodes=2 \
  --min-nodes=1 \
  --max-nodes=5 \
  --enable-autoscaling

# Install NVIDIA GPU drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

# Get cluster credentials
gcloud container clusters get-credentials $CLUSTER_NAME --region=$REGION
```

### 4. Create Cloud Storage Bucket

```bash
export BUCKET_NAME="verify-ai-media-uploads"

gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME/
gsutil iam ch allUsers:objectViewer gs://$BUCKET_NAME  # Optional: make public
```

### 5. Setup Cloud Pub/Sub

```bash
# Create topic for video processing
gcloud pubsub topics create video-processing

# Create subscription
gcloud pubsub subscriptions create video-processing-sub \
  --topic=video-processing \
  --ack-deadline=600
```

### 6. Setup Cloud SQL (PostgreSQL)

```bash
# Create Cloud SQL instance
gcloud sql instances create verify-ai-db \
  --database-version=POSTGRES_15 \
  --tier=db-custom-2-7680 \
  --region=$REGION \
  --storage-size=20GB \
  --storage-type=SSD \
  --storage-auto-increase

# Create database
gcloud sql databases create verify_ai_db --instance=verify-ai-db

# Create user
gcloud sql users create verify_user \
  --instance=verify-ai-db \
  --password=SECURE_PASSWORD_HERE
```

### 7. Setup Secret Manager

```bash
# Store API keys securely
echo -n "your-gemini-api-key" | gcloud secrets create gemini-api-key \
  --data-file=- \
  --replication-policy="automatic"

echo -n "your-tavily-api-key" | gcloud secrets create tavily-api-key \
  --data-file=- \
  --replication-policy="automatic"

echo -n "your-jwt-secret-key" | gcloud secrets create jwt-secret-key \
  --data-file=- \
  --replication-policy="automatic"

# Grant access to GKE service account
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:$PROJECT_ID@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### 8. Setup Cloud Memorystore (Redis)

```bash
gcloud redis instances create verify-ai-redis \
  --size=1 \
  --region=$REGION \
  --tier=basic \
  --redis-version=redis_7_0
```

---

## Production Deployment

### 1. Build and Push Docker Images

```bash
# Authenticate Docker with GCR
gcloud auth configure-docker

# Set image registry
export IMAGE_REGISTRY="gcr.io/$PROJECT_ID"

# Build all services
./scripts/build_images.sh

# Push to GCR
docker push $IMAGE_REGISTRY/verify-ai-gateway:latest
docker push $IMAGE_REGISTRY/verify-ai-text-liar:latest
docker push $IMAGE_REGISTRY/verify-ai-text-brain2:latest
docker push $IMAGE_REGISTRY/verify-ai-image-detector:latest
docker push $IMAGE_REGISTRY/verify-ai-video-detector:latest
docker push $IMAGE_REGISTRY/verify-ai-voice-detector:latest
docker push $IMAGE_REGISTRY/verify-ai-video-worker:latest
docker push $IMAGE_REGISTRY/verify-ai-trending:latest
```

### 2. Deploy to Kubernetes

```bash
# Update image references in K8s manifests
cd infrastructure/kubernetes

# Apply ConfigMap and Secrets
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml

# Deploy database migrations job
kubectl apply -f migrations-job.yaml
kubectl wait --for=condition=complete job/db-migrations --timeout=300s

# Deploy services
kubectl apply -f deployments/
kubectl apply -f services/
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods
kubectl get services
```

### 3. Setup Ingress and Load Balancer

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Get external IP
kubectl get service -n ingress-nginx ingress-nginx-controller

# Configure DNS (add A record pointing to external IP)
# Example: api.verify-ai.com -> EXTERNAL_IP
```

### 4. Setup Auto-scaling

```bash
# Horizontal Pod Autoscaler
kubectl apply -f hpa/gateway-hpa.yaml
kubectl apply -f hpa/model-hpa.yaml

# Verify HPA
kubectl get hpa
```

---

## Environment Variables

### Production `.env` Template

```env
# Environment
ENVIRONMENT=production

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_VERSION=v1
CORS_ORIGINS=https://verify-ai.com,https://app.verify-ai.com

# Security
JWT_SECRET_KEY=<from-secret-manager>
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Database (Cloud SQL)
DATABASE_URL=postgresql+asyncpg://verify_user:PASSWORD@/verify_ai_db?host=/cloudsql/PROJECT_ID:REGION:INSTANCE_NAME
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis (Cloud Memorystore)
REDIS_URL=redis://REDIS_IP:6379/0

# Google Cloud
GCP_PROJECT_ID=verify-ai-production
GCP_REGION=us-central1
GCS_BUCKET=verify-ai-media-uploads
PUBSUB_VIDEO_TOPIC=video-processing
PUBSUB_VIDEO_SUBSCRIPTION=video-processing-sub

# AI APIs
GEMINI_API_KEY=<from-secret-manager>
GEMINI_MODEL=gemini-2.0-flash
TAVILY_API_KEY=<from-secret-manager>

# Model Service URLs (internal K8s DNS)
MODEL_TEXT_LIAR_URL=http://text-liar-service:8001
MODEL_TEXT_BRAIN2_URL=http://text-brain2-service:8002
MODEL_IMAGE_URL=http://image-detector-service:8003
MODEL_VIDEO_URL=http://video-detector-service:8004
MODEL_VOICE_URL=http://voice-detector-service:8005

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
RATE_LIMIT_PER_DAY=10000

# Monitoring
ENABLE_METRICS=true
ENABLE_TRACING=true
LOG_LEVEL=INFO
```

---

## Model Deployment

### Model Server Architecture

Each model runs in its own container with:
- FastAPI server for inference
- Model loaded from Hugging Face Hub
- GPU acceleration via CUDA
- Health checks and metrics

### Model Endpoints

All models expose:
- `POST /predict` - Run inference
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

### Example: Text Model (LIAR)

**Labels:**
- `0` = FAKE
- `1` = REAL

**Request:**
```json
{
  "text": "Your text to analyze here"
}
```

**Response:**
```json
{
  "prediction": 0,
  "confidence": 0.92,
  "probabilities": [0.92, 0.08],
  "label": "FAKE"
}
```

---

## Monitoring & Logging

### Prometheus Metrics

Access metrics at: `http://GATEWAY_IP:8000/metrics`

Key metrics:
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `model_inference_duration_seconds` - Model inference time
- `model_inference_total` - Total model predictions

### Grafana Dashboards

Access Grafana: `http://GRAFANA_IP:3001`

Default credentials: `admin / admin`

Pre-built dashboards:
1. API Gateway Overview
2. Model Performance
3. Database Metrics
4. System Resources

### Cloud Logging

View logs in GCP Console:
```bash
# Stream logs
gcloud logging read "resource.type=k8s_container" --limit=50 --format=json

# Query specific service
gcloud logging read "resource.labels.container_name=gateway" --limit=100
```

---

## Scaling & Performance

### Horizontal Scaling

```bash
# Scale gateway manually
kubectl scale deployment gateway --replicas=5

# Scale model servers
kubectl scale deployment text-liar-service --replicas=3
```

### Vertical Scaling

Update resource requests/limits in deployment YAML:

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1
  limits:
    memory: "8Gi"
    cpu: "4000m"
    nvidia.com/gpu: 1
```

### Performance Optimization

1. **Caching**: Redis caches repeated queries for 1 hour
2. **Connection Pooling**: Database pool size = 20
3. **Batch Processing**: Video processing uses queue
4. **GPU Optimization**: Mixed precision (BF16) for faster inference
5. **CDN**: Use Cloud CDN for media delivery

---

## Troubleshooting

### Common Issues

#### 1. Models Not Loading

```bash
# Check model pod logs
kubectl logs -f deployment/text-liar-service

# Verify GPU access
kubectl exec -it POD_NAME -- nvidia-smi
```

#### 2. Database Connection Issues

```bash
# Check Cloud SQL Proxy
kubectl logs -f deployment/gateway | grep database

# Test connection
kubectl exec -it POD_NAME -- psql $DATABASE_URL
```

#### 3. High Latency

```bash
# Check Prometheus metrics
kubectl port-forward svc/prometheus 9090:9090

# View slow queries
kubectl logs -f deployment/gateway | grep "Process-Time"
```

#### 4. Out of Memory

```bash
# Check resource usage
kubectl top pods

# Increase memory limits
kubectl edit deployment DEPLOYMENT_NAME
```

### Debug Mode

Enable debug logging:

```bash
kubectl set env deployment/gateway LOG_LEVEL=DEBUG
```

---

## Cost Optimization

### Estimated Monthly Costs (GCP)

| Service | Configuration | Est. Cost |
|---------|--------------|-----------|
| GKE Cluster (2 nodes) | n1-standard-4 | $150 |
| GPU Node Pool (2 nodes) | T4 GPU | $400 |
| Cloud SQL | db-custom-2-7680 | $120 |
| Cloud Storage | 100 GB | $2 |
| Cloud Memorystore | 1 GB Basic | $25 |
| Load Balancer | Standard | $20 |
| **Total** | | **~$717/month** |

### Cost Reduction Strategies

1. **Use Preemptible GPU nodes** for video processing (-70% cost)
2. **Auto-scale down** during low traffic
3. **Archive old data** to Cloud Storage Nearline
4. **Use regional deployment** instead of multi-region
5. **Implement aggressive caching**

---

## Security Checklist

- [ ] All secrets in Secret Manager (not in code)
- [ ] HTTPS/TLS enabled on all endpoints
- [ ] JWT tokens with short expiration
- [ ] Rate limiting enabled
- [ ] CORS configured for trusted origins only
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention (parameterized queries)
- [ ] File upload malware scanning
- [ ] API key rotation policy
- [ ] Regular security audits
- [ ] Database encryption at rest
- [ ] Network policies in Kubernetes
- [ ] Pod security policies enforced

---

## Support & Maintenance

### Regular Maintenance Tasks

**Daily:**
- Monitor error rates in Grafana
- Check disk usage

**Weekly:**
- Review slow query logs
- Check for security updates
- Verify backup integrity

**Monthly:**
- Update dependencies
- Review and optimize costs
- Conduct security audit
- Archive old data

### Backup Strategy

```bash
# Database backups (automated in Cloud SQL)
gcloud sql backups list --instance=verify-ai-db

# Manual backup
gcloud sql backups create --instance=verify-ai-db

# Restore from backup
gcloud sql backups restore BACKUP_ID --backup-instance=verify-ai-db --backup-location=us-central1
```

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Kubernetes Engine Docs](https://cloud.google.com/kubernetes-engine/docs)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)

---

**Need Help?** Open an issue on GitHub or contact: support@verify-ai.com
