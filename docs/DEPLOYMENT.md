# Deployment Guide

## Architecture

```
┌─────────────────┐  HTTPS   ┌───────────────────────────────┐
│  Vercel          │─────────►│  Hugging Face Spaces          │
│  (Frontend)      │          │  (Backend + ML Models)        │
│  Next.js 14      │          │  FastAPI + PyTorch + MediaPipe│
│  your-app.vercel │          │  hf.co/spaces/you/deepfake    │
└─────────────────┘          └───────────────────────────────┘
                                       │
                              Downloads weights from
                                       ▼
                             ┌─────────────────────┐
                             │  HF Model Repo       │
                             │  you/deepfake-weights │
                             │  .pth + .task files   │
                             └─────────────────────┘
```

**Why Hugging Face Spaces?**
- **Free tier**: 2 vCPU, 16 GB RAM — plenty for ML inference
- **No cold starts** on Docker Spaces (always-on)
- **Native model hosting** — weights live on HF Hub (Git LFS)
- **GPU upgrade** available ($0.60/hr for T4, $1.05/hr for A10G)
- **Community visibility** — your Space gets a public demo page

---

## Step 1: Push Code to GitHub

```bash
cd /path/to/DeepFackeDetector
git add .
git commit -m "Configure for HF Spaces + Vercel deployment"
git push origin main
```

---

## Step 2: Upload Weights to Hugging Face

### 2a. Create a HF Model Repository

1. Go to [huggingface.co/new](https://huggingface.co/new) → **New Model**
2. Name it: `deepfake-weights`
3. Visibility: **Public** (or Private with an access token)

### 2b. Upload Weight Files

```bash
# Install the HF CLI (if not already)
pip install huggingface_hub

# Login
huggingface-cli login

# Upload weights
huggingface-cli upload JonSunilThomas/deepfake-weights \
    ml/deepfake/weights/xception_celeb_df.pth \
    xception_celeb_df.pth

huggingface-cli upload JonSunilThomas/deepfake-weights \
    ml/audio/weights/audio_cnn_lstm.pth \
    audio_cnn_lstm.pth

huggingface-cli upload JonSunilThomas/deepfake-weights \
    backend/weights/face_landmarker.task \
    face_landmarker.task
```

Your model repo: `https://huggingface.co/JonSunilThomas/deepfake-weights`

---

## Step 3: Deploy Backend on Hugging Face Spaces

### 3a. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Configure:
   - **Space name**: `deepfake-detector`
   - **SDK**: **Docker**
   - **Hardware**: **CPU basic** (free) — upgrade to GPU later if needed
   - **Visibility**: Public

### 3b. Connect to GitHub (Recommended)

**Option A — GitHub Actions sync (auto-deploy on push):**
- In the Space Settings → **Repository** → link to your GitHub repo
- HF will mirror your repo and rebuild on every push

**Option B — Push directly to HF:**
```bash
# Clone your HF Space
git clone https://huggingface.co/spaces/JonSunilThomas/deepfake-detector
cd deepfake-detector

# Copy your project files
cp -r /path/to/DeepFackeDetector/* .

# Push to HF (this triggers a build)
git add .
git commit -m "Deploy backend"
git push
```

### 3c. Set Environment Variables

In the Space **Settings** → **Variables and secrets**:

| Key | Value |
|-----|-------|
| `HF_WEIGHTS_REPO` | `JonSunilThomas/deepfake-weights` |
| `PYTHONUNBUFFERED` | `1` |
| `JAX_PLATFORMS` | `cpu` |

### 3d. Wait for Build

- The Docker build takes ~8-12 minutes (PyTorch is large)
- Watch the **Build logs** tab for progress
- Once running, your backend is live at:

```
https://JonSunilThomas-deepfake-detector.hf.space
```

### 3e. Test It

```bash
# Health check
curl https://JonSunilThomas-deepfake-detector.hf.space/api/v1/health

# Model status
curl https://JonSunilThomas-deepfake-detector.hf.space/api/v1/models/status

# API docs (Swagger UI)
open https://JonSunilThomas-deepfake-detector.hf.space/docs
```

---

## Step 4: Deploy Frontend on Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New** → **Project**
2. Import your GitHub repo
3. Configure:
   - **Framework Preset**: Next.js (auto-detected)
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`

4. Set **Environment Variables**:

   | Key | Value |
   |-----|-------|
   | `NEXT_PUBLIC_API_URL` | `https://JonSunilThomas-deepfake-detector.hf.space` |

5. Click **Deploy**

6. Your frontend: `https://your-project.vercel.app`

---

## Step 5: Verify End-to-End

1. Open your Vercel URL in a browser
2. Check the **Model Status** sidebar — all 4 models should show green ✅
3. Upload a test image → deepfake/liveness/emotion scores appear
4. Upload a test video → frame-by-frame timeline with per-frame analysis

---

## Upgrading to GPU (Optional)

For faster inference (especially video with 20+ frames):

1. HF Space **Settings** → **Hardware** → **T4 GPU** ($0.60/hr)
2. Remove the CPU-forcing env vars:
   - Delete `JAX_PLATFORMS=cpu`
   - Delete `CUDA_VISIBLE_DEVICES=""`
3. The models will automatically use CUDA when available

---

## Custom Domain (Optional)

### Vercel (frontend)
1. Vercel Dashboard → Settings → Domains → Add `deepfake.yourdomain.com`
2. Add CNAME: `deepfake.yourdomain.com` → `cname.vercel-dns.com`

### Hugging Face (backend)
HF Spaces don't support custom domains directly. Use a reverse proxy
(Cloudflare Workers, Vercel rewrites, etc.) if needed:

```js
// In frontend/next.config.js — add API proxy rewrites
async rewrites() {
  return [
    {
      source: '/api/:path*',
      destination: 'https://JonSunilThomas-deepfake-detector.hf.space/api/:path*',
    },
  ];
},
```

---

## Cost Estimate

| Service | Plan | Cost |
|---------|------|------|
| Vercel (frontend) | Hobby (free) | $0/mo |
| HF Spaces (backend) | CPU Basic (free) | $0/mo |
| HF Model Repo (weights) | Free | $0/mo |
| **Total** | | **$0/mo** |

| Upgrade | Cost |
|---------|------|
| HF T4 GPU | ~$0.60/hr (≈$432/mo if always on) |
| HF A10G GPU | ~$1.05/hr |
| HF Persistent Storage (small) | $5/mo |
| Vercel Pro | $20/mo |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Space shows "Building…" for >15 min | Check build logs; PyTorch download can be slow |
| Models show "No weights" | Verify `HF_WEIGHTS_REPO` is set and repo is public |
| CORS errors in browser | Backend already has `allow_origins=["*"]` — check browser console |
| Space goes to sleep | Free CPU spaces sleep after 48hr idle; upgrade to persistent |
| "Out of memory" on video | Upgrade to more RAM or GPU hardware |
