# Deployment Guide

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Vercel          │  HTTPS  │  Render           │
│  (Frontend)      │────────►│  (Backend + ML)   │
│  Next.js 14      │         │  FastAPI + PyTorch │
│  deepfake.app    │         │  api.deepfake.app  │
└─────────────────┘         └──────────────────┘
```

---

## Step 1: Push to GitHub

```bash
# Initialize repo (if not already)
cd /path/to/DeepFackeDetector
git init
git add .
git commit -m "Initial commit"

# Create repo on GitHub, then:
git remote add origin https://github.com/<YOUR_USERNAME>/DeepFackeDetector.git
git branch -M main
git push -u origin main
```

### Upload ML Weights to GitHub Releases

The `.pth` weight files are gitignored (too large for git). Host them as
GitHub Release assets:

```bash
# Create a release and upload weights
gh release create v1.0 \
  ml/deepfake/weights/xception_celeb_df.pth \
  ml/audio/weights/audio_cnn_lstm.pth \
  backend/weights/face_landmarker.task \
  --title "v1.0 — Model Weights" \
  --notes "Trained model weights for deployment"
```

Your weights URL will be:
```
https://github.com/<YOUR_USERNAME>/DeepFackeDetector/releases/download/v1.0
```

---

## Step 2: Deploy Backend on Render

1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub repo
3. Configure:
   - **Name**: `deepfake-backend`
   - **Root Directory**: `backend`
   - **Runtime**: Docker
   - **Dockerfile Path**: `./backend/Dockerfile`
   - **Docker Context**: `.` (repo root — needed for `ml/` weights path)
   - **Plan**: **Standard** ($25/mo) — minimum 2 GB RAM for ML models
   - **Region**: Oregon (or closest to your users)

4. Set **Environment Variables**:
   | Key | Value |
   |-----|-------|
   | `WEIGHTS_BASE_URL` | `https://github.com/<user>/DeepFackeDetector/releases/download/v1.0` |
   | `PYTHONUNBUFFERED` | `1` |
   | `JAX_PLATFORMS` | `cpu` |
   | `CUDA_VISIBLE_DEVICES` | *(empty)* |

5. Click **Create Web Service**

6. Wait for build + deploy (~5-10 min for Docker + weight download)

7. Note your backend URL: `https://deepfake-backend.onrender.com`

8. Test it:
   ```bash
   curl https://deepfake-backend.onrender.com/api/v1/health
   curl https://deepfake-backend.onrender.com/api/v1/models/status
   ```

### ⚠️ Render Important Notes
- **Free tier won't work** — ML models need ≥2 GB RAM
- **Cold starts**: Render spins down free/starter services after 15 min idle.
  Standard plan stays warm.
- **Build time**: First build takes ~10 min (PyTorch is large).
  Subsequent builds use Docker layer cache.
- **Disk**: If you added a persistent disk in `render.yaml`, weights survive
  redeploys.

---

## Step 3: Deploy Frontend on Vercel

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
   | `NEXT_PUBLIC_API_URL` | `https://deepfake-backend.onrender.com` |

5. Click **Deploy**

6. Your frontend will be live at: `https://your-project.vercel.app`

### ⚠️ Vercel Important Notes
- **Free tier works fine** for the frontend — it's just Next.js
- Vercel auto-deploys on every push to `main`
- Set the `NEXT_PUBLIC_API_URL` env var **before** the first build

---

## Step 4: Verify End-to-End

1. Open your Vercel URL in a browser
2. Check the **Model Status** sidebar — all 4 models should show green
3. Upload a test image → you should get deepfake/liveness/emotion scores
4. Upload a test video → frame-by-frame timeline should appear

---

## Custom Domain (Optional)

### Vercel (frontend)
1. Vercel Dashboard → Settings → Domains → Add `deepfake.yourdomain.com`
2. Add CNAME record: `deepfake.yourdomain.com` → `cname.vercel-dns.com`

### Render (backend)
1. Render Dashboard → Settings → Custom Domains → Add `api.deepfake.yourdomain.com`
2. Add CNAME record: `api.deepfake.yourdomain.com` → `deepfake-backend.onrender.com`

Then update Vercel env var:
```
NEXT_PUBLIC_API_URL=https://api.deepfake.yourdomain.com
```

---

## Cost Estimate

| Service | Plan | Cost |
|---------|------|------|
| Vercel (frontend) | Hobby (free) | $0/mo |
| Render (backend) | Standard | $25/mo |
| GitHub | Free | $0/mo |
| **Total** | | **$25/mo** |

For GPU inference (faster, better for video): Render GPU instances start at ~$100/mo.
