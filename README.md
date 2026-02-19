# ============================================================
#  DeepFake Detector — Decentralized Proof-of-Life Platform
# ============================================================
#
# A multi-layer biometric verification pipeline that:
#   • Verifies identity via face matching (ArcFace / Facenet)
#   • Detects liveness through active challenges + rPPG
#   • Catches deepfakes in video (XceptionNet) and audio (CNN-LSTM)
#   • Scores emotional authenticity
#   • Mints a soulbound, 5-minute Proof-of-Life NFT on-chain
#
# ──────────────────────── Quick Start ────────────────────────
#
# 1.  Copy environment template:
#         cp .env.template .env
#     Fill in secrets.
#
# 2.  Start everything with Docker Compose:
#         docker compose up --build
#
# 3.  Open the frontend:
#         http://localhost:3000
#
# 4.  Connect MetaMask to local Hardhat network (chain 31337).
#
# ──────────────────────── Manual Setup ───────────────────────
#
# Backend (Python 3.11+):
#     cd backend
#     python -m venv .venv && source .venv/bin/activate
#     pip install -r requirements.txt
#     uvicorn main:app --reload
#
# Frontend (Node 20+):
#     cd frontend
#     npm install
#     npm run dev
#
# Smart Contracts:
#     cd contracts
#     npm install
#     npx hardhat compile
#     npx hardhat node                    # terminal 1
#     npx hardhat run scripts/deploy.ts --network localhost  # terminal 2
#
# ML Training:
#     cd ml
#     pip install -r requirements.txt
#     python deepfake/train.py
#     python audio/train.py
#     python emotion/train.py
#
# ──────────────────── Architecture Overview ──────────────────
#
#   ┌──────────┐   WebRTC    ┌──────────┐   gRPC/REST   ┌──────────┐
#   │ Frontend │ ──────────► │ Backend  │ ────────────► │ ML Svc   │
#   │ Next.js  │ ◄────ws──── │ FastAPI  │ ◄──────────── │ PyTorch  │
#   └────┬─────┘             └────┬─────┘               └──────────┘
#        │                        │
#        │  Ethers.js             │  Sign payload
#        ▼                        ▼
#   ┌──────────┐           ┌──────────┐
#   │ MetaMask │           │ Hardhat  │
#   │  Wallet  │ ────────► │ EVM Node │
#   └──────────┘           └──────────┘
#
# ──────────────────── API Endpoints ──────────────────────────
#
#   POST   /api/v1/challenge          – Generate randomized challenge
#   POST   /api/v1/verify/stream      – Submit video frames for verification
#   POST   /api/v1/verify/audio       – Submit audio for voice verification
#   POST   /api/v1/aggregate          – Aggregate all scores
#   POST   /api/v1/mint               – Request Proof-of-Life NFT mint
#   WS     /api/v1/ws/scores          – Real-time score streaming
#
# ──────────────────── Monorepo Structure ─────────────────────
#
#   /frontend      Next.js 14 + TypeScript + Tailwind + WebRTC
#   /backend       FastAPI + async + Redis + PostgreSQL
#   /ml            PyTorch / TF training scripts & models
#     /face        Face match (DeepFace wrapper)
#     /deepfake    XceptionNet video deepfake detector
#     /audio       CNN-LSTM audio deepfake detector
#     /emotion     FER2013 emotion classifier
#   /contracts     Solidity + Hardhat
#   /scripts       Deploy, seed, utility scripts
#   /docker        Dockerfiles & compose
#   /docs          API spec, threat model, security checklist
#
# ──────────────────── License ────────────────────────────────
#   MIT — see LICENSE
# ============================================================
