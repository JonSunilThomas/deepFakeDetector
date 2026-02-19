"use client";

import { useState } from "react";
import UploadAnalyzer from "../components/UploadAnalyzer";
import AnalysisResults from "../components/AnalysisResults";
import ModelStatus from "@/components/ModelStatus";
import StatusBanner from "@/components/StatusBanner";

export interface AnalysisResult {
  type: "image" | "video";
  filename: string;
  total_time_ms: number;
  frames_analyzed: number;
  deepfake: {
    fake_probability?: number;
    avg_fake_probability?: number;
    confidence?: number;
    min_fake?: number;
    max_fake?: number;
    per_frame?: number[];
    inference_ms?: number;
    error?: string;
  };
  liveness: {
    score?: number;
    avg_score?: number;
    face_detected?: boolean;
    inference_ms?: number;
    error?: string;
  };
  emotion: {
    detected?: string;
    dominant?: string;
    confidence?: number;
    all_emotions?: Record<string, number>;
    duchenne?: any;
    per_frame?: string[];
    inference_ms?: number;
    error?: string;
  };
  verdict: {
    is_likely_fake: boolean;
    fake_probability: number;
    real_probability: number;
    label: string;
  };
  video_info?: {
    duration_s: number;
    total_frames: number;
    fps: number;
  };
  frame_details?: Array<{
    frame_idx: number;
    timestamp_s: number;
    fake_probability?: number;
    liveness_score?: number;
    emotion?: string;
  }>;
}

export default function Home() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [analyzing, setAnalyzing] = useState(false);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const handleAnalysisComplete = (data: AnalysisResult) => {
    setResult(data);
    setError("");
  };

  const handleReset = () => {
    setResult(null);
    setError("");
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-950 p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            DeepFake Detector
          </h1>
          <p className="text-slate-400 mt-2 text-lg">
            Upload an image or video to check if it&apos;s AI-generated
          </p>
          <p className="text-slate-500 mt-1 text-sm">
            Powered by XceptionNet &bull; MediaPipe &bull; CNN-LSTM &bull; Emotion Analysis
          </p>
        </header>

        {error && (
          <StatusBanner type="error" message={error} onDismiss={() => setError("")} />
        )}

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Sidebar */}
          <div className="space-y-6">
            {/* Model Status */}
            <div className="glass p-5">
              <h2 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <span>🧠</span> Model Status
              </h2>
              <ModelStatus apiUrl={API_URL} />
            </div>

            {/* How it works */}
            <div className="glass p-5">
              <h2 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <span>📖</span> How It Works
              </h2>
              <div className="space-y-3 text-xs text-slate-400">
                <div className="flex gap-2">
                  <span className="text-indigo-400 font-bold shrink-0">1.</span>
                  <span>Upload any image or video file</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-indigo-400 font-bold shrink-0">2.</span>
                  <span>XceptionNet analyzes each frame for deepfake artifacts</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-indigo-400 font-bold shrink-0">3.</span>
                  <span>MediaPipe checks for liveness cues &amp; facial landmarks</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-indigo-400 font-bold shrink-0">4.</span>
                  <span>Emotion CNN detects expression authenticity</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-indigo-400 font-bold shrink-0">5.</span>
                  <span>Get a verdict: Real, Fake, or Uncertain</span>
                </div>
              </div>
            </div>

            {/* Supported formats */}
            <div className="glass p-5">
              <h2 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <span>📁</span> Supported Formats
              </h2>
              <div className="space-y-2 text-xs text-slate-400">
                <p><span className="text-slate-300 font-medium">Images:</span> JPG, PNG, WebP, BMP</p>
                <p><span className="text-slate-300 font-medium">Video:</span> MP4, WebM, AVI, MOV</p>
                <p className="text-slate-500 mt-2">Max file size: 50 MB</p>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            {!result ? (
              <UploadAnalyzer
                apiUrl={API_URL}
                onResult={handleAnalysisComplete}
                onError={setError}
                analyzing={analyzing}
                setAnalyzing={setAnalyzing}
              />
            ) : (
              <AnalysisResults result={result} onReset={handleReset} />
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-12 text-slate-500 text-sm">
          <p>DeepFake Detector — Open Source AI Media Analysis</p>
          <p className="mt-1 text-slate-600">
            Models: XceptionNet (CelebDF) &bull; MediaPipe Face Landmarker &bull; CNN-LSTM Audio &bull; EmotionCNN (FER2013)
          </p>
        </footer>
      </div>
    </main>
  );
}
