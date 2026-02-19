"use client";

import { useEffect, useState } from "react";

interface ModelStatusProps {
  apiUrl: string;
}

interface ModelInfo {
  loaded: boolean;
  weights?: string;
}

interface StatusData {
  deepfake_detector: ModelInfo;
  audio_detector: ModelInfo;
  liveness_engine: ModelInfo;
  emotion_scorer: ModelInfo;
}

export default function ModelStatus({ apiUrl }: ModelStatusProps) {
  const [status, setStatus] = useState<StatusData | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${apiUrl}/api/v1/models/status`);
        if (res.ok) {
          setStatus(await res.json());
          setBackendOnline(true);
        } else {
          setBackendOnline(false);
        }
      } catch {
        setBackendOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, [apiUrl]);

  const models = status
    ? [
        { name: "Deepfake (XceptionNet)", ...status.deepfake_detector },
        { name: "Audio (CNN-LSTM)", ...status.audio_detector },
        { name: "Liveness (MediaPipe)", ...status.liveness_engine },
        { name: "Emotion (CNN)", ...status.emotion_scorer },
      ]
    : [];

  const allLoaded = models.every((m) => m.loaded);

  return (
    <div className="space-y-3">
      {/* Backend connection */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-slate-400">Backend API</span>
        <span className="flex items-center gap-1.5">
          <span
            className={`w-2 h-2 rounded-full ${
              backendOnline === null
                ? "bg-slate-500 animate-pulse"
                : backendOnline
                ? "bg-emerald-400"
                : "bg-red-400"
            }`}
          />
          <span
            className={`text-xs font-medium ${
              backendOnline === null
                ? "text-slate-500"
                : backendOnline
                ? "text-emerald-400"
                : "text-red-400"
            }`}
          >
            {backendOnline === null
              ? "Checking…"
              : backendOnline
              ? "Online"
              : "Offline"}
          </span>
        </span>
      </div>

      {/* Model list */}
      {status && (
        <>
          <div className="border-t border-white/5 pt-2" />
          {models.map((m) => (
            <div key={m.name} className="flex items-center justify-between">
              <span className="text-xs text-slate-400">{m.name}</span>
              <span className="flex items-center gap-1.5">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    m.loaded ? "bg-emerald-400" : "bg-amber-400"
                  }`}
                />
                <span
                  className={`text-[11px] font-mono ${
                    m.loaded ? "text-emerald-400" : "text-amber-400"
                  }`}
                >
                  {m.loaded ? "Loaded" : "No weights"}
                </span>
              </span>
            </div>
          ))}
          <div className="border-t border-white/5 pt-2">
            <div className="flex items-center gap-2">
              <span className="text-xs">
                {allLoaded ? "🟢" : "🟡"}
              </span>
              <span
                className={`text-xs font-semibold ${
                  allLoaded ? "text-emerald-400" : "text-amber-400"
                }`}
              >
                {allLoaded
                  ? "All models ready"
                  : "Some models using random init"}
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
