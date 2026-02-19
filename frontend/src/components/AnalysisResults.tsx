"use client";

import type { AnalysisResult } from "@/app/page";

interface AnalysisResultsProps {
  result: AnalysisResult;
  onReset: () => void;
}

function VerdictBadge({ verdict }: { verdict: AnalysisResult["verdict"] }) {
  const isReal = verdict.label === "LIKELY REAL";
  const isFake = verdict.label === "LIKELY FAKE";

  const bgColor = isFake
    ? "from-red-600 to-red-500"
    : isReal
    ? "from-emerald-600 to-emerald-500"
    : "from-amber-600 to-amber-500";

  const icon = isFake ? "⚠️" : isReal ? "✅" : "🤔";

  return (
    <div className={`bg-gradient-to-r ${bgColor} rounded-2xl p-6 text-center`}>
      <div className="text-4xl mb-2">{icon}</div>
      <h3 className="text-2xl font-bold text-white">{verdict.label}</h3>
      <div className="mt-3 flex items-center justify-center gap-6">
        <div>
          <p className="text-white/60 text-xs uppercase tracking-wide">Fake Probability</p>
          <p className="text-2xl font-bold font-mono text-white">
            {(verdict.fake_probability * 100).toFixed(1)}%
          </p>
        </div>
        <div className="w-px h-10 bg-white/20" />
        <div>
          <p className="text-white/60 text-xs uppercase tracking-wide">Real Probability</p>
          <p className="text-2xl font-bold font-mono text-white">
            {(verdict.real_probability * 100).toFixed(1)}%
          </p>
        </div>
      </div>
    </div>
  );
}

function ScoreBar({
  label,
  value,
  icon,
  inverted = false,
  suffix = "",
}: {
  label: string;
  value: number;
  icon: string;
  inverted?: boolean;
  suffix?: string;
}) {
  const display = Math.max(0, Math.min(1, value));
  const barWidth = inverted ? (1 - display) * 100 : display * 100;
  const effective = inverted ? 1 - display : display;

  const color =
    effective >= 0.7
      ? "from-emerald-500 to-emerald-400"
      : effective >= 0.45
      ? "from-amber-500 to-yellow-400"
      : "from-red-600 to-red-400";

  const textColor =
    effective >= 0.7
      ? "text-emerald-400"
      : effective >= 0.45
      ? "text-amber-400"
      : "text-red-400";

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-sm text-slate-300 flex items-center gap-1.5">
          <span>{icon}</span>
          {label}
        </span>
        <span className={`text-sm font-mono font-semibold ${textColor}`}>
          {inverted
            ? `${((1 - display) * 100).toFixed(1)}%`
            : `${(display * 100).toFixed(1)}%`}
          {suffix}
        </span>
      </div>
      <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-700 ease-out`}
          style={{ width: `${barWidth}%` }}
        />
      </div>
    </div>
  );
}

function FrameTimeline({ frames }: { frames: NonNullable<AnalysisResult["frame_details"]> }) {
  if (frames.length === 0) return null;

  const maxFake = Math.max(...frames.map((f) => f.fake_probability ?? 0));
  const minFake = Math.min(...frames.map((f) => f.fake_probability ?? 1));

  return (
    <div className="glass p-5">
      <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
        <span>📊</span> Frame-by-Frame Analysis
      </h3>
      <div className="flex items-end gap-0.5 h-24">
        {frames.map((frame, i) => {
          const fakeProb = frame.fake_probability ?? 0.5;
          const height = Math.max(8, fakeProb * 100);
          const isFake = fakeProb > 0.5;

          return (
            <div
              key={i}
              className="flex-1 group relative"
              title={`Frame ${frame.frame_idx} (${frame.timestamp_s}s): ${(fakeProb * 100).toFixed(1)}% fake`}
            >
              <div
                className={`w-full rounded-t transition-all duration-200 ${
                  isFake
                    ? "bg-gradient-to-t from-red-600 to-red-400"
                    : "bg-gradient-to-t from-emerald-600 to-emerald-400"
                } group-hover:opacity-80`}
                style={{ height: `${height}%` }}
              />
              {/* Tooltip */}
              <div className="absolute -top-8 left-1/2 -translate-x-1/2 hidden group-hover:block bg-slate-800 text-[10px] text-white px-2 py-1 rounded whitespace-nowrap z-10">
                {(fakeProb * 100).toFixed(1)}% fake
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex justify-between mt-1 text-[10px] text-slate-500">
        <span>{frames[0]?.timestamp_s}s</span>
        <span>{frames[frames.length - 1]?.timestamp_s}s</span>
      </div>
      <div className="flex items-center gap-4 mt-2 text-[11px] text-slate-400">
        <div className="flex items-center gap-1">
          <div className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />
          <span>Real</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2.5 h-2.5 rounded-sm bg-red-500" />
          <span>Fake</span>
        </div>
        <span className="text-slate-500">
          Range: {(minFake * 100).toFixed(0)}% – {(maxFake * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

function EmotionBreakdown({ emotions }: { emotions: Record<string, number> }) {
  const sorted = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
  const top = sorted[0];

  return (
    <div className="space-y-2">
      {sorted.map(([emotion, score]) => (
        <div key={emotion} className="flex items-center gap-2">
          <span className="text-xs text-slate-400 w-16 text-right capitalize">{emotion}</span>
          <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                emotion === top[0] ? "bg-indigo-400" : "bg-slate-600"
              }`}
              style={{ width: `${score * 100}%` }}
            />
          </div>
          <span className="text-[11px] font-mono text-slate-500 w-12">
            {(score * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

export default function AnalysisResults({ result, onReset }: AnalysisResultsProps) {
  const fakeProbability =
    result.deepfake.fake_probability ?? result.deepfake.avg_fake_probability ?? 0.5;
  const livenessScore = result.liveness.score ?? result.liveness.avg_score ?? 0;
  const emotionDetected = result.emotion.detected ?? result.emotion.dominant ?? "unknown";

  return (
    <div className="space-y-6">
      {/* Verdict */}
      <VerdictBadge verdict={result.verdict} />

      {/* Meta info */}
      <div className="glass p-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-slate-400">
        <span>📄 {result.filename}</span>
        <span>📐 {result.type === "video" ? "Video" : "Image"}</span>
        {result.video_info && (
          <>
            <span>⏱ {result.video_info.duration_s}s</span>
            <span>🎞 {result.video_info.fps} FPS</span>
            <span>📷 {result.video_info.total_frames} total frames</span>
          </>
        )}
        <span>🔬 {result.frames_analyzed} frames analyzed</span>
        <span>⚡ {(result.total_time_ms / 1000).toFixed(1)}s total</span>
      </div>

      {/* Score meters */}
      <div className="glass p-6 space-y-4">
        <h3 className="text-sm font-semibold text-white mb-1 flex items-center gap-2">
          <span>📊</span> Detailed Scores
        </h3>
        <ScoreBar
          label="Deepfake Detection"
          value={fakeProbability}
          inverted
          icon="🛡️"
        />
        <ScoreBar
          label="Liveness Score"
          value={livenessScore}
          icon="💓"
        />
        {result.emotion.confidence != null && (
          <ScoreBar
            label="Emotion Confidence"
            value={result.emotion.confidence}
            icon="😊"
          />
        )}
      </div>

      {/* Emotion breakdown */}
      {result.emotion.all_emotions && Object.keys(result.emotion.all_emotions).length > 0 && (
        <div className="glass p-6">
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span>🎭</span> Emotion Analysis
          </h3>
          <p className="text-xs text-slate-400 mb-3">
            Detected emotion:{" "}
            <span className="text-indigo-400 font-medium capitalize">{emotionDetected}</span>
            {result.emotion.duchenne && (
              <span className="ml-2">
                {result.emotion.duchenne.is_duchenne ? "😊 Genuine smile" : ""}
              </span>
            )}
          </p>
          <EmotionBreakdown emotions={result.emotion.all_emotions} />
        </div>
      )}

      {/* Video frame timeline */}
      {result.type === "video" && result.frame_details && result.frame_details.length > 0 && (
        <FrameTimeline frames={result.frame_details} />
      )}

      {/* Video per-frame emotions */}
      {result.type === "video" && result.emotion.per_frame && result.emotion.per_frame.length > 0 && (
        <div className="glass p-5">
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span>🎭</span> Emotion Timeline
          </h3>
          <p className="text-xs text-slate-400 mb-2">
            Dominant emotion: <span className="text-indigo-400 font-medium capitalize">{result.emotion.dominant}</span>
          </p>
          <div className="flex flex-wrap gap-1.5">
            {result.emotion.per_frame.map((emo, i) => (
              <span
                key={i}
                className="text-[10px] px-2 py-0.5 rounded-full bg-white/5 text-slate-400 border border-white/5"
              >
                {emo}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Analyze another */}
      <button
        onClick={onReset}
        className="w-full py-4 px-6 glass glass-hover text-white font-semibold rounded-xl text-base flex items-center justify-center gap-2"
      >
        ← Analyze Another File
      </button>
    </div>
  );
}
