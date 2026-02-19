"use client";

import { useCallback, useRef, useState } from "react";
import type { AnalysisResult } from "@/app/page";

interface UploadAnalyzerProps {
  apiUrl: string;
  onResult: (result: AnalysisResult) => void;
  onError: (msg: string) => void;
  analyzing: boolean;
  setAnalyzing: (v: boolean) => void;
}

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB

const ACCEPTED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/bmp",
  "video/mp4",
  "video/webm",
  "video/x-msvideo",
  "video/quicktime",
  "video/x-matroska",
];

export default function UploadAnalyzer({
  apiUrl,
  onResult,
  onError,
  analyzing,
  setAnalyzing,
}: UploadAnalyzerProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [previewType, setPreviewType] = useState<"image" | "video" | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [progress, setProgress] = useState("");

  const handleFile = useCallback((file: File) => {
    if (file.size > MAX_FILE_SIZE) {
      onError(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum is 50 MB.`);
      return;
    }

    setSelectedFile(file);

    const isVideo = file.type.startsWith("video/");
    const isImage = file.type.startsWith("image/");

    if (isImage) {
      setPreviewType("image");
      const url = URL.createObjectURL(file);
      setPreview(url);
    } else if (isVideo) {
      setPreviewType("video");
      const url = URL.createObjectURL(file);
      setPreview(url);
    } else {
      // Try by extension
      const ext = file.name.split(".").pop()?.toLowerCase();
      if (["mp4", "webm", "avi", "mov", "mkv"].includes(ext || "")) {
        setPreviewType("video");
        setPreview(URL.createObjectURL(file));
      } else {
        setPreviewType("image");
        setPreview(URL.createObjectURL(file));
      }
    }
  }, [onError]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const analyze = useCallback(async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    setProgress("Uploading file…");
    onError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      setProgress("Analyzing with AI models… This may take a minute.");

      const res = await fetch(`${apiUrl}/api/v1/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `Server error (${res.status})` }));
        throw new Error(err.detail || "Analysis failed");
      }

      const data = await res.json();
      setProgress("");
      onResult(data);
    } catch (err: any) {
      onError(err.message || "Analysis failed. Make sure the backend is running.");
      setProgress("");
    } finally {
      setAnalyzing(false);
    }
  }, [selectedFile, apiUrl, onResult, onError, setAnalyzing]);

  const clearSelection = () => {
    setSelectedFile(null);
    setPreview(null);
    setPreviewType(null);
    setProgress("");
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="space-y-6">
      {/* Drop zone */}
      {!selectedFile ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`glass cursor-pointer transition-all duration-300 ${
            dragOver
              ? "border-indigo-400 bg-indigo-500/10 scale-[1.01]"
              : "hover:bg-white/[0.07] hover:border-white/20"
          }`}
        >
          <div className="flex flex-col items-center justify-center py-20 px-8">
            <div className="w-20 h-20 rounded-full bg-indigo-500/10 flex items-center justify-center mb-5">
              <svg
                className="w-10 h-10 text-indigo-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                />
              </svg>
            </div>
            <p className="text-lg font-medium text-white mb-2">
              Drop a file here or click to browse
            </p>
            <p className="text-sm text-slate-400">
              Images (JPG, PNG, WebP) or Videos (MP4, WebM, AVI)
            </p>
            <p className="text-xs text-slate-500 mt-1">Up to 50 MB</p>
          </div>
        </div>
      ) : (
        /* Preview + Analyze */
        <div className="glass p-6 space-y-5">
          {/* File info bar */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded-lg bg-indigo-500/10 flex items-center justify-center shrink-0">
                <span className="text-lg">
                  {previewType === "video" ? "🎬" : "🖼️"}
                </span>
              </div>
              <div className="min-w-0">
                <p className="text-sm font-medium text-white truncate">
                  {selectedFile.name}
                </p>
                <p className="text-xs text-slate-400">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB &bull;{" "}
                  {previewType === "video" ? "Video" : "Image"}
                </p>
              </div>
            </div>
            <button
              onClick={clearSelection}
              disabled={analyzing}
              className="text-slate-400 hover:text-white transition-colors p-2 rounded-lg hover:bg-white/10 disabled:opacity-30"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Preview */}
          <div className="rounded-xl overflow-hidden border border-white/10 bg-black/50 max-h-[400px] flex items-center justify-center">
            {previewType === "image" && preview && (
              <img
                src={preview}
                alt="Preview"
                className="max-h-[400px] w-auto object-contain"
              />
            )}
            {previewType === "video" && preview && (
              <video
                src={preview}
                controls
                className="max-h-[400px] w-auto"
                style={{ maxWidth: "100%" }}
              />
            )}
          </div>

          {/* Analyze button */}
          <button
            onClick={analyze}
            disabled={analyzing}
            className="w-full py-4 px-6 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold rounded-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed text-lg"
          >
            {analyzing ? (
              <span className="flex items-center justify-center gap-3">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Analyzing…
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                🔍 Analyze for Deepfakes
              </span>
            )}
          </button>

          {/* Progress message */}
          {progress && (
            <div className="flex items-center gap-3 px-4 py-3 bg-indigo-500/10 border border-indigo-500/20 rounded-xl">
              <div className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse" />
              <p className="text-sm text-indigo-300">{progress}</p>
            </div>
          )}
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*,video/*,.mp4,.webm,.avi,.mov,.mkv"
        onChange={handleInputChange}
        className="hidden"
      />
    </div>
  );
}
