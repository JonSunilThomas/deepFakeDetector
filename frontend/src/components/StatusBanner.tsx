"use client";

import { useEffect, useState } from "react";

interface StatusBannerProps {
  type: "success" | "error" | "warning" | "info";
  message: string;
  onDismiss?: () => void;
  autoDismiss?: number; // ms
}

const ICON_MAP = {
  success: "✅",
  error: "❌",
  warning: "⚠️",
  info: "ℹ️",
};

const STYLE_MAP = {
  success:
    "bg-emerald-500/10 border-emerald-500/40 text-emerald-300",
  error:
    "bg-red-500/10 border-red-500/40 text-red-300",
  warning:
    "bg-amber-500/10 border-amber-500/40 text-amber-300",
  info:
    "bg-blue-500/10 border-blue-500/40 text-blue-300",
};

export default function StatusBanner({
  type,
  message,
  onDismiss,
  autoDismiss = 0,
}: StatusBannerProps) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (autoDismiss > 0) {
      const timer = setTimeout(() => {
        setVisible(false);
        onDismiss?.();
      }, autoDismiss);
      return () => clearTimeout(timer);
    }
  }, [autoDismiss, onDismiss]);

  if (!visible) return null;

  return (
    <div
      role="alert"
      className={`flex items-center justify-between gap-3 px-5 py-3 mb-6 border rounded-2xl transition-all duration-300 animate-in ${STYLE_MAP[type]}`}
    >
      <div className="flex items-center gap-3">
        <span className="text-lg" aria-hidden="true">
          {ICON_MAP[type]}
        </span>
        <p className="text-sm font-medium">{message}</p>
      </div>

      {onDismiss && (
        <button
          onClick={() => {
            setVisible(false);
            onDismiss();
          }}
          className="shrink-0 w-7 h-7 flex items-center justify-center rounded-full hover:bg-white/10 transition-colors text-current opacity-60 hover:opacity-100"
          aria-label="Dismiss"
        >
          ✕
        </button>
      )}
    </div>
  );
}
