import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DeepFake Detector — AI-Powered Media Analysis",
  description: "Upload images or videos to detect AI-generated deepfakes using XceptionNet, MediaPipe, and emotion analysis.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
