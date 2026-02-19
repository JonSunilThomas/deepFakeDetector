/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // "standalone" is for Docker — Vercel handles this automatically.
  // Uncomment the next line only for Docker / self-hosted deployments:
  // output: "standalone",
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860",
    NEXT_PUBLIC_CONTRACT_ADDRESS: process.env.NEXT_PUBLIC_CONTRACT_ADDRESS || "",
    NEXT_PUBLIC_CHAIN_ID: process.env.NEXT_PUBLIC_CHAIN_ID || "31337",
  },
  images: {
    unoptimized: true,
  },
};

module.exports = nextConfig;
