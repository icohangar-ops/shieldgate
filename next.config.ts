import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
  serverExternalPackages: [
    "@authzed/authzed-node",
    "@grpc/grpc-js",
    "@grpc/proto-loader",
    "splunk-sdk",
  ],
};

export default nextConfig;
