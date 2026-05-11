import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ShieldGate — Least-Privilege Agentic SOC",
  description: "AuthZed-powered authorization gateway for Splunk AI agents. Every tool call, every query, every access — governed by least-privilege policies.",
  keywords: ["AuthZed", "Splunk", "SOC", "AI Agent", "Zero Trust", "Least Privilege", "MCP"],
  authors: [{ name: "cubiczan" }],
  icons: {
    icon: "https://z-cdn.chatglm.cn/z-ai/static/logo.svg",
  },
  openGraph: {
    title: "ShieldGate — Least-Privilege Agentic SOC",
    description: "AuthZed x Splunk: Zero-trust authorization for AI-powered security operations",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
