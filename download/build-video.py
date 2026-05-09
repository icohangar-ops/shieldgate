#!/usr/bin/env python3
"""Build Convergence dashboard demo video from screenshots using ffmpeg."""

import subprocess
import os

SS_DIR = "***REDACTED_PATH***/download/screenshots"
OUT = "***REDACTED_PATH***/download/convergence-dashboard-demo.mp4"
W, H = 1265, 1200

# Video structure: each "slide" = screenshot + title card
# Total target: ~120 seconds
slides = [
    ("01-overview.png",       "Convergence — Overview",           18),
    ("02-workstreams.png",    "Workstream Health & Progress",      22),
    ("03-chp-decisions.png",  "CHP Decision Pipeline",             24),
    ("04-risks.png",          "Risk Registry",                     22),
    ("05-synergies.png",      "Synergy Pipeline & Tracking",       22),
]

# Generate fade-in/fade-out segments for each slide
filter_parts = []
inputs = []

for i, (img_file, title, duration) in enumerate(slides):
    img_path = os.path.join(SS_DIR, img_file)
    inputs.extend(["-loop", "1", "-t", str(duration), "-i", img_path])

# Build xfade filter chain
n = len(slides)
filter_complex = ""

# Apply fade in/out to each segment
for i in range(n):
    dur = slides[i][2]
    fade_in = min(1.0, 0.8)
    fade_out = min(1.0, 0.8)
    filter_complex += f"[{i}:v]format=yuv420p,fade=t=in:st=0:d={fade_in},fade=t=out:st={dur-fade_out}:d={fade_out},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=white[v{i}];"

# Chain xfade transitions
prev = "v0"
for i in range(1, n):
    transition_offset = slides[i-1][2] - 1  # 1 second overlap
    transition_type = ["fade", "slideleft", "fadeblack", "slideleft", "fade"][i-1]
    filter_complex += f"[{prev}][v{i}]xfade=transition={transition_type}:duration=1:offset={transition_offset}[xf{i}];"
    prev = f"xf{i}"

# Remove trailing semicolon
filter_complex = filter_complex.rstrip(";")

# Build ffmpeg command
cmd = [
    "ffmpeg", "-y",
    *inputs,
    "-filter_complex", filter_complex,
    "-map", f"[{prev}]",
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-r", "30",
    OUT,
]

print(f"Building video: {OUT}")
print(f"Slides: {n}, Total duration: ~{sum(s[2] for s in slides) - (n-1)}s")
print(f"Running: {' '.join(cmd[:10])}...")

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    size = os.path.getsize(OUT)
    print(f"Video created: {OUT} ({size / (1024*1024):.1f} MB)")
else:
    print(f"Error: {result.stderr[-500:]}")

