"""
utils/download_video.py — Download a public YouTube video for testing
======================================================================
Requires:  pip install yt-dlp

Usage:
    python utils/download_video.py --url "https://youtu.be/XXXXXXXXXXX"
    python utils/download_video.py --url "https://youtu.be/XXXXXXXXXXX" --out my_video.mp4 --max-height 720
"""

import argparse
import subprocess
import sys
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a public YouTube video via yt-dlp"
    )
    parser.add_argument("--url",        required=True,        help="YouTube (or other public) video URL")
    parser.add_argument("--out",        default="input_video.mp4", help="Output filename")
    parser.add_argument("--max-height", type=int, default=720,  help="Max vertical resolution (default 720p)")
    args = parser.parse_args()

    if shutil.which("yt-dlp") is None:
        print("[ERROR] yt-dlp not found. Install with:  pip install yt-dlp")
        sys.exit(1)

    fmt = f"best[height<={args.max_height}][ext=mp4]/best[height<={args.max_height}]"
    cmd = ["yt-dlp", "-f", fmt, "-o", args.out, args.url]

    print(f"Downloading: {args.url}")
    print(f"Output     : {args.out}")
    print(f"Max height : {args.max_height}p\n")

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\n✅ Saved to: {args.out}")
    else:
        print("\n❌ Download failed. Check the URL and your yt-dlp installation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
