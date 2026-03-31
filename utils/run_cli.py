"""
utils/run_cli.py — Command-line runner (no Streamlit required)
==============================================================
Usage:
    python utils/run_cli.py --input path/to/video.mp4 --output tracked.mp4

This lets you run the full pipeline headlessly — useful for testing
on a server or before wiring up the Streamlit UI.
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2

from tracker.pipeline  import SportsPipeline
from tracker.analytics import Analytics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sports Multi-Object Tracker — CLI runner"
    )
    parser.add_argument("--input",        required=True,        help="Path to input video")
    parser.add_argument("--output",       default="output_tracked.mp4", help="Output video path")
    parser.add_argument("--model",        default="yolov8s.pt", help="YOLO model (e.g. yolov8s.pt)")
    parser.add_argument("--tracker",      default="botsort.yaml",help="Tracker config")
    parser.add_argument("--conf",         type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--stride",       type=int,   default=2,   help="Frame stride")
    parser.add_argument("--tail",         type=int,   default=40,  help="Tail length (frames)")
    parser.add_argument("--no-heatmap",   action="store_true",     help="Skip heatmap generation")
    parser.add_argument("--no-count",     action="store_true",     help="Skip count-over-time plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input video not found: {args.input}")
        sys.exit(1)

    # Read video metadata
    cap    = cv2.VideoCapture(args.input)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"\n{'='*55}")
    print(f"  Sports Multi-Object Tracker — CLI")
    print(f"{'='*55}")
    print(f"  Input   : {args.input}")
    print(f"  Output  : {args.output}")
    print(f"  Model   : {args.model}  |  Tracker: {args.tracker}")
    print(f"  Conf    : {args.conf}   |  Stride : {args.stride}")
    print(f"  Video   : {width}x{height} @ {fps:.1f} fps  ({total} frames)")
    print(f"{'='*55}\n")

    analytics = Analytics(frame_width=width, frame_height=height)

    pipeline = SportsPipeline(
        model_path=args.model,
        tracker_cfg=args.tracker,
        conf=args.conf,
        frame_stride=args.stride,
        tail_length=args.tail,
    )

    processed = 0
    for progress, frame_idx, _, frame_dets in pipeline.process(args.input, args.output):
        analytics.update(frame_dets, frame_idx, fps)
        processed += 1
        bar_len = 40
        filled  = int(bar_len * progress)
        bar     = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{bar}] {progress*100:.1f}%  "
            f"frame {frame_idx}/{total}  "
            f"IDs: {analytics.unique_id_count()}",
            end="", flush=True,
        )

    print(f"\n\n  ✅ Done — {processed} frames processed")
    print(f"  📹 Output saved to: {args.output}")
    print(f"  🆔 Unique IDs tracked: {analytics.unique_id_count()}")
    print(f"  📌 Peak simultaneous IDs: {analytics.peak_simultaneous_ids()}\n")

    # Save analytics plots
    if not args.no_heatmap:
        hm_path = args.output.replace(".mp4", "_heatmap.png")
        fig = analytics.plot_heatmap()
        fig.savefig(hm_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
        print(f"  🔥 Heatmap saved to: {hm_path}")

    if not args.no_count:
        ct_path = args.output.replace(".mp4", "_id_count.png")
        fig2 = analytics.plot_id_count_over_time()
        fig2.savefig(ct_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
        print(f"  📈 ID count plot saved to: {ct_path}")

    print()


if __name__ == "__main__":
    main()
