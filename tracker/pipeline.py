"""
tracker/pipeline.py — YOLOv8 + BoT-SORT core pipeline
=======================================================
Handles frame-by-frame detection, persistent ID tracking,
trajectory tail drawing, and annotated video writing.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# Colour palette: 40 visually distinct colours for track IDs
# ─────────────────────────────────────────────────────────────
_PALETTE = [
    (255, 56,  56),  (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210,  49), (72,  249, 10),  (146, 204, 23),  (61,  219, 134),
    (26,  147, 52),  (0,   212, 187), (44,  153, 168), (0,   194, 255),
    (52,  69,  147), (100, 115, 255), (0,   24,  236), (132,  56, 255),
    (82,   0, 133),  (203,  56, 255), (255, 149, 200), (255,  55, 199),
    (255, 255, 0),   (0,   255, 255), (255, 0,   255), (0,   128, 255),
    (255, 128, 0),   (128, 255, 0),   (0,   255, 128), (128, 0,   255),
    (255, 0,   128), (0,   255, 0),   (255, 0,   0),   (0,   0,   255),
    (200, 100, 50),  (50,  200, 100), (100, 50,  200), (200, 200, 50),
    (50,  200, 200), (200, 50,  200), (150, 150, 0),   (0,   150, 150),
]


def _colour(track_id: int) -> tuple[int, int, int]:
    """Return a consistent BGR colour for a given track ID."""
    return _PALETTE[int(track_id) % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────
# SportsPipeline
# ─────────────────────────────────────────────────────────────
class SportsPipeline:
    """
    End-to-end sports video tracking pipeline.

    Parameters
    ----------
    model_path   : Ultralytics model identifier, e.g. 'yolov8s.pt'
    tracker_cfg  : Tracker config, 'botsort.yaml' or 'bytetrack.yaml'
    conf         : Detection confidence threshold (0–1)
    frame_stride : Process every Nth frame (1 = every frame)
    tail_length  : Number of past positions to keep for trajectory tails
                   (0 = disable tails)
    classes      : YOLO class indices to detect (None = all; 0 = person)
    """

    def __init__(
        self,
        model_path:   str   = "yolov8s.pt",
        tracker_cfg:  str   = "botsort.yaml",
        conf:         float = 0.3,
        frame_stride: int   = 2,
        tail_length:  int   = 40,
        classes:      list  | None = None,
    ) -> None:
        self.model        = YOLO(model_path)
        self.tracker_cfg  = tracker_cfg
        self.conf         = conf
        self.frame_stride = max(1, frame_stride)
        self.tail_length  = tail_length
        self.classes      = classes  # e.g. [0] to detect only "person"

        # track_id → deque of (cx, cy) centre points for trajectory tails
        self._tails: dict[int, collections.deque] = {}

    # ── public API ───────────────────────────────────────────

    def process(
        self,
        input_path:  str,
        output_path: str,
    ) -> Generator[tuple[float, int, np.ndarray, list[dict]], None, None]:
        """
        Process a video file frame-by-frame.

        Yields
        ------
        (progress_fraction, frame_index, annotated_frame, detections_this_frame)

        detections_this_frame is a list of dicts:
            {track_id, class_name, confidence, bbox_xyxy, centre}
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = self._make_writer(output_path, fps, width, height)

        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # ── Skip frames per stride ───────────────────
                if frame_idx % self.frame_stride != 0:
                    writer.write(frame)       # write unmodified to keep timing
                    continue

                # ── Run tracker ──────────────────────────────
                results = self.model.track(
                    frame,
                    persist=True,             # CRITICAL: keeps IDs alive between frames
                    conf=self.conf,
                    tracker=self.tracker_cfg,
                    classes=self.classes,
                    verbose=False,
                )

                detections, annotated = self._annotate(frame, results)

                writer.write(annotated)

                progress = frame_idx / max(total, 1)
                yield progress, frame_idx, annotated, detections

        finally:
            cap.release()
            writer.release()

    # ── private helpers ──────────────────────────────────────

    def _make_writer(
        self, path: str, fps: float, w: int, h: int
    ) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {path}")
        return writer

    def _annotate(
        self, frame: np.ndarray, results
    ) -> tuple[list[dict], np.ndarray]:
        """
        Draw bounding boxes, IDs, confidence scores, and trajectory tails.
        Returns (detections_list, annotated_frame).
        """
        annotated = frame.copy()
        detections: list[dict] = []

        if results is None or len(results) == 0:
            return detections, annotated

        r = results[0]

        # No track IDs available yet (first frame or tracker not ready)
        if r.boxes is None or r.boxes.id is None:
            return detections, annotated

        boxes      = r.boxes.xyxy.cpu().numpy()   # (N, 4)
        track_ids  = r.boxes.id.cpu().numpy().astype(int)
        confs      = r.boxes.conf.cpu().numpy()
        class_ids  = r.boxes.cls.cpu().numpy().astype(int)
        names      = r.names                       # dict: int → str

        for box, tid, conf, cid in zip(boxes, track_ids, confs, class_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            colour = _colour(tid)
            cls_name = names.get(cid, "obj")

            # ── Trajectory tail ──────────────────────────────
            if self.tail_length > 0:
                if tid not in self._tails:
                    self._tails[tid] = collections.deque(maxlen=self.tail_length)
                self._tails[tid].append((cx, cy))
                self._draw_tail(annotated, tid, colour)

            # ── Bounding box ─────────────────────────────────
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            # ── Label background ─────────────────────────────
            label      = f"ID:{tid} {cls_name} {conf:.2f}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness  = 1
            (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(
                annotated,
                (x1, y1 - lh - baseline - 4),
                (x1 + lw + 2, y1),
                colour, -1,
            )
            # ── Label text ───────────────────────────────────
            cv2.putText(
                annotated, label,
                (x1 + 1, y1 - baseline - 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )

            detections.append({
                "track_id":   tid,
                "class_name": cls_name,
                "confidence": float(conf),
                "bbox_xyxy":  (x1, y1, x2, y2),
                "centre":     (cx, cy),
            })

        return detections, annotated

    def _draw_tail(
        self, frame: np.ndarray, tid: int, colour: tuple[int, int, int]
    ) -> None:
        """Draw fading polyline of past centre positions for a track ID."""
        pts = list(self._tails[tid])
        if len(pts) < 2:
            return
        n = len(pts)
        for i in range(1, n):
            alpha    = i / n                            # fade in towards current pos
            thick    = max(1, int(alpha * 3))
            fade_col = tuple(int(c * alpha) for c in colour)
            cv2.line(frame, pts[i - 1], pts[i], fade_col, thick, cv2.LINE_AA)
