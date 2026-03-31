"""
tracker/analytics.py — Post-processing analytics
=================================================
Builds movement heatmaps, ID-count-over-time charts,
and summary statistics from per-frame detection data.
"""

from __future__ import annotations

import collections
from typing import Any

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


class Analytics:
    """
    Accumulates tracking detections across frames and produces analytics.

    Parameters
    ----------
    frame_width   : Width of the source video in pixels
    frame_height  : Height of the source video in pixels
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.W = frame_width
        self.H = frame_height

        # Density accumulation array (float32 for GaussianBlur precision)
        self._density: np.ndarray = np.zeros((frame_height, frame_width), dtype=np.float32)

        # All unique track IDs ever seen
        self._all_ids: set[int] = set()

        # Per-second active ID counts: second_index → set of IDs
        self._ids_per_second: dict[int, set[int]] = collections.defaultdict(set)

        # Peak number of simultaneous IDs in any single frame
        self._peak: int = 0

    # ── public API ───────────────────────────────────────────

    def update(
        self,
        detections: list[dict[str, Any]],
        frame_idx:  int,
        fps:        float,
    ) -> None:
        """
        Call once per processed frame.

        Parameters
        ----------
        detections : List of dicts from pipeline.process()
        frame_idx  : 1-based frame index
        fps        : Video frames-per-second (for time bucketing)
        """
        if not detections:
            return

        second_idx = int(frame_idx / max(fps, 1))
        frame_ids: set[int] = set()

        for det in detections:
            tid    = det["track_id"]
            cx, cy = det["centre"]

            # Accumulate density (guard bounds)
            if 0 <= cy < self.H and 0 <= cx < self.W:
                self._density[cy, cx] += 1.0

            self._all_ids.add(tid)
            frame_ids.add(tid)
            self._ids_per_second[second_idx].add(tid)

        self._peak = max(self._peak, len(frame_ids))

    def unique_id_count(self) -> int:
        """Total number of unique track IDs seen so far."""
        return len(self._all_ids)

    def peak_simultaneous_ids(self) -> int:
        """Highest number of IDs active in any single frame."""
        return self._peak

    # ── visualisations ───────────────────────────────────────

    def plot_heatmap(self, figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Return a Matplotlib figure of the movement heatmap.
        Applies Gaussian blur to smooth point density into heat zones.
        """
        if self._density.max() == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No detections yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return fig

        # Resize to a standard display resolution for speed
        display_w, display_h = min(self.W, 960), min(self.H, 540)
        density_small = cv2.resize(self._density, (display_w, display_h))

        # Gaussian blur: radius proportional to display size
        ksize = _odd(max(display_w, display_h) // 20)
        blurred = cv2.GaussianBlur(density_small, (ksize, ksize), 0)

        # Normalise to 0–1
        blurred_norm = blurred / (blurred.max() + 1e-9)

        fig, ax = plt.subplots(figsize=figsize, facecolor="#0e1117")
        ax.imshow(blurred_norm, cmap="inferno", aspect="auto", origin="upper")
        cbar = plt.colorbar(
            cm.ScalarMappable(cmap="inferno"), ax=ax, fraction=0.03, pad=0.02
        )
        cbar.set_label("Relative Activity", color="white", fontsize=11)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax.set_title("🔥 Player Movement Heatmap", color="white", fontsize=14, pad=10)
        ax.set_xlabel("Horizontal Position", color="white")
        ax.set_ylabel("Vertical Position", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        fig.tight_layout()
        return fig

    def plot_id_count_over_time(self, figsize: tuple = (10, 4)) -> plt.Figure:
        """
        Return a Matplotlib figure showing unique active IDs per second.
        """
        if not self._ids_per_second:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return fig

        seconds = sorted(self._ids_per_second.keys())
        counts  = [len(self._ids_per_second[s]) for s in seconds]

        fig, ax = plt.subplots(figsize=figsize, facecolor="#0e1117")
        ax.set_facecolor("#0e1117")

        ax.fill_between(seconds, counts, alpha=0.35, color="#00c8ff")
        ax.plot(seconds, counts, color="#00c8ff", linewidth=2, marker="o",
                markersize=4, label="Active IDs")

        ax.axhline(
            y=sum(counts) / max(len(counts), 1),
            color="#ff6b35", linestyle="--", linewidth=1.5, label="Mean",
        )

        ax.set_title("📈 Active Tracked IDs Over Time", color="white", fontsize=14, pad=10)
        ax.set_xlabel("Time (seconds)", color="white")
        ax.set_ylabel("Active IDs", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1e1e1e", edgecolor="#444", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        fig.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _odd(n: int) -> int:
    """Return n if odd, else n+1 (cv2.GaussianBlur requires odd kernel size)."""
    return n if n % 2 == 1 else n + 1
