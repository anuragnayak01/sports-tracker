"""
app.py — Streamlit entry point for Sports Multi-Object Tracker
=============================================================
Supports:
  • Upload a local video file
  • Paste a YouTube (or any yt-dlp-supported) URL

Run with:  streamlit run app.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

from tracker.pipeline import SportsPipeline
from tracker.analytics import Analytics


# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sports Multi-Object Tracker",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# Helper — download video via yt-dlp
# ─────────────────────────────────────────────────────────────
def download_video(url: str, out_path: str, max_height: int = 720) -> tuple[bool, str]:
    """
    Download a public video with yt-dlp.
    Returns (success: bool, message: str).
    """
    if shutil.which("yt-dlp") is None:
        return False, (
            "`yt-dlp` is not installed. Run:  `pip install yt-dlp`  "
            "then restart the app."
        )

    fmt = f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best"
    cmd = [
        "yt-dlp",
        "-f", fmt,
        "-o", out_path,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(out_path):
            return True, "Download complete."
        err = result.stderr.strip() or result.stdout.strip()
        return False, f"yt-dlp error: {err[:400]}"
    except subprocess.TimeoutExpired:
        return False, "Download timed out (>5 min). Try a shorter clip."
    except Exception as e:
        return False, f"Unexpected error: {e}"


# ─────────────────────────────────────────────────────────────
# Helper — read video metadata
# ─────────────────────────────────────────────────────────────
def video_meta(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    meta = {
        "fps":    cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total":  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    meta["duration"] = meta["total"] / meta["fps"]
    cap.release()
    return meta


# ─────────────────────────────────────────────────────────────
# Sidebar — pipeline configuration
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Pipeline Settings")

    model_size = st.selectbox(
        "YOLO Model",
        ["yolov8s.pt", "yolov8m.pt", "yolov8n.pt"],
        index=0,
        help="yolov8s = best speed/accuracy balance. yolov8m for max accuracy.",
    )

    tracker_type = st.selectbox(
        "Tracker",
        ["botsort.yaml", "bytetrack.yaml"],
        index=0,
        help="BoT-SORT: camera motion compensation, better for broadcast sports.",
    )

    conf_threshold = st.slider(
        "Detection Confidence", 0.1, 0.9, 0.3, 0.05,
        help="Lower = detect more subjects (occluded). Higher = fewer false positives.",
    )

    frame_stride = st.slider(
        "Frame Stride", 1, 5, 2,
        help="Process every Nth frame. stride=2 halves compute.",
    )

    max_height = st.selectbox(
        "Download Quality (URL mode)",
        [480, 720, 1080],
        index=1,
        help="Max vertical resolution when downloading from a URL.",
    )

    st.divider()
    st.subheader("🎨 Visual Enhancements")
    show_trajectory = st.checkbox("Trajectory Tails",   value=True)
    show_heatmap    = st.checkbox("Movement Heatmap",   value=True)
    show_count_plot = st.checkbox("ID Count Over Time", value=True)
    tail_length     = st.slider("Tail Length (frames)", 10, 100, 40, 5)

    st.divider()
    st.markdown("**Tracker:** BoT-SORT (CVPR 2025 · KTH benchmarks)")
    st.markdown("**Detector:** YOLOv8 — Ultralytics")


# ─────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────
st.title("🏃 Sports Multi-Object Tracker")
st.caption(
    "Detect and persistently track all subjects in sports/event footage using "
    "**YOLOv8 + BoT-SORT**. Upload a video file **or** paste a public video URL."
)

input_tab, info_tab = st.tabs(["🎬 Run Tracker", "ℹ️ How It Works"])

# ── Info tab ─────────────────────────────────────────────────
with info_tab:
    st.markdown("""
### Pipeline Architecture
```
Video Input → Frame Extraction → YOLOv8 Detection → BoT-SORT Tracker
    → Persistent ID Assignment → Frame Annotation → Analytics → Output Video
```

### Why BoT-SORT over ByteTrack?
| Feature | BoT-SORT | ByteTrack |
|---|---|---|
| Camera Motion Compensation | ✅ Yes | ❌ No |
| Appearance Re-ID | ✅ Yes | ❌ No |
| Sports HOTA Score | **49.98%** | 42.92% |
| ID Switches (basketball) | **11.81** | 15.06 |

### Suggested Public Test Clips
| Sport | URL |
|---|---|
| ⚽ Football | `https://youtu.be/QCG8QarzeHs` |
| 🏏 Cricket | ICC T20 highlights — youtube.com/@ICC |
| 🏀 Basketball | NBA clips — youtube.com/@nba |

### Known Limitations
- Long-term off-screen exits → player re-enters with new ID
- Broadcast camera cuts → temporary ID spike  
- Identical uniforms may cause ID swaps in dense crowds
- Ball tracking unreliable (too small / fast for standard YOLO)
""")

# ── Run Tracker tab ───────────────────────────────────────────
with input_tab:

    # ── Input mode toggle ─────────────────────────────────────
    mode = st.radio(
        "Video source",
        ["📁 Upload File", "🔗 YouTube / Public URL"],
        horizontal=True,
    )

    input_path: str | None = None
    source_label: str = ""

    # ── MODE 1 : File upload ──────────────────────────────────
    if mode == "📁 Upload File":
        uploaded_file = st.file_uploader(
            "Upload a sports / event video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Short clips (30–90 s) recommended.",
        )
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.read())
            tmp.flush()
            tmp.close()
            input_path   = tmp.name
            source_label = uploaded_file.name
            # Clear any stale URL-download session state
            st.session_state.pop("downloaded_path",  None)
            st.session_state.pop("downloaded_label", None)

    # ── MODE 2 : YouTube / URL ────────────────────────────────
    else:
        st.markdown(
            "Paste any **YouTube** or public video URL. "
            "The video will be downloaded at up to the quality selected in the sidebar."
        )

        # Suggested clips quick-fill
        st.markdown("**Quick suggestions:**")
        q1, q2, q3 = st.columns(3)
        if q1.button("⚽ Football demo", use_container_width=True):
            st.session_state["url_input"] = "https://youtu.be/QCG8QarzeHs"
        if q2.button("🏏 ICC Cricket", use_container_width=True):
            st.session_state["url_input"] = "https://www.youtube.com/@ICC"
        if q3.button("🏀 NBA clip", use_container_width=True):
            st.session_state["url_input"] = "https://www.youtube.com/@nba"

        url_col, btn_col = st.columns([4, 1])
        with url_col:
            video_url = st.text_input(
                "Video URL",
                value=st.session_state.get("url_input", ""),
                placeholder="https://youtu.be/QCG8QarzeHs",
                key="url_input",
            )
        with btn_col:
            st.write("")
            st.write("")
            download_btn = st.button("⬇️ Download", type="secondary", use_container_width=True)

        # Download button pressed
        if video_url and download_btn:
            tmp_dir = tempfile.mkdtemp()
            dl_path = os.path.join(tmp_dir, "downloaded_video.mp4")
            with st.spinner("Downloading… (this may take 30–60 s for a 720p clip)"):
                ok, msg = download_video(video_url, dl_path, max_height=max_height)
            if ok:
                st.success("✅ Downloaded — ready to track!")
                st.session_state["downloaded_path"]  = dl_path
                st.session_state["downloaded_label"] = video_url
            else:
                st.error(f"❌ {msg}")

        # Restore from session state after reruns
        if "downloaded_path" in st.session_state:
            cached = st.session_state["downloaded_path"]
            if os.path.exists(cached):
                input_path   = cached
                source_label = st.session_state.get("downloaded_label", cached)
                st.info(f"📥 Video ready: `{source_label}`")

    # ── Preview + metadata ────────────────────────────────────
    if input_path and os.path.exists(input_path):
        meta = video_meta(input_path)

        col_prev, col_info = st.columns([2, 1])
        with col_prev:
            st.video(input_path)
        with col_info:
            st.metric("Duration",          f"{meta['duration']:.1f} s")
            st.metric("Resolution",        f"{meta['width']}×{meta['height']}")
            st.metric("Total Frames",      meta["total"])
            st.metric("Frames to Process", f"~{meta['total'] // frame_stride}")
            if source_label:
                st.caption(f"**Source:** {source_label}")

        st.divider()

        # ── RUN ───────────────────────────────────────────────
        if st.button("🚀 Run Tracker", type="primary", use_container_width=True):

            suffix      = Path(input_path).suffix or ".mp4"
            output_path = input_path.replace(suffix, "_tracked.mp4")

            progress_bar = st.progress(0, text="Initialising pipeline…")
            status_box   = st.empty()

            analytics = Analytics(frame_width=meta["width"], frame_height=meta["height"])
            pipeline  = SportsPipeline(
                model_path=model_size,
                tracker_cfg=tracker_type,
                conf=conf_threshold,
                frame_stride=frame_stride,
                tail_length=tail_length if show_trajectory else 0,
            )

            start_time = time.time()

            for progress, frame_idx, _, frame_dets in pipeline.process(input_path, output_path):
                analytics.update(frame_dets, frame_idx, meta["fps"])
                progress_bar.progress(
                    progress,
                    text=f"Processing frame {frame_idx} / {meta['total']}",
                )
                status_box.info(
                    f"⚡ Active IDs this frame: **{len(frame_dets)}**  |  "
                    f"Unique IDs so far: **{analytics.unique_id_count()}**"
                )

            elapsed = time.time() - start_time
            progress_bar.progress(1.0, text="✅ Done!")
            status_box.success(
                f"Processed **{meta['total'] // frame_stride}** frames in "
                f"**{elapsed:.1f} s** · "
                f"Unique subjects tracked: **{analytics.unique_id_count()}**"
            )

            # ── Results ───────────────────────────────────────
            r1, r2, r3, r4 = st.tabs(
                ["📹 Annotated Video", "🔥 Heatmap", "📈 ID Count Over Time", "📊 Summary"]
            )

            with r1:
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Annotated Video",
                            f, file_name="tracked_output.mp4", mime="video/mp4",
                            use_container_width=True,
                        )
                    st.video(output_path)
                else:
                    st.warning("Output video not found — check console for errors.")

            with r2:
                if show_heatmap:
                    st.pyplot(analytics.plot_heatmap())
                else:
                    st.info("Enable 'Movement Heatmap' in the sidebar.")

            with r3:
                if show_count_plot:
                    st.pyplot(analytics.plot_id_count_over_time())
                else:
                    st.info("Enable 'ID Count Over Time' in the sidebar.")

            with r4:
                st.markdown(f"""
| Metric | Value |
|---|---|
| Source | `{source_label or input_path}` |
| Model | `{model_size}` |
| Tracker | `{tracker_type}` |
| Confidence threshold | `{conf_threshold}` |
| Frame stride | `{frame_stride}` |
| Total frames processed | `{meta['total'] // frame_stride}` |
| Processing time | `{elapsed:.1f} s` |
| Unique subjects tracked | `{analytics.unique_id_count()}` |
| Peak simultaneous IDs | `{analytics.peak_simultaneous_ids()}` |
""")
