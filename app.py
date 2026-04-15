"""
app.py - Sports Multi-Object Tracker (upload only)
Run with: streamlit run app.py
"""

import os
import tempfile
import time
from pathlib import Path

try:
    import cv2
except Exception as e:
    import streamlit as st
    st.error("❌ OpenCV failed to load. Deployment issue.")
    st.code(str(e))
    st.stop()
import streamlit as st

from tracker.pipeline import SportsPipeline
from tracker.analytics import Analytics

st.set_page_config(
    page_title="Sports Multi-Object Tracker",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


# Sidebar
with st.sidebar:
    st.title("Pipeline Settings")
    model_size     = st.selectbox("YOLO Model",
                                  ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    tracker_type   = st.selectbox("Tracker",
                                  ["botsort.yaml", "bytetrack.yaml"], index=0)
    conf_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.3, 0.05)
    frame_stride   = st.slider("Frame Stride", 1, 5, 2)
    st.divider()
    st.subheader("Visual Enhancements")
    show_trajectory = st.checkbox("Trajectory Tails",   value=True)
    show_heatmap    = st.checkbox("Movement Heatmap",   value=True)
    show_count_plot = st.checkbox("ID Count Over Time", value=True)
    tail_length     = st.slider("Tail Length (frames)", 10, 100, 40, 5)
    st.divider()
    st.caption("YOLOv8 + BoT-SORT | CVPR 2025")


# Main
st.title("Sports Multi-Object Tracker")
st.caption(
    "Upload a sports video to detect and track all subjects using "
    "YOLOv8 + BoT-SORT with unique IDs, heatmaps, and trajectory tails."
)

run_tab, info_tab = st.tabs(["Run Tracker", "How It Works"])

with info_tab:
    st.markdown("""
### Pipeline
```
Upload -> YOLOv8 Detection -> BoT-SORT Tracker
       -> Persistent IDs -> Annotation -> Heatmap + Charts -> Output Video
```
### BoT-SORT vs ByteTrack
| Feature | BoT-SORT | ByteTrack |
|---|---|---|
| Camera Motion Compensation | Yes | No |
| Appearance Re-ID | Yes | No |
| Sports HOTA | 49.98% | 42.92% |
| ID Switches | 11.81 | 15.06 |

### Best Video Types
- Football/Soccer wide broadcast angle
- Cricket fielding sequences
- Basketball full-court view
- 30 to 90 seconds at 720p is ideal

### Limitations
- Player exits and re-enters gets a new ID
- Camera cuts cause temporary ID spike
- Identical uniforms cause ID swaps in crowds
""")

with run_tab:
    st.markdown("### Upload Your Video")
    st.info(
        "Download a sports clip using yt-dlp on your local machine, "
        "then upload the MP4 file here. Best: 30-90 sec, 720p, wide-angle.",
        icon="💡",
    )

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()
        input_path = tmp.name
        meta = video_meta(input_path)

        col_prev, col_info = st.columns([2, 1])
        with col_prev:
            st.video(input_path)
        with col_info:
            st.metric("Duration",          f"{meta['duration']:.1f} s")
            st.metric("Resolution",        f"{meta['width']}x{meta['height']}")
            st.metric("Total Frames",      meta["total"])
            st.metric("Frames to Process", f"~{meta['total'] // frame_stride}")
            st.caption(f"File: {uploaded_file.name}")

        st.divider()

        if st.button("Run Tracker", type="primary", use_container_width=True):
            output_path  = input_path.replace(suffix, "_tracked.mp4")
            progress_bar = st.progress(0, text="Initialising...")
            status_box   = st.empty()

            analytics = Analytics(frame_width=meta["width"], frame_height=meta["height"])
            pipeline  = SportsPipeline(
                model_path   = model_size,
                tracker_cfg  = tracker_type,
                conf         = conf_threshold,
                frame_stride = frame_stride,
                tail_length  = tail_length if show_trajectory else 0,
            )

            start_time = time.time()
            for progress, frame_idx, _, frame_dets in pipeline.process(input_path, output_path):
                analytics.update(frame_dets, frame_idx, meta["fps"])
                progress_bar.progress(
                    progress, text=f"Frame {frame_idx} / {meta['total']}"
                )
                status_box.info(
                    f"Active IDs: {len(frame_dets)}  |  "
                    f"Unique IDs so far: {analytics.unique_id_count()}"
                )

            elapsed = time.time() - start_time
            progress_bar.progress(1.0, text="Done!")
            status_box.success(
                f"{meta['total'] // frame_stride} frames in {elapsed:.1f}s  |  "
                f"Unique subjects: {analytics.unique_id_count()}"
            )

            r1, r2, r3, r4 = st.tabs([
                "Annotated Video", "Heatmap", "ID Count Over Time", "Summary"
            ])

            with r1:
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "Download Annotated Video", f,
                            file_name="tracked_output.mp4", mime="video/mp4",
                            use_container_width=True,
                        )
                    st.video(output_path)

            with r2:
                if show_heatmap:
                    st.pyplot(analytics.plot_heatmap())
                else:
                    st.info("Enable Movement Heatmap in sidebar.")

            with r3:
                if show_count_plot:
                    st.pyplot(analytics.plot_id_count_over_time())
                else:
                    st.info("Enable ID Count Over Time in sidebar.")

            with r4:
                st.markdown(f"""
| Metric | Value |
|---|---|
| File | `{uploaded_file.name}` |
| Model | `{model_size}` |
| Tracker | `{tracker_type}` |
| Confidence | `{conf_threshold}` |
| Frame stride | `{frame_stride}` |
| Frames processed | `{meta['total'] // frame_stride}` |
| Time taken | `{elapsed:.1f} s` |
| Unique IDs tracked | `{analytics.unique_id_count()}` |
| Peak simultaneous | `{analytics.peak_simultaneous_ids()}` |
""")
            try:
                os.unlink(input_path)
            except Exception:
                pass
