# 🏃 Sports Multi-Object Tracker

**Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage**

> Assignment: AI / Computer Vision / Data Science  
> Stack: YOLOv8s + BoT-SORT · OpenCV · Streamlit  
> Research basis: CVPR 2025 · arXiv 2025 · KTH benchmarks

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Running the App](#running-the-app)
4. [CLI Usage](#cli-usage)
5. [Download a Test Video](#download-a-test-video)
6. [Project Structure](#project-structure)
7. [Model & Tracker Choices](#model--tracker-choices)
8. [Pipeline Architecture](#pipeline-architecture)
9. [Assumptions & Limitations](#assumptions--limitations)
10. [Optional Enhancements](#optional-enhancements)

---

## Quick Start

```bash
# 1. Clone / unzip the project
cd sports-tracker

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Download a public test video
python utils/download_video.py --url "https://youtu.be/YOUR_VIDEO_ID" --out input_video.mp4

# 4. Run the Streamlit app
streamlit run app.py

# OR use the CLI directly
python utils/run_cli.py --input input_video.mp4 --output tracked_output.mp4
```

---

## Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| pip | latest |
| GPU (recommended) | NVIDIA CUDA 11.8+ |

### Steps

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# Install all dependencies
pip install -r requirements.txt
```

> **GPU acceleration:** Uncomment the `torch` lines in `requirements.txt` and install the CUDA-enabled wheels for a 5–10× speedup.

---

## Running the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**Streamlit UI workflow:**
1. Upload a sports video (MP4, AVI, MOV, MKV)
2. Configure model size, tracker, confidence, and frame stride in the sidebar
3. Toggle optional enhancements (trajectory tails, heatmap, ID count over time)
4. Click **Run Tracker**
5. View the annotated video, heatmap, and ID-over-time chart in the results tabs
6. Download the annotated output video

---

## CLI Usage

```bash
python utils/run_cli.py \
  --input  input_video.mp4 \
  --output tracked_output.mp4 \
  --model  yolov8s.pt \
  --tracker botsort.yaml \
  --conf   0.3 \
  --stride 2 \
  --tail   40
```

| Argument | Default | Description |
|---|---|---|
| `--input` | — | Path to input video (required) |
| `--output` | `output_tracked.mp4` | Output annotated video path |
| `--model` | `yolov8s.pt` | YOLO model variant |
| `--tracker` | `botsort.yaml` | Tracker config file |
| `--conf` | `0.3` | Detection confidence threshold |
| `--stride` | `2` | Process every Nth frame |
| `--tail` | `40` | Trajectory tail length in frames |
| `--no-heatmap` | `False` | Skip saving heatmap PNG |
| `--no-count` | `False` | Skip saving ID-count plot PNG |

---

## Download a Test Video

```bash
# Install yt-dlp if needed
pip install yt-dlp

# Download a 720p public sports clip
python utils/download_video.py \
  --url "https://youtu.be/YOUR_VIDEO_ID" \
  --out input_video.mp4 \
  --max-height 720
```

Suggested public video categories: IPL cricket highlights, Premier League clips, NBA game footage, marathon race footage.

> **Include the source URL in your submission** as required by the assignment brief.

---

## Project Structure

```
sports-tracker/
├── app.py                   # Streamlit application entry point
│
├── tracker/
│   ├── __init__.py
│   ├── pipeline.py          # YOLOv8 + BoT-SORT core (detection + tracking + annotation)
│   └── analytics.py         # Heatmap, ID-count-over-time, summary statistics
│
├── utils/
│   ├── __init__.py
│   ├── run_cli.py           # Headless CLI runner
│   └── download_video.py    # yt-dlp video download helper
│
├── report/
│   └── technical_report.md  # 1–2 page technical report
│
├── requirements.txt
└── README.md
```

---

## Model & Tracker Choices

### Detector — YOLOv8s (Small)

- **Why YOLOv8?** Best-in-class real-time object detector with native multi-object tracking via the Ultralytics library.
- **Why Small (s) not Nano (n) or Large (l)?** Research shows YOLOv8s gives the best speed/accuracy balance for a 2–3 day project. Researchers at KTH found YOLOv8x caused GPU memory constraints; yolov8m reached mAP50 ~0.90 in sports tasks with just 50 training epochs.
- Switch to `yolov8m.pt` for higher accuracy if you have the hardware.

### Tracker — BoT-SORT

- **Why BoT-SORT over ByteTrack?** BoT-SORT is the clear winner for sports tracking:

| Feature | BoT-SORT | ByteTrack |
|---|---|---|
| Camera Motion Compensation (CMC) | ✅ Yes | ❌ No |
| Appearance Re-ID embeddings | ✅ Yes | ❌ No |
| Sports HOTA score | **49.98%** | 42.92% |
| ID switches (3×3 basketball) | **11.81** | 15.06 |

- BoT-SORT's CMC corrects for the continuous pan/zoom of broadcast sports cameras, preventing the false position drift that causes ByteTrack to lose tracks.
- Source: KTH comparison study (2024) + arXiv:2503.18282 (CVPR 2025)

### Key code flag — `persist=True`

```python
results = model.track(frame, persist=True, tracker="botsort.yaml")
```

Without `persist=True`, tracking IDs reset on every frame call. This single flag is what enables cross-frame ID persistence.

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Input Video                          │
└──────────────────────┬───────────────────────────────────┘
                       │  cv2.VideoCapture
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Frame Extraction & Stride                   │
│         (process every Nth frame for efficiency)         │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│           YOLOv8s Object Detection                       │
│   → bounding boxes, class names, confidence scores       │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│      BoT-SORT Multi-Object Tracker (persist=True)        │
│   → Camera Motion Compensation (CMC)                     │
│   → Kalman Filter position prediction                    │
│   → Re-ID embedding matching                             │
│   → Persistent unique ID assignment                      │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│               Frame Annotation                           │
│   → Colour-coded bounding boxes per ID                   │
│   → "ID:N  class  conf" label overlay                    │
│   → Fading trajectory tail polylines                     │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────┴──────────────┐
          ▼                           ▼
┌─────────────────┐        ┌──────────────────────┐
│  Annotated MP4  │        │  Analytics Engine    │
│  (VideoWriter)  │        │  → Density heatmap   │
└─────────────────┘        │  → ID count / time   │
                           │  → Peak / total IDs  │
                           └──────────────────────┘
```

---

## Assumptions & Limitations

### Assumptions

- Input video contains human subjects (YOLO class `person`; class index 0). For vehicles or other subjects, update the `classes` parameter in `SportsPipeline`.
- Video is publicly accessible and ≤ 5 minutes for reasonable CLI processing time.
- Processing occurs on CPU by default; GPU significantly improves throughput.
- Frame stride of 2 is sufficient for typical sports footage at 25–30 fps.

### Known Limitations

| Failure Case | Cause | Severity |
|---|---|---|
| ID reassignment after off-screen exit | No long-term Re-ID memory | Medium |
| ID switches in dense crowds | Near-identical bounding box overlap | Medium |
| Ball tracking unreliable | Too small / fast for standard YOLO | Low (ball not primary objective) |
| Broadcast camera cuts | Temporal continuity break | Medium |
| Identical uniforms | Appearance embeddings confused | Medium |
| False positives | Spectators, referees, shadows | Low |

---

## Optional Enhancements

Implemented in this submission:

| Enhancement | Status | Location |
|---|---|---|
| Trajectory tails | ✅ Implemented | `tracker/pipeline.py` → `_draw_tail()` |
| Movement heatmap | ✅ Implemented | `tracker/analytics.py` → `plot_heatmap()` |
| Object count over time | ✅ Implemented | `tracker/analytics.py` → `plot_id_count_over_time()` |
| Streamlit deployment | ✅ Implemented | `app.py` |
| CLI runner | ✅ Implemented | `utils/run_cli.py` |

Not implemented (future work):

- Bird's-eye / top-view homography projection
- Speed estimation (requires calibration data)
- Team clustering (requires jersey colour segmentation)
- Formal HOTA/MOTA metric evaluation

---

## Public Video Source

> 📎 **Video URL:** *(add your chosen YouTube / public URL here)*  
> Format: MP4, resolution: 720p, duration: ~60 seconds  
> Category: *(e.g., IPL cricket highlights / Premier League match)*

---

## References

1. Zhang et al., *BoT-SORT: Robust Associations Multi-Pedestrian Tracking* (2022)
2. Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* (2022)
3. KTH thesis — *Object Tracking Evaluation: BoT-SORT & ByteTrack with YOLOv8* (2024)
4. arXiv:2503.18282 — Sports tracker comparison (CVPR 2025)
5. arXiv:2406.19655 — Basketball multi-object occlusion handling (2024)
6. FieldMOT — Field-Registered MOT for Sports Videos (CVPRW 2025)
7. Ultralytics YOLOv8 Documentation — [docs.ultralytics.com](https://docs.ultralytics.com)
