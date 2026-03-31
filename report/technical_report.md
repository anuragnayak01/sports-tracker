# Technical Report
## Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage

**Author:** *(your name)*  
**Date:** *(submission date)*  
**Video Source:** *(paste YouTube / public URL here)*  

---

## 1. Overview

This report describes the design and implementation of a real-time computer vision pipeline for detecting and persistently tracking multiple subjects in publicly available sports footage. The pipeline uses YOLOv8s for detection and BoT-SORT for multi-object tracking, deployed as a Streamlit web application.

---

## 2. Model — Detector

**Chosen model:** YOLOv8s (`yolov8s.pt`) from the Ultralytics library.

YOLOv8 (You Only Look Once, version 8) is a single-stage anchor-free detector that performs detection, segmentation, and classification in a single forward pass. It was selected over alternatives for the following reasons:

- **Native tracker integration:** Ultralytics provides BoT-SORT and ByteTrack as built-in tracker configs, reducing boilerplate code.
- **Pretrained on COCO:** The model detects `person` (class 0) out of the box — the primary subject class in sports footage — without requiring fine-tuning.
- **Speed/accuracy trade-off:** YOLOv8s achieves a strong balance. Research shows YOLOv8m reaches mAP50 ≈ 0.90 on sports tasks with just 50 training epochs; YOLOv8s is marginally lighter while maintaining strong detection quality (KTH, 2024).
- **`yolov8l` and `yolov8x` were ruled out** due to hardware constraints: both larger variants have been shown to hit GPU memory limits in broadcast footage pipelines (arXiv:2406.19655, 2024).

---

## 3. Tracking Algorithm

**Chosen tracker:** BoT-SORT (`botsort.yaml`)

BoT-SORT (Zhang et al., 2022) extends the SORT framework with two major additions:

1. **Camera Motion Compensation (CMC):** Warps the predicted Kalman filter state based on estimated homography between consecutive frames. This is critical for broadcast sports where the camera continuously pans, zooms, and cuts — without CMC, the position prior drifts and causes false track terminations.

2. **Re-Identification (Re-ID) embeddings:** Appearance features are extracted per detection and matched against stored gallery embeddings. This allows the tracker to re-associate a subject after a partial occlusion, even when IoU-based matching fails.

### Why BoT-SORT over ByteTrack?

ByteTrack is a strong baseline that associates low-confidence detections to maintain tracks through partial occlusion. However, a direct comparison on sports data reveals BoT-SORT's superiority:

| Metric | BoT-SORT | ByteTrack |
|---|---|---|
| HOTA (3×3 basketball, drone) | **49.98%** | 42.92% |
| ID Switches | **11.81** | 15.06 |
| Camera Motion Compensation | ✅ | ❌ |
| Appearance Re-ID | ✅ | ❌ |

*Source: KTH thesis — Object Tracking Evaluation: BoT-SORT & ByteTrack with YOLOv8 (2024)*

In team sports where all players wear identical uniforms, IoU-only association (ByteTrack's primary mechanism) fails frequently in dense player clusters. BoT-SORT's appearance embeddings provide an additional discriminative signal.

---

## 4. ID Consistency Strategy

Persistent ID assignment is the central challenge of this task. The following mechanisms ensure consistency:

### 4.1 `persist=True` (Ultralytics)
Setting `persist=True` in `model.track()` maintains the internal Kalman filter state and track history between frame calls. Without this flag, the tracker reinitialises on every frame and all IDs are reset.

### 4.2 Kalman Filter Prediction
Between frames, BoT-SORT predicts each track's next position using a constant-velocity Kalman filter. Predicted boxes are matched against new detections via IoU gating, allowing the tracker to bridge gaps caused by temporary occlusion (typically 1–10 frames).

### 4.3 Appearance Re-ID Matching
When IoU matching alone is ambiguous (two nearby players with overlapping bounding boxes), BoT-SORT falls back to appearance embedding cosine distance. This is especially important when players cross paths or emerge from a crowd.

### 4.4 Track Lifecycle Management
- **New tracks** are created for high-confidence detections (≥ `new_track_thresh`) that cannot be matched to existing tracks.
- **Lost tracks** are kept alive in a buffer for `track_buffer` frames before being deleted, allowing re-association after short disappearances.
- **Low-confidence detections** are used for matching existing tracks but do not spawn new ones — this handles partially occluded players without introducing false positives.

---

## 5. Challenges Faced

### 5.1 Complex Multi-Object Occlusions (CMOO)
Dense player clusters — common in penalty areas, scrums, and jump balls — cause multiple bounding boxes to overlap significantly. BoT-SORT's Re-ID partially mitigates this, but ID switches still occur when ≥3 players with identical uniforms occlude each other simultaneously. This is classified as a *Complex Multi-Object Occlusion* (CMOO) in literature (arXiv:2406.19655, 2024).

### 5.2 Broadcast Camera Cuts
Hard camera cuts (abrupt scene transitions) instantly invalidate all Kalman filter predictions. After a cut, the tracker treats all players as new detections and assigns fresh IDs. BoT-SORT's CMC handles continuous panning but cannot compensate for discontinuous cuts.

### 5.3 Similar Appearance (Uniform Jerseys)
Players on the same team wear identical jerseys, making appearance-based Re-ID inherently ambiguous. This is a known open problem in sports MOT — current solutions involve fine-tuned Re-ID models trained on jersey-number recognition, which is beyond the scope of this assignment.

### 5.4 Small and Fast Objects
Ball detection is unreliable: the ball is small (often <20×20 pixels), moves faster than inter-frame intervals can capture, and is frequently occluded by player bodies. Standard YOLO models, trained primarily on COCO-scale objects, have low sensitivity to sub-30-pixel objects.

---

## 6. Failure Cases Observed

| Scenario | Observed Effect |
|---|---|
| Player exits frame edge | Re-enters with new ID (no cross-boundary Re-ID) |
| 3+ players in tight cluster | ID switch on separation (1–2 swaps per cluster event) |
| Camera hard cut | All IDs reset; "staircase" spike in unique ID count |
| Referee mixed with players | Referee occasionally assigned a player ID |
| Fast lateral run + blur | Track lost for 2–5 frames; recovered or new ID assigned |

---

## 7. Possible Improvements

### Short-term (1–2 days additional)
- **Confidence threshold tuning:** Lower `track_low_thresh` to 0.05 to recover more occluded detections.
- **Frame stride reduction:** Stride=1 (every frame) improves temporal continuity at the cost of compute.
- **Domain fine-tuning:** Fine-tune YOLOv8 on a sports-specific dataset (e.g., SportsMOT) to improve recall for partially visible players.

### Medium-term
- **Jersey number Re-ID:** Use OCR on jersey numbers as a supplementary Re-ID signal to resolve uniform-similarity ambiguity.
- **Team clustering:** Segment player jerseys by dominant colour (k-means on HSV histograms) to separate teams and improve cross-team Re-ID.
- **Long-term track memory:** Store appearance embeddings in a gallery and match returning players against the full gallery, not just active tracks.

### Long-term / Research
- **FieldMOT-style field registration:** Use field line homography to project detections onto a canonical top-down pitch view, enabling position-based Re-ID across camera cuts (CVPRW 2025).
- **SAM-based occlusion recovery:** Use Segment Anything Model (SAM) to maintain instance masks through occlusion events, recovering identity from partial silhouettes (arXiv:2512.08467).

---

## 8. References

1. Zhang, Y. et al. *BoT-SORT: Robust Associations Multi-Pedestrian Tracking.* arXiv:2206.14651 (2022).
2. Zhang, Y. et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022.
3. KTH — *Object Tracking Evaluation: BoT-SORT & ByteTrack with YOLOv8.* DiVA:1886982 (2024).
4. arXiv:2503.18282 — Sports tracking comparison study (2025).
5. arXiv:2406.19655 — *An Association Method for Complex Multi-object Occlusion Problems in Basketball MOT* (2024).
6. Chen et al. *FieldMOT: A Field-Registered Multi-Object Tracking for Sports Videos.* CVPRW 2025.
7. Cui et al. *SportsMOT: A Large MOT Dataset in Multiple Sports Scenes.* SemanticScholar (2023).
8. arXiv:2512.08467 — *Team-Aware Football Player Tracking with SAM* (2024).
9. Ultralytics — *YOLOv8 Tracking Documentation.* docs.ultralytics.com.
