# Project Report Outline

Use this as a chapter skeleton for your final-year report / thesis.

## 1. Introduction
- 1.1 Motivation: marine oil spills, ecological/economic impact
- 1.2 Problem statement
- 1.3 Objectives
- 1.4 Scope and limitations (be explicit: folder-based MOSDAC ingestion,
  not a live API; pixel-based stitching, not geo-referenced mosaicking)

## 2. Literature Review
- 2.1 Satellite-based oil spill detection approaches (SAR vs optical)
- 2.2 CNN-based classification for remote sensing imagery
- 2.3 Classical CV segmentation (HSV thresholding, morphology) for
  slick boundary detection
- 2.4 Image stitching / mosaicking techniques (feature-based panorama methods)

## 3. System Requirements
- 3.1 Functional requirements (upload, MOSDAC scan, detection, history, alerts)
- 3.2 Non-functional requirements (response time, extensibility, logging)
- 3.3 Hardware/software requirements

## 4. System Design
- 4.1 Architecture overview (reference `docs/ARCHITECTURE.md` diagram 1)
- 4.2 Data flow (reference diagram 2)
- 4.3 Module design (`backend/` package breakdown)
- 4.4 Class design (reference diagram 3)
- 4.5 Database/storage design (CSV/JSON history schema)

## 5. Implementation
- 5.1 Preprocessing pipeline (denoise, Gaussian blur, CLAHE, HSV threshold,
  morphology, ROI masking) - describe why each stage exists
- 5.2 Model architecture (attention-based MobileNet) and how it's invoked
- 5.3 Decision logic (confidence + area double-gate, severity bands)
- 5.4 Stitching module (OpenCV Stitcher + ORB/homography fallback,
  explain when/why the fallback triggers)
- 5.5 MOSDAC ingestion module and its designed extension point for a
  live API
- 5.6 Dashboard/UI implementation

## 6. Testing and Results
- 6.1 Test methodology (reference `docs/TESTING.md`)
- 6.2 Sample detection results (screenshots of dashboard, actual `area_pct`/
  `confidence` numbers from your own test images)
- 6.3 Stitching results (success and fallback/failure cases)
- 6.4 Performance (processing time per image, from `processing_time_ms`
  logged in history)

## 7. Discussion
- 7.1 Strengths of the double-gate decision approach
- 7.2 Known limitations (texture-dependent stitching, folder-based ingestion,
  no live alerting channels yet)
- 7.3 Comparison with the original single-file prototype

## 8. Future Work
- 8.1 Live MOSDAC/Bhoonidhi API integration
- 8.2 Real-time satellite feed monitoring (scheduled ingestion)
- 8.3 Email/SMS alert channel implementation
- 8.4 Map-based visualization of detected spills (lat/lon overlay)
- 8.5 Spill trajectory prediction (ocean current modeling)
- 8.6 Cloud deployment (containerization, autoscaling)

## 9. Conclusion

## Appendices
- A. Full source code listing (or GitHub link)
- B. `requirements.txt`
- C. User manual (screenshots of upload flow, MOSDAC scan, history page)
