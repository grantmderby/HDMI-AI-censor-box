# AI Inline HDMI Censor Box

A real-time embedded AI device that intercepts an HDMI video stream, runs neural network inference to detect content, and outputs a filtered feed to a display. Built as a content moderation tool with applications in family-friendly viewing environments.

The device sits inline between any HDMI source (game console, streaming device, computer) and a TV, processing the video stream in real time with no impact on the source device.

---

## Architecture

```
HDMI Source → HDCP Splitter → USB Capture Card → Raspberry Pi 5 → HDMI Out → Display
                                                       │
                                                       ├── Hailo-10H NPU (40 TOPS)
                                                       │   └── YOLO segmentation
                                                       │
                                                       └── ARM CPU
                                                           └── Secondary classifier
```

### Hardware Stack

| Component | Role |
|-----------|------|
| Raspberry Pi 5 (8 GB) | Main compute, async classifier, pipeline orchestration |
| Hailo-10H AI HAT+ 2 | 40 TOPS INT4 NPU running primary detection model |
| MS2130 USB 3.0 dongle | HDMI capture (1080p60 MJPEG + ALSA audio) |
| ViewHD HDCP splitter | HDCP 1.4 stripping, signal duplication |
| Argon ONE V5 case | Active cooling under sustained inference load |

### Software Stack

- **Detection:** YOLO segmentation on Hailo NPU + secondary classifier on ARM CPU (asynchronous, frame-skipped)
- **Pipeline:** GStreamer with custom Python callbacks for inference and censor application
- **OS:** Raspberry Pi OS Trixie (Debian 13)
- **Service Management:** systemd (three services: pipeline, web UI, audio sync)
- **Web UI:** Flask, accessible at `blurbox.local:5000` for runtime configuration

---

## Key Technical Decisions

### Single Linear GStreamer Pipeline

Earlier iterations used a tee/fakesink split pipeline that forked video into separate inference and display branches. This was fundamentally broken — the display branch could receive frames before inference completed, defeating the purpose of the device. The current architecture uses a single linear path: every frame flows through capture → decode → inference → censor application → display, guaranteeing no frame reaches the screen until processing is complete.

### Asynchronous Secondary Classifier

The secondary classifier runs on the ARM CPU because its underlying ONNX operations are not supported by the Hailo Dataflow Compiler. To prevent CPU inference latency from blocking the video pipeline, the classifier runs asynchronously on a worker thread, with results applied to future frames via a persistence cache. This trades a small amount of latency for sustained 25+ fps display rate.

### Hybrid Censor Rendering

The implementation uses GStreamer's `cairooverlay` element for solid-fill censoring on the common path (no pixel reading required, ~0.5 ms per frame) and OpenCV pixel manipulation only for regions where the secondary classifier confirms specific content types. This minimizes per-frame cost while preserving visual quality where it matters.

---

## Repository Contents

```
.
├── Detect.py            # Desktop simulation (validates detection pipeline)
├── detect_screen.py     # Screen-capture variant for live testing
├── web_ui.py            # Flask configuration interface
└── docs/                # Hardware build guide (v4.0)
    ├── chunk1.docx      # Architecture + hardware list
    ├── chunk2.docx      # OS setup + Hailo stack
    ├── chunk3.docx      # GStreamer pipeline
    ├── chunk4.docx      # Detection strategy + audio sync
    └── chunk5.docx      # Systemd autostart + roadmap
```

The desktop simulation (`Detect.py`) validates the full detection and censoring pipeline on a development machine. The hardware build guide documents the deployment to Raspberry Pi 5.

---

## Status

**Current phase:** Simulation → Hardware migration. The desktop simulation is validated end-to-end. The hardware build guide (v4.0) is drafted with confidence-tagged claims for every architectural decision.

**Next milestones:**
- v4.1 — Real-hardware latency measurement and audio sync calibration
- v4.5 — Falconsai ViT compiled for NPU as first-pass classifier (reduces CPU load)
- v5.0 — Custom YOLO-seg model trained for direct content classification

---

## What I Learned Building This

This project pushed me into territory I had no formal training in: real-time embedded systems, GStreamer pipeline construction, neural network inference on dedicated hardware, Linux service management, and the hard reality that benchmarks from one platform rarely translate cleanly to another.

The most valuable lesson was the importance of **honest confidence assessment**. The project documentation tags every architectural claim with a confidence level (CONFIRMED, NEEDS VERIFICATION, OPEN QUESTION) based on whether it is verified, theoretically sound, or genuinely uncertain. A build guide that pretends everything is certain is worse than one that admits uncertainty — because when reality disagrees with the confident guide, the builder has no framework for understanding why.

---

## License

MIT

---

*Built by Grant Derby. Computer Engineering student, Brigham Young University.*
