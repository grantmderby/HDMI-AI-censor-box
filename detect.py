# ─── PATH FIX: Must run BEFORE any onnxruntime import ────────────
import os, torch
cuda_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['PATH'] = cuda_lib + os.pathsep + os.environ.get('PATH', '')
# ─────────────────────────────────────────────────────────────────

import numpy as np
import cv2
import json
import threading
import time
from ultralytics import YOLO
from nudenet import NudeDetector


class AsyncStream:
    def __init__(self, src):
        self.src     = src
        self.frame   = None
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        cap = cv2.VideoCapture(self.src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while not self.stopped:
            grabbed, frame = cap.read()
            if grabbed:
                self.frame = frame
            else:
                self.stopped = True
        cap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class AsyncNudeNet:
    """Runs NudeNet in a background thread. Main loop never blocks waiting."""

    def __init__(self, detector, confidence=0.35):
        self.detector   = detector
        self.confidence = confidence
        self.mask       = None
        self.stopped    = False
        self.lock       = threading.Lock()
        self.event      = threading.Event()  # signals new frame available
        self._frame     = None
        self._active    = []
        self._persons   = None

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()
        return self

    def submit(self, frame, active_classes, person_boxes=None):
        with self.lock:
            self._frame   = frame.copy()
            self._active  = active_classes
            self._persons = person_boxes
        self.event.set()  # wake up the background thread

    def run(self):
        while not self.stopped:
            # Sleep until a frame is submitted — no busy-wait
            self.event.wait(timeout=0.5)
            self.event.clear()

            with self.lock:
                frame   = self._frame
                active  = self._active
                persons = self._persons

            if frame is None:
                continue

            h, w = frame.shape[:2]
            local_mask = np.zeros((h, w), dtype=np.uint8)

            # PASS 1: Crop-based scan on YOLO person regions
            if persons is not None and len(persons) > 0:
                for (px1, py1, px2, py2) in persons:
                    px1, py1 = max(0, px1), max(0, py1)
                    px2, py2 = min(w, px2), min(h, py2)

                    crop = frame[py1:py2, px1:px2]
                    if crop.size == 0:
                        continue

                    ch, cw = crop.shape[:2]
                    scale = max(1, 320 // max(min(ch, cw), 1))
                    if scale > 1:
                        crop = cv2.resize(crop, (cw * scale, ch * scale),
                                          interpolation=cv2.INTER_LINEAR)

                    detections = self.detector.detect(crop)
                    crop_h, crop_w = crop.shape[:2]

                    for det in detections:
                        label = det['class']
                        score = det['score']
                        if label in active and score >= self.confidence:
                            bx, by, bw, bh = det['box']
                            rx = px1 + int(bx * (px2 - px1) / crop_w)
                            ry = py1 + int(by * (py2 - py1) / crop_h)
                            rw = int(bw * (px2 - px1) / crop_w)
                            rh = int(bh * (py2 - py1) / crop_h)
                            x1 = max(0, rx)
                            y1 = max(0, ry)
                            x2 = min(w, rx + rw)
                            y2 = min(h, ry + rh)
                            local_mask[y1:y2, x1:x2] = 1

            # PASS 2: Full-frame safety net
            full_detections = self.detector.detect(frame)
            for det in full_detections:
                label = det['class']
                score = det['score']
                if label in active and score >= self.confidence:
                    x, y, bw, bh = det['box']
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w, x + bw)
                    y2 = min(h, y + bh)
                    local_mask[y1:y2, x1:x2] = 1

            with self.lock:
                self.mask = local_mask

    def get_mask(self):
        with self.lock:
            return self.mask

    def stop(self):
        self.stopped = True
        self.event.set()  # wake thread so it can exit


# ─── MODELS ──────────────────────────────────────────────────────
COCO_MODEL_PATH = 'yolo26n-seg.pt'

USE_COCO_MODEL = True
USE_NUDE_MODEL = True

# ─── CONFIGURATION ───────────────────────────────────────────────
CONFIG_FILE   = 'config.json'
CONFIDENCE    = 0.35
WEBCAM_INDEX  = 0
SHOW_WINDOW   = True
USE_FACE_BLUR = False

# ─── BLUR SETTINGS ───────────────────────────────────────────────
BLUR_STYLE  = 'pixelate'
PIXEL_SIZE  = 16
BLUR_KERNEL = (31, 31)

# ─── NUDENET PERFORMANCE ─────────────────────────────────────────
MAX_PERSONS = 3

# ─── LEWDNESS LEVELS ─────────────────────────────────────────────
LEVELS = {

    'KIDS': [
        'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED',
        'FEMALE_GENITALIA_COVERED', 'FEMALE_BREAST_EXPOSED',
        'FEMALE_BREAST_COVERED',    'MALE_BREAST_EXPOSED',
        'BUTTOCKS_EXPOSED',         'BUTTOCKS_COVERED',
        'BELLY_EXPOSED',            'ANUS_EXPOSED',
    ],

    'STRICT': [
        'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED',
        'FEMALE_BREAST_EXPOSED',
        'BUTTOCKS_EXPOSED',         'ANUS_EXPOSED',
    ],

    'STANDARD': [
        'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED',
        'FEMALE_BREAST_EXPOSED',    'ANUS_EXPOSED',
    ],

    'LIGHT': ['banana'],
}

# ─── SHARED STATE (JSON IPC) ─────────────────────────────────────
cached_level  = 'STRICT'
frame_counter = 0
level_lock    = threading.Lock()


def get_current_level():
    global frame_counter, cached_level
    frame_counter += 1
    if frame_counter % 30 == 0:
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            with level_lock:
                cached_level = data.get('level', cached_level)
        except:
            pass
    with level_lock:
        return cached_level


def get_face_detector():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)


def force_nudenet_gpu(detector):
    import onnxruntime as ort
    import nudenet

    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in providers:
        print('  ONNX CUDA not available — NudeNet stays on CPU.')
        return detector

    model_path = os.path.join(os.path.dirname(nudenet.__file__), '320n.onnx')
    if not os.path.exists(model_path):
        print(f'  Model not found at {model_path}')
        return detector

    print(f'  Found: {model_path}')
    print(f'  Forcing CUDA provider...')

    detector.onnx_session = ort.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    active = detector.onnx_session.get_providers()
    print(f'  Active providers: {active}')
    return detector


def apply_censor(frame, mask_bool):
    h, w = frame.shape[:2]
    if BLUR_STYLE == 'pixelate':
        small     = cv2.resize(frame,
                               (max(1, w // PIXEL_SIZE), max(1, h // PIXEL_SIZE)),
                               interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[mask_bool] = pixelated[mask_bool]
    elif BLUR_STYLE == 'box':
        blurred = cv2.blur(frame, BLUR_KERNEL)
        frame[mask_bool] = blurred[mask_bool]
    elif BLUR_STYLE == 'stack':
        blurred = cv2.stackBlur(frame, BLUR_KERNEL)
        frame[mask_bool] = blurred[mask_bool]
    else:
        blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
        frame[mask_bool] = blurred[mask_bool]
    return frame


def build_yolo_mask(model, frame, active_classes, combined_mask, device):
    detections_found = False
    person_boxes     = []
    results = model(
        frame, imgsz=640, verbose=False, retina_masks=True, device=device
    )
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                if label == 'person' and float(box.conf) > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))

        if result.masks is not None:
            for mask, box in zip(result.masks.data, result.boxes):
                label = model.names[int(box.cls)]
                if label in active_classes and float(box.conf) > CONFIDENCE:
                    mask_np      = mask.cpu().numpy()
                    mask_resized = cv2.resize(
                        mask_np,
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    combined_mask = cv2.bitwise_or(
                        combined_mask,
                        (mask_resized > 0.5).astype(np.uint8)
                    )
                    detections_found = True

    person_boxes = sorted(
        person_boxes,
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        reverse=True
    )[:MAX_PERSONS]

    return combined_mask, detections_found, person_boxes


# ─── MAIN ────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    import onnxruntime as ort
    ort_providers = ort.get_available_providers()
    print(f'ONNX Runtime providers: {ort_providers}')

    coco_model = None
    async_nn   = None
    nn_device  = 'OFF'
    dummy      = np.zeros((640, 640, 3), dtype=np.uint8)

    if USE_COCO_MODEL:
        print(f'Loading YOLO26: {COCO_MODEL_PATH}')
        coco_model = YOLO(COCO_MODEL_PATH)
        coco_model(dummy, imgsz=640, verbose=False, device=device)
        print(f'YOLO26 classes: {coco_model.names}')

    if USE_NUDE_MODEL:
        print('Loading NudeNet...')
        nude_detector = NudeDetector()
        nude_detector = force_nudenet_gpu(nude_detector)
        nude_detector.detect(dummy)
        nn_providers = nude_detector.onnx_session.get_providers()
        nn_device = 'GPU' if 'CUDAExecutionProvider' in nn_providers else 'CPU'
        print(f'NudeNet on: {nn_device}. Max persons: {MAX_PERSONS}.')
        async_nn = AsyncNudeNet(nude_detector, CONFIDENCE).start()

    print(f'Blur: {BLUR_STYLE} | Pixel: {PIXEL_SIZE}')

    stream        = AsyncStream(WEBCAM_INDEX).start()
    face_detector = get_face_detector()

    fps_start       = time.time()
    fps_count       = 0
    fps             = 0.0
    cached_mask     = None
    blur_ttl        = 0
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))

    print('Running. Press Q to quit.')
    print('Web UI: http://127.0.0.1:5000')

    while True:
        frame = stream.read()
        if frame is None:
            continue

        fps_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps       = fps_count / elapsed
            fps_start = time.time()
            fps_count = 0
        else:
            fps = fps_count / max(elapsed, 0.001)

        if USE_FACE_BLUR:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (99, 99), 0)
            hud = f'Mode: FACE BLUR | {fps:.1f} fps'
            cv2.rectangle(frame, (0, 0), (340, 30), (0, 0, 0), -1)
            cv2.putText(frame, hud, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if SHOW_WINDOW:
                cv2.imshow('AI Censor', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        current_level  = get_current_level()
        active_classes = LEVELS[current_level]

        combined_mask    = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        detections_found = False
        person_boxes     = []

        # Stage 1: YOLO26 — blocks briefly (~10ms on GPU), finds persons + objects
        if coco_model is not None:
            combined_mask, found, person_boxes = build_yolo_mask(
                coco_model, frame, active_classes, combined_mask, device
            )
            detections_found = detections_found or found

        # Stage 2: NudeNet async — submit frame, read latest completed result
        if async_nn is not None:
            async_nn.submit(frame, active_classes, person_boxes)
            nn_mask = async_nn.get_mask()
            if nn_mask is not None and nn_mask.shape == combined_mask.shape:
                combined_mask = cv2.bitwise_or(combined_mask, nn_mask)
                if np.any(nn_mask > 0):
                    detections_found = True

        if detections_found:
            combined_mask = cv2.dilate(combined_mask, dilation_kernel, iterations=1)
            cached_mask   = combined_mask
            blur_ttl      = 12

        if cached_mask is not None and blur_ttl > 0:
            frame = apply_censor(frame, cached_mask > 0)

        blur_ttl = max(0, blur_ttl - 1)

        models_active = []
        if USE_COCO_MODEL: models_active.append('YOLO26')
        if USE_NUDE_MODEL: models_active.append('NudeNet')
        model_str = '+'.join(models_active)

        hud = f'Level: {current_level} | {fps:.1f} fps | {BLUR_STYLE} | {model_str} | YOLO:{device.upper()} NN:{nn_device}'
        cv2.rectangle(frame, (0, 0), (720, 30), (0, 0, 0), -1)
        cv2.putText(frame, hud, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if SHOW_WINDOW:
            cv2.imshow('AI Censor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if async_nn:
        async_nn.stop()
    stream.stop()
    cv2.destroyAllWindows()
    print('Stopped.')


if __name__ == '__main__':
    main()