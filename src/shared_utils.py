"""
shared_utils.py
===============
Shared utilities for both sarcopenia monitoring pipelines.

  - MoveNet loading via TensorFlow Hub
  - Per-frame and per-video keypoint extraction
  - Keypoint normalisation (camera/size invariant)

Why MoveNet (not MediaPipe):
  MoveNet outputs exactly 17 COCO keypoints — the same joint space used
  during training (NWU OpenPose 18 → COCO 17 mapping is near 1:1).
  This means NO adapter layer is needed between training and inference.
  MoveNet also has two speed/accuracy variants:
    'thunder'   → higher accuracy  (training / offline batch)
    'lightning' → real-time speed  (home webcam inference)

Install:
  pip install tensorflow tensorflow-hub opencv-python numpy pandas tqdm scikit-learn
"""

import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

# ─────────────────────────────────────────────────────────────────────────────
# MoveNet COCO-17 keypoint index map
# ─────────────────────────────────────────────────────────────────────────────
KP = {
    "nose"          : 0,
    "left_eye"      : 1,  "right_eye"      : 2,
    "left_ear"      : 3,  "right_ear"      : 4,
    "left_shoulder" : 5,  "right_shoulder" : 6,
    "left_elbow"    : 7,  "right_elbow"    : 8,
    "left_wrist"    : 9,  "right_wrist"    : 10,
    "left_hip"      : 11, "right_hip"      : 12,
    "left_knee"     : 13, "right_knee"     : 14,
    "left_ankle"    : 15, "right_ankle"    : 16,
}

NUM_KEYPOINTS = 17
COORDS        = 2                        # x, y only (confidence dropped)
INPUT_DIM     = NUM_KEYPOINTS * COORDS   # 34 — base model input width


# ─────────────────────────────────────────────────────────────────────────────
# MoveNet loader
# ─────────────────────────────────────────────────────────────────────────────
def load_movenet(variant: str = "thunder"):
    """
    Load MoveNet SinglePose from TensorFlow Hub.

    Args:
        variant: 'thunder'   — higher accuracy, ~20ms/frame
                              use for training + recorded video assessment
                 'lightning' — real-time, ~6ms/frame
                              use for live home webcam inference

    Returns:
        Callable TF serving signature.
    """
    urls = {
        "thunder"  : "https://tfhub.dev/google/movenet/singlepose/thunder/4",
        "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
    }
    url = urls.get(variant, urls["thunder"])
    print(f"[MoveNet] Loading '{variant}' from TF Hub …")
    module = hub.load(url)
    fn     = module.signatures["serving_default"]
    print("[MoveNet] Ready.\n")
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame keypoint extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_keypoints_from_frame(movenet_fn, frame_bgr: np.ndarray) -> np.ndarray:
    """
    Run MoveNet on a single OpenCV BGR frame.

    Returns:
        (17, 3) float32 — [y, x, confidence] per keypoint
        Coordinates are normalised to [0, 1] relative to frame dimensions.
    """
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256))
    tensor  = tf.cast(tf.expand_dims(resized, axis=0), dtype=tf.int32)
    output  = movenet_fn(input=tensor)
    return output["output_0"].numpy()[0, 0]   # (17, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Per-video keypoint extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_keypoints_from_video(movenet_fn, video_path: str,
                                  max_frames: int = 600) -> np.ndarray:
    """
    Extract keypoints from every frame of a video file or webcam stream.

    Args:
        movenet_fn:  loaded MoveNet signature
        video_path:  path to .mp4 file, or 0 for webcam
        max_frames:  cap at this many frames

    Returns:
        (T, 17, 3) float32
    """
    cap    = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(extract_keypoints_from_frame(movenet_fn, frame))

    cap.release()

    if not frames:
        raise ValueError(f"[MoveNet] No frames extracted from: {video_path}")

    return np.array(frames, dtype=np.float32)   # (T, 17, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Keypoint normalisation
# ─────────────────────────────────────────────────────────────────────────────
def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalise keypoints to be invariant to camera position and subject size.

    Strategy:
      1. Center on mid-hip       → removes where the person is in the frame
      2. Scale by torso length   → removes subject height / camera distance
         (torso = mid-hip to mid-shoulder)

    Applied consistently at both training time (on NWU/Health&Gait data)
    and inference time (on patient video), ensuring the model sees the
    same data distribution in both phases.

    Input:  (T, 17, 3)  raw MoveNet output  [y, x, confidence]
    Output: (T, 17, 2)  normalised [y, x]   confidence dropped
    """
    xy = keypoints[:, :, :2].copy()   # (T, 17, 2)

    mid_hip = (xy[:, KP["left_hip"],      :] +
               xy[:, KP["right_hip"],     :]) / 2   # (T, 2)

    mid_shoulder = (xy[:, KP["left_shoulder"], :] +
                    xy[:, KP["right_shoulder"],:]) / 2   # (T, 2)

    torso_len = np.linalg.norm(mid_shoulder - mid_hip, axis=1, keepdims=True)
    torso_len = np.clip(torso_len, 1e-6, None)

    xy -= mid_hip[:, np.newaxis, :]
    xy /= torso_len[:, np.newaxis, :]

    return xy   # (T, 17, 2)