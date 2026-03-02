# 5-Times Sit-to-Stand (5STS) Monitoring

Computer vision pipeline for monitoring sarcopenia progression and fall risk using the 5-Times Sit-to-Stand clinical functional test. Designed for remote home monitoring — the patient only needs a webcam or phone camera.

---

## Clinical Background

The 5STS test measures how long a patient takes to rise from a chair and sit back down 5 complete times without using their arms.

| Result | Risk Category |
|---|---|
| < 12 seconds | Low fall risk |
| ≥ 12 seconds | **High fall risk** |

Worsening is flagged when a follow-up assessment is ≥ 2 seconds slower than baseline, or when risk category escalates from Low → High.

---

## How It Works

```
TRAINING (one-time, offline)
  NWU CSV  ──→  build_synthetic_windows()  ──→  Conv1D → GRU → Dense  ──→  5sts_best.keras

INFERENCE (each patient visit, at home)
  Patient video  ──→  MoveNet Lightning  ──→  17 keypoints/frame
                                                      │
                                               normalize_keypoints()
                                                      │
                                              trained model (per window)
                                                      │
                                            state machine (count transitions)
                                                      │
                                              duration + risk category
```

---

## Dataset

**NWU Human Pose Dataset**
- 50,727 rows of pre-extracted OpenPose keypoints
- Labels: `sit` (12,979 rows) and `stand` (37,748 rows)
- Source: https://dayta.nwu.ac.za/articles/dataset/Human_pose_dataset_sit_stand_pose_classes_/23290937
- Download and rename to `nwu_sit_stand.csv`

The NWU dataset provides static pose frames, not video. Synthetic temporal windows of 60 frames are constructed during training to match the sequential input the model sees during live inference.

---

## Model Architecture

```
Input: (batch, 60, 34)   ← 60-frame window, 17 COCO joints × 2 coords

Conv1D(64,  kernel=3)  ← captures local frame-to-frame sit↔stand transitions
BatchNorm + ReLU
Conv1D(128, kernel=3)  ← deeper local patterns
BatchNorm + ReLU + Dropout(0.3)

GRU(64)                ← learns sustained posture pattern across the window
Dropout(0.3)

Dense(64, relu)
Dense(1, sigmoid)      ← probability of 'standing' state
```

---

## Performance (v2, seed=42)

| Metric | Sit | Stand | Overall |
|---|---|---|---|
| Precision | 0.92 | 0.99 | — |
| Recall | 0.99 | 0.91 | — |
| F1-score | 0.95 | 0.95 | — |
| Accuracy | — | — | **95%** |

> Note: `train_loss > val_loss` is expected and not a sign of underfitting. It is caused by label smoothing (0.1) making training targets harder to fit. The accuracy gap (train 0.96 vs val 0.97) is the correct health check and shows clean generalisation.

---

## Installation

```bash
pip install tensorflow tensorflow-hub opencv-python numpy pandas scikit-learn tqdm
```

---

## File Structure

```
├── README.md 
├── .gitignore
├── requirements.txt                                    <- The requirements file for reproducing the environment.
├── nwu_human_pose_dataset
│   ├── dataset_HumanPose_SampleImages/
│   └── dataset_HumanPose_KeypointCoordinates.xlsx     
├── models                                              <- saved model checkpoints (5sts_best_YYMMDD_HHMMSS.keras)
├── notebooks                                           <- Jupyter notebooks for exploration and analysis
│   └── 5sts.ipynb 
├── report                                              <- Generated analysis reports, figures, and visualizations.
├── requirements.txt      
└── src                                                 <- Source code for use in this project.
    └── shared_utils.py                                 <- MoveNet loader, keypoint extraction, normalisation

```

---

## Usage

### Training

```python
from pipeline_5sts import load_nwu_dataset, build_synthetic_windows, build_model, train_model

X, y         = load_nwu_dataset("nwu_sit_stand.csv")
X_win, y_win = build_synthetic_windows(X, y)
model        = build_model()
model        = train_model(model, X_train, y_train, X_val, y_val)
```

### Patient Inference

```python
from shared_utils import load_movenet
from pipeline_5sts import infer_5sts, compare

movenet_fn = load_movenet("lightning")   # fast variant for home use

baseline = infer_5sts(model, movenet_fn, "patient_baseline.mp4")
followup = infer_5sts(model, movenet_fn, "patient_week8.mp4")
report   = compare(baseline, followup)
```

### Example output

```json
{
  "test": "5STS",
  "baseline_s": 10.4,
  "followup_s": 13.1,
  "delta_s": 2.7,
  "worsened": true,
  "baseline_risk": "Low",
  "followup_risk": "High",
  "action": "⚠️  Alert clinician"
}
```

---

## Key Design Decisions

**Why MoveNet over MediaPipe**
MoveNet outputs 17 COCO keypoints — the same joint space used by the NWU training data. This eliminates the need for an adapter layer between training and inference, and MoveNet Lightning runs in real-time (~6ms/frame) on low-end home devices.

**Why NWU over Kinetics**
Both Kinetics-400 and Kinetics-700 were verified to contain no sit, stand, sitting down, or standing up labels. NWU provides 50k+ pre-extracted keypoints with exact sit/stand labels and no video processing overhead.

**Sit class oversampling**
The NWU dataset has a 2.9:1 stand:sit imbalance. A tighter sliding window step for the sit class (seq_len//6 vs seq_len//2) balances the training windows to ~1:1, directly improving Sit F1 from 0.83 to 0.95.

**Label smoothing (0.1)**
Reduces overconfidence and acts as regularisation. Causes train_loss to appear higher than val_loss — this is expected behaviour, not underfitting.

**State machine thresholds**
Stand is called at prob ≥ 0.65 and sit at prob ≤ 0.35 (tighter than the default 0.5 boundary) to reduce false transitions that would corrupt the rep count and duration measurement.

---

## Limitations

- Training data (NWU) consists of static public images, not real patient videos. A domain gap exists between NWU keypoints and home webcam footage. Fine-tuning on a small set of real patient recordings is recommended before clinical deployment.
- MoveNet SinglePose works best when only one person is visible in the frame. Ensure the patient is filmed alone against a clear background.
- Gait speed monitoring is handled separately in `pipeline_gait.py`. A unified clinical report combining both tests is available in `pipeline_assessment.py`.