# Intelligent Video Surveillance

This project follows the graduation-topic requirements for an intelligent video surveillance system:

- YOLO-based person detection
- ByteTrack-style lightweight multi-object tracking
- Trajectory-based abnormal behavior analysis
- Training-first workflow built on MOT17

The current priority is the model training module. A UI is intentionally not included yet.

## Project Structure

- `app/`: runtime pipeline
- `behavior/`: intrusion, cross-line, loitering, and running detection
- `detector/`: YOLO detector wrapper
- `tracker/`: ByteTrack-style tracker with Kalman prediction and two-stage association
- `training/`: dataset preparation, training, validation, and behavior-threshold calibration
- `data/MOT17/`: raw MOT17 dataset
- `data/processed/mot17_person/`: generated YOLO-format dataset
- `output/training/`: detector training outputs
- `output/calibration/`: behavior calibration outputs

## Environment

Use the repository virtual environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Or run commands directly with:

```powershell
venv\Scripts\python.exe
```

## Training Workflow

### 1. Prepare MOT17 for YOLO

```powershell
venv\Scripts\python.exe train.py prepare --overwrite
```

This command:

- reads `data/MOT17/train`
- keeps `FRCNN` sequences by default to avoid triple-duplicating the same scene
- converts pedestrian annotations into a single `person` class
- creates a YOLO dataset under `data/processed/mot17_person`

### 2. Calibrate behavior thresholds

```powershell
venv\Scripts\python.exe train.py calibrate
```

This writes `output/calibration/behavior_thresholds.json` with suggested values for:

- `loiter_frames`
- `loiter_radius`
- `loiter_speed`
- `running_speed`

Because MOT17 does not contain explicit abnormal-event labels, this step is calibration rather than supervised behavior training.

### 3. Train the detector

```powershell
venv\Scripts\python.exe train.py train --epochs 50 --imgsz 960 --batch 8
```

Output weights are saved under `output/training/mot17_person/weights/`.

### 4. Validate the detector

```powershell
venv\Scripts\python.exe train.py validate --weights output\training\mot17_person\weights\best.pt
```

### 5. Validate anomaly behavior on CUHK Avenue

```powershell
venv\Scripts\python.exe train.py validate-avenue --model models\yolov8n.pt --enable-loitering --enable-running --max-videos 1
```

This command runs detection, tracking, and rule-based anomaly logic on CUHK Avenue and compares predicted anomalous frames against Avenue anomaly masks.

### 6. Generate track-level pseudo labels from CUHK Avenue

```powershell
venv\Scripts\python.exe train.py generate-avenue-pseudo-labels --model models\yolov8n.pt --device 0
```

This command uses Avenue testing videos plus anomaly masks to generate track-level pseudo labels for:

- `running`
- `loitering`
- `normal`
- `unknown`

Outputs are saved under `output/avenue_pseudo_labels/` as JSONL manifests and a `summary.json` report.

### 7. Train a lightweight behavior classifier

```powershell
venv\Scripts\python.exe train.py train-behavior --dataset-path output\avenue_pseudo_labels\tracks.jsonl --device 0
```

This trains a small MLP on trajectory-level features using the pseudo labels generated from CUHK Avenue.

Artifacts are saved under `output/behavior_training/`.

### 8. Expand behavior pseudo labels

```powershell
venv\Scripts\python.exe train.py expand-behavior-dataset --input-path output\avenue_pseudo_labels\tracks.jsonl
```

This expands the pseudo-label set by:

- slicing additional windows from long `running` and `loitering` tracks
- re-checking abnormal-overlap `unknown` tracks with relaxed heuristics
- preserving `source_track_id` so training can split by original track

## Runtime

After training, update `config.yaml` so `model.model_path` points to the trained `best.pt`, then run:

```powershell
venv\Scripts\python.exe main.py
```

During the current training-first phase, abnormal events can be disabled independently in `config.yaml`:

- `enable_intrusion`
- `enable_cross_line`
- `enable_loitering`
- `enable_running`

If you want to enable the learned behavior model later, set:

- `behavior.behavior_mode` to `hybrid` or `model`
- `behavior.behavior_model_path` to the trained `best.pt`

## Notes

- `ByteTrack` is an association algorithm and does not require supervised training.
- The abnormal-behavior module is rule-based, but the thresholds are now calibratable from MOT17 trajectories.
- `cross_line` is supported in `config.yaml`; leave it empty to disable it.
- The tracker implementation now includes Kalman prediction, high/low score matching, and tracked/lost/removed states, which is much closer to the algorithmic flow expected in a graduation defense.
