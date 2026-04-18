# Intelligent Video Surveillance

This project follows the graduation-topic requirements for an intelligent video surveillance system:

- YOLO-based person detection
- ByteTrack-style lightweight multi-object tracking
- Trajectory-based abnormal behavior analysis
- Training-first workflow built on DanceTrack and external behavior datasets

The current priority is the model training module. A UI is intentionally not included yet.

## Project Structure

- `app/`: runtime pipeline
- `behavior/`: intrusion, cross-line, loitering, and running detection
- `detector/`: YOLO detector wrapper
- `tracker/`: ByteTrack-style tracker with Kalman prediction and two-stage association
- `training/`: detector training, tracking evaluation, and behavior-model utilities
- `data/external/dancetrack/`: raw DanceTrack dataset
- `data/processed/dancetrack_person/`: generated YOLO-format detection dataset
- `output/training/`: detector training outputs

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

### 1. Train the detector

```powershell
venv\Scripts\python.exe train.py train --epochs 50 --imgsz 640 --batch 8
```

By default this command trains on `data/processed/dancetrack_person/dancetrack_person.yaml`.

### 2. Validate the detector

```powershell
venv\Scripts\python.exe train.py validate --weights output\training\dancetrack_person\weights\best.pt
```

### 3. Evaluate tracking

```powershell
venv\Scripts\python.exe train.py eval-mot-tracking --config-path config_tracking_dancetrack.yaml --mot-root data\external\dancetrack --split-name val
```

This command evaluates detector + tracker performance on DanceTrack with MOT-style metrics such as `MOTA`, `IDF1`, and `ID Switches`.

### 4. Validate anomaly behavior on CUHK Avenue

```powershell
venv\Scripts\python.exe train.py validate-avenue --model models\yolov8n.pt --enable-loitering --enable-running --max-videos 1
```

### 5. Generate track-level pseudo labels from CUHK Avenue

```powershell
venv\Scripts\python.exe train.py generate-avenue-pseudo-labels --model models\yolov8n.pt --device 0
```

This command uses Avenue testing videos plus anomaly masks to generate track-level pseudo labels for:

- `running`
- `loitering`
- `normal`
- `unknown`

Outputs are saved under `output/avenue_pseudo_labels/` as JSONL manifests and a `summary.json` report.

### 6. Train a lightweight behavior classifier

```powershell
venv\Scripts\python.exe train.py train-behavior --dataset-path output\avenue_pseudo_labels\tracks.jsonl --device 0
```

This trains a small MLP on trajectory-level features using the pseudo labels generated from CUHK Avenue.

Artifacts are saved under `output/behavior_training/`.

### 7. Expand behavior pseudo labels

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
- `cross_line` is supported in `config.yaml`; leave it empty to disable it.
- The tracker implementation includes Kalman prediction, high/low score matching, ReID support, and optional BoT-SORT / StrongSORT++ style add-ons.
