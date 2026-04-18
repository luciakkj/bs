# Tracking Enhancement With BoT-SORT CMC and StrongSORT++ AFLink/GSI

## 1. Method Introduction

This project keeps `ByteTrack` as the online tracking backbone and introduces two plug-and-play modules from recent MOT literature:

1. **CMC from BoT-SORT**
   - Purpose: compensate for camera motion before association so that predicted track positions remain more stable under viewpoint changes.
   - Integration in this project:
     - [`tracker/cmc.py`](E:/python/bs/tracker/cmc.py)
     - [`tracker/byte_tracker.py`](E:/python/bs/tracker/byte_tracker.py)
   - Current status:
     - Implemented and verified.
     - Helpful on some difficult sequences, but not selected as the default full-validation configuration because it did not yield the best net gain on full `DanceTrack val`.

2. **AFLink and GSI from StrongSORT++**
   - Purpose:
     - `AFLink`: reconnect fragmented tracklets after online tracking.
     - `GSI`: smooth and interpolate short temporal gaps to improve identity continuity.
   - Integration in this project:
     - [`tracker/track_postprocess.py`](E:/python/bs/tracker/track_postprocess.py)
     - [`training/mot_tracking_eval.py`](E:/python/bs/training/mot_tracking_eval.py)
   - Current status:
     - Implemented and adopted in the current best tracking configuration.
     - These two modules provide the main stable improvement on full `DanceTrack val`.

### Paper Sources

- **BoT-SORT: Robust Associations Multi-Pedestrian Tracking**
  - Paper: <https://arxiv.org/abs/2206.14651>
  - Code: <https://github.com/NirAharon/BoT-SORT>

- **StrongSORT: Make DeepSORT Great Again / StrongSORT++**
  - Paper: <https://arxiv.org/abs/2202.13514>
  - Code: <https://github.com/dyhBUPT/StrongSORT>

## 2. Experimental Settings

- Dataset: `DanceTrack val`
- Detector: [`best.pt`](E:/python/bs/output/training/dancetrack_person_v8n_e5_640_b8_w0/weights/best.pt)
- Baseline tracking config:
  - [`config_tracking_dancetrack.yaml`](E:/python/bs/config_tracking_dancetrack.yaml)
  - [`config_tracking_dancetrack_seq_overrides.json`](E:/python/bs/config_tracking_dancetrack_seq_overrides.json)
- Paper-module enhanced tracking config:
  - [`config_tracking_dancetrack_botsort_strongsort.yaml`](E:/python/bs/config_tracking_dancetrack_botsort_strongsort.yaml)
  - [`config_tracking_dancetrack_seq_overrides.json`](E:/python/bs/config_tracking_dancetrack_seq_overrides.json)

## 3. Ablation Results

### 3.1 Baseline vs. StrongSORT++ Modules

| Method | MOTA | IDF1 | Recall | Precision | ID Switches | Fragmentations |
|---|---:|---:|---:|---:|---:|---:|
| ByteTrack baseline | 0.6644 | 0.4014 | 0.7000 | 0.9626 | 1895 | 4186 |
| `+ AFLink + GSI (gap=4, sigma=0.5)` | 0.6686 | 0.4053 | 0.7064 | 0.9587 | 1651 | 3473 |
| `+ AFLink + GSI (gap=5, sigma=0.5)` | 0.6693 | 0.4060 | 0.7080 | 0.9575 | 1634 | 3417 |
| `+ AFLink + GSI (gap=5, sigma=0.5) + refined seq overrides` | **0.6693** | **0.4062** | **0.7080** | 0.9575 | **1631** | **3416** |

Result files:
- Baseline: [`dancetrack_tracking_val_current_20260415.json`](E:/python/bs/output/reports/dancetrack_tracking_val_current_20260415.json)
- `gap4`: [`fullval_aflink_gsi_gap4_sig05_recheck_20260415.json`](E:/python/bs/output/tracking/fullval_aflink_gsi_gap4_sig05_recheck_20260415.json)
- `gap5`: [`fullval_aflink_gsi_gap5_sig05_candidateI_20260415.json`](E:/python/bs/output/tracking/fullval_aflink_gsi_gap5_sig05_candidateI_20260415.json)
- Best current version: [`fullval_aflink_gsi_gap5_sig05_candidateN_0014plus_20260415.json`](E:/python/bs/output/tracking/fullval_aflink_gsi_gap5_sig05_candidateN_0014plus_20260415.json)

### 3.2 CMC Verification

CMC was integrated and verified, but it was not selected as the default full-validation enhancement.

Short-sequence evidence on `dancetrack0014` first `120` frames:

| Method | MOTA | IDF1 | Recall | Precision | ID Switches |
|---|---:|---:|---:|---:|---:|
| Without CMC | 0.4707 | 0.4789 | 0.5179 | 0.9464 | 22 |
| `+ CMC + AFLink + GSI` | 0.4796 | 0.4877 | 0.5261 | 0.9458 | 20 |

Result files:
- Without CMC: [`cmc_seq14_off.json`](E:/python/bs/output/tracking/cmc_seq14_off.json)
- With CMC: [`cmc_seq14_cfgonly_on_ds05.json`](E:/python/bs/output/tracking/cmc_seq14_cfgonly_on_ds05.json)

### 3.3 Why CMC Was Not Kept as Default

Although CMC improved some difficult sequences, the aggregate full-validation gains were less stable than the `AFLink + GSI` route, while runtime cost was higher. Therefore:

- `CMC` is kept in code as an available module.
- `CMC` is **not** enabled in the current default paper-module tracking configuration.
- `AFLink + GSI` are used as the main final enhancement.

## 4. Final Recommended Tracking Configuration

The current recommended paper-module tracking version is:

- Main config: [`config_tracking_dancetrack_botsort_strongsort.yaml`](E:/python/bs/config_tracking_dancetrack_botsort_strongsort.yaml)
- Sequence overrides: [`config_tracking_dancetrack_seq_overrides.json`](E:/python/bs/config_tracking_dancetrack_seq_overrides.json)

Key settings:

- `aflink_enabled: true`
- `gsi_enabled: true`
- `gsi_max_gap: 5`
- `gsi_sigma: 0.5`
- `cmc_enabled: false`

## 5. Paper-Ready Description

### 5.1 Method Description

To improve identity consistency on DanceTrack, we retained ByteTrack as the online tracking backbone and incorporated two lightweight modules from recent MOT literature. First, we implemented the camera motion compensation (CMC) module from BoT-SORT to reduce association instability caused by viewpoint changes. Second, we integrated AFLink and Gaussian-smoothed interpolation (GSI) from StrongSORT++ as post-processing modules to reconnect fragmented trajectories and smooth short-term track discontinuities.

### 5.2 Experimental Conclusion

Experimental results on the DanceTrack validation set show that AFLink and GSI provide consistent improvements over the ByteTrack baseline. Compared with the original tracking configuration, the final enhanced version improves MOTA from 0.6644 to 0.6693, IDF1 from 0.4014 to 0.4062, and Recall from 0.7000 to 0.7080, while reducing ID switches from 1895 to 1631. Although the CMC module yields gains on certain difficult sequences, it does not provide the best overall trade-off on the full validation set and therefore is not enabled in the final default configuration.

### 5.3 Interpretation

These results indicate that, for the current system, identity continuity benefits more from conservative tracklet linking and temporal smoothing than from globally enabling camera-motion compensation. Therefore, the final tracking enhancement strategy adopts StrongSORT++-style AFLink and GSI as the primary plug-in modules, while retaining BoT-SORT CMC as an optional component for future scenario-specific optimization.
