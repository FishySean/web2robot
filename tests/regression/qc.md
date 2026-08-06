# Stage-1 video quality report

- clips: **10**  |  wall time: 110.6s (11.1s per clip)
- verdicts: **defer** 6, **reject** 2, **trim** 2

> DEFER is not a rejection: hands-only framing cannot be separated from egocentric footage by body pose alone, and first-person video is a valid route in step 2.

## Duplicates dropped

Byte-identical re-uploads under different names. Reported rather than silently skipped -- they inflate the apparent sample size and every rate computed from it.

- `cand2_AZpkh40AlLE.mp4` == `chores_fold_frontal.mp4`
- `cand_cards.mp4` == `chores_cards_frontal.mp4`
- `chores_cupstack_frontal.mp4` == `cup_SUDRM59MIc8.mp4`

| clip | verdict | view | camera | texture | route | usable | reasons |
|---|---|---|---|---|---|---|---|
| chores_cupstack_frontal.mp4 | **trim** | third_person_body | static | rich | egoinfinity/wilor_moge | 24.0s | - |
| cup_cpvH8gzUTko.mp4 | **trim** | third_person_body | static | rich | egoinfinity/wilor_moge | 66.5s | span_conflict |
| cand2_AZpkh40AlLE.mp4 | **defer** | hands_only | static | rich | pending_view/wilor_moge | 15.1s | hands_only, hand_truncated |
| cand2_ZKCmHESpYgM.mp4 | **defer** | hands_only | static | rich | pending_view/wilor_moge | 22.4s | hands_only, no_torso |
| cand_cards.mp4 | **defer** | hands_only | static | rich | pending_view/wilor_moge | 135.2s | hands_only, hand_truncated |
| cand_knit.mp4 | **defer** | hands_only | static | rich | pending_view/wilor_moge | 79.6s | hands_only, hand_truncated |
| fold_uz6.mp4 | **defer** | hands_only | static | rich | pending_view/wilor_moge | 11.8s | hands_only |
| pos_twohands.mp4 | **defer** | hands_only | static | rich | pending_view/wilor_moge | 8.8s | hands_only, low_hand_ratio, span_conflict |
| neg_noperson.mp4 | **reject** | no_stable_hands | unknown | unknown | - | - | no_person |
| neg_onehand.mp4 | **reject** | no_stable_hands | unknown | unknown | - | - | no_stable_hands, low_hand_ratio, no_torso |

## Framing detail (body-pose gate)

| clip | n | step | body_rate | body_span | torso | head | forearm | elbow@btm | wrist_rate† |
|---|---|---|---|---|---|---|---|---|---|
| chores_cupstack_frontal.mp4 | 24 | 1.9s | **0.83** | 25s | 0.83 | 0.75 | 0.192 | 0.00 | 0.96 |
| cup_cpvH8gzUTko.mp4 | 58 | 3.07s | **0.31** | 9s | 0.48 | 0.55 | 0.385 | 0.02 | 0.57 |
| cand2_AZpkh40AlLE.mp4 | 24 | 1.68s | **0.12** | 3s | 0.25 | 0.67 | 0.331 | 0.08 | 0.46 |
| cand2_ZKCmHESpYgM.mp4 | 24 | 2.89s | **0.00** | 0s | 0.00 | 0.33 | 0.464 | 0.12 | 0.50 |
| cand_cards.mp4 | 96 | 7.51s | **0.00** | 0s | 0.14 | 0.85 | 0.292 | 0.02 | 0.47 |
| cand_knit.mp4 | 64 | 3.06s | **0.00** | 0s | 0.06 | 0.45 | 0.195 | 0.03 | 0.42 |
| fold_uz6.mp4 | 24 | 2.37s | **0.00** | 0s | 0.04 | 0.50 | 0.326 | 0.00 | 0.29 |
| pos_twohands.mp4 | 24 | 0.51s | **0.00** | 0s | 0.04 | 0.33 | 0.304 | 0.00 | 0.21 |
| neg_noperson.mp4 | 24 | 0.51s | **0.00** | 0s | 0.00 | 0.00 | 0.000 | 0.00 | 0.00 |
| neg_onehand.mp4 | 24 | 0.51s | **0.00** | 0s | 0.00 | 0.33 | 0.127 | 0.00 | 0.25 |

`head` is reported but never gates: a clip whose camera missed the head is still usable.

† `wrist_rate` (both wrist KEYPOINTS visible) is reported and **decides nothing**. Measured inverted on hands-only footage: paired controls scored 0.21 for two hands vs 0.25 for one hand at det 0.7, and the inversion held at 0.3/0.1/0.05. That boundary moved to the hand detector below.

## Hand detail (hand gate — decides hands_only vs no_stable_hands)

| clip | n | both_hand | hands_span | any_hand | mean_n | avg_size | trunc | dup |
|---|---|---|---|---|---|---|---|---|
| chores_cupstack_frontal.mp4 | 24 | **1.00** | 44s | 1.00 | 2.04 | 0.0155 | 0.00 | 1 |
| cup_cpvH8gzUTko.mp4 | 58 | **0.43** | 18s | 0.83 | 1.28 | 0.0958 | 0.24 | 11 |
| cand2_AZpkh40AlLE.mp4 | 24 | **0.54** | 15s | 0.88 | 1.42 | 0.0674 | 0.54 | 2 |
| cand2_ZKCmHESpYgM.mp4 | 24 | **0.42** | 29s | 0.75 | 1.17 | 0.0561 | 0.25 | 0 |
| cand_cards.mp4 | 96 | **0.48** | 135s | 0.95 | 1.43 | 0.1959 | 0.61 | 39 |
| cand_knit.mp4 | 64 | **0.72** | 80s | 0.92 | 1.64 | 0.2314 | 0.86 | 3 |
| fold_uz6.mp4 | 24 | **0.38** | 12s | 0.79 | 1.17 | 0.0201 | 0.29 | 4 |
| pos_twohands.mp4 | 24 | **0.42** | 3s | 0.58 | 1.00 | 0.0667 | 0.29 | 1 |
| neg_noperson.mp4 | 24 | **0.00** | 0s | 0.00 | 0.00 | 0.0000 | 0.00 | 0 |
| neg_onehand.mp4 | 24 | **0.00** | 0s | 0.54 | 0.54 | 0.0702 | 0.33 | 1 |

`any_hand` is the official `hand_ratio` (>=1 hand) kept at its official definition for comparability; it **cannot** tell one hand from two (0.58 vs 0.54 on the paired controls), which is why `both_hand` exists. `avg_size` / `trunc` are recorded, never fatal. `dup` counts duplicate boxes merged away (same hand detected twice) -- without that merge a single pair of hands can report as three.

A class is reached by EITHER a high whole-file rate OR a long enough contiguous span. `*_span` is interpolated between samples (`step` apart), not measured frame by frame.

## Config

```json
{
  "n_frames": 24,
  "sample_every_sec": 3.0,
  "n_frames_max": 96,
  "sample_lo": 0.08,
  "sample_hi": 0.92,
  "kpt_score_thresh": 3.0,
  "det_score_thresh": 0.7,
  "body_frame_rate_min": 0.6,
  "hands_only_rate_min": 0.4,
  "hand_weights": "/mnt/vlm/fanshaoheng/HaWoR/weights/external/detector.pt",
  "hand_conf": 0.3,
  "hand_iou_merge": 0.1,
  "both_hand_rate_min": 0.25,
  "min_hand_ratio": 0.75,
  "min_hand_size": 0.005,
  "max_hand_size": 0.4,
  "trunc_edge_px": 5,
  "max_trunc_ratio": 0.5,
  "min_usable_sec": 5.0,
  "span_gap_tol": 1,
  "min_run_density": 0.6,
  "scene_threshold": 0.4,
  "min_subseg_sec": 5.0,
  "cut_ignore_before_sec": 0.1,
  "max_bg_flow": 2.0,
  "flow_grid": [
    4,
    4
  ],
  "flow_percentile": 20,
  "flow_max_pairs": 24,
  "min_corner_density": 0.0002,
  "min_hand_lapvar": 40.0,
  "min_duration_sec": 10.0,
  "min_side_px": 240,
  "device": "auto",
  "early_exit": true
}
```
