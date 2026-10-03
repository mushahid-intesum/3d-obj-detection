# MonoMH-Enhanced Ablation Results

Generated: 2026-10-03 00:25:14

## Ablation Matrix

| Exp | Name | A1 | B2 | C2 | D1 | Status | Duration | AP₃D Easy | AP₃D Mod | AP₃D Hard |
|-----|------|----|----|----|----|---------|---------:|----------:|---------:|----------:|
| 0 | baseline | — | — | — | — | dry_run | — | — | — | — |
| 1 | A1_only | ✅ | — | — | — | dry_run | — | — | — | — |
| 2 | B2_only | — | ✅ | — | — | dry_run | — | — | — | — |
| 3 | C2_only | — | — | — | — | dry_run | — | — | — | — |
| 4 | D1_only | — | — | — | ✅ | dry_run | — | — | — | — |
| 5 | A1_D1 | ✅ | — | — | ✅ | dry_run | — | — | — | — |
| 6 | full | ✅ | ✅ | — | ✅ | dry_run | — | — | — | — |

## Improvement Analysis (vs. Baseline)

*Baseline results not yet available — run experiment 0 first.*

## Enhancement-Specific Metrics

### C2 (Depth Prior Warm-Start)
- Compare Exp 0 vs Exp 3 at epochs 10, 20, 30 to measure convergence acceleration
- The prior loss should decay to 0 by epoch 50 (configurable)

### D1 (Early Exit)
- Compare inference FPS between Exp 0 and Exp 4
- Check AP₃D retention: should lose < 0.5% while gaining 50-70% FPS

### A1 (Prototype Filter)
- Compare Exp 0 Hard vs Exp 1 Hard — expect biggest improvement on hard cases
- Bank should saturate within 5-10 epochs after warmup

### A1 + D1 Combo (Exp 5)
- Should show both AP₃D improvement AND inference speedup simultaneously
- Key experiment for paper: practical gains with minimal overhead

---

## How to Run

```bash
# Full ablation (all 7 experiments)
python tools/run_ablations.py --data_dir /path/to/kitti --gpu 0

# Quick sanity check (30 epochs each)
python tools/run_ablations.py --data_dir /path/to/kitti --quick

# Specific experiments only
python tools/run_ablations.py --data_dir /path/to/kitti --exps 0 3 6

# Analyze results after training completes
python tools/analyze_ablations.py --results_dir outputs/ablation
```