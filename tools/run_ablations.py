"""
tools/run_ablations.py
Ablation experiment runner for MonoMH-Enhanced.

Generates per-experiment YAML configs with the correct enhancement toggles,
launches training/evaluation for each experiment, and collects results.

Ablation matrix:
  Exp 0: Baseline       — all OFF
  Exp 1: A1 only        — prototype filter ON
  Exp 2: B2 only        — intrinsic conditioner ON
  Exp 3: C2 only        — depth prior warm-start ON
  Exp 4: D1 only        — early exit ON
  Exp 5: A1 + D1        — filtering + speed combo
  Exp 6: Full           — all ON

Usage:
    # Run all experiments
    python tools/run_ablations.py --data_dir /path/to/kitti --gpu 0

    # Run specific experiments
    python tools/run_ablations.py --data_dir /path/to/kitti --gpu 0 --exps 0 3 6

    # Dry run (generate configs only, don't train)
    python tools/run_ablations.py --data_dir /path/to/kitti --dry_run

    # Quick ablation (fewer epochs for sanity check)
    python tools/run_ablations.py --data_dir /path/to/kitti --quick
"""

import os
import sys
import copy
import yaml
import argparse
import subprocess
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# ──────────────────────────────────────────────────
# Ablation Experiment Definitions
# ──────────────────────────────────────────────────

ABLATION_MATRIX = {
    0: {
        'name': 'baseline',
        'desc': 'Baseline reproduction (all enhancements OFF)',
        'toggles': {'A1': False, 'B2': False, 'C2': False, 'D1': False},
        'measures': ['AP3D Easy/Mod/Hard', 'Convergence epoch', 'FPS'],
    },
    1: {
        'name': 'A1_only',
        'desc': 'Prototype-based hypothesis filtering',
        'toggles': {'A1': True, 'B2': False, 'C2': False, 'D1': False},
        'measures': ['AP3D Hard improvement', 'Hypothesis quality'],
    },
    2: {
        'name': 'B2_only',
        'desc': 'Intrinsic-conditioned hypothesis diversity',
        'toggles': {'A1': False, 'B2': True, 'C2': False, 'D1': False},
        'measures': ['AP3D improvement', 'Threshold distribution'],
    },
    3: {
        'name': 'C2_only',
        'desc': 'Depth prior warm-start',
        'toggles': {'A1': False, 'B2': False, 'C2': True, 'D1': False},
        'measures': ['Convergence speed', 'Early-epoch AP3D'],
    },
    4: {
        'name': 'D1_only',
        'desc': 'Hypothesis early exit',
        'toggles': {'A1': False, 'B2': False, 'C2': False, 'D1': True},
        'measures': ['Inference time reduction', 'AP3D retention'],
    },
    5: {
        'name': 'A1_D1',
        'desc': 'Filtering + speed combo',
        'toggles': {'A1': True, 'B2': False, 'C2': False, 'D1': True},
        'measures': ['Combined AP3D + FPS improvement'],
    },
    6: {
        'name': 'full',
        'desc': 'Full integration (all enhancements ON)',
        'toggles': {'A1': True, 'B2': True, 'C2': True, 'D1': True},
        'measures': ['Overall AP3D', 'FPS', 'Convergence', 'Hypothesis quality'],
    },
}


def build_experiment_config(base_cfg, exp_id, data_dir, quick=False):
    """
    Build a per-experiment config by toggling enhancement flags.

    Args:
        base_cfg: dict — base YAML config
        exp_id: int — experiment index from ABLATION_MATRIX
        data_dir: str — path to KITTI dataset root
        quick: bool — if True, reduce epochs for fast validation

    Returns:
        dict — modified config for this experiment
    """
    exp = ABLATION_MATRIX[exp_id]
    cfg = copy.deepcopy(base_cfg)

    # Set data path
    cfg['dataset']['root_dir'] = data_dir

    # Quick mode: reduce epochs significantly
    if quick:
        cfg['trainer']['max_epoch'] = 30
        cfg['trainer']['eval_start'] = 5
        cfg['trainer']['eval_frequency'] = 5
        cfg['trainer']['save_frequency'] = 10

    # Apply enhancement toggles
    toggles = exp['toggles']

    cfg['enhancements']['prototype_filter']['enabled'] = toggles['A1']
    cfg['enhancements']['intrinsic_conditioner']['enabled'] = toggles['B2']
    cfg['enhancements']['depth_prior']['enabled'] = toggles['C2']
    cfg['enhancements']['early_exit']['enabled'] = toggles['D1']

    # C2 needs depth priors precomputed
    if toggles['C2']:
        cfg['enhancements']['depth_prior']['prior_dir'] = 'depth_prior'
        # Adjust warmstart_epochs for quick mode
        if quick:
            cfg['enhancements']['depth_prior']['warmstart_epochs'] = 15

    # A1 warmup adjustment for quick mode
    if quick and toggles['A1']:
        cfg['enhancements']['prototype_filter']['warmup_epochs'] = 3

    return cfg


def save_experiment_config(cfg, exp_id, output_dir):
    """Save experiment-specific YAML config."""
    exp_name = ABLATION_MATRIX[exp_id]['name']
    config_dir = os.path.join(output_dir, f'exp{exp_id}_{exp_name}')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'config.yaml')

    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return config_path, config_dir


def run_experiment(config_path, exp_dir, exp_id, gpu_id, dry_run=False):
    """
    Launch a single experiment's training run.

    Returns:
        dict with timing and status info
    """
    exp = ABLATION_MATRIX[exp_id]
    exp_name = exp['name']

    cmd = [
        sys.executable, os.path.join(ROOT_DIR, 'tools', 'train_val.py'),
        '--config', config_path,
        '--work-date', 'ablation',
        '--work-dir', f'exp{exp_id}_{exp_name}',
        '--save-path', exp_dir,
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # Disable wandb in ablation mode to avoid clutter (use offline mode)
    env['WANDB_MODE'] = 'offline'

    print(f'\n{"="*70}')
    print(f'  Experiment {exp_id}: {exp_name}')
    print(f'  Description: {exp["desc"]}')
    toggles_str = ' | '.join(f'{k}={"ON" if v else "off"}' for k, v in exp['toggles'].items())
    print(f'  Toggles: {toggles_str}')
    print(f'  Config: {config_path}')
    print(f'  Output: {exp_dir}')
    print(f'  Command: {" ".join(cmd)}')
    print(f'{"="*70}')

    if dry_run:
        print('  [DRY RUN] Skipping training')
        return {'status': 'dry_run', 'duration': 0}

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, env=env,
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=86400  # 24h max
        )
        duration = time.time() - start_time

        # Save stdout log
        log_path = os.path.join(exp_dir, f'exp{exp_id}_{exp_name}', 'stdout.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            f.write(result.stdout)

        status = 'success' if result.returncode == 0 else 'failed'
        print(f'\n  [{status.upper()}] Exp {exp_id} ({exp_name}) — {duration/60:.1f} min')

        return {
            'status': status,
            'duration': duration,
            'returncode': result.returncode,
            'log_path': log_path,
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f'\n  [TIMEOUT] Exp {exp_id} ({exp_name}) — exceeded 24h')
        return {'status': 'timeout', 'duration': duration}

    except Exception as e:
        duration = time.time() - start_time
        print(f'\n  [ERROR] Exp {exp_id} ({exp_name}) — {e}')
        return {'status': 'error', 'duration': duration, 'error': str(e)}


def parse_results_from_log(log_path):
    """
    Parse training log to extract best AP3D results.

    Returns:
        dict with best results per difficulty level
    """
    results = {
        'best_easy': -1, 'best_moderate': -1, 'best_hard': -1,
        'best_epoch': -1, 'total_epochs': 0,
    }

    if not os.path.exists(log_path):
        return results

    with open(log_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Look for AP3D results in log
        if '3d@0.70' in line.lower() or "'3d@0.70'" in line:
            try:
                # Parse the results dict
                import ast
                # Find dict-like patterns
                start = line.find('{')
                end = line.rfind('}') + 1
                if start >= 0 and end > start:
                    res_dict = ast.literal_eval(line[start:end])
                    if '3d@0.70' in res_dict:
                        vals = res_dict['3d@0.70']
                        if vals[1] > results['best_moderate']:
                            results['best_easy'] = vals[0]
                            results['best_moderate'] = vals[1]
                            results['best_hard'] = vals[2]
            except (ValueError, SyntaxError):
                pass

        if 'TRAIN EPOCH' in line:
            try:
                epoch = int(line.split('EPOCH')[1].strip().split()[0])
                results['total_epochs'] = max(results['total_epochs'], epoch)
            except (ValueError, IndexError):
                pass

    return results


def generate_summary(all_results, output_dir):
    """Generate a markdown summary of all ablation results."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        '# MonoMH-Enhanced Ablation Results',
        f'',
        f'Generated: {timestamp}',
        f'',
        '## Ablation Matrix',
        '',
        '| Exp | Name | A1 | B2 | C2 | D1 | Status | Duration | AP₃D Easy | AP₃D Mod | AP₃D Hard |',
        '|-----|------|----|----|----|----|---------|---------:|----------:|---------:|----------:|',
    ]

    for exp_id in sorted(all_results.keys()):
        exp = ABLATION_MATRIX[exp_id]
        res = all_results[exp_id]
        t = exp['toggles']

        a1 = '✅' if t['A1'] else '—'
        b2 = '✅' if t['B2'] else '—'
        c2 = '✅' if t['C2'] else '—'
        d1 = '✅' if t['D1'] else '—'

        status = res.get('status', 'unknown')
        duration = res.get('duration', 0)
        dur_str = f'{duration/60:.1f}m' if duration > 0 else '—'

        ap_res = res.get('ap_results', {})
        easy = f"{ap_res.get('best_easy', -1):.2f}" if ap_res.get('best_easy', -1) > 0 else '—'
        mod = f"{ap_res.get('best_moderate', -1):.2f}" if ap_res.get('best_moderate', -1) > 0 else '—'
        hard = f"{ap_res.get('best_hard', -1):.2f}" if ap_res.get('best_hard', -1) > 0 else '—'

        lines.append(
            f'| {exp_id} | {exp["name"]} | {a1} | {b2} | {c2} | {d1} | '
            f'{status} | {dur_str} | {easy} | {mod} | {hard} |'
        )

    # Improvement analysis
    lines.extend([
        '',
        '## Improvement Analysis (vs. Baseline)',
        '',
    ])

    baseline = all_results.get(0, {}).get('ap_results', {})
    if baseline.get('best_moderate', -1) > 0:
        base_mod = baseline['best_moderate']
        lines.append(f'Baseline AP₃D Moderate: **{base_mod:.2f}**')
        lines.append('')
        lines.append('| Exp | Name | ΔAP₃D Mod | Δ% |')
        lines.append('|-----|------|----------:|---:|')

        for exp_id in sorted(all_results.keys()):
            if exp_id == 0:
                continue
            ap_res = all_results[exp_id].get('ap_results', {})
            if ap_res.get('best_moderate', -1) > 0:
                delta = ap_res['best_moderate'] - base_mod
                pct = (delta / base_mod) * 100
                lines.append(
                    f'| {exp_id} | {ABLATION_MATRIX[exp_id]["name"]} | '
                    f'{delta:+.2f} | {pct:+.1f}% |'
                )
    else:
        lines.append('*Baseline results not yet available — run experiment 0 first.*')

    # Enhancement-specific notes
    lines.extend([
        '',
        '## Enhancement-Specific Metrics',
        '',
        '### C2 (Depth Prior Warm-Start)',
        '- Compare Exp 0 vs Exp 3 at epochs 10, 20, 30 to measure convergence acceleration',
        '- The prior loss should decay to 0 by epoch 50 (configurable)',
        '',
        '### D1 (Early Exit)',
        '- Compare inference FPS between Exp 0 and Exp 4',
        '- Check AP₃D retention: should lose < 0.5% while gaining 50-70% FPS',
        '',
        '### A1 (Prototype Filter)',
        '- Compare Exp 0 Hard vs Exp 1 Hard — expect biggest improvement on hard cases',
        '- Bank should saturate within 5-10 epochs after warmup',
        '',
        '### A1 + D1 Combo (Exp 5)',
        '- Should show both AP₃D improvement AND inference speedup simultaneously',
        '- Key experiment for paper: practical gains with minimal overhead',
        '',
        '---',
        '',
        '## How to Run',
        '',
        '```bash',
        '# Full ablation (all 7 experiments)',
        'python tools/run_ablations.py --data_dir /path/to/kitti --gpu 0',
        '',
        '# Quick sanity check (30 epochs each)',
        'python tools/run_ablations.py --data_dir /path/to/kitti --quick',
        '',
        '# Specific experiments only',
        'python tools/run_ablations.py --data_dir /path/to/kitti --exps 0 3 6',
        '',
        '# Analyze results after training completes',
        'python tools/analyze_ablations.py --results_dir outputs/ablation',
        '```',
    ])

    summary_path = os.path.join(output_dir, 'ablation_results.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f'\nSummary saved to: {summary_path}')
    return summary_path


def check_depth_priors(data_dir):
    """Check if depth priors are precomputed (needed for C2 experiments)."""
    prior_dir = os.path.join(data_dir, 'training', 'depth_prior')
    if os.path.isdir(prior_dir):
        count = len([f for f in os.listdir(prior_dir) if f.endswith('.npy')])
        return count > 0, count
    return False, 0


def main():
    parser = argparse.ArgumentParser(description='MonoMH-Enhanced Ablation Runner')
    parser.add_argument('--data_dir', type=str, default='/mnt/Stuff/3d-mono-obj-det/kitti',
                        help='Path to KITTI dataset root')
    parser.add_argument('--base_config', type=str, default=os.path.join(ROOT_DIR, 'lib', 'kitti.yaml'),
                        help='Base config file (default: lib/kitti.yaml)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(ROOT_DIR, 'outputs', 'ablation'),
                        help='Output directory for all experiments')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--exps', nargs='+', type=int, default=None,
                        help='Specific experiment IDs to run (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 30 epochs for fast validation')
    parser.add_argument('--dry_run', action='store_true',
                        help='Generate configs only, do not train')
    args = parser.parse_args()

    # Validate data directory
    if not os.path.isdir(os.path.join(args.data_dir, 'training', 'image_2')):
        print(f'[ERROR] KITTI training images not found at {args.data_dir}/training/image_2')
        sys.exit(1)

    # Load base config
    with open(args.base_config, 'r') as f:
        base_cfg = yaml.load(f, Loader=yaml.Loader)

    # Determine experiments to run
    exp_ids = args.exps if args.exps else sorted(ABLATION_MATRIX.keys())

    # Check if any C2 experiments are requested
    c2_exps = [eid for eid in exp_ids if ABLATION_MATRIX[eid]['toggles']['C2']]
    if c2_exps:
        has_priors, prior_count = check_depth_priors(args.data_dir)
        if not has_priors:
            print(f'[WARNING] C2 experiments ({c2_exps}) require precomputed depth priors.')
            print(f'  Run: python tools/precompute_depth_priors.py --data_dir {args.data_dir}/training')
            print(f'  C2 will be disabled for these experiments until priors are available.')
            # Disable C2 for safety
            for eid in c2_exps:
                ABLATION_MATRIX[eid]['toggles']['C2'] = False
                ABLATION_MATRIX[eid]['desc'] += ' [C2 DISABLED — no priors]'
        else:
            print(f'[OK] Found {prior_count} precomputed depth priors')

    # Print experiment plan
    print(f'\n{"="*70}')
    print(f'  MonoMH-Enhanced Ablation Experiments')
    print(f'  Data: {args.data_dir}')
    print(f'  Output: {args.output_dir}')
    print(f'  GPU: {args.gpu}')
    print(f'  Mode: {"QUICK (30 epochs)" if args.quick else "FULL (200 epochs)"}')
    print(f'  Experiments: {exp_ids}')
    print(f'{"="*70}')

    os.makedirs(args.output_dir, exist_ok=True)

    # Run experiments
    all_results = {}
    for exp_id in exp_ids:
        # Build config
        cfg = build_experiment_config(base_cfg, exp_id, args.data_dir, quick=args.quick)
        config_path, config_dir = save_experiment_config(cfg, exp_id, args.output_dir)

        # Run training
        result = run_experiment(config_path, args.output_dir, exp_id, args.gpu, dry_run=args.dry_run)

        # Parse results from log
        if result['status'] == 'success':
            log_path = result.get('log_path', '')
            result['ap_results'] = parse_results_from_log(log_path)
        else:
            result['ap_results'] = {}

        all_results[exp_id] = result

    # Generate summary
    summary_path = generate_summary(all_results, args.output_dir)

    # Save raw results as YAML
    results_yaml_path = os.path.join(args.output_dir, 'ablation_raw_results.yaml')
    with open(results_yaml_path, 'w') as f:
        # Convert to serializable format
        serializable = {}
        for k, v in all_results.items():
            serializable[k] = {
                'name': ABLATION_MATRIX[k]['name'],
                'status': v.get('status', 'unknown'),
                'duration_seconds': v.get('duration', 0),
                'ap_results': v.get('ap_results', {}),
            }
        yaml.dump(serializable, f, default_flow_style=False)

    print(f'\nRaw results saved to: {results_yaml_path}')
    print(f'Summary saved to: {summary_path}')
    print(f'\nDone! Run `python tools/analyze_ablations.py --results_dir {args.output_dir}` for detailed analysis.')


if __name__ == '__main__':
    main()
