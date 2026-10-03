"""
tools/analyze_ablations.py
Post-hoc analysis of MonoMH-Enhanced ablation experiments.

Parses training logs, collects per-epoch metrics, generates comparison
tables, convergence plots (saved as ASCII/CSV), and a final markdown report.

Usage:
    python tools/analyze_ablations.py --results_dir outputs/ablation
    python tools/analyze_ablations.py --results_dir outputs/ablation --compare 0 3 6
"""

import os
import sys
import argparse
import yaml
import json
import re
from collections import defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def find_experiment_dirs(results_dir):
    """Find all experiment directories in the results folder."""
    experiments = {}
    for entry in sorted(os.listdir(results_dir)):
        if entry.startswith('exp') and os.path.isdir(os.path.join(results_dir, entry)):
            match = re.match(r'exp(\d+)_(.*)', entry)
            if match:
                exp_id = int(match.group(1))
                exp_name = match.group(2)
                exp_path = os.path.join(results_dir, entry)
                experiments[exp_id] = {
                    'name': exp_name,
                    'path': exp_path,
                    'config': os.path.join(exp_path, 'config.yaml'),
                }
    return experiments


def parse_training_log(exp_path):
    """
    Parse training logs to extract per-epoch loss values and eval results.

    Returns:
        dict with 'losses' (per-epoch), 'eval_results' (per-eval-epoch), 'config'
    """
    data = {
        'losses': defaultdict(list),  # {loss_name: [(epoch, value), ...]}
        'eval_results': [],  # [{epoch, easy, moderate, hard}, ...]
        'total_epochs': 0,
        'convergence_epoch': None,  # first epoch reaching >50% of best moderate
    }

    # Find log files
    log_files = []
    for root, dirs, files in os.walk(exp_path):
        for f in files:
            if f.endswith('.log') or f == 'stdout.log':
                log_files.append(os.path.join(root, f))

    for log_path in log_files:
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
        except Exception:
            continue

        current_epoch = 0
        for line in lines:
            # Extract epoch
            epoch_match = re.search(r'TRAIN EPOCH\s+(\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                data['total_epochs'] = max(data['total_epochs'], current_epoch)

            # Extract loss values from batch display
            loss_matches = re.findall(r'(\w+_loss):\s*([\d.]+)', line)
            for loss_name, loss_val in loss_matches:
                try:
                    data['losses'][loss_name].append((current_epoch, float(loss_val)))
                except ValueError:
                    pass

            # Extract eval results
            if '3d@0.70' in line:
                try:
                    import ast
                    start = line.find('{')
                    end = line.rfind('}') + 1
                    if start >= 0 and end > start:
                        res_dict = ast.literal_eval(line[start:end])
                        if '3d@0.70' in res_dict:
                            vals = res_dict['3d@0.70']
                            data['eval_results'].append({
                                'epoch': current_epoch,
                                'easy': vals[0],
                                'moderate': vals[1],
                                'hard': vals[2],
                            })
                except (ValueError, SyntaxError):
                    pass

    # Compute convergence epoch (first epoch reaching >50% of best moderate)
    if data['eval_results']:
        best_mod = max(r['moderate'] for r in data['eval_results'])
        threshold = best_mod * 0.5
        for r in data['eval_results']:
            if r['moderate'] >= threshold:
                data['convergence_epoch'] = r['epoch']
                break

    return data


def compute_experiment_stats(parsed_data):
    """Compute summary statistics from parsed training data."""
    stats = {
        'total_epochs': parsed_data['total_epochs'],
        'convergence_epoch': parsed_data.get('convergence_epoch'),
        'best_easy': -1,
        'best_moderate': -1,
        'best_hard': -1,
        'best_epoch': -1,
        'final_easy': -1,
        'final_moderate': -1,
        'final_hard': -1,
    }

    evals = parsed_data['eval_results']
    if evals:
        # Best results (by moderate)
        best = max(evals, key=lambda r: r['moderate'])
        stats['best_easy'] = best['easy']
        stats['best_moderate'] = best['moderate']
        stats['best_hard'] = best['hard']
        stats['best_epoch'] = best['epoch']

        # Final results
        final = evals[-1]
        stats['final_easy'] = final['easy']
        stats['final_moderate'] = final['moderate']
        stats['final_hard'] = final['hard']

    return stats


def generate_analysis_report(experiments, all_stats, all_parsed, output_dir, compare_ids=None):
    """Generate a comprehensive markdown analysis report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        '# MonoMH-Enhanced — Ablation Analysis Report',
        '',
        f'Generated: {timestamp}',
        '',
    ]

    # ─── Main Results Table ───
    lines.extend([
        '## 1. Main Results (AP₃D @ IoU 0.70)',
        '',
        '| Exp | Name | Best Easy | Best Mod | Best Hard | Best Epoch | Conv. Epoch | Total Epochs |',
        '|-----|------|----------:|---------:|----------:|-----------:|------------:|-------------:|',
    ])

    for exp_id in sorted(all_stats.keys()):
        s = all_stats[exp_id]
        name = experiments[exp_id]['name']
        easy = f"{s['best_easy']:.2f}" if s['best_easy'] > 0 else '—'
        mod = f"{s['best_moderate']:.2f}" if s['best_moderate'] > 0 else '—'
        hard = f"{s['best_hard']:.2f}" if s['best_hard'] > 0 else '—'
        be = str(s['best_epoch']) if s['best_epoch'] > 0 else '—'
        ce = str(s['convergence_epoch']) if s['convergence_epoch'] else '—'
        te = str(s['total_epochs']) if s['total_epochs'] > 0 else '—'

        lines.append(f'| {exp_id} | {name} | {easy} | {mod} | {hard} | {be} | {ce} | {te} |')

    # ─── Improvement vs Baseline ───
    baseline_stats = all_stats.get(0, {})
    baseline_mod = baseline_stats.get('best_moderate', -1)

    if baseline_mod > 0:
        lines.extend([
            '',
            '## 2. Improvement vs. Baseline (Exp 0)',
            '',
            '| Exp | Name | ΔEasy | ΔMod | ΔHard | ΔMod % |',
            '|-----|------|------:|-----:|------:|-------:|',
        ])

        for exp_id in sorted(all_stats.keys()):
            if exp_id == 0:
                continue
            s = all_stats[exp_id]
            if s['best_moderate'] <= 0:
                continue
            name = experiments[exp_id]['name']
            de = s['best_easy'] - baseline_stats['best_easy']
            dm = s['best_moderate'] - baseline_mod
            dh = s['best_hard'] - baseline_stats['best_hard']
            pct = (dm / baseline_mod) * 100

            lines.append(
                f'| {exp_id} | {name} | {de:+.2f} | {dm:+.2f} | {dh:+.2f} | {pct:+.1f}% |'
            )

    # ─── Convergence Analysis ───
    lines.extend([
        '',
        '## 3. Convergence Analysis',
        '',
    ])

    conv_data = []
    for exp_id in sorted(all_stats.keys()):
        s = all_stats[exp_id]
        if s['convergence_epoch']:
            conv_data.append((exp_id, experiments[exp_id]['name'], s['convergence_epoch']))

    if conv_data:
        lines.append('| Exp | Name | 50% Convergence Epoch |')
        lines.append('|-----|------|----------------------:|')
        for eid, name, ce in conv_data:
            lines.append(f'| {eid} | {name} | {ce} |')

        # Compare C2 convergence vs baseline
        baseline_conv = baseline_stats.get('convergence_epoch')
        c2_stats = all_stats.get(3, {})
        c2_conv = c2_stats.get('convergence_epoch')
        if baseline_conv and c2_conv:
            speedup = baseline_conv - c2_conv
            lines.append(f'\n**C2 convergence acceleration:** {speedup} epochs faster than baseline')
    else:
        lines.append('*No convergence data available yet.*')

    # ─── Per-Epoch AP Trajectories ───
    lines.extend([
        '',
        '## 4. Per-Epoch AP₃D Trajectories',
        '',
    ])

    compare = compare_ids or sorted(all_stats.keys())
    for exp_id in compare:
        if exp_id not in all_parsed:
            continue
        evals = all_parsed[exp_id]['eval_results']
        if not evals:
            continue
        name = experiments[exp_id]['name']
        lines.append(f'### Exp {exp_id}: {name}')
        lines.append('')
        lines.append('| Epoch | Easy | Moderate | Hard |')
        lines.append('|------:|-----:|---------:|-----:|')
        for r in evals:
            lines.append(f"| {r['epoch']} | {r['easy']:.2f} | {r['moderate']:.2f} | {r['hard']:.2f} |")
        lines.append('')

    # ─── Loss Analysis ───
    lines.extend([
        '## 5. Enhancement-Specific Loss Tracking',
        '',
    ])

    # Check for enhancement-specific losses
    for exp_id in sorted(all_parsed.keys()):
        losses = all_parsed[exp_id]['losses']
        enhancement_losses = [k for k in losses.keys() if k in ['depth_prior_loss', 'amb_loss']]
        if enhancement_losses:
            name = experiments[exp_id]['name']
            lines.append(f'### Exp {exp_id}: {name}')
            for loss_name in enhancement_losses:
                vals = losses[loss_name]
                if vals:
                    # Get last N values
                    recent = vals[-10:]
                    avg = sum(v for _, v in recent) / len(recent)
                    lines.append(f'- **{loss_name}**: final avg = {avg:.4f} (over last {len(recent)} batches)')
            lines.append('')

    # ─── Recommendations ───
    lines.extend([
        '## 6. Recommendations',
        '',
        '> [!NOTE]',
        '> These recommendations are generated based on the numerical results above.',
        '',
    ])

    if baseline_mod > 0:
        best_exp = max(
            ((eid, s) for eid, s in all_stats.items() if s['best_moderate'] > 0),
            key=lambda x: x[1]['best_moderate'],
            default=(0, baseline_stats)
        )
        lines.append(
            f'- **Best overall:** Exp {best_exp[0]} ({experiments.get(best_exp[0], {}).get("name", "unknown")}) '
            f'with AP₃D Mod = {best_exp[1]["best_moderate"]:.2f}'
        )

        # Best per category
        hard_best = max(
            ((eid, s) for eid, s in all_stats.items() if s['best_hard'] > 0),
            key=lambda x: x[1]['best_hard'],
            default=(0, baseline_stats)
        )
        if hard_best[0] != 0:
            lines.append(
                f'- **Best Hard AP₃D:** Exp {hard_best[0]} ({experiments[hard_best[0]]["name"]}) '
                f'— A1 prototype filtering is likely responsible'
            )
    else:
        lines.append('*Run experiments to generate recommendations.*')

    # Save
    report_path = os.path.join(output_dir, 'ablation_analysis.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f'Analysis report saved to: {report_path}')

    # Also save per-epoch data as CSV for external plotting
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)

    for exp_id in sorted(all_parsed.keys()):
        evals = all_parsed[exp_id]['eval_results']
        if evals:
            name = experiments[exp_id]['name']
            csv_path = os.path.join(csv_dir, f'exp{exp_id}_{name}_ap3d.csv')
            with open(csv_path, 'w') as f:
                f.write('epoch,easy,moderate,hard\n')
                for r in evals:
                    f.write(f"{r['epoch']},{r['easy']:.4f},{r['moderate']:.4f},{r['hard']:.4f}\n")
            print(f'  CSV: {csv_path}')

    return report_path


def main():
    parser = argparse.ArgumentParser(description='Analyze MonoMH-Enhanced ablation results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing ablation experiment outputs')
    parser.add_argument('--compare', nargs='+', type=int, default=None,
                        help='Specific experiment IDs to compare (default: all)')
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f'[ERROR] Results directory not found: {args.results_dir}')
        sys.exit(1)

    # Find experiments
    experiments = find_experiment_dirs(args.results_dir)
    if not experiments:
        print(f'[ERROR] No experiment directories found in {args.results_dir}')
        print('  Expected format: exp0_baseline/, exp1_A1_only/, ...')
        sys.exit(1)

    print(f'Found {len(experiments)} experiments:')
    for eid, info in sorted(experiments.items()):
        print(f'  Exp {eid}: {info["name"]} ({info["path"]})')

    # Parse all experiments
    all_parsed = {}
    all_stats = {}
    for exp_id, info in sorted(experiments.items()):
        print(f'\nParsing Exp {exp_id} ({info["name"]})...')
        parsed = parse_training_log(info['path'])
        all_parsed[exp_id] = parsed
        stats = compute_experiment_stats(parsed)
        all_stats[exp_id] = stats
        print(f'  Total epochs: {stats["total_epochs"]}')
        print(f'  Best AP₃D Moderate: {stats["best_moderate"]:.2f}' if stats['best_moderate'] > 0 else '  Best AP₃D Moderate: N/A')
        print(f'  Best epoch: {stats["best_epoch"]}')

    # Generate report
    print(f'\n{"="*60}')
    print('  Generating Analysis Report')
    print(f'{"="*60}')

    report_path = generate_analysis_report(
        experiments, all_stats, all_parsed,
        args.results_dir, compare_ids=args.compare
    )

    print(f'\n✅ Analysis complete. Report: {report_path}')


if __name__ == '__main__':
    main()
