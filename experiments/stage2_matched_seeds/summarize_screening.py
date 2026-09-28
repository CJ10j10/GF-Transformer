#!/usr/bin/env python3
"""Summarize every preregistered K2-minus-B0 seed pair after all runs finish."""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
METRICS = ('F1b', 'F1d', 'F1s', 'F1_0', 'F1_1', 'F1_2', 'F1_3')


def load_result(variant, seed):
    if seed == 3:
        path = ROOT / ('experiments/stage2_b0_ddp2/results/ddp2_single_view.json'
                       if variant == 'b0' else
                       'experiments/stage2_k2_kalman_refine/results/ddp2_single_view.json')
    else:
        path = EXP / 'runs' / f'{variant}_seed{seed}' / 'results' / 'single_view.json'
    result = json.loads(path.read_text())
    assert result['mode'] == 'single' and result['views'] == ['original']
    if seed != 3:
        assert result['variant'] == variant and result['train_seed'] == seed
    return result


def main():
    plan = json.loads((EXP / 'seed_plan.json').read_text())
    frozen = json.loads((ROOT / 'experiments/stage2_b0_ddp2/results/freeze_manifest.json').read_text())
    rows = []
    for seed in plan['train_seeds_for_all_variants']:
        b0, k2 = load_result('b0', seed), load_result('k2', seed)
        for result in (b0, k2):
            assert result['validation_images'] == 917
            assert result['validation_split_sha256'] == frozen['image_ids_sha256']
            assert result['localization_masks'] == frozen['localization_masks']
            assert result['metric_code_sha256'] == frozen['metric_code_sha256']
            if seed != 3:
                assert result['validation_mask_manifest_sha256'] == frozen['validation_mask_manifest_sha256']
        row = {'train_seed': seed,
               'b0_checkpoint': b0['checkpoint'],
               'k2_checkpoint': k2['checkpoint']}
        for key in METRICS:
            row[f'b0_{key}'] = b0['metrics'][key]
            row[f'k2_{key}'] = k2['metrics'][key]
            row[f'k2_minus_b0_{key}'] = round(k2['metrics'][key] - b0['metrics'][key], 4)
        rows.append(row)
    aggregate = {}
    for key in METRICS:
        differences = [row[f'k2_minus_b0_{key}'] for row in rows]
        aggregate[key] = {
            'mean_k2_minus_b0': statistics.mean(differences),
            'median_k2_minus_b0': statistics.median(differences),
            'positive_pairs': sum(value > 0 for value in differences),
            'per_seed_k2_minus_b0': dict(zip(plan['train_seeds_for_all_variants'], differences)),
        }
    output_dir = EXP / 'results'
    output_dir.mkdir(exist_ok=True)
    with (output_dir / 'screening_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'delta_definition': 'K2-D2 minus B0-D2',
        'train_seeds': plan['train_seeds_for_all_variants'],
        'validation_images': 917,
        'validation_split_sha256': frozen['image_ids_sha256'],
        'validation_mask_manifest_sha256': frozen['validation_mask_manifest_sha256'],
        'metric_code_sha256': frozen['metric_code_sha256'],
        'metrics': aggregate,
        'limitation': 'Three training seeds support screening, not a precise population confidence interval; checkpoints were selected on the same validation split.',
    }
    (output_dir / 'screening_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: aggregate[key] for key in ('F1s', 'F1_1')}, indent=2))


if __name__ == '__main__':
    main()
