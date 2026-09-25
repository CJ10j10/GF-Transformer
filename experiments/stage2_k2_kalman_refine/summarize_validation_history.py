#!/usr/bin/env python3
"""Extract compact validation metrics from a completed tqdm training log."""

import argparse
import csv
import re
from pathlib import Path

METRICS = ('F1b', 'F1d', 'F1s', 'F1_0', 'F1_1', 'F1_2', 'F1_3')
FIELDS = ('epoch_0based', 'checkpoint_epoch_1based', *METRICS,
          'best_score', 'improved')
VAL = re.compile(
    r'^Val Score: (?P<F1s>\d+\.\d+), Dice: (?P<F1b>\d+\.\d+), '
    r'F1: (?P<F1d>\d+\.\d+), F1_0:(?P<F1_0>\d+\.\d+) '
    r'F1_1:(?P<F1_1>\d+\.\d+) F1_2:(?P<F1_2>\d+\.\d+) '
    r'F1_3:(?P<F1_3>\d+\.\d+)$')
SCORE = re.compile(r'^score: (?P<score>\d+\.\d+)\s+score_best: (?P<best>\d+\.\d+)$')
EPOCH = re.compile(r'^epoch: (?P<epoch>\d+);.*micro_batches (?P<batches>\d+); '
                   r'optimizer_updates (?P<updates>\d+);')


def extract(log_path):
    current_epoch = None
    pending = None
    rows = []
    text = Path(log_path).read_text(errors='replace').replace('\r', '\n')
    for line in text.splitlines():
        epoch = EPOCH.match(line)
        if epoch:
            if epoch.group('batches') != epoch.group('updates'):
                raise RuntimeError(f'Update mismatch at epoch {epoch.group("epoch")}')
            current_epoch = int(epoch.group('epoch'))
            continue
        val = VAL.fullmatch(line)
        if val:
            if pending is not None or current_epoch is None:
                raise RuntimeError('Validation line has no unique completed epoch')
            pending = {'epoch_0based': current_epoch,
                       'checkpoint_epoch_1based': current_epoch + 1,
                       **{key: float(value) for key, value in val.groupdict().items()}}
            continue
        score = SCORE.fullmatch(line)
        if score:
            if pending is None or abs(float(score.group('score')) - pending['F1s']) > 0.00005:
                raise RuntimeError('score line does not match validation metrics')
            best = float(score.group('best'))
            if best + 0.00005 < pending['F1s']:
                raise RuntimeError('best_score is below validation score')
            old_best = float(rows[-1]['best_score']) if rows else 0.0
            if best + 0.00005 < old_best:
                raise RuntimeError('best_score decreased')
            pending['best_score'] = best
            pending['improved'] = int(best > old_best + 0.00005)
            rows.append(pending)
            pending = None
    if pending:
        raise RuntimeError('Last validation has no score line')
    if not rows:
        raise RuntimeError('No validation rows found')
    if any(row['epoch_0based'] != 2 * i for i, row in enumerate(rows)):
        raise RuntimeError('Validation epoch sequence differs from expected even epochs')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-rows', type=int, default=25)
    args = parser.parse_args()
    rows = extract(args.log)
    if len(rows) != args.expected_rows:
        raise RuntimeError(f'Expected {args.expected_rows} rows, got {len(rows)}')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} validation rows to {output}; '
          f'best={max(row["F1s"] for row in rows):.4f}')


if __name__ == '__main__':
    main()
