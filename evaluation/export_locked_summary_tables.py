#!/usr/bin/env python3
"""Export locked mean/sample-SD strings and count existing per-seed BH decisions.

This presentation-only tool never evaluates a model, estimates an interval,
computes a test, reapplies BH, or averages p-values, q-values or CI endpoints.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from decimal import Decimal
from pathlib import Path

PAIRED_METRICS = ('accuracy', 'macro_recall', 'macro_f1')


def paired_counts(records, cohort, comparator, seeds):
    """Count q < .05 within each endpoint, preserving all five seed identities."""
    expected = {str(seed) for seed in seeds}
    counts = []
    for metric in PAIRED_METRICS:
        selected = [row for row in records if row['record_type'] == 'paired_difference'
                    and row['model'] == 'Full RL-LHM'
                    and row['cohort'] == cohort and row['comparator'] == comparator
                    and row['metric'] == metric]
        observed = [row['seed'] for row in selected]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError(f'Incomplete or duplicate paired seeds: {cohort}/{comparator}/{metric}')
        qs = [Decimal(row['bh_q']) for row in selected]
        if any(not q.is_finite() or not 0 <= q <= 1 for q in qs):
            raise ValueError('BH q must be finite and in [0,1]')
        counts.append(f'{sum(q < Decimal("0.05") for q in qs)}/{len(expected)}')
    return ', '.join(counts)


def tex_escape(value):
    return ''.join({'&':r'\&', '%':r'\%', '_':r'\_', '#':r'\#',
                    '$':r'\$', '{':r'\{', '}':r'\}'}.get(c,c) for c in str(value))


def metric_cell(metric, include_range=False):
    # Preserve decimal strings exactly; formatting cannot replace a locked value.
    mean, sd = metric['mean'], metric['sd']
    for value in (mean, sd):
        if not isinstance(value, str) or not re.fullmatch(r'-?\d+\.\d+', value):
            raise ValueError('Locked summaries must use decimal strings')
    if Decimal(sd) < 0:
        raise ValueError('Sample SD cannot be negative')
    mean_tex = r'\textbf{' + mean + '}' if metric.get('best', False) else mean
    cell = mean_tex + r'$\,\pm\,$' + sd
    if include_range and 'seed_range' in metric:
        low, high = metric['seed_range']
        cell = r'\shortstack{' + cell + r'\\{[' + low + '--' + high + ']}}'
    return cell


def render_tables(source, records):
    seeds = source['training_seeds']
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('Expected five distinct training seeds')
    outputs = {}
    for name, table in source['tables'].items():
        rendered = ['% Generated from locked summary strings; do not hand-edit.']
        for row in table['rows']:
            cells = [tex_escape(row['model'])]
            if name.startswith('perclass_'):
                cells.extend([tex_escape(row['class']), str(row['test_patients'])])
            cells.extend(metric_cell(row['metrics'][m], name.startswith('perclass_'))
                         for m in table['metrics'])
            if name.startswith('classification_'):
                cohort = name.removeprefix('classification_')
                cells.append(paired_counts(records,cohort,row['model'],seeds)
                             if row['model'] in ('SFT','vanilla GRPO') else '---')
            rendered.append(' & '.join(cells) + r' \\')
        outputs[name + '.tex'] = '\n'.join(rendered) + '\n'
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Verify generated files without writing')
    args = parser.parse_args()
    source = json.loads(args.summary.read_text(encoding='utf-8'))
    with args.supplement.open(encoding='utf-8', newline='') as handle:
        records = list(csv.DictReader(handle))
    outputs = render_tables(source, records)
    if args.check:
        for name, content in outputs.items():
            if (args.output_dir / name).read_text(encoding='utf-8') != content:
                raise SystemExit(f'Generated table is stale: {name}')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            (args.output_dir / name).write_text(content, encoding='utf-8')
    print(f'{len(outputs)} locked summary table fragments verified' if args.check else f'{len(outputs)} table fragments exported')

if __name__ == '__main__':
    main()
