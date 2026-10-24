#!/usr/bin/env python3
"""Export legacy grouped summaries while preserving supplied decimal strings.

This presentation-only tool validates existing paired records and metric domains;
it performs no evaluation, interval estimation, statistical test or BH correction.
Its 60-comparison family contains separate per-seed tests, not joint five-seed
inference. Supplied q-values determine the notes; no conclusion is prescribed.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from decimal import Decimal
from pathlib import Path

PAIRED_METRICS = ('accuracy', 'macro_recall', 'macro_f1')
CLASSIFICATION_METRICS = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')
MODEL_GROUPS = (
    ('Qwen3-VL-4B', ('SFT', 'vanilla GRPO', 'Full RL-LHM')),
    ('Auxiliary VLM baselines', ('LLaVA-OneVision-1.5-4B SFT',
                                'Qwen2.5-VL-3B SFT', 'InternVL3.5-4B SFT')),
    ('Task-specific detectors', ('YOLOv8-X', 'YOLOv11-X', 'RT-DETR-L',
                                 'Faster R-CNN (R50-FPN)')),
)
DECIMAL_PATTERN = r'-?\d+\.\d+'
GENERATED = '% Generated from locked summary strings; do not hand-edit.'


def decimal_string(value):
    if not isinstance(value, str) or not re.fullmatch(DECIMAL_PATTERN, value):
        raise ValueError('Locked summaries must use finite decimal strings')
    return Decimal(value)


def validate_metric(metric, *, domain='performance'):
    """Validate domains without repairing signs or recomputing any value.

    Explicit signed_difference permits negative differences of proportions.
    Every source table in this exporter contains performance metrics in [0,1].
    """
    if domain not in ('performance', 'signed_difference'):
        raise ValueError(f'Unknown metric domain: {domain}')
    lower = Decimal('0') if domain == 'performance' else Decimal('-1')
    mean, sd = decimal_string(metric['mean']), decimal_string(metric['sd'])
    if not lower <= mean <= 1:
        raise ValueError(f'{domain} mean outside [{lower},1]')
    if sd < 0:
        raise ValueError('Sample SD cannot be negative')
    if 'seed_range' in metric:
        bounds = metric['seed_range']
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError('A seed range requires exactly two bounds')
        low, high = map(decimal_string, bounds)
        if not lower <= low <= high <= 1:
            raise ValueError(f'{domain} seed range outside [{lower},1] or reversed')
        if not low <= mean <= high:
            raise ValueError('Mean must fall within its seed range')


def parse_locked_tex_summary(cell, *, domain='performance'):
    """Import an explicit TeX min--max separator without treating it as a sign.

    Read the bracketed range separately from mean/SD. A signed endpoint is
    accepted only in the explicit signed_difference domain; no absolute value
    or inferred sign correction is applied. Decimal strings are retained.
    """
    range_match = re.search(
        rf'\[\s*({DECIMAL_PATTERN})\s*--\s*({DECIMAL_PATTERN})\s*\]', cell)
    if ('[' in cell or ']' in cell) and not range_match:
        raise ValueError('Malformed TeX seed range')
    core = cell[:range_match.start()] if range_match else cell
    nums = re.findall(DECIMAL_PATTERN, core)
    if len(nums) != 2:
        raise ValueError('Expected one locked mean and sample SD')
    metric = dict(mean=nums[0], sd=nums[1], best=r'\best{' in cell)
    if range_match:
        metric['seed_range'] = list(range_match.groups())
    validate_metric(metric, domain=domain)
    return metric


def validate_paired_family(records, seeds):
    """Check the 60 existing cohort/seed/comparator/endpoint identities and q's."""
    if len(seeds) != 5 or len({str(seed) for seed in seeds}) != 5:
        raise ValueError('Expected five distinct training seeds')
    expected = {(cohort, comparator, metric, str(seed))
                for cohort in ('private', 'public')
                for comparator in ('SFT', 'vanilla GRPO')
                for metric in PAIRED_METRICS for seed in seeds}
    selected = [r for r in records if r['record_type'] == 'paired_difference']
    observed = [(r['cohort'], r['comparator'], r['metric'], r['seed']) for r in selected]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('Incomplete, duplicate or unexpected paired-family identity')
    if any(r['model'] != 'Full RL-LHM' for r in selected):
        raise ValueError('Paired comparison candidate must be Full RL-LHM')
    qs = [Decimal(r['bh_q']) for r in selected]
    if any(not q.is_finite() or not 0 <= q <= 1 for q in qs):
        raise ValueError('BH q must be finite and in [0,1]')
    return {'tests': len(qs), 'below_threshold': sum(q < Decimal('0.05') for q in qs)}


def tex_escape(value):
    return ''.join({'&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#',
                    '$': r'\$', '{': r'\{', '}': r'\}'}.get(c, c) for c in str(value))


def paired_group_summaries(records, seeds, cohort):
    """Describe each five-record group without changing the 60-test family.

    Keep exact supplied q-range endpoints so display rounding cannot obscure
    which side of 0.05 a value is on. Counts use the existing strict q<0.05 rule;
    equality is reported separately rather than silently grouped with either side.
    """
    validate_paired_family(records, seeds)
    if cohort not in ('private', 'public'):
        raise ValueError('Unknown paired-family cohort')
    cohort_rows = [r for r in records if r['record_type'] == 'paired_difference'
                   and r['cohort'] == cohort]
    groups = []
    for comparator in ('SFT', 'vanilla GRPO'):
        for metric in PAIRED_METRICS:
            rows = [r for r in cohort_rows if r['comparator'] == comparator and r['metric'] == metric]
            qs = [Decimal(r['bh_q']) for r in rows]
            groups.append({
                'cohort': cohort, 'comparator': comparator, 'metric': metric,
                'tests': len(rows),
                'below_threshold': sum(q < Decimal('0.05') for q in qs),
                'equal_threshold': sum(q == Decimal('0.05') for q in qs),
                'above_threshold': sum(q > Decimal('0.05') for q in qs),
                'q_min': min(rows, key=lambda row: Decimal(row['bh_q']))['bh_q'],
                'q_max': max(rows, key=lambda row: Decimal(row['bh_q']))['bh_q'],
            })
    return groups


def classification_note(records, seeds, cohort, *, language='en'):
    """Report existing per-seed tests; group ranges are not new joint tests."""
    groups = paired_group_summaries(records, seeds, cohort)
    labels = {'accuracy': 'Accuracy', 'macro_recall': 'macro-Recall', 'macro_f1': 'macro-F1'}
    if language == 'en':
        intro = ('Legacy per-seed paired comparisons of RL-LHM, using the supplied '
                 '60-comparison BH family; these are not joint five-seed tests. '
                 'For each comparator and endpoint, counts out of five are reported '
                 'at $q<0.05$, $q=0.05$ and $q>0.05$, followed by the existing $q$ range. ')
        details = [
            f'Versus {row["comparator"]}, {labels[row["metric"]]}: '
            f'{row["below_threshold"]}/5, {row["equal_threshold"]}/5, {row["above_threshold"]}/5; '
            f'$q$ {row["q_min"]}--{row["q_max"]}.' for row in groups]
        return intro + ' '.join(details) + '\n'
    if language == 'zh':
        intro = ('旧版逐seed配对比较沿用所给60项BH比较家族，不是五seed联合检验。'
                 '以下各比较方法和指标依次列出五个既有检验中$q<0.05$、$q=0.05$和$q>0.05$的数量，'
                 '以及既有$q$值范围。')
        details = [
            f'对{row["comparator"]}，{labels[row["metric"]]}：'
            f'{row["below_threshold"]}/5、{row["equal_threshold"]}/5、{row["above_threshold"]}/5；'
            f'$q$ {row["q_min"]}--{row["q_max"]}。' for row in groups]
        return intro + ''.join(details) + '\n'
    raise ValueError('Unsupported note language')


def model_cell(model):
    return r'\textbf{RL-LHM (ours)}' if model == 'Full RL-LHM' else tex_escape(model)


def range_cell(metric, *, domain='performance'):
    validate_metric(metric, domain=domain)
    low, high = metric['seed_range']
    # Explicit math separator cannot fuse with a signed endpoint into ---.
    return '$[' + low + r'\,\text{--}\,' + high + ']$'


def metric_cell(metric, include_range=False, *, domain='performance'):
    validate_metric(metric, domain=domain)
    mean, sd = metric['mean'], metric['sd']
    mean_tex = r'\textbf{' + mean + '}' if metric.get('best', False) else mean
    cell = mean_tex + r'$\,\pm\,$' + sd
    if include_range and 'seed_range' in metric:
        cell = r'\shortstack{' + cell + r'\\' + range_cell(metric, domain=domain) + '}'
    return cell


def classification_rows(table, *, primary_only=False):
    if tuple(table['metrics']) != CLASSIFICATION_METRICS:
        raise ValueError('Classification requires its four reported metrics in order')
    rows = {row['model']: row for row in table['rows']}
    expected = {model for _, models in MODEL_GROUPS for model in models}
    if len(rows) != len(table['rows']) or set(rows) != expected:
        raise ValueError('Classification requires all ten distinct named models')
    rendered = [GENERATED]
    groups = MODEL_GROUPS[:1] if primary_only else MODEL_GROUPS
    for i, (group, models) in enumerate(groups):
        if i:
            rendered.append(r'\addlinespace[3pt]')
        rendered.append(r'\multicolumn{5}{@{}l}{\textit{' + group + r'}} \\')
        for model in models:
            cells = [model_cell(model)]
            cells.extend(metric_cell(rows[model]['metrics'][m]) for m in table['metrics'])
            rendered.append(' & '.join(cells) + r' \\')
    return '\n'.join(rendered) + '\n'


def render_tables(source, records):
    seeds = source['training_seeds']
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('Expected five distinct training seeds')
    if any(name.startswith('classification_') for name in source['tables']):
        validate_paired_family(records, seeds)
    outputs = {}
    for name, table in source['tables'].items():
        if name.startswith('classification_'):
            outputs[name + '.tex'] = classification_rows(table)
            outputs[name + '_response.tex'] = classification_rows(table, primary_only=True)
            cohort = name.removeprefix('classification_')
            outputs[name + '_note.tex'] = classification_note(records, seeds, cohort)
            outputs[name + '_note_zh.tex'] = classification_note(records, seeds, cohort, language='zh')
            continue
        rendered = [GENERATED]
        for row in table['rows']:
            cells = [model_cell(row['model'])]
            if name.startswith('perclass_'):
                cells.extend([tex_escape(row['class']), str(row['test_patients'])])
            cells.extend(metric_cell(row['metrics'][m]) for m in table['metrics'])
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
    print(f'{len(outputs)} locked summary table fragments verified' if args.check
          else f'{len(outputs)} table fragments exported')


if __name__ == '__main__':
    main()
