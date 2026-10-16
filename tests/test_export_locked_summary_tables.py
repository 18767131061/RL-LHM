"""Synthetic CPU checks for domain-safe locked summaries and TeX presentation."""
import copy
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from evaluation.export_locked_summary_tables import (
    CLASSIFICATION_METRICS, MODEL_GROUPS, metric_cell, parse_locked_tex_summary,
    range_cell, render_tables, validate_paired_family, classification_note,
)

SEEDS = [42, 2026, 3407, 1234, 5678]


def fixtures():
    return [dict(record_type='paired_difference', cohort=cohort,
                 model='Full RL-LHM', comparator=comparator, metric=metric,
                 seed=str(seed), bh_q='0.2000', raw_p='0.0200',
                 estimate='-0.0100', ci_lower='-0.1000', ci_upper='0.0800')
            for cohort in ('private', 'public')
            for comparator in ('SFT', 'vanilla GRPO')
            for metric in ('accuracy', 'macro_recall', 'macro_f1')
            for seed in SEEDS]


def source_fixture():
    rows = [dict(model=model, metrics={m: dict(mean=f'0.{5000 + i:04d}', sd='0.0100')
                                      for m in CLASSIFICATION_METRICS})
            for i, model in enumerate(model for _, models in MODEL_GROUPS for model in models)]
    return dict(training_seeds=SEEDS, tables={
        'classification_private': dict(metrics=list(CLASSIFICATION_METRICS), rows=rows),
        'perclass_private': dict(metrics=['precision', 'recall', 'f1'], rows=[
            dict(model='SFT', **{'class': 'HCC'}, test_patients=91,
                 metrics={m: dict(mean='0.9106', sd='0.0742', seed_range=['0.8137', '1.0000'])
                          for m in ('precision', 'recall', 'f1')})])})


class LockedTableTests(unittest.TestCase):
    def test_locked_precision_and_explicit_signed_difference(self):
        self.assertEqual(metric_cell(dict(mean='0.6000', sd='0.0100')), r'0.6000$\,\pm\,$0.0100')
        signed = dict(mean='-0.0100', sd='0.0300', seed_range=['-0.1000', '0.0800'])
        self.assertIn('-0.0100', metric_cell(signed, True, domain='signed_difference'))
        with self.assertRaises(ValueError):
            metric_cell(signed)
        self.assertEqual(signed['seed_range'], ['-0.1000', '0.0800'])

    def test_tex_parser_separates_separator_from_sign(self):
        metric = parse_locked_tex_summary(r'0.9106$\pm$0.0742 [0.8137--1.0000]')
        self.assertEqual(metric['seed_range'], ['0.8137', '1.0000'])
        self.assertEqual(metric['mean'], '0.9106')
        negative = parse_locked_tex_summary(r'-0.0500$\pm$0.0300 [-0.1000---0.0200]',
                                             domain='signed_difference')
        self.assertEqual(negative['seed_range'], ['-0.1000', '-0.0200'])
        self.assertNotIn('---', range_cell(negative, domain='signed_difference'))
        with self.assertRaises(ValueError):
            parse_locked_tex_summary(r'0.9106$\pm$0.0742 [0.8137---1.0000]')

    def test_performance_boundaries_and_equal_range(self):
        for value in ('0.0000', '1.0000'):
            with self.subTest(value=value):
                metric_cell(dict(mean=value, sd='0.0000', seed_range=[value, value]), True)
        metric_cell(dict(mean='0.5000', sd='0.5000', seed_range=['0.0000', '1.0000']), True)

    def test_invalid_bounds_negative_sd_and_reversed_order_rejected(self):
        for low, high in (('-0.0001', '1.0000'), ('0.0000', '1.0001'),
                          ('0.9000', '0.8000'), ('0.8137', '-1.0000')):
            with self.subTest(bounds=(low, high)), self.assertRaises(ValueError):
                metric_cell(dict(mean='0.8500', sd='0.0100', seed_range=[low, high]))
        for value in ('NaN', 'Infinity', '0.1e2', 0.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                metric_cell(dict(mean=value, sd='0.0100'))
        with self.assertRaises(ValueError):
            metric_cell(dict(mean='0.5000', sd='-0.0100'))
        with self.assertRaises(ValueError):
            metric_cell(dict(mean='0.9000', sd='0.0100', seed_range=['0.5000', '0.8000']))

    def test_q_threshold_and_no_input_mutation(self):
        records = fixtures()
        records[0]['bh_q'], records[1]['bh_q'] = '0.0499', '0.0500'
        original = copy.deepcopy(records)
        self.assertEqual(validate_paired_family(records, SEEDS), {'tests': 60, 'below_threshold': 1})
        self.assertEqual(records, original)
        self.assertEqual(records[0]['estimate'], '-0.0100')

    def test_missing_duplicate_wrong_candidate_and_invalid_q_rejected(self):
        for change in ('missing', 'duplicate', 'wrong_candidate', 'invalid_q', 'nan_q'):
            records = fixtures()
            if change == 'missing': records.pop(0)
            if change == 'duplicate': records[0]['seed'] = records[1]['seed']
            if change == 'wrong_candidate': records[0]['model'] = 'SFT'
            if change == 'invalid_q': records[0]['bh_q'] = '1.0001'
            if change == 'nan_q': records[0]['bh_q'] = 'NaN'
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_paired_family(records, SEEDS)

    def test_group_order_four_metrics_and_no_significance_column(self):
        source, records = source_fixture(), fixtures()
        source['tables']['classification_private']['rows'].reverse()
        before, records_before = copy.deepcopy(source), copy.deepcopy(records)
        output = render_tables(source, records)
        full, response = (output[n] for n in ('classification_private.tex', 'classification_private_response.tex'))
        self.assertEqual(len([line for line in full.splitlines() if ' & ' in line]), 10)
        self.assertEqual(len([line for line in response.splitlines() if ' & ' in line]), 3)
        for line in full.splitlines():
            if ' & ' in line: self.assertEqual(line.count(' & '), 4)
        expected = [group for group, _ in MODEL_GROUPS]
        self.assertEqual(sorted(expected, key=full.index), expected)
        self.assertLess(full.index('SFT &'), full.index('vanilla GRPO &'))
        self.assertLess(full.index('vanilla GRPO &'), full.index('RL-LHM (ours)'))
        self.assertIn('Faster R-CNN (R50-FPN)', full)
        self.assertNotIn('LLaVA', response)
        for absent in ('0/5', '---', 'n.s.', 'Significant seeds', '0.0200'):
            self.assertNotIn(absent, full)
        self.assertEqual(source, before)
        self.assertEqual(records, records_before)

    def test_note_uses_only_correct_cohort_comparator_and_metric(self):
        records = fixtures()
        for row in records:
            if row['cohort'] == 'private' and row['comparator'] == 'SFT' and row['metric'] == 'macro_f1':
                row['bh_q'] = {42: '0.1000', 2026: '0.2000', 3407: '0.3000', 1234: '0.4000', 5678: '0.5000'}[int(row['seed'])]
            elif row['cohort'] == 'public':
                row['bh_q'] = '0.9000'
        before = copy.deepcopy(records)
        note = classification_note(records, SEEDS, 'private')
        self.assertIn('0.1000--0.5000 versus SFT', note)
        self.assertIn('0.2000--0.2000 versus vanilla GRPO', note)
        self.assertNotIn('0.9000', note)
        self.assertEqual(records, before)
        zh = classification_note(records, SEEDS, 'private', language='zh')
        self.assertIn('0.1000--0.5000', zh)
        for forbidden in ('0/5', 'n.s.', 'Significant seeds'):
            self.assertNotIn(forbidden, note)

    def test_note_refuses_threshold_or_identity_mismatch(self):
        for q in ('0.0499', '0.0500'):
            records = fixtures()
            records[0]['bh_q'] = q
            with self.assertRaises(ValueError):
                classification_note(records, SEEDS, 'private')
        records = fixtures()
        records[0]['seed'] = records[1]['seed']
        with self.assertRaises(ValueError):
            classification_note(records, SEEDS, 'private')
        with self.assertRaises(ValueError):
            classification_note(fixtures(), SEEDS, 'unknown')

    def test_missing_extra_or_duplicate_model_not_silently_dropped(self):
        for change in ('missing', 'extra', 'duplicate'):
            source = source_fixture()
            rows = source['tables']['classification_private']['rows']
            if change == 'missing': rows.pop()
            if change == 'extra': rows.append(dict(model='Unspecified model', metrics=rows[0]['metrics']))
            if change == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                render_tables(source, fixtures())

    def test_class_mean_sd_and_counts_with_ranges_preserved_separately(self):
        source = source_fixture()
        outputs = render_tables(source, fixtures())
        main = outputs['perclass_private.tex']
        supplement = outputs['perclass_seed_ranges.tex']
        self.assertIn('HCC & 91 & 0.9106', main)
        self.assertNotIn('0.8137', main)
        self.assertIn(r'$[0.8137\,\text{--}\,1.0000]$', supplement)
        source['tables']['perclass_private']['rows'][0]['metrics']['f1']['seed_range'][1] = '-1.0000'
        with self.assertRaises(ValueError):
            render_tables(source, fixtures())  # Invalid hidden ranges cannot pass.

    @unittest.skipUnless(shutil.which('pdflatex'), 'TeX engine unavailable')
    def test_tex_rendering_of_grouped_tables_and_signed_ranges(self):
        outputs = render_tables(source_fixture(), fixtures())
        signed = metric_cell(dict(mean='-0.0500', sd='0.0300',
                                  seed_range=['-0.1000', '-0.0200']),
                             True, domain='signed_difference')
        tex = (r'\documentclass{article}\usepackage[margin=15mm]{geometry}'
               r'\usepackage{booktabs,amsmath,array}\begin{document}\small'
               r'\begin{tabular}{@{}p{5.2cm}cccc@{}}\toprule '
               r'Model & Accuracy & Precision & Recall & F1 \\\midrule ' +
               outputs['classification_private.tex'] + r'\bottomrule\end{tabular}\par\bigskip ' +
               r'\begin{tabular}{l}'+signed+r'\\\end{tabular}\end{document}')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'render.tex').write_text(tex)
            run = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                                  'render.tex'], cwd=path, capture_output=True, text=True, timeout=45)
            self.assertEqual(run.returncode, 0, run.stdout[-4000:])
            self.assertGreater((path/'render.pdf').stat().st_size, 1000)
            log = (path/'render.log').read_text(errors='replace')
            self.assertNotIn('Overfull \\hbox', log)
            self.assertNotIn('Extra alignment tab', log)


if __name__ == '__main__':
    unittest.main()
