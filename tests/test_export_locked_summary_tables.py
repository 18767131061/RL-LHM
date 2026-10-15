"""Synthetic checks for presentation grouping and preservation (no study data)."""
import copy
import unittest
from evaluation.export_locked_summary_tables import metric_cell, paired_counts, render_tables

SEEDS = [42, 2026, 3407, 1234, 5678]


def fixtures():
    records = []
    for cohort in ('private', 'public'):
        for comparator in ('SFT', 'vanilla GRPO'):
            for metric in ('accuracy', 'macro_recall', 'macro_f1'):
                for seed in SEEDS:
                    records.append(dict(record_type='paired_difference', cohort=cohort,
                        model='Full RL-LHM', comparator=comparator, metric=metric,
                        seed=str(seed), bh_q='0.2000', raw_p='0.0200',
                        estimate='-0.0100', ci_lower='-0.1000', ci_upper='0.0800'))
    return records


class LockedTableTests(unittest.TestCase):
    def test_preserves_locked_precision_and_negative_values(self):
        self.assertEqual(metric_cell(dict(mean='0.6000',sd='0.0100')),r'0.6000$\,\pm\,$0.0100')
        self.assertEqual(metric_cell(dict(mean='-0.0100',sd='0.0000')),r'-0.0100$\,\pm\,$0.0000')
        cell=metric_cell(dict(mean='0.6000',sd='0.0100',seed_range=['0.5900','0.6100']),True)
        self.assertIn(r'\\{[0.5900--0.6100]}',cell)

    def test_q_threshold_grouping_and_no_input_mutation(self):
        records=fixtures()
        records[0]['bh_q']='0.0499'
        records[1]['bh_q']='0.0500'
        original=copy.deepcopy(records)
        self.assertEqual(paired_counts(records,'private','SFT',SEEDS),'1/5, 0/5, 0/5')
        self.assertEqual(paired_counts(records,'public','SFT',SEEDS),'0/5, 0/5, 0/5')
        self.assertEqual(paired_counts(records,'private','vanilla GRPO',SEEDS),'0/5, 0/5, 0/5')
        self.assertEqual(records,original)

    def test_missing_duplicate_or_wrong_candidate_rejected(self):
        for change in ('missing','duplicate','wrong_candidate'):
            records=fixtures()
            if change=='missing':records.pop(0)
            if change=='duplicate':records[0]['seed']=records[1]['seed']
            if change=='wrong_candidate':records[0]['model']='SFT'
            with self.subTest(change=change),self.assertRaises(ValueError):
                paired_counts(records,'private','SFT',SEEDS)

    def test_untested_rows_use_dash_not_nonsignificance(self):
        rows=[dict(model=name,metrics={'accuracy':dict(mean='0.5000',sd='0.0100')})
              for name in ('SFT','vanilla GRPO','Full RL-LHM','Auxiliary')]
        source=dict(training_seeds=SEEDS,tables={'classification_private':dict(metrics=['accuracy'],rows=rows)})
        text=render_tables(source,fixtures())['classification_private.tex']
        self.assertEqual(text.count('0/5, 0/5, 0/5'),2)
        self.assertEqual(text.count(' & ---'),2)
        self.assertNotIn('n.s.',text)
        self.assertNotIn('0.0200',text)  # raw p never reinterpreted as summary

if __name__=='__main__':
    unittest.main()
