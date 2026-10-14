# Code documentation

- [Method-to-code mapping](METHOD_MAPPING.md) and
  [implementation consistency matrix](implementation_consistency_matrix.md).
- [Paper method and comparison settings](paper_comparison_settings.md):
  Selected/OAT/controls, input construction, comparison interfaces and
  classification-table locations.
- [Table presentation](table_presentation.md): main classification panels,
  complementary appendix fields, factorial controls and aggregation views.
- [Metric pipeline](metrics_pipeline.md): evaluation schemas and protocols.
- [Training pipeline](training_pipeline.md),
  [inference pipeline](inference_pipeline.md) and
  [reproducibility guide](reproducibility_guide.md).

Run `scripts/run_required_tests.sh` for the configured synthetic regression
suite. Its current execution log is written to `outputs/test_logs/` (or the
explicit `RL_LHM_TEST_LOG_DIR`); test counts depend on the selected suite and
installed dependencies. No historical test log is bundled in this directory.
