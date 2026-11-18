# Week 2 — Schema audit and time validation

## Done
- `src/data/inspect_schema.py`: confirmed all 6 data CSVs have 87 columns
  (timestamp + 86 SCADA tags). Generated `config/tag_schema.yaml` (86 tags)
  and `reports/schema_audit.md`.
- `src/data/validate_time.py`: all train/test files are continuous 1 Hz
  (train1 280,800; train2 291,600; train3 126,000; train4 198,000;
  test1 54,000; test2 230,400 rows). `label-test1.csv` aligned;
  `label-test2.csv` uses minute-resolution repeated timestamps and is accepted
  under a documented special rule (row-aligned, max gap <= 60 s).

## Issues
- `label-test2.csv` has 3,840 unique timestamps vs 230,400 rows; handled as a
  known format characteristic, not a defect.
