# ASSYST-EvsV

## Repo Layout

- `scripts/`: runnable Python scripts for reports and plots
- `TrainingData/`: source dataset files (`ase.db`, `data.pckl.gz`) plus dataset notes
- `reports/`: generated text exports such as large-volume and top-stress summaries
- `plots/`: generated figures
- `docs/`: PDFs and reference documents

## Common Commands

From the repo root:

```bash
source .venv/bin/activate
python scripts/data.py
python scripts/top_stress_structures.py
python scripts/plot_training_energy_volume.py
```
