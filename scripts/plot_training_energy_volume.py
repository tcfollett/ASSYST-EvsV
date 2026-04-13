#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault(
    "MPLCONFIGDIR", str((ROOT / ".matplotlib").resolve())
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from ase.db import connect
from plot_EvV_FvV_RAW_Data import plot_energy_volume_density


def load_structures(db_path: Path, limit: int | None = None):
    db = connect(str(db_path))
    structures = []

    for index, row in enumerate(db.select(), start=1):
        if limit is not None and index > limit:
            break
        structures.append(row.toatoms())

    return structures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot energy-vs-volume views for the training data."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / "TrainingData" / "ase.db",
        help="Path to the ASE database.",
    )
    parser.add_argument(
        "--output-energy-volume",
        type=Path,
        default=ROOT / "plots" / "energy_vs_volume.png",
        help="Where to save the total energy vs total volume figure.",
    )
    parser.add_argument(
        "--output-energy-volume-per-atom",
        type=Path,
        default=ROOT / "plots" / "energy_vs_volume_per_atom.png",
        help="Where to save the energy per atom vs volume per atom figure.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of structures to plot for a quicker preview.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    structures = load_structures(args.db, args.limit)

    if not structures:
        raise SystemExit("No structures were loaded from the training database.")

    args.output_energy_volume.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plot_energy_volume_density(structures)
    plt.xlabel(r"Volume ($\AA^3$)")
    plt.ylabel("Energy (eV)")
    plt.title(f"Training Data Energy vs Volume ({len(structures):,} structures)")
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(args.output_energy_volume, dpi=300)
    plt.close()

    args.output_energy_volume_per_atom.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plot_energy_volume_density(structures, per_atom=True)
    plt.xlabel(r"Atomic Volume ($\AA^3$/atom)")
    plt.ylabel("Atomic Energy (eV/atom)")
    plt.title(
        f"Training Data Energy vs Volume per Atom ({len(structures):,} structures)"
    )
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(args.output_energy_volume_per_atom, dpi=300)
    plt.close()

    print(f"Saved energy-vs-volume plot to {args.output_energy_volume}")
    print(
        "Saved energy-vs-volume-per-atom plot to "
        f"{args.output_energy_volume_per_atom}"
    )


if __name__ == "__main__":
    main()
