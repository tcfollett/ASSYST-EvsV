from pathlib import Path

import numpy as np
import pandas as pd
from ase.db import connect

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "TrainingData" / "ase.db"
PICKLE_PATH = ROOT / "TrainingData" / "data.pckl.gz"
OUTPUT_PATH = ROOT / "reports" / "large_volume_structures.txt"


def format_structure_report(row, exported) -> str:
    atoms = row.toatoms()
    volume_per_atom = row.volume / row.natoms
    energy_per_atom = row.energy / row.natoms

    return (
        f"id: {row.id}\n"
        f"name: {exported.name}\n"
        f"formula: {atoms.get_chemical_formula()}\n"
        f"volume/atom: {volume_per_atom}\n"
        f"energy/atom: {energy_per_atom}\n"
        f"cell: {atoms.cell}\n"
        f"positions: {atoms.positions}\n"
        f"forces: {row.forces}\n"
        f"stress: {exported.stress}\n"
        f"metadata: {row.key_value_pairs}\n\n"
    )


def main() -> None:
    db = connect(str(DB_PATH))
    df = pd.read_pickle(PICKLE_PATH, compression="gzip")
    reports = []

    if len(df) != db.count():
        raise ValueError(
            f"Row count mismatch between {DB_PATH} ({db.count()}) and "
            f"{PICKLE_PATH} ({len(df)})."
        )

    for row, exported in zip(db.select(), df.itertuples(index=False)):
        # The ASE database has NULL stress for every row, so we enrich the
        # selected structures with the aligned value from data.pckl.gz.
        if row.energy is not None and not np.isclose(row.energy, exported.energy):
            raise ValueError(
                f"Pickle row order does not match ASE DB at row id {row.id}."
            )

        if row.natoms != exported.number_of_atoms:
            raise ValueError(
                f"Atom-count mismatch between ASE DB and pickle at row id {row.id}."
            )

        if row.natoms and row.volume / row.natoms >= 200:
            reports.append(format_structure_report(row, exported))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("".join(reports), encoding="utf-8")
    print(f"Wrote {len(reports)} structures to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
