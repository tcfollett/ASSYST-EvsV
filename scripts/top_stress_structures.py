from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "TrainingData" / "data.pckl.gz"
DEFAULT_OUTPUT_PATH = ROOT / "reports" / "top_100_max_stress_structures.txt"


def parse_args():
    parser = ArgumentParser(
        description="Find the structures with the largest stress components."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Input pickle file (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output text file (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of top-stress structures to write.",
    )
    return parser.parse_args()


def max_abs_stress(stress) -> float:
    array = np.asarray(stress, dtype=float)
    return float(np.max(np.abs(array)))


def format_entry(rank: int, row_index: int, row) -> str:
    atoms = row["atoms"]
    stress = np.asarray(row["stress"], dtype=float)
    return (
        f"rank: {rank}\n"
        f"row_index: {row_index}\n"
        f"name: {row['name']}\n"
        f"formula: {atoms.get_chemical_formula()}\n"
        f"number_of_atoms: {row['number_of_atoms']}\n"
        f"energy: {row['energy']}\n"
        f"max_abs_stress: {row['max_abs_stress']}\n"
        f"stress_norm: {row['stress_norm']}\n"
        f"stress: {stress}\n"
        f"cell: {atoms.cell}\n"
        f"positions: {atoms.positions}\n\n"
    )


def main() -> None:
    args = parse_args()
    df = pd.read_pickle(args.data, compression="gzip").copy()

    df["max_abs_stress"] = df["stress"].apply(max_abs_stress)
    df["stress_norm"] = df["stress"].apply(lambda x: float(np.linalg.norm(x)))

    top = df.nlargest(args.limit, ["max_abs_stress", "stress_norm"]).reset_index()

    report = "".join(
        format_entry(rank, int(row["index"]), row)
        for rank, (_, row) in enumerate(top.iterrows(), start=1)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(
        f"Wrote top {len(top)} structures by max absolute stress component to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
