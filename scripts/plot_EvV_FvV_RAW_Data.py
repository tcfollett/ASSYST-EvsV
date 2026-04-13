from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"

os.environ.setdefault(
    "MPLCONFIGDIR", str((ROOT / ".matplotlib").resolve())
)

import matplotlib.pyplot as plt
from ase import Atoms
from matplotlib.colors import LogNorm  # Import LogNorm for the log scale

import matplotlib.colors as colors

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import seaborn as sns
except ImportError:
    sns = None


def get_energy_volume_data(structures, per_atom: bool = False):
    volumes = []
    energies = []

    for atoms in structures:
        natoms = len(atoms)
        if natoms == 0:
            continue

        volume = atoms.get_volume()
        energy = atoms.get_potential_energy()

        if per_atom:
            volume /= natoms
            energy /= natoms

        volumes.append(volume)
        energies.append(energy)

    return volumes, energies


def plot_energy_volume_density(
    structures,
    *,
    per_atom: bool = False,
    gridsize: int = 110,
    cmap: str = "viridis",
    colorbar_label: str = "# Structures",
):
    volumes, energies = get_energy_volume_data(structures, per_atom=per_atom)
    hexbin = plt.hexbin(
        volumes,
        energies,
        gridsize=gridsize,
        cmap=cmap,
        mincnt=1,
        norm=LogNorm(),
    )
    colorbar = plt.colorbar(hexbin)
    colorbar.set_label(colorbar_label)
    return hexbin


def process_alloy_data(file_pattern="FeC*_DFT.json"):
    if pd is None:
        raise ImportError("pandas is required to process raw alloy JSON data.")

    data_list = []
    
    # 1. Iterate through all matching JSON files
    for file_path in glob.glob(file_pattern):
        # Extract CALCTYPE from filename (e.g., prefix_PBE_DFT.json -> PBE)
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        # Standard format assumption: {prefix}_{CALCTYPE}_DFT.json
        calctype = parts[-2] if len(parts) >= 2 else "Unknown"

        with open(file_path, 'r') as f:
            data = json.load(f)

        for alloy_id, content in data.items():
            # Extract configuration data
            cell = content['cfg'].get('cell', [])
            pos = content['cfg'].get('pos', [])
            types = content['cfg'].get('types', [])
            
            # Use ASE to create an Atoms object for volume calculation
            # 'types' are interpreted as atomic numbers
            if not cell or not pos: continue
            atoms = Atoms(symbols=types, positions=pos, cell=cell, pbc=True)
            
            num_atoms = len(atoms)
            volume_total = atoms.get_volume()
            volume_per_atom = volume_total / num_atoms
            energy_per_atom = content['energy'] / num_atoms
            max_force = content['max_force_val']
            
            # Extract the shortest distance from the min_distances dictionary
            dist_vals = [v for v in content['min_distances'].values() if v is not None]
            min_dist = min(dist_vals) if dist_vals else np.nan
            
            data_list.append({
                "Alloy": alloy_id,
                "CALCTYPE": calctype,
                "Volume_total": volume_total,
                "Energy_per_atom": energy_per_atom,
                "Volume_per_atom": volume_per_atom,
                "Max_Force": max_force,
                "Min_Distance": min_dist
            })

    return pd.DataFrame(data_list)

def main():
    if pd is None or sns is None:
        raise ImportError(
            "pandas and seaborn are required to run plot_EvV_FvV_RAW_Data.py."
        )

    df = process_alloy_data()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.2)
    plot_configs = [
        ("Volume_per_atom", "Energy_per_atom", r"Energy per Atom vs. Volume",
         r"Volume per Atom ($\AA^3$)", r"Energy per Atom (eV)", "energy_vs_volume.png"),
        ("Volume_per_atom", "Max_Force", r"Max Force vs. Volume",
         r"Volume per Atom ($\AA^3$)", r"Max Force Val (eV/$\AA$)", "force_vs_volume.png"),
        ("Min_Distance", "Energy_per_atom", r"Energy per Atom vs. Min Distance",
         r"Minimum Atomic Distance ($\AA$)", r"Energy per Atom (eV)", "energy_vs_min_dist.png"),
    ]

    for x, y, title, xlabel, ylabel, fname in plot_configs:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df, x=x, y=y, hue="CALCTYPE", style="CALCTYPE", s=100, alpha=0.7
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / fname, dpi=300)
        plt.close()

    df["Energy_Bin"] = pd.cut(df["Energy_per_atom"], bins=500)
    df["Structure_Count"] = df.groupby(
        "Energy_Bin", observed=True
    )["Energy_per_atom"].transform("count")

    sns.set_theme(style="whitegrid", font_scale=1.2)

    plt.figure(figsize=(10, 6))
    norm = colors.LogNorm(
        vmin=df["Structure_Count"].min(), vmax=df["Structure_Count"].max()
    )
    sc = plt.scatter(
        data=df,
        x="Volume_per_atom",
        y="Energy_per_atom",
        c="Structure_Count",
        cmap="viridis",
        norm=norm,
        s=100,
        alpha=0.8,
        edgecolor="w",
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Number of Structures (Log Scale)", rotation=270, labelpad=20)
    plt.title("Energy/Atom vs. Volume")
    plt.xlabel(r"Volume per Atom ($\AA^3$)")
    plt.ylabel(r"Energy per Atom (eV)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "z_energy_volume_per_atom_colorbar.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    norm = colors.LogNorm(
        vmin=df["Structure_Count"].min(), vmax=df["Structure_Count"].max()
    )
    sc = plt.scatter(
        data=df,
        x="Volume_total",
        y="Energy_per_atom",
        c="Structure_Count",
        cmap="viridis",
        norm=norm,
        s=100,
        alpha=0.8,
        edgecolor="w",
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Number of Structures (Log Scale)", rotation=270, labelpad=20)
    plt.title("Energy/Atom vs. Volume")
    plt.xlabel(r"Volume($\AA^3$)")
    plt.ylabel(r"Energy per Atom (eV)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "z_energy_volume_total_colorbar.png", dpi=300)
    plt.close()

    df.to_csv(PLOTS_DIR / "z_properties_summary.csv", index=False)


if __name__ == "__main__":
    main()
