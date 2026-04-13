# Training Data

## Pandas
You can read the whole training set into a pandas dataframe with

```python
import pandas as pd
df = pd.read_pickle("data.pckl.gz", compression='gzip')
```

Data was exported with Python 3.10 and pandas version 2.2.2.

## ASE

```python
import ase.db
db = ase.db.connect("ase.db")
db.select('Al==1')
```

Data was export with Python 3.10 and ASE 3.22.1

## Energy vs Volume Plots

You can generate both the total `energy vs volume` plot and the `energy per atom vs volume per atom` plot for the full ASE database with

```bash
../.venv/bin/python ../scripts/plot_training_energy_volume.py
```

This writes `energy_vs_volume.png` and `energy_vs_volume_per_atom.png` in the repo's `plots` directory by default.

## Finding Large-Volume-Per-Atom Structures

There are 63 structures in `ase.db` with `volume / natoms >= 200`.

To find them quickly with SQLite:

```bash
sqlite3 ase.db '
select id, natoms, volume, energy,
       volume * 1.0 / natoms as volume_per_atom,
       energy * 1.0 / natoms as energy_per_atom
from systems
where volume * 1.0 / natoms >= 200
order by volume_per_atom desc;
'
```

To inspect the full structure information with ASE:

```python
from ase.db import connect

db = connect("ase.db")

for row in db.select():
    if row.natoms and row.volume / row.natoms >= 200:
        atoms = row.toatoms()
        print("id:", row.id)
        print("formula:", atoms.get_chemical_formula())
        print("volume/atom:", row.volume / row.natoms)
        print("energy/atom:", row.energy / row.natoms)
        print("cell:", atoms.cell)
        print("positions:", atoms.positions)
        print("forces:", row.forces)
        print("metadata:", row.key_value_pairs)
        print()
```
