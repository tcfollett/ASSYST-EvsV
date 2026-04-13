from ase.db import connect

db = connect("TrainingData/ase.db")

for row in db.select():
    if row.natoms and row.volume / row.natoms >= 200:
        atoms = row.toatoms()
        print(row.id, atoms.get_chemical_formula())
        print("cell:", atoms.cell)
        print("positions:", atoms.positions)
        print("forces:", row.forces)
        print("metadata:", row.key_value_pairs)
