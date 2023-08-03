import numpy as np

import libcasm.xtal as xtal

# Lattice vectors
lattice_column_vector_matrix = np.array(
    [
        [3.233986860000, 0.000000000000, 0.000000000000],  # a, along x
        [-1.616993430000, 2.800714770000, 0.000000000000],  # a
        [0.000000000000, 0.000000000000, 5.168678340000],  # c, along z
    ]
).transpose()  # <--- note transpose
lattice = xtal.Lattice(lattice_column_vector_matrix)

# Basis sites positions, as columns of a matrix,
# in fractional coordinates with respect to the lattice vectors
coordinate_frac = np.array(
    [
        [0.0, 0.0, 0.0],  # coordinates of basis site, b=0
        [2.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0],  # coordinates of basis site, b=1
        [1.0 / 3.0, 2.0 / 3.0, 1.0 / 4.0],  # coordinates of basis site, b=2
        [1.0 / 3.0, 2.0 / 3.0, 3.0 / 4.0],  # coordinates of basis site, b=3
    ]
).transpose()

# Occupation degrees of freedom (DoF)
occ_dof = [
    ["Zr"],  # no variation allowed on basis site, b=0
    ["Zr"],  # no variation allowed on basis site, b=1
    ["Va", "O"],  # occupants allowed on basis site, b=2
    ["Va", "O"],  # occupants allowed on basis site, b=3
]

# Construct the prim
prim = xtal.Prim(
    lattice=lattice, coordinate_frac=coordinate_frac, occ_dof=occ_dof, title="ZrO"
)

# Print the factor group
i = 1
factor_group = xtal.make_factor_group(prim)
for op in factor_group:
    syminfo = xtal.SymInfo(op, lattice)
    print(str(i) + ":", syminfo.brief_cart())
    i += 1

# Format as JSON
with open("../../doc/examples/prim/json/ZrO_prim.json", "w") as f:
    f.write(prim.to_json())
