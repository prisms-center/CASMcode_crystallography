import numpy as np

import libcasm.xtal as xtal

# Lattice vectors
lattice_column_vector_matrix = np.array(
    [
        [1.0, 0.0, 0.0],  # a
        [0.0, 1.0, 0.0],  # a
        [0.0, 0.0, 1.0],  # a
    ]
).transpose()  # <--- note transpose
lattice = xtal.Lattice(lattice_column_vector_matrix)

# Basis sites positions, as columns of a matrix,
# in fractional coordinates with respect to the lattice vectors
coordinate_frac = np.array(
    [
        [0.0, 0.0, 0.0],
    ]
).transpose()  # coordinates of basis site, b=0

# Occupation degrees of freedom (DoF)
A_up = xtal.Occupant("A", properties={"Cmagspin": np.array([1.0])})
A_down = xtal.Occupant("A", properties={"Cmagspin": np.array([-1.0])})
occupants = {
    "A.up": A_up,  # A atom, spin up
    "A.down": A_down,  # A atom, spin down
}
occ_dof = [
    ["A.up", "A.down"],
]

# Construct the prim
prim = xtal.Prim(
    lattice=lattice,
    coordinate_frac=coordinate_frac,
    occ_dof=occ_dof,
    occupants=occupants,
    title="simple_cubic_ising",
)

# Print the factor group
i = 1
factor_group = xtal.make_factor_group(prim)
for op in factor_group:
    syminfo = xtal.SymInfo(op, lattice)
    print(str(i) + ":", syminfo.brief_cart())
    i += 1

# Format as JSON
with open("../../doc/examples/prim/json/simple_cubic_ising.json", "w") as f:
    f.write(prim.to_json())
