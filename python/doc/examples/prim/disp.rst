Atomic displacement DoF
=======================

Simple cubic with atomic displacement DoF
-----------------------------------------

This example constructs the prim for simple cubic crystal system with atomic displacement degrees of freedom (DoF).

To construct this prim, the following must be specified:

- the lattice vectors
- basis site coordinates
- occupant DoF
- atomic displacement DoF

This example uses a fixed "A" sublattice for occ_dof, which by default are created as isotropic atoms.

.. code-block:: Python

    import numpy as np
    import libcasm.xtal as xtal

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [1., 0., 0.],  # a
        [0., 1., 0.],  # a
        [0., 0., 1.],  # a
    ]).transpose()  # <--- note transpose
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array([
        [0., 0., 0.],  # coordinates of basis site, b=0
    ]).transpose()

    # Occupation degrees of freedom (DoF)
    occ_dof = [
        ["A"],  # occupants allowed on basis site, b=0
    ]

    # Local continuous degrees of freedom (DoF)
    disp_dof = xtal.DoFSetBasis("disp")  # Atomic displacement
    local_dof = [
        [disp_dof],  # allow displacements on basis site b=0
    ]

    # Construct the prim
    prim = xtal.Prim(lattice=lattice,
                     coordinate_frac=coordinate_frac,
                     occ_dof=occ_dof,
                     local_dof=local_dof,
                     title="simple_cubic_disp")


This prim as JSON: :download:`simple_cubic_disp.json <json/simple_cubic_disp.json>`
