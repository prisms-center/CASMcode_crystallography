Coupled DoF
===========

Coupled occupation, displacement, and strain DoF
------------------------------------------------


This example constructs the prim for simple cubic crystal system with binary occupation degrees of freedom (DoF), atomic displacement DoF, and strain DoF, using the Green-Lagrange strain metric, and the symmetry-adapted strain basis.

To construct this prim, the following must be specified:

- the lattice vectors
- basis site coordinates
- occupant DoF
- atomic displacement DoF
- strain DoF


.. code-block:: Python

    import numpy as np
    import libcasm.xtal as xtal
    from math import sqrt

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
        [0., 0., 0.],
    ]).transpose()  # coordinates of basis site, b=0

    # Occupation degrees of freedom (DoF)
    occ_dof = [
        ["A", "B"],  # occupants allowed on basis site, b=0
    ]

    # Local continuous degrees of freedom (DoF)
    disp_dof = xtal.DoFSetBasis("disp")  # Atomic displacement
    local_dof = [
        [disp_dof],  # basis site, b=0
    ]

    # Global continuous degrees of freedom (DoF)
    GLstrain_dof = xtal.DoFSetBasis(
        dofname="GLstrain",
        axis_names=["e_{1}", "e_{2}", "e_{3}", "e_{4}", "e_{5}", "e_{6}"],
        basis=np.array([
            [1. / sqrt(3), 1. / sqrt(3), 1. / sqrt(3), 0.0, 0.0, 0.0],
            [1. / sqrt(2), -1. / sqrt(2), 0.0, 0.0, 0.0, 0.0],
            [-1. / sqrt(6), -1. / sqrt(6), 2. / sqrt(6), 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]).transpose())
    global_dof = [GLstrain_dof]

    # Construct the prim
    prim = xtal.Prim(lattice=lattice,
                     coordinate_frac=coordinate_frac,
                     occ_dof=occ_dof,
                     local_dof=local_dof,
                     global_dof=global_dof,
                     title="simple_cubic_coupled")

This prim as JSON: :download:`simple_cubic_coupled.json <json/simple_cubic_coupled.json>`
