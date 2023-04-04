
Occupation DoF
==============


Simple cubic binary
-------------------

This example constructs the prim for simple cubic crystal system with binary ("A", "B") occupation degrees of freedom (DoF).

To construct this prim, the following must be specified:

- the lattice vectors
- basis site coordinates
- occupant DoF

This example uses "A" and "B" for occ_dof, which by default are created as isotropic atoms.

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
        ["A", "B"],  # occupants allowed on basis site, b=0
    ]

    # Construct the prim
    prim = xtal.Prim(lattice=lattice,
                     coordinate_frac=coordinate_frac,
                     occ_dof=occ_dof,
                     title="simple_cubic_binary")


This prim as JSON: :download:`simple_cubic_binary.json <json/simple_cubic_binary.json>`


ZrO, binary with vacancies
--------------------------

This example constructs the prim for the Zr-O system, where hcp Zr is a fixed lattice, and O-vacancy disorder is allowed on the octahedral interstitial sites.

To construct this prim, the following must be specified:

- the lattice vectors
- basis site coordinates
- occupant DoF

This example uses "Va", the reserved name for vacancies, in occ_dof.

.. code-block:: Python

    import numpy as np
    import libcasm.xtal as xtal

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [3.233986860000, 0.000000000000, 0.000000000000],   # a, along x
        [-1.616993430000, 2.80071477000, 0.000000000000],   # a
        [0.000000000000, 0.000000000000, 5.168678340000],   # c, along z
    ]).transpose()  # <--- note transpose
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array([
        [0., 0., 0.],  # coordinates of basis site, b=0
        [2. / 3., 1. / 3., 1. / 2.],  # coordinates of basis site, b=1
        [1. / 3., 2. / 3., 1. / 4.],  # coordinates of basis site, b=2
        [1. / 3., 2. / 3., 3. / 4.],  # coordinates of basis site, b=3
    ]).transpose()

    # Occupation degrees of freedom (DoF)
    occ_dof = [
        ["Zr"],  # no variation allowed on basis site, b=0
        ["Zr"],  # no variation allowed on basis site, b=1
        ["Va", "O"],  # occupants allowed on basis site, b=2
        ["Va", "O"],  # occupants allowed on basis site, b=3
    ]

    # Construct the prim
    prim = xtal.Prim(lattice=lattice,
                     coordinate_frac=coordinate_frac,
                     occ_dof=occ_dof,
                     title="ZrO")


This prim as JSON: :download:`ZrO_prim.json <json/ZrO_prim.json>`


Ising model
-----------

This example constructs the prim for a simple cubic crystal system occupied by "A.up" and "A.down" occupants, where "A.up" indicates an "A" atom with magnetic spin up, and "A.down" indicates an "A" atom with magnetic spin down.

To construct this prim, the following must be specified:

- the lattice vectors
- basis site coordinates
- occupants
- occupant DoF

The occupants list includes the value of the fixed collinear magnetic spin "Cmagspin" associated with the occupants. The occ_dof uses the occupants keys as labels to specify which occupants are allowed on each basis site.

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
        [0., 0., 0.],
    ]).transpose()  # coordinates of basis site, b=0

    # Occupation degrees of freedom (DoF)
    A_up = xtal.Occupant("A", properties={"Cmagspin": np.array([1.])})
    A_down = xtal.Occupant("A", properties={"Cmagspin": np.array([-1.])})
    occupants = {
        "A.up": A_up,  # A atom, spin up
        "A.down": A_down,  # A atom, spin down
    }
    occ_dof = [
        ["A.up", "A.down"],
    ]

    # Construct the prim
    prim = xtal.Prim(lattice=lattice,
                     coordinate_frac=coordinate_frac,
                     occ_dof=occ_dof,
                     occupants=occupants,
                     title="simple_cubic_ising")


This prim as JSON: :download:`simple_cubic_ising.json <json/simple_cubic_ising.json>`
