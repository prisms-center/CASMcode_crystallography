Strain DoF
==========

Simple cubic with strain DoF
----------------------------

This example constructs the prim for simple cubic crystal system with strain degrees of freedom (DoF), using the Green-Lagrange strain metric.

To construct this prim, the following must be specified:

- the lattice vectors
- basis site coordinates
- occupant DoF
- strain DoF

This example uses a fixed "A" sublattice for occ_dof, which by default are created as isotropic atoms.

.. code-block:: Python

    import numpy as np
    import casm.xtal as xtal

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [1., 0., 0.], # a
        [0., 1., 0.], # a
        [0., 0., 1.]] # a
        ).transpose() # <--- note transpose
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array([
        [0., 0., 0.]]).transpose()  # coordinates of basis site, b=0

    # Occupation degrees of freedom (DoF)
    occ_dof = [
        ["A"]  # occupants allowed on basis site, b=0
    ]

    # Global continuous degrees of freedom (DoF)
    GLstrain_dof = xtal.DoFSetBasis("GLstrain")     # Green-Lagrange strain metric
    global_dof = [GLstrain_dof]

    # Construct the prim
    prim = xtal.Prim(lattice=lattice, coordinate_frac=coordinate_frac, occ_dof=occ_dof,
                     global_dof=global_dof, title="simple_cubic_GLstrain")


This prim as JSON: :download:`simple_cubic_GLstrain.json <json/simple_cubic_GLstrain.json>`


Simple cubic with strain DoF, symmetry-adapted basis
----------------------------------------------------

This example constructs the prim for simple cubic crystal system with strain DoF, using the Green-Lagrange strain metric, with the symmetry-adapted basis :math:`\vec{e}`, as described by :cite:t:`THOMAS201776`:

.. math::

    \vec{e} = \left( \begin{array}{ccc} e_1 \\ e_2 \\ e_3 \\ e_4 \\ e_5 \\ e_6 \end{array} \right) = \left( \begin{array}{ccc} \left( E_{xx} + E_{yy} + E_{zz} \right)/\sqrt{3} \\ \left( E_{xx} - E_{yy} \right)/\sqrt{2} \\ \left( 2E_{zz} - E_{xx} - E_{yy} + \right)/\sqrt{6} \\ \sqrt{2}E_{yz} \\ \sqrt{2}E_{xz} \\ \sqrt{2}E_{xy} \end{array} \right),

The symmetry-adapted basis :math:`\vec{e}` decomposes strain space into irreducible subspaces which do not mix under application of symmetry.

To construct this prim, the following must be specified:

- the lattice vectors
- basis site coordinates
- occupant DoF
- strain DoF

This example uses a fixed "A" sublattice for occ_dof, which by default are created as isotropic atoms.

.. code-block:: Python

    import numpy as np
    import casm.xtal as xtal
    from math import sqrt

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [1., 0., 0.], # a
        [0., 1., 0.], # a
        [0., 0., 1.]] # a
        ).transpose() # <--- note transpose
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array([
        [0., 0., 0.]]).transpose()  # coordinates of basis site, b=0

    # Occupation degrees of freedom (DoF)
    occ_dof = [
        ["A"]  # occupants allowed on basis site, b=0
    ]

    # Global continuous degrees of freedom (DoF)
    GLstrain_dof = xtal.DoFSetBasis(
        dofname="GLstrain",
        axis_names=["e_{1}", "e_{2}", "e_{3}", "e_{4}", "e_{5}", "e_{6}"],
        basis=np.array([
            [1./sqrt(3), 1./sqrt(3), 1./sqrt(3), 0.0, 0.0, 0.0],
            [1./sqrt(2), -1./sqrt(2), 0.0, 0.0, 0.0, 0.0],
            [-1./sqrt(6), -1./sqrt(6), 2./sqrt(6), 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).transpose())
    global_dof = [GLstrain_dof]

    # Construct the prim
    prim = xtal.Prim(lattice=lattice, coordinate_frac=coordinate_frac, occ_dof=occ_dof,
                     global_dof=global_dof, title="simple_cubic_GLstrain_symadapted")


This prim as JSON: :download:`simple_cubic_GLstrain_symadapted.json <json/simple_cubic_GLstrain_symadapted.json>`
