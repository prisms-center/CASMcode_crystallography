Primitive crystal structure construction and symmetry analysis
==============================================================

The :py:class:`~libcasm.xtal.Prim` class is used to represent a primitive crystal structure and allowed degrees of freedom (DoF). It specifies the:

- lattice vectors
- crystal basis sites
- occupation DoF
- continuous local (site) DoF
- continuous global DoF

The prim is the starting point for constructing a cluster expansion effective Hamiltonian. The allowed DoF determine which configurations are possible, and the symmetry of the prim determines the symmetry of the cluster basis functions. The :class:`~libcasm.xtal.Prim` class constructor documentation is :ref:`here <prim-init>`, and this section gives an introduction through examples.


Occupation DoF
--------------

The following is an example of prim construction, including atomic occupation DoF only:

.. code-block:: Python

    import numpy as np
    import libcasm.xtal as xtal

    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [ 1.0, 0.0, 0.0], # a
            [ 0.0, 1.1, 0.0], # b
            [ 0.0, 0.0, 1.3], # c
        ]
    ).transpose()    # <--- note transpose
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array(
        [
            [0., 0., 0.],  # coordinates of basis site, b=0
        ]
    ).transpose() # <--- note transpose

    # Occupation degrees of freedom (DoF)
    occ_dof = [
        ["A", "B"],    # occupants allowed on basis site, b=0
    ]

    return xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        title="simple_cubic_binary",
    )

The parameter `lattice` gives the primitive cell Lattice.

The parameter `coordinate_frac` gives basis site positions, as columns of a matrix, in fractional coordinates with respect to the lattice vectors.

The parameter `occ_dof` gives labels of occupants allowed on each basis site. The value occ_dof[b] is the list of occupants allowed on the b-th basis site. The values may either be (i) the name of an isotropic atom (i.e. “Mg”) or vacancy (“Va”), or (ii) a key in the optional parameter, `occupants`, (see below). The names are case sensitive, and “Va” is reserved for vacancies.

The optional parameter, `occupants`, is a dictionary containing :class:`~libcasm.xtal.Occupant` allowed in the crystal. The keys are labels used in the occ_dof parameter. This may include isotropic atoms, vacancies, atoms with fixed anisotropic properties, and molecular occupants. A seperate key and value is required for all species with distinct anisotropic properties (i.e. “H2_xy”, “H2_xz”, and “H2_yz” for distinct orientations, or “A.up”, and “A.down” for distinct collinear magnetic spins, etc.). See the CASM `Degrees of Freedom (DoF) and Properties`_ documentation for the full list of supported property types and their definitions.

**Example: Isotropic atomic occupant**

.. code-block:: Python

    A_occ = xtal.Occupant(name="A")

Or, equivalently:

.. code-block:: Python

    A_occ = xtal.make_atom("A")

**Example: Vacancy occupant**

.. code-block:: Python

    Va_occ = xtal.Occupant(name="Va")

Or, equivalently:

.. code-block:: Python

    Va_occ = xtal.make_vacancy()

**Example: Atomic occupants with fixed collinear magnetic spin**

The value "Cmagspin" is string indicating the CASM supported collinear magnetic spin property type. See the `Degrees of Freedom (DoF) and Properties`_ documentation for the full list of supported property types and their definitions.

.. code-block:: Python

    A_up_occ = xtal.Occupant(
        name="A",                       # "chemical name" of occupant
        properties={
            "Cmagspin": np.array([1.])  # fixed properties of the occupant
        },
    )
    A_down_occ = xtal.Occupant(
        name="A",                       # "chemical name" of occupant
        properties={
            "Cmagspin": np.array([-1.]) # fixed properties of the occupant
        },
    )
    occupants = {
      "A.up": A_up_occ,     # <label> : occupant
      "A.down": A_down_occ, # <label> : occupant
    }
    occ_dof = [
      ["A.up", "A.down"],   # occupants allowed on basis site, b=0
    ]
    prim = xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        occupants=occupants,
        title="ising",
    )

The :class:`~libcasm.xtal.Occupant` constructor parameter `name` is a "chemical name" which must be equal for occupants to be found symmetrically equivalent.


**Example: Molecular occupants with distinct orientations**

The :class:`~libcasm.xtal.AtomComponent` can be used to specify the positions of individual atoms in a molecular :class:`~libcasm.xtal.Occupant`. The following specifies three orientations of O2, aligned along the x, y, and z axes, respectively.

.. code-block:: Python

    delta = 0.6   # Cartesian distance
    O2_xx_occ = xtal.Occupant(
        name="O2",
        atoms=[
            libcasm.xtal.AtomComponent(name="O", coordinate=np.array([delta, 0., 0.])),
            libcasm.xtal.AtomComponent(name="O", coordinate=np.array([-delta, 0., 0.])),
        ],
    )
    O2_yy_occ = xtal.Occupant(
        name="O2",
        atoms=[
            libcasm.xtal.AtomComponent(name="O", coordinate=np.array([0., delta, 0.])),
            libcasm.xtal.AtomComponent(name="O", coordinate=np.array([0., -delta, 0.])),
        ],
    )
    O2_zz_occ = xtal.Occupant(
        name="O2",
        atoms=[
            libcasm.xtal.AtomComponent(name="O", coordinate=np.array([0., 0., delta])),
            libcasm.xtal.AtomComponent(name="O", coordinate=np.array([0., 0., -delta])),
        ],
    )
    occupants = {
      "O2_xx": O2_xx_occ,     # <label> : occupant
      "O2_yy": O2_yy_occ,     # <label> : occupant
      "O2_zz": O2_zz_occ,     # <label> : occupant
    }
    occ_dof = [
      ["O2_xx", "O2_yy", "O2_zz"],   # occupants allowed on basis site, b=0
    ]
    prim = xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        occupants=occupants,
        title="ternary_orientation",
    )


Continuous local DoF
--------------------

The optional local_dof parameter enables constructing a :class:`~libcasm.xtal.Prim` which includes continuous local DoF (DoF associated with a particular site). There is no effect if local_dof is empty. If not empty, the value local_dof[b] is a list of :class:`~libcasm.xtal.DoFSetBasis` objects describing the continuous local DoF allowed on the b-th basis site.

This section provides examples construting a prim with:

- "disp": Atomic displacement DoF
- "Cmagspin": Collinear magnetic spin DoF
- "SOmagspin": Non-collinear magnetic spin DoF, with spin-orbit coupling

See the `Degrees of Freedom (DoF) and Properties`_ documentation for the full list of supported DoF types and their definitions.


**Example: Atomic displacement DoF**

Atomic displacement DoF, with the standard basis :math:`[d_{x}, d_{y}, d_{z}]` can be added using:

.. code-block:: Python

    # Local continuous degrees of freedom (DoF)
    disp_dof = xtal.DoFSetBasis("disp")    # Atomic displacement
    local_dof = [
        [disp_dof], # allow displacements on basis site b=0
        [disp_dof], # allow displacements on basis site b=1
    ]
    prim = xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        local_dof=local_dof,
    )


**Example: Collinear magnetic spin DoF**

Collinear magnetic spin DoF, with the standard basis :math:`[m]` can be added using:

.. code-block:: Python

    # Local continuous degrees of freedom (DoF)
    Cmagspin_dof = xtal.DoFSetBasis("Cmagspin")    # Collinear magnetic spin
    local_dof = [
        [Cmagspin_dof], # allow collinear magnetic spin on basis site b=0
        [Cmagspin_dof], # allow collinear magnetic spin on basis site b=1
    ]
    prim = xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        local_dof=local_dof,
    )


**Example: Non-collinear magnetic spin DoF, with spin-orbit coupling**

Non-collinear magnetic spin DoF, with spin-orbit coupling, with the standard basis :math:`[m]` can be added using:

.. code-block:: Python

    # Local continuous degrees of freedom (DoF)
    SOmagspin_dof = xtal.DoFSetBasis("SOmagspin")
    local_dof = [
        [SOmagspin_dof], # allow SOmagspin on basis site b=0
        [SOmagspin_dof], # allow SOmagspin on basis site b=1
    ]
    prim = xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        local_dof=local_dof,
    )


**Example: Atomic displacement DoF, user-specified basis**

It is possible to restrict the dimension of allowed DoF, or rotate the basis, by providing a user-specified basis. The following restricts atomic displacements to 1-dimensions displacements along the x-axis:

.. code-block:: Python

    # Local continuous degrees of freedom (DoF)
    disp_dof = xtal.DoFSetBasis(
        "disp",
        axis_names=["d_{1}"],  # 1d displacments
        basis=np.array(
            [
                [1.0, 0.0, 0.0], # displacements along x
            ]
        ).transpose())
    local_dof = [
        [disp_dof], # basis site 1
        [disp_dof], # basis site 2
    ]
    prim = xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        local_dof=local_dof,
    )

If a user-specified basis is provided, configurations, and the cluster expansion, are restricted to the specified space.


.. _sec-strain-dof:

Strain DoF
----------

CASM supports strain global continuous DoF, with the following choices of symmetric strain metrics, :math:`E`:

- `"GLstrain"`: Green-Lagrange strain metric, :math:`E = \frac{1}{2}(F^{\mathsf{T}} F - I)`
- `"Hstrain"`: Hencky strain metric, :math:`E = \frac{1}{2}\ln(F^{\mathsf{T}} F)`
- `"EAstrain"`: Euler-Almansi strain metric, :math:`E = \frac{1}{2}(I−(F F^{\mathsf{T}})^{-1})`

Where:

- :math:`L`: Lattice vectors, as columns of a matrix, shape=(3,3)
- :math:`F`: deformation tensor, :math:`L^{strained} = F L^{ideal}`, shape=(3,3)
- :math:`I`: identity matrix, shape=(3,3)
- :math:`E`: symmetric strain metric, shape=(3,3)

Two additional strain metrics are supported as properties which can be transformed by symmetry operations, but not as DoF:

- `"Bstrain"`: Biot strain metric, :math:`E = U - I`
- `"Ustrain"`: Right stretch tensor, :math:`E = U`

The deformation tensor, F, can be decomposed into a pure isometry (rigid transformation), :math:`Q`, shape=(3,3), and either the right stretch tensor, :math:`U`, shape=(3,3), or the left stretch tensor, :math:`V`, shape=(3,3), according to:

.. math::

    F &= Q U = V Q

    Q^{-1} &= Q^{\mathsf{T}}

The strain metric, :math:`E`, can be represented by the vector, :math:`\vec{E}`, which is the CASM standard strain basis:

.. math::

    \vec{E} = [E_{xx}, E_{yy}, E_{zz}, \sqrt{2}E_{yz}, \sqrt{2}E_{xz}, \sqrt{2}E_{xy}]


**Example: Strain DoF, using the Green-Lagrange strain metric**

The following constructs a prim with strain DoF, using the Green-Lagrange strain metric, with the standard basis, :math:`\vec{E}`:

.. code-block:: Python

    # Global continuous degrees of freedom (DoF)
    Hstrain_dof = xtal.DoFSetBasis("Hstrain")     # Hencky strain metric
    global_dof = [Hstrain_dof]
    prim = xtal.Prim(lattice=lattice, coordinate_frac=coordinate_frac, global_dof=global_dof)

**Example: Strain DoF, symmetry-adapted basis**

As described by :cite:t:`THOMAS2017a`, the symmetry-adapted strain basis,

.. math::

    B^{\vec{e}} = \left(
      \begin{array}{cccccc}
      1/\sqrt{3} & 1/\sqrt{2} & -1/\sqrt{6} & 0 & 0 & 0 \\
      1/\sqrt{3} & -1/\sqrt{2} & -1/\sqrt{6} & 0 & 0 & 0  \\
      1/\sqrt{3} & 0 & 2/\sqrt{6} & 0 & 0 & 0  \\
      0 & 0 & 0 & 1 & 0 & 0 \\
      0 & 0 & 0 & 0 & 1 & 0 \\
      0 & 0 & 0 & 0 & 0 & 1
      \end{array}
    \right),

is a transformation which decomposes strain space into irreducible subspaces which do not mix under application of symmetry. Using the symmetry-adapted strain basis results in symmetry-adapted strain metric vectors,

.. math::

    \vec{e} = \left( \begin{array}{ccc} e_1 \\ e_2 \\ e_3 \\ e_4 \\ e_5 \\ e_6 \end{array} \right) = \left( \begin{array}{ccc} \left( E_{xx} + E_{yy} + E_{zz} \right)/\sqrt{3} \\ \left( E_{xx} - E_{yy} \right)/\sqrt{2} \\ \left( 2E_{zz} - E_{xx} - E_{yy} + \right)/\sqrt{6} \\ \sqrt{2}E_{yz} \\ \sqrt{2}E_{xz} \\ \sqrt{2}E_{xy} \end{array} \right).

The same symmetry-adapted strain basis holds for all crystal point groups, but the irreducible subspaces vary. As an example, for cubic point groups, there are three irreducible subspaces: :math:`\{e_1\}`, :math:`\{e_2, e_3\}`, and :math:`\{e_4, e_5, e_6\}`. For hexagonal point groups, there are four irreducible subspaces: :math:`\{e_1\}`, :math:`\{e_3\}`, :math:`\{e_2, e_6\}`, and :math:`\{e_4, e_5\}`.

The following uses :func:`~libcasm.xtal.make_symmetry_adapted_strain_basis` to construct a prim with strain DoF, using the Hencky strain metric, and the symmetry-adapted basis:

.. code-block:: Python

    # Global continuous degrees of freedom (DoF)
    Hstrain_dof = xtal.DoFSetBasis(
        dofname="Hstrain",
        axis_names=["e_{1}", "e_{2}", "e_{3}", "e_{4}", "e_{5}", "e_{6}"],
        basis=xtal.make_symmetry_adapted_strain_basis(),
    )
    global_dof = [Hstrain_dof]
    prim = xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        global_dof=global_dof,
    )


**Example: Strain DoF, user-specified basis**

It is possible to restrict the dimension of allowed strain DoF, or rotate the strain basis, by providing a user-specified basis. The following restricts strain to exclude shear strains:

.. code-block:: Python

    from math import sqrt
    # Global continuous degrees of freedom (DoF)
    Hstrain_dof = xtal.DoFSetBasis(
        dofname="Hstrain",
        axis_names=["e_{1}", "e_{2}", "e_{3}"],
        basis=np.array(
            [
                [1./sqrt(3), 1./sqrt(3), 1./sqrt(3), 0.0, 0.0, 0.0],
                [1./sqrt(2), -1./sqrt(2), 0.0, 0.0, 0.0, 0.0],
                [-1./sqrt(6), -1./sqrt(6), 2./sqrt(6), 0.0, 0.0, 0.0],
            ]
        ).transpose()
    )
    global_dof = [Hstrain_dof]
    prim = xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        global_dof=global_dof,
    )


Common prim
-----------

Some common prim can be constructed using the convenience methods in :py:mod:`libcasm.xtal.prims`. For example, a binary FCC prim with conventional cubic lattice parameter `a` equal to 6.60 can be constructed using the following:

.. code-block:: Python

    >>> import libcasm.xtal.prims as xtal_prims
    >>> fcc_prim = xtal_prims.FCC(a=6.60, occ_dof=["A", "B"])
    >>> print(fcc_prim.to_json())
    {
      "basis": [
        {
          "coordinate": [0.0, 0.0, 0.0],
          "occupants": ["A", "B"]
        }
      ],
      "coordinate_mode": "Fractional",
      "lattice_vectors": [
        [0.0, 3.3, 3.3],
        [3.3, 0.0, 3.3],
        [3.3, 3.3, 0.0]
      ],
      "title": "prim"
    }


Primitive cell
--------------

A :class:`~libcasm.xtal.Prim` object is not forced to be the primitive equivalent cell at construction. The :func:`~libcasm.xtal.make_primitive` method finds and returns the primitive equivalent cell by checking for internal translations that map all basis sites onto equivalent basis sites, including allowed occupants and equivalent local degrees of freedom (DoF), if they exist.


Canonical cell
--------------

The :func:`~libcasm.xtal.make_canonical` method finds the canonical right-handed Niggli cell of the lattice, applying lattice point group operations to find the equivalent lattice in a standardized orientation. The canonical orientation prefers lattice vectors that form symmetric matrices with large positive values on the diagonal and small values off the diagonal. See also `Lattice Canonical Form`_.

.. _`Lattice Canonical Form`: https://prisms-center.github.io/CASMcode_docs/formats/lattice_canonical_form/


Factor group
------------

The `crystal space group` is the set of all rigid transformations that map the infinite crystal onto itself. The crystal space group is not limited to operations that keep the origin fixed, so due to the periodicity of the crystal the crystal space group is infinite.

The `factor group` is a finite description of the crystal space group in which all operations that differ only by a translation are represented by a single operation whose translation lies within the primitive unit cell.

The `factor group` of the prim is the set of transformations, with translation lying within the primitive unit cell, that leave the lattice vectors, basis site coordinates, and all DoF invariant. It is found by a check of the combination of lattice point group operations and translations between basis sites. For cluster expansions of global crystal properties, such as the energy, the cluster basis functions are constructed to have the same symmetry as the prim factor group.

The factor group can be generated using the :func:`~libcasm.xtal.make_factor_group` method, and a description of the operations printed using :class:`~libcasm.xtal.SymInfo` (described :ref:`previously <lattice-symmetry-operation-information>`):

    >>> i = 1
    >>> factor_group = xtal.make_factor_group(prim)
    >>> for op in factor_group:
    ...     syminfo = xtal.SymInfo(op, lattice)
    ...     print(str(i) + ":", syminfo.brief_cart())
    ...     i += 1
    1: 1
    2: 6⁺ (0.0000000 0.0000000 2.5843392) 0, 1.867143, z
    3: 6⁻ (0.0000000 0.0000000 2.5843392) 1.616993, -0.9335716, z
    4: 3⁺ 0, 0, z
    5: 3⁻ 0, 0, z
    6: 2 0.8084967+0.5*x, 0.4667858-0.8660254*x, 1.29217
    ...
    19: g (-0.0000000 -0.0000000  2.5843392) 0.8084967+0.5*x, 0.4667858-0.8660254*x, z
    20: -3⁺ 1.616993, -0.9335716, z;  1.6169934 -0.9335716  1.2921696
    21: -3⁻ 0, 1.867143, z; 0.0000000 1.8671432 1.2921696
    22: -6⁺ 0, 0, z;  0.0000000 -0.0000000  0.0000000
    23: -6⁻ 0, 0, z; -0.0000000 -0.0000000  0.0000000
    24: -1 0.8084967 0.4667858 1.2921696


Crystal point group
-------------------

The `crystal point group` is the group constructed from the prim factor group operations with translation vector set to zero. This is the appropriate point group to use for checking the equivalence of superlattices while taking into account the symmetry of the prim basis site coordinates and DoF.

The crystal point group can be generated using the :func:`~libcasm.xtal.make_crystal_point_group` method:

.. code-block:: Python

    crystal_point_group = xtal.make_crystal_point_group(prim)

.. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/
