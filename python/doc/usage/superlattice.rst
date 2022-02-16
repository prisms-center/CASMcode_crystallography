Superlattice enumeration
========================

Superlattice relationships
--------------------------

Superlattices satisfy:

.. math::

    S = L T,

where :math:`S` and :math:`L` are, respectively, the superlattice and unit lattice vectors as columns of :math:`3 \times 3` matrices, and :math:`T` is an integer :math:`3 \times 3` transformation matrix. The :func:`~casm.xtal.Lattice.is_superlattice_of`, and :func:`~casm.xtal.Lattice.make_transformation_matrix_to_super` methods of :class:`~casm.xtal.Lattice` can be used to check if a superlattice relationship exists between two lattices and find T.

Superlattices :math:`S_1` and :math:`S_2` may have different lattice points but be symmetrically equivalent if there exists p and :math:`U` such that:

.. math::

    S_2 = A_p S_1 U,

where :math:`A_p` the operation matrix of the p-th element in the relevant point group, and :math:`U` is a unimodular matrix (integer matrix, with :math:`\det(U) = \pm 1`). The :func:`~casm.xtal.Lattice.is_equivalent_superlattice_of` method of :class:`~casm.xtal.Lattice` can be used to check if a lattice is symmetrically equivalent to a superlattice of another lattice and identify p.

The :func:`~casm.xtal.enumerate_superlattices` function enumerates symmetrically unique superlattices given:

- a unit lattice
- a point group defining which lattices are symmetrically equivalent
- a maximum volume (as a multiple of the unit lattice volume) to enumerate

The appropriate point group for superlattice enumeration depends on the use case. For enumeration of degrees of freedom (DoF) values given a particular prim, the appropriate point group is the crystal point group. If there is no basis or DoF to consider, then the unit lattice point group may be the appropriate point group.

Usage
-----

To enumerate superlattices of a prim (casm.xtal.Prim), taking into account the symmetry of the basis and DoF:

.. code-block:: Python

    >>> unit_lattice = prim.lattice()
    >>> point_group = prim.make_crystal_point_group()
    >>> superlattices = xtal.enumerate_superlattices(
    ...     unit_lattice, point_group, max_volume=4, min_volume=1, dirs="abc")

To enumerate superlattices of a lattice (casm.xtal.Lattice), with no basis or DoF:

.. code-block:: Python

    >>> unit_lattice = lattice
    >>> point_group = lattice.make_crystal_point_group()
    >>> superlattices = xtal.enumerate_superlattices(
    ...     unit_lattice, point_group, max_volume=4, min_volume=1, dirs="abc")

The minimum volume is optional, with default=1. The dirs parameter, with default="abc", specifies which lattice vectors to enumerate over ("a", "b", and "c" indicate the first, second, and third lattice vectors, respectively). This allows restriction of the enumeration to 1d (i.e. dirs="b") or 2d superlattices (i.e. dirs="ac").

The output, superlattices, is a list of :class:`~casm.xtal.Lattice`, which will be in canonical form.


Super-duper lattice
-------------------

It is often useful to find superlattices that are commensurate with multiple ordered phases. The :func:`~casm.xtal.make_superduperlattice` function finds a minimum volume lattice that is a superlattice of 2 or more input lattices.

.. code-block:: Python

    >>> # make super-duper lattices
    >>> superduperlattice = xtal.make_superduperlattice(
    ...     lattices=[lattice1, lattice2, lattice3],
    ...     mode="fully_commensurate",
    ...     point_group=point_group)

It includes three modes:

- (default) "commensurate": Finds the mininum volume superlattice of all the input lattices, without any application of symmetry. The point_group parameter is ignored if provided.
- "minimal_commensurate": Returns the lattice that is the smallest possible superlattice of an equivalent lattice to all input lattices.
- "fully_commensurate": Returns the lattice that is a superlattice of all equivalents of
  all input lattices.

The point_group is used to generate equivalent lattices for the the "minimal_commensurate" and
"fully_commensurate" modes. This would typically be the prim crystal point group.
