Superlattice enumeration
========================

Superlattice relationships
--------------------------

Superlattices satisfy:

.. math::

    S = L T,

where :math:`S` and :math:`L` are, respectively, the superlattice and unit lattice vectors as columns of shape=(3,3) matrices, and :math:`T` is a shape=(3,3) integer transformation matrix. The :func:`~libcasm.xtal.is_superlattice_of` and :func:`~libcasm.xtal.make_transformation_matrix_to_super` methods can be used to check if a superlattice relationship exists between two lattices and find :math:`T`.

Superlattices :math:`S_1` and :math:`S_2` may have different lattice vectors but be symmetrically equivalent if there exists `p` and :math:`U` such that:

.. math::

    S_1 = A_p S_2 U,

where :math:`A_p` is the operation matrix of the `p`-th element in the relevant point group, and :math:`U` is a unimodular matrix (integer matrix, with :math:`\det(U) = \pm 1`). The :func:`~libcasm.xtal.is_equivalent_superlattice_of` method can be used to check if a lattice is symmetrically equivalent to a superlattice of another lattice and identify `p`.

The :func:`~libcasm.xtal.enumerate_superlattices` function enumerates symmetrically unique superlattices given:

- a unit lattice
- a point group defining which lattices are symmetrically equivalent
- a maximum volume (as a multiple of the unit lattice volume) to enumerate

The appropriate point group for superlattice enumeration depends on the use case. For enumeration of degrees of freedom (DoF) values given a particular prim, the appropriate point group is the crystal point group. If there is no basis or DoF to consider, then the unit lattice point group may be the appropriate point group.

Enumerating superlattices
-------------------------

To enumerate superlattices of a :class:`~libcasm.xtal.Lattice`, with no basis or DoF:

.. code-block:: Python

    unit_lattice = lattice
    point_group = xtal.make_point_group(lattice)
    superlattices = xtal.enumerate_superlattices(
        unit_lattice, point_group, max_volume=4, min_volume=1, dirs="abc")


To enumerate superlattices of a :class:`~libcasm.xtal.Prim`, taking into account the symmetry of the basis and DoF:

.. code-block:: Python

    unit_lattice = prim.lattice()
    point_group = xtal.make_crystal_point_group(prim)
    superlattices = xtal.enumerate_superlattices(
        unit_lattice, point_group, max_volume=4, min_volume=1, dirs="abc")


The minimum volume is optional, with default=1. The `dirs` parameter, with default="abc", specifies which lattice vectors to enumerate over ("a", "b", and "c" indicate the first, second, and third lattice vectors, respectively). This allows restriction of the enumeration to 1d (i.e. ``dirs="b"``) or 2d superlattices (i.e. ``dirs="ac"``).

The output, `superlattices`, is a list of :class:`~libcasm.xtal.Lattice`, which will be in canonical form with respect to point_group.


Super-duper lattice
-------------------

It is often useful to find superlattices that are commensurate with multiple ordered phases. The :func:`~libcasm.xtal.make_superduperlattice` function finds a minimum volume lattice that is a superlattice of 2 or more input lattices.

.. code-block:: Python

    >>> # make super-duper lattices
    >>> superduperlattice = xtal.make_superduperlattice(
    ...     lattices=[lattice1, lattice2, lattice3],
    ...     mode="fully_commensurate",
    ...     point_group=point_group)

This function implements three modes:

- (default) "commensurate": Finds the mininum volume superlattice of all the input lattices, without any application of symmetry. The `point_group` parameter is ignored if provided.
- "minimal_commensurate": Returns the lattice that is the smallest possible superlattice of an equivalent lattice to all input lattices.
- "fully_commensurate": Returns the lattice that is a superlattice of all equivalents of
  all input lattices.

The `point_group` parameter is used to generate equivalent lattices for the the "minimal_commensurate" and "fully_commensurate" modes. This would typically be the prim crystal point group.
