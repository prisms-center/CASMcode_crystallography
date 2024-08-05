Structure construction and symmetry analysis
============================================

The :py:class:`~libcasm.xtal.Structure` class is used to represent a crystal of molecular and/or atomic occupants, and any additional properties.

Both enumerated configurations and calculation results may be represented using this format.

:py:class:`~libcasm.xtal.Structure` may specify atom and / or molecule coordinates and properties:

- lattice vectors
- atom coordinates
- atom type names
- continuous atom properties
- molecule coordinates
- molecule type names
- continuous molecule properties
- continuous global properties

Atom representation is most widely supported in CASM methods. In some limited cases the molecule representation is used.



Structure construction
----------------------

As an example, a Si-Ge ordering in the diamond cubic structure can be constructed as follows:

.. code-block:: Python

    import math
    import numpy as np
    import libcasm.xtal as xtal

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [ 5.600000000000, 0.000000000000, 0.000000000000 ],
        [ 0.000000000000, 5.600000000000, 0.000000000000 ],
        [ 0.000000000000, 0.000000000000, 5.600000000000 ]
    ]).transpose()  # <--- note transpose
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Atom coordinates, as columns of a matrix, in Cartesian coordinates
    atom_coordinate_cart = np.array([
        [ 0.000000000000, 0.000000000000, 0.000000000000 ],
        [ 2.800000000000, 0.000000000000, 2.800000000000 ],
        [ 0.000000000000, 2.800000000000, 2.800000000000 ],
        [ 2.800000000000, 2.800000000000, 0.000000000000 ],
        [ 1.400000000000, 1.400000000000, 1.400000000000 ],
        [ 4.200000000000, 1.400000000000, 4.200000000000 ],
        [ 1.400000000000, 4.200000000000, 4.200000000000 ],
        [ 4.200000000000, 4.200000000000, 1.400000000000 ]
    ]).transpose()

    # Atom coordinates, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    atom_coordinate_frac = xtal.cartesian_to_fractional(
      lattice,
      atom_coordinate_cart,
    )

    # Atom types
    atom_type = [ "Ge", "Si", "Si", "Si", "Ge", "Si", "Si", "Si" ]

    # Construct the structure
    structure = xtal.Structure(
        lattice=lattice,
        atom_coordinate_frac=atom_coordinate_frac,
        atom_type=atom_type,
    )


Structure properties
--------------------

Local or global properties of the structure obtained from calculations can be added to a structure:

.. code-block:: Python

    import math
    import numpy as np
    import libcasm.xtal as xtal

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [ -0.000801520000, 5.528984430000, -0.000801520000 ],
        [ -0.000801520000, -0.000801520000, 5.528984430000 ],
        [ 5.528984430000, -0.000801520000, -0.000801520000 ]
    ]).transpose()  # <--- note transpose
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Atom coordinates, as columns of a matrix, in Cartesian coordinates
    atom_coordinate_cart = np.array([
        [ 5.509869982471, 5.509869982471, 5.509869982471 ],
        [ 2.779560277905, 2.779560277905, 5.511669529816 ],
        [ 5.511669529816, 2.779560277905, 2.779560277905 ],
        [ 2.779560277905, 5.511669529816, 2.779560277905 ],
        [ 1.399356755029, 1.399356755029, 1.399356755029 ],
        [ 4.129666459595, 4.129666459595, 1.397557207684 ],
        [ 1.397557207684, 4.129666459595, 4.129666459595 ],
        [ 4.129666459595, 1.397557207684, 4.129666459595 ]
    ]).transpose()

    # Atom coordinates, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    atom_coordinate_frac = xtal.cartesian_to_fractional(
      lattice,
      atom_coordinate_cart,
    )

    # Atom types
    atom_type = [ "Ge", "Si", "Si", "Si", "Ge", "Si", "Si", "Si" ]

    # Global properties
    global_properties = {
        "energy": {"value": -41.486033340000},
        "Ustrain" : {
            "value": [0.987318648214, 0.987318648214, 0.987318648214, -0.000202414367, -0.000202414367, -0.000202414367]
        }
    }

    # Construct the structure with properties
    structure_with_properties = xtal.Structure(
        lattice=lattice,
        atom_coordinate_frac=atom_coordinate_frac,
        atom_type=atom_type,
        atom_properties=atom_properties,
        global_properties=global_properties,
    )

The positions of atoms or molecules in the crystal state is defined by the lattice and atom coordinates or molecule coordinates. If included, strain and displacement properties, which are defined in reference to an ideal state, should be interpreted as the strain and displacement that takes the crystal from the ideal state to the state specified by the structure lattice and atom or molecule coordinates. The convention used by CASM is that displacements are applied first, and then the displaced coordinates and lattice vectors are strained.

See the `CASM Degrees of Freedom (DoF) and Properties documentation`_ for the full list of supported properties and their definitions.

.. _`CASM Degrees of Freedom (DoF) and Properties documentation`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/


Common structures
-----------------

Some common structures can be constructed using the convenience methods in :py:mod:`libcasm.xtal.structures`:

.. code-block:: Python

    >>> import libcasm.xtal.structures as xtal_structures

    # Zr HCP structure, specified by conventional cubic lattice parameter `a`
    >>> Zr_hcp = xtal_structures.HCP(a=3.23398686, c=5.16867834, atom_type="Zr")
    >>> print(Zr_hcp.to_json())
    {
      "atom_coords": [
        [0.0, 1.8671431841767125, 1.292169585],
        [1.6169934299999997, 0.9335715920883563, 3.8765087549999997]
      ],
      "atom_type": ["Zr", "Zr"],
      "coordinate_mode": "Cartesian",
      "lattice_vectors": [
        [3.23398686, 0.0, 0.0],
        [-1.61699343, 2.800714776265069, 0.0],
        [0.0, 0.0, 5.16867834]
      ]
    }


Structure factor group generation
---------------------------------

The `factor group` of a structure is the set of transformations, with translation lying within the primitive unit cell, that leave the lattice vectors, basis site coordinates, and atom types invariant. It is found by creating a :py:class:`~libcasm.xtal.Prim` with only the current atom types as allowed DoF and constructing the prim's factor group. Currently this method only considers atom coordinates and types. Molecular coordinates and types are not considered.

.. code-block:: Python

    >>> factor_group = xtal.make_structure_factor_group(structure)
    >>> i = 1
    ... for op in factor_group:
    ...     syminfo = xtal.SymInfo(op, lattice)
    ...     print(str(i) + ":", syminfo.brief_cart())
    ...     i += 1

::

    1: 1
    2: 3⁺ 0.5773503*x, 0.5773503*x, 0.5773503*x
    3: 3⁻ 0.5773503*x, 0.5773503*x, 0.5773503*x
    4: 2 0.7, 0.7+0.7071068*y, 0.7-0.7071068*y
    5: 2 0.7+0.7071068*x, 0.7-0.7071068*x, 0.7
    6: 2 0.7+0.7071068*x, 0.7, 0.7-0.7071068*x
    7: m x, 0.7071068*y, 0.7071068*y
    8: m 0.7071068*x, 0.7071068*x, z
    9: m 0.7071068*x, y, 0.7071068*x
    10: -3⁺ 0.7+0.5773503*x, 0.7+0.5773503*x, 0.7+0.5773503*x; 0.7000000 0.7000000 0.7000000
    11: -3⁻ 0.7+0.5773503*x, 0.7+0.5773503*x, 0.7+0.5773503*x; 0.7000000 0.7000000 0.7000000
    12: -1 0.7000000 0.7000000 0.7000000


Crystal point group
-------------------

The crystal point group of a structure can be generated using the :func:`~libcasm.xtal.make_structure_crystal_point_group` method:

.. code-block:: Python

    crystal_point_group = xtal.make_structure_crystal_point_group(structure)


Structure manipulation
----------------------

Move coordinates within the unit cell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~libcasm.xtal.make_structure_within` method returns an equivalent :py:class:`~libcasm.xtal.Structure` with all atom and mol site coordinates within the unit cell.


Make a superstructure
^^^^^^^^^^^^^^^^^^^^^

The :func:`~libcasm.xtal.make_superstructure` method can be used to create superstructures.


Apply a transformation to a structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rotations and other transformations can be applied to a :class:`Structure` using the :py:class:`~libcasm.xtal.SymOp` class. For example, to rotate a structure by 90 degrees counterclockwise about the z-axis, a :py:class:`~libcasm.xtal.SymOp` can be constructed with the rotation matrix and applied to the initial structure with the ``*`` operator:

.. code-block:: Python

    rotation_matrix = np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]
    )
    rotation_op = xtal.SymOp(rotation_matrix)
    rotated_structure = rotation_op * structure


Filter or sort atoms in a structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several methods provided in libcasm-xtal to help with filtering and sorting atoms in a :class:`Structure`:

- :func:`~libcasm.xtal.filter_structure_by_atom_info`: Filter atoms by atom type, coordinate, or properties.
- :func:`~libcasm.xtal.sort_structure_by_atom_coordinate_cart`: Sort atoms by Cartesian coordinates.
- :func:`~libcasm.xtal.sort_structure_by_atom_coordinate_frac`: Sort atoms by fractional coordinates.
- :func:`~libcasm.xtal.sort_structure_by_atom_type`: Sort atoms by type name.
- :func:`~libcasm.xtal.sort_structure_by_atom_info`: Sort atoms by custom function of atom type, coordinate, or properties.


Substitute species
^^^^^^^^^^^^^^^^^^

The method :func:`~libcasm.xtal.substitute_structure_species` can be used to quickly create a copy of a structure with atomic and molecular species renamed according to a dict containing the substitutions. For example, using ``substitutions = { "A": "Mg", "B": "Cd" }`` a generic A-B structure can be converted to a MgCd structure.


Combine structures
^^^^^^^^^^^^^^^^^^

The method :func:`~libcasm.xtal.combine_structures` can be used to make superstructures of two or more atomic substructures. The resulting structure contains the species of all input structures with Cartesian coordinates fixed to the same values as in the input structures. All atom or molecule properties remain the same as in the input structures.


Custom manipulations
^^^^^^^^^^^^^^^^^^^^

:class:`~libcasm.xtal.Structure` objects are immutable, so manipulations are performed by creating a copy.
For atomic structures

For atomic structures, the :func:`~libcasm.xtal.make_structure_atom_info` and :func:`~libcasm.xtal.make_structure_from_atom_info` methods can be used to:

- iterate over the atoms in one or more structures,
- perform some operation, and
- create a new structure.

The method :func:`~libcasm.xtal.make_structure_atom_info` takes a structure and creates a list of :class:`~libcasm.xtal.StructureAtomInfo` namedtuple which have atom type, coordinates, and properties. These can be iterated over, filterd, sorted, combined or otherwise manipulated and then passed to :func:`~libcasm.xtal.make_structure_from_atom_info` to create a new structure.

For example, the :func:`~libcasm.xtal.combine_structures` method can be implemented as:

.. code-block:: Python

    import libcasm.xtal
    import typing
    import numpy as np

    def combine_structures(
        structures: list[libcasm.xtal.Structure],
        lattice: typing.Optional[libcasm.xtal.Lattice] = None,
        global_properties: dict[str, np.ndarray[np.float64]] = {},
    ) -> libcasm.xtal.Structure:
        """
        Combine `structures` into a new `combined_structure`
        with given `lattice` and `global_properties`.
        If `lattice` is None, use the lattice of the first structure.
        """
        atoms = []
        for structure in structures:
            if lattice is None:
                lattice = structure.lattice()
            atoms += libcasm.xtal.make_structure_atom_info(structure)
        return libcasm.xtal.make_structure_from_atom_info(
            lattice=lattice,
            atoms=atoms,
            global_properties=global_properties,
        )

