Input and output
================

Converting to and from Python types
-----------------------------------

Important CASM classes have ``to_dict`` / ``to_list`` and ``from_dict`` / ``from_list`` methods so they can be converted to and from plain Python types.

For example, a :class:`xtal.Lattice <libcasm.xtal.Lattice>` can be constructed from a Python dict containing the lattice vectors using:

.. code-block:: Python

    import libcasm.xtal as xtal
    lattice = xtal.Lattice.from_dict({
        "lattice_vectors": [
            [4.471006, 0.0, 0.0],         # vector a
            [0.0, 1.411149, 5.179652],    # vector b
            [-2.235503, -9.345034, 0.0],  # vector c
        ]
    })
    assert isinstance(lattice, xtal.Lattice)

Similarly, a Python dict can be obtained from a :class:`xtal.Lattice <libcasm.xtal.Lattice>` using:

.. code-block:: Python

    data = lattice.to_dict()
    assert isinstance(data, dict)


JSON serialization
------------------

CASM projects primarily use the JSON format to save data in files.

For example, to read :class:`xtal.Prim <libcasm.xtal.Prim>` from a ``prim.json`` JSON file, use:

.. code-block:: Python

    import json
    import libcasm.xtal as xtal
    with open("prim.json", "r") as f:
        prim = xtal.Prim.from_dict(json.load(f))

To write :class:`xtal.Prim <libcasm.xtal.Prim>` to a JSON file, use:

.. code-block:: Python

    with open("prim.json", "w") as f:
        f.write(xtal.pretty_json(lattice.to_dict()))

Here, the function :func:`~libcasm.xtal.pretty_json` is used to convert the Python dict to a JSON string formatted for human readability. It provides a JSON string formatted with indentation, but with arrays printed on a single line if they do not contains sub-objects or sub-arrays.

.. code-block:: Python

    >>> print(xtal.pretty_json(lattice.to_dict()))
    {
      "lattice_vectors": [
        [4.471006, 0.0, 0.0],
        [0.0, 1.411149, 5.179652],
        [-2.235503, -9.345034, 0.0]
      ]
    }


Printing CASM objects
---------------------

Many CASM classes use the ``to_dict`` or ``to_list`` methods and :func:`~libcasm.xtal.pretty_json` when given to the ``print`` function to provide a human-readable representation of the object. For example:

.. code-block:: Python

    >>> print(lattice)
    {
      "lattice_vectors": [
        [4.471006, 0.0, 0.0],
        [0.0, 1.411149, 5.179652],
        [-2.235503, -9.345034, 0.0]
      ]
    }


Column-vector vs row-vector representations
-------------------------------------------

.. attention::

    Take care to use the correct matrix representation for collections of vectors.
    Unless noted otherwise, it is the convention used throughout CASM that:

    - In memory, vectors are stored as *columns* of a `np.ndarray`.
    - For input and output purposes, each individual vector is output in a single Python list or JSON array, so they appear as *rows*.

    This includes lattice vectors, atom and site coordinates, displacements, etc. For
    example:

    - The :class:`~libcasm.xtal.Lattice` class is constructed using the `column_vector_matrix` parameter to specify the lattice vectors as *columns* of a shape=(3,3) `np.ndarray`.
    - For input and output using :func:`~libcasm.xtal.Lattic.to_dict` and :func:`~libcasm.xtal.Lattic.from_dict`, each lattice vector is in single Python list or JSON array so they appear as *rows*.


Reference
---------

The JSON / plain Python format used by CASM objects is documented `here <https://prisms-center.github.io/CASMcode_docs/pages/reference/>`_.


