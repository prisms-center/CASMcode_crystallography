"""Data structure conversions

Users of libcasm.xtal.convert should take care to double-check
that conversions are performed correctly!

This package is a work in progress. Particularly for more
complicated structures or molecules, such as those with mixed
occupation, charge, spin, and magnetic moments, a limited number
or no test cases may exist. Users should double-check that
conversions are correct and make adjustments as needed!

Notes
-----

- This package is intended to work on plain old Python data structures,
  meaning it should work without needing to import libcasm, pymatgen,
  ase, etc.
- Use of standard Python modules and numpy is allowed

"""

import warnings

warnings.warn(
    """
    Users of libcasm.xtal.convert should take care to double-check
    that conversions are performed correctly!
    
    This package is a work in progress. Particularly for more
    complicated structures or molecules, such as those with mixed
    occupation, charge, spin, and magnetic moments, a limited number
    or no test cases may exist. Users should double-check that
    conversions are correct and make adjustments as needed!
    
    Suppress this warning with:
    
        import warnings
        warnings.simplefilter("ignore")
    
    Or by setting the PYTHONWARNINGS environment variable:
    
        export PYTHONWARNINGS="ignore"
    
    """
)
