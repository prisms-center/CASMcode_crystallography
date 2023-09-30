# Changelog

All notable changes to `libcasm-xtal` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0a4] - 2023-09-29

### Fixed

- Fixed xtal::Site::has_dof to check for either occ or continuous DoF 


## [2.0a3] - 2023-08-11

This release separates out casm/crystallography from CASM v1, in particular removing remaining dependencies on casm/symmetry for constructing symmetry representations and getting basic symmetry operation info. It creates a Python package, libcasm.xtal, that enables using casm/crystallography and may be installed via pip install, using scikit-build, CMake, and pybind11. This release also includes usage and API documentation for using libcasm.xtal, built using Sphinx.

### Added

- Added xtal::SymOpSortKeyCompare for sorting xtal::SymOp
- Added xtal::make_crystal_point_group to make the crystal point group from a sorted_factor_group (std::vector<xtal::SymOp>)
- Added xtal::make_internal_translations to get translations from the factor group
- Added standalone methods xtal::fast_pbc_displacement_cart and xtal::robust_pbc_displacement_cart using Eigen::Vector3d for coordinates
- Added xtal::Molecule::identical overload for checking for identical Molecule up to a permutation
- Added OccupantDoFIsEquivalent::atom_position_perm to get Molecule atom position permuation
- Added xtal::make_simple_structure from a POSCAR stream
- Added xtal::apply and xtal::copy_apply methods for transforming xtal::SimpleStructure including properties
- Added xtal::make_inverse for xtal::SymOp
- Added xtal::is_equivalent for comparing xtal::SimpleStructure
- Added Python package libcasm.xtal to use CASM crystallography methods for building lattices, structures, and parent crystal structures and allowed degrees of freedom (prim); enumerating superlattices; creating superstructures; finding primitive and reduced cells; and determining symmetry operations.
- Added scikit-build, CMake, and pybind11 build process
- Added GitHub Actions for unit testing
- Added GitHub Action build_wheels.yml for Python x86_64 wheel building using cibuildwheel
- Added Cirrus-CI .cirrus.yml for Python aarch64 and arm64 wheel building using cibuildwheel
- Added Python documentation


### Changed

- Moved StrainConverter into crystallography, removing symmetry dependencies
- Moved SymInfo into crystallography from symmetry, using only xtal::SymOp
- Moved class SymBasisPermute from symmetry to struct UnitCellCoordRep

### Removed

- Removed autotools build process
- Removed boost dependencies
