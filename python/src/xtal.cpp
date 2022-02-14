#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/BasicStructureTools.hh"
#include "casm/crystallography/CanonicalForm.hh"
#include "casm/crystallography/io/BasicStructureIO.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

double default_tol() { return TOL; }

xtal::Lattice make_canonical_lattice(xtal::Lattice const &lattice) {
  return xtal::canonical::equivalent(lattice);
}

/// \brief Convert fractional coordinates to Cartesian coordinates
///
/// \param lattice Lattice
/// \param coordinate_frac Fractional coordinates, as columns of a matrix
Eigen::MatrixXd fractional_to_cartesian(
    xtal::Lattice const &lattice, Eigen::MatrixXd const &coordinate_frac) {
  return lattice.lat_column_mat() * coordinate_frac;
}

/// \brief Convert Cartesian coordinates to fractional coordinates
///
/// \param lattice Lattice
/// \param coordinate_cart Cartesian coordinates, as columns of a matrix
Eigen::MatrixXd cartesian_to_fractional(
    xtal::Lattice const &lattice, Eigen::MatrixXd const &coordinate_cart) {
  return lattice.inv_lat_column_mat() * coordinate_cart;
}

/// \brief Translate fractional coordinates within the lattice unit cell
///
/// \param lattice Lattice
/// \param coordinate_frac Fractional coordinates, as columns of a matrix
Eigen::MatrixXd fractional_within(xtal::Lattice const &lattice,
                                  Eigen::MatrixXd coordinate_frac) {
  double tshift;
  for (Index col = 0; col < coordinate_frac.cols(); ++col) {
    for (Index i = 0; i < 3; i++) {
      tshift = floor(coordinate_frac(i, col) + 1E-6);
      if (!almost_zero(tshift, TOL)) {
        coordinate_frac(i, col) -= tshift;
      }
    }
  }
  return coordinate_frac;
}

// {"axis_names", "basis"}
// typedef std::pair<std::vector<std::string>, Eigen::MatrixXd> DoFSetBasis;

struct DoFSetBasis {
  DoFSetBasis(
      std::string const &_dofname,
      std::vector<std::string> const &_axis_names = std::vector<std::string>{},
      Eigen::MatrixXd const &_basis = Eigen::MatrixXd(0, 0))
      : dofname(_dofname), axis_names(_axis_names), basis(_basis) {}

  /// The type of DoF
  std::string dofname;

  /// A name for each basis vector (i.e. for each column of basis).
  std::vector<std::string> axis_names;

  /// Basis vectors, as columns of a matrix, such that `x_standard = basis *
  /// x_prim`. If `basis.cols() == 0`, the standard basis will be used when
  /// constructing a prim.
  Eigen::MatrixXd basis;
};

std::string get_dofsetbasis_dofname(DoFSetBasis const &dofsetbasis) {
  return dofsetbasis.dofname;
}
std::vector<std::string> get_dofsetbasis_axis_names(
    DoFSetBasis const &dofsetbasis) {
  return dofsetbasis.axis_names;
}

Eigen::MatrixXd get_dofsetbasis_basis(DoFSetBasis const &dofsetbasis) {
  return dofsetbasis.basis;
}

/// \brief Construct DoFSetBasis
///
/// \param dofname DoF name. Must be a CASM-supported DoF type.
/// \param axis_names DoFSet axis names. Size equals number of columns in basis.
/// \param basis Basis vectors, as columns of a matrix, such that `x_standard =
/// basis * x_prim`. If `basis.cols() == 0`, the standard basis will be used.
///
DoFSetBasis make_dofsetbasis(
    std::string dofname,
    std::vector<std::string> const &axis_names = std::vector<std::string>{},
    Eigen::MatrixXd const &basis = Eigen::MatrixXd(0, 0)) {
  return DoFSetBasis(dofname, axis_names, basis);
}

std::map<std::string, xtal::SpeciesProperty> make_species_properties(
    std::map<std::string, Eigen::MatrixXd> species_properties) {
  std::map<std::string, xtal::SpeciesProperty> result;
  for (auto const &pair : species_properties) {
    result.emplace(pair.first, xtal::SpeciesProperty{AnisoValTraits(pair.first),
                                                     pair.second});
  }
  return result;
}

xtal::AtomPosition make_atom_position(
    std::string name, Eigen::Vector3d pos,
    std::map<std::string, Eigen::MatrixXd> properties = {}) {
  xtal::AtomPosition atom(pos, name);
  atom.set_properties(make_species_properties(properties));
  return atom;
}

std::map<std::string, Eigen::MatrixXd> get_atom_position_properties(
    xtal::AtomPosition const &atom) {
  std::map<std::string, Eigen::MatrixXd> result;
  for (auto const &pair : atom.properties()) {
    result.emplace(pair.first, pair.second.value());
  }
  return result;
}

xtal::Molecule make_molecule(
    std::string name, std::vector<xtal::AtomPosition> atoms = {},
    bool divisible = false,
    std::map<std::string, Eigen::MatrixXd> properties = {}) {
  xtal::Molecule mol(name, atoms, divisible);
  mol.set_properties(make_species_properties(properties));
  return mol;
}

std::map<std::string, Eigen::MatrixXd> get_molecule_properties(
    xtal::Molecule const &mol) {
  std::map<std::string, Eigen::MatrixXd> result;
  for (auto const &pair : mol.properties()) {
    result.emplace(pair.first, pair.second.value());
  }
  return result;
}

/// \brief Construct xtal::BasicStructure from JSON string
xtal::BasicStructure basicstructure_from_json(std::string const &prim_json_str,
                                              double xtal_tol) {
  jsonParser json{prim_json_str};
  ParsingDictionary<AnisoValTraits> const *aniso_val_dict = nullptr;
  return read_prim(json, xtal_tol, aniso_val_dict);
}

/// \brief Format xtal::BasicStructure as JSON string
std::string basicstructure_to_json(xtal::BasicStructure const &prim) {
  jsonParser json;
  write_prim(prim, json, FRAC);
  std::stringstream ss;
  ss << json;
  return ss.str();
}

xtal::BasicStructure make_basicstructure(
    xtal::Lattice const &lattice, Eigen::MatrixXd const &coordinate_frac,
    std::vector<std::vector<std::string>> const &occ_dof,
    std::vector<std::vector<DoFSetBasis>> const &local_dof,
    std::vector<DoFSetBasis> const &global_dof,
    std::map<std::string, xtal::Molecule> const &molecules,
    std::string title = std::string("prim")) {
  // validation
  if (coordinate_frac.rows() != 3) {
    throw std::runtime_error(
        "Error in make_basicstructure: coordinate_frac.rows() != 3");
  }
  if (coordinate_frac.cols() != Index(occ_dof.size())) {
    throw std::runtime_error(
        "Error in make_basicstructure: coordinate_frac.cols() != "
        "occ_dof.size()");
  }
  if (local_dof.size() && coordinate_frac.cols() != Index(local_dof.size())) {
    throw std::runtime_error(
        "Error in make_basicstructure: local_dof.size() && "
        "coordinate_frac.cols() != occ_dof.size()");
  }

  // construct prim
  xtal::BasicStructure prim{lattice};
  prim.set_title(title);

  // set basis sites
  for (Index b = 0; b < coordinate_frac.cols(); ++b) {
    xtal::Coordinate coord{coordinate_frac.col(b), prim.lattice(), FRAC};
    std::vector<xtal::Molecule> site_occ;
    for (std::string label : occ_dof[b]) {
      if (molecules.count(label)) {
        site_occ.push_back(molecules.at(label));
      } else {
        site_occ.push_back(xtal::Molecule{label});
      }
    }

    std::vector<xtal::SiteDoFSet> site_dofsets;
    if (local_dof.size()) {
      for (auto const &dofsetbasis : local_dof[b]) {
        if (dofsetbasis.basis.cols()) {
          site_dofsets.emplace_back(AnisoValTraits(dofsetbasis.dofname),
                                    dofsetbasis.axis_names, dofsetbasis.basis,
                                    std::unordered_set<std::string>{});
        } else {
          site_dofsets.emplace_back(AnisoValTraits(dofsetbasis.dofname));
        }
      }
    }

    xtal::Site site{coord, site_occ, site_dofsets};
    prim.push_back(site, FRAC);
  }
  prim.set_unique_names(occ_dof);

  // set global dof
  std::vector<xtal::DoFSet> global_dofsets;
  for (auto const &dofsetbasis : global_dof) {
    if (dofsetbasis.basis.cols()) {
      global_dofsets.emplace_back(AnisoValTraits(dofsetbasis.dofname),
                                  dofsetbasis.axis_names, dofsetbasis.basis);
    } else {
      global_dofsets.emplace_back(AnisoValTraits(dofsetbasis.dofname));
    }
  }

  prim.set_global_dofs(global_dofsets);

  return prim;
}

Eigen::MatrixXd get_basicstructure_coordinate_frac(
    xtal::BasicStructure const &prim) {
  Eigen::MatrixXd coordinate_frac(3, prim.basis().size());
  Index b = 0;
  for (auto const &site : prim.basis()) {
    coordinate_frac.col(b) = site.const_frac();
    ++b;
  }
  return coordinate_frac;
}

Eigen::MatrixXd get_basicstructure_coordinate_cart(
    xtal::BasicStructure const &prim) {
  return prim.lattice().lat_column_mat() *
         get_basicstructure_coordinate_frac(prim);
}

std::vector<std::vector<std::string>> get_basicstructure_occ_dof(
    xtal::BasicStructure const &prim) {
  return xtal::allowed_molecule_names(prim);
}

std::vector<std::vector<DoFSetBasis>> get_basicstructure_local_dof(
    xtal::BasicStructure const &prim) {
  std::vector<std::vector<DoFSetBasis>> local_dof;
  Index b = 0;
  for (auto const &site : prim.basis()) {
    std::vector<DoFSetBasis> site_dof;
    for (auto const &pair : site.dofs()) {
      std::string const &dofname = pair.first;
      xtal::SiteDoFSet const &dofset = pair.second;
      site_dof.emplace_back(dofname, dofset.component_names(), dofset.basis());
    }
    local_dof.push_back(site_dof);
    ++b;
  }
  return local_dof;
}

std::vector<DoFSetBasis> get_basicstructure_global_dof(
    xtal::BasicStructure const &prim) {
  std::vector<DoFSetBasis> global_dof;
  for (auto const &pair : prim.global_dofs()) {
    std::string const &dofname = pair.first;
    xtal::DoFSet const &dofset = pair.second;
    global_dof.emplace_back(dofname, dofset.component_names(), dofset.basis());
  }
  return global_dof;
}

std::map<std::string, xtal::Molecule> get_basicstructure_molecules(
    xtal::BasicStructure const &prim) {
  std::map<std::string, xtal::Molecule> molecules;
  std::vector<std::vector<std::string>> mol_names = prim.unique_names();
  if (mol_names.empty()) {
    mol_names = xtal::allowed_molecule_unique_names(prim);
  }
  Index b = 0;
  for (auto const &site_mol_names : mol_names) {
    Index i = 0;
    for (auto const &name : site_mol_names) {
      if (!molecules.count(name)) {
        molecules.emplace(name, prim.basis()[b].occupant_dof()[i]);
      }
      ++i;
    }
    ++b;
  }
  return molecules;
}

xtal::BasicStructure make_within(xtal::BasicStructure prim) {
  prim.within();
  return prim;
}

xtal::BasicStructure make_primitive(xtal::BasicStructure prim) {
  return xtal::make_primitive(prim, prim.lattice().tol());
}

xtal::BasicStructure make_canonical_basicstructure(xtal::BasicStructure prim) {
  xtal::Lattice lattice{prim.lattice()};
  lattice.make_right_handed();
  lattice = xtal::canonical::equivalent(lattice);
  prim.set_lattice(xtal::canonical::equivalent(lattice), CART);
  return prim;
}

std::vector<std::vector<Index>> asymmetric_unit_indices(
    xtal::BasicStructure prim) {
  // Note: pybind11 doesn't nicely convert sets of set,
  // so return vector of vector, which is converted to List[List[int]]
  std::vector<std::vector<Index>> result;
  std::set<std::set<Index>> asym_unit = make_asymmetric_unit(prim);
  for (auto const orbit : asym_unit) {
    result.push_back(std::vector<Index>(orbit.begin(), orbit.end()));
  }
  return result;
}

}  // namespace CASMpy

PYBIND11_MODULE(xtal, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        casm.xtal
        ---------

        The casm.xtal module is a Python interface to the crystallography
        classes and methods in the CASM::xtal namespace of the CASM C++ libraries.
        This includes:

        - Data structures for representing coordinates, lattices,
          degrees of freedom (DoF), and crystal structures.
        - Methods for enumerating lattices, making super structures,
          finding primitive and reduced cells, and finding symmetry
          operations.

    )pbdoc";

  m.attr("TOL") = TOL;

  py::class_<xtal::Lattice>(m, "Lattice", R"pbdoc(
      A 3-dimensional lattice
      )pbdoc")
      .def(py::init<Eigen::Matrix3d const &, double, bool>(),
           "Construct a Lattice", py::arg("column_vector_matrix"),
           py::arg("tol") = TOL, py::arg("force") = false, R"pbdoc(

      Parameters
      ----------
      column_vector_matrix : array_like, shape=(3,3)
          The lattice vectors, as columns of a 3x3 matrix.
      tol : float, default=xtal.TOL
          Tolerance to be used for crystallographic comparisons.
      )pbdoc")
      .def("column_vector_matrix", &xtal::Lattice::lat_column_mat,
           "Return the lattice vectors, as columns of a 3x3 matrix.")
      .def("tol", &xtal::Lattice::tol,
           "Return the tolerance used for crystallographic comparisons.")
      .def("set_tol", &xtal::Lattice::set_tol, py::arg("tol"),
           "Set the tolerance used for crystallographic comparisons.")
      .def("make_canonical", &make_canonical_lattice, R"pbdoc(
        Return the canonical equivalent lattice

        Finds the canonical right-handed Niggli cell of the lattice, applying
        lattice point group operations to find the equivalent lattice in a
        standardized orientation. The canonical orientation prefers lattice
        vectors that form symmetric matrices with large positive values on the
        diagonal and small values off the diagonal. See also `Lattice Canonical Form`_.

        Notes
        -----
        The returned lattice is not canonical in the context of Prim supercell
        lattices, in which case the crystal point group must be used in
        determining the canonical orientation of the supercell lattice.

        .. _`Lattice Canonical Form`: https://prisms-center.github.io/CASMcode_docs/formats/lattice_canonical_form/

        )pbdoc")
      .def("fractional_to_cartesian", &fractional_to_cartesian,
           py::arg("coordinate_frac"), R"pbdoc(
        Convert fractional coordinates to Cartesian coordinates

        The result is equal to:

            lattice.column_vector_matrix() @ coordinate_frac

        Parameters
        ----------
        coordinate_frac : numpy.ndarray[numpy.float64[3, n]]
            Coordinates, as columns of a matrix, in fractional coordinates
            with respect to the lattice vectors.

        Returns
        -------
        coordinate_cart : numpy.ndarray[numpy.float64[3, n]]
            Coordinates, as columns of a matrix, in Cartesian coordinates.
        )pbdoc")
      .def("cartesian_to_fractional", &cartesian_to_fractional,
           py::arg("coordinate_cart"), R"pbdoc(
        Convert Cartesian coordinates to fractional coordinates

        The result is equal to:

            np.linalg.pinv(lattice.column_vector_matrix()) @ coordinate_cart

        Parameters
        ----------
        coordinate_cart : numpy.ndarray[numpy.float64[3, n]]
            Coordinates, as columns of a matrix, in Cartesian coordinates.

        Returns
        -------
        coordinate_frac : numpy.ndarray[numpy.float64[3, n]]
            Coordinates, as columns of a matrix, in fractional coordinates
            with respect to the lattice vectors.
        )pbdoc")
      .def("fractional_within", &fractional_within,
           py::arg("init_coordinate_frac"), R"pbdoc(
        Translate fractional coordinates within the lattice unit cell

        Parameters
        ----------
        init_coordinate_frac : numpy.ndarray[numpy.float64[3, n]]
            Coordinates, as columns of a matrix, in fractional coordinates
            with respect to the lattice vectors.

        Returns
        -------
        coordinate_frac : numpy.ndarray[numpy.float64[3, n]]
            Coordinates, as columns of a matrix, in fractional coordinates
            with respect to the lattice vectors, translatd within the
            lattice unit cell.
        )pbdoc");

  py::class_<xtal::AtomPosition>(m, "AtomComponent", R"pbdoc(
      An atomic component of a molecular :class:`~casm.xtal.Occupant`
      )pbdoc")
      .def(py::init(&make_atom_position), py::arg("name"),
           py::arg("coordinate"), py::arg("properties"), R"pbdoc(

      Parameters
      ----------
      name : str
          A \"chemical name\", which must be identical for atoms to
          be found symmetrically equivalent. The names are case
          sensitive, and "Va" is reserved for vacancies.
      coordinate : array_like, shape (3,)
          Position of the atom, in Cartesian coordinates, relative
          to the basis site at which the occupant containing this
          atom is placed.
      properties : Dict[str, array_like]
          Fixed properties of the atom, such as magnetic sping or
          selective dynamics flags. Keys must be the name of a
          CASM-supported property type. Values are arrays with
          dimensions matching the standard dimension of the property
          type.

          See the CASM `Degrees of Freedom (DoF) and Properties`_
          documentation for the full list of supported properites and their
          definitions.

          .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/
      )pbdoc")
      .def("name", &xtal::AtomPosition::name,
           "Return the \"chemical name\" of the atom.")
      .def("coordinate", &xtal::AtomPosition::cart, R"pbdoc(
           Return the position of the atom

           The osition is in Cartesian coordinates, relative to the
           basis site at which the occupant containing this atom
           is placed.
           )pbdoc")
      .def("properties", &get_atom_position_properties,
           "Return the fixed properties of the atom");

  py::class_<xtal::Molecule>(m, "Occupant", R"pbdoc(
      A site occupant, which may be a vacancy, atom, or molecule

      The Occupant class is used to represent all chemical species,
      including single atoms, vacancies, and molecules.

      )pbdoc")
      .def(py::init(&make_molecule), py::arg("name"),
           py::arg("atoms") = std::vector<xtal::AtomPosition>{},
           py::arg("is_divisible") = false,
           py::arg("properties") = std::map<std::string, Eigen::MatrixXd>{},
           R"pbdoc(

      Parameters
      ----------
      name : str
          A \"chemical name\", which must be identical for occupants to
          be found symmetrically equivalent. The names are case
          sensitive, and "Va" is reserved for vacancies.
      atoms : List[casm.xtal.AtomComponent], optional
          The atomic components of a molecular occupant. Atoms and
          vacancies are represented with a single AtomComponent with the
          same name for the Occupant and the AtomComponent. If atoms is
          an empty list (the default value), then an atom or vacancy is
          created, based on the name parameter.
      is_divisible : bool, default=False
          If True, indicates an Occupant that may split into components
          during kinetic Monte Carlo calculations.
      properties : Dict[str, array_like], default={}
          Fixed properties of the occupant, such as magnetic
          spin or selective dynamics flags. Keys must be the name of a
          CASM-supported property type. Values are arrays with
          dimensions matching the standard dimension of the property
          type.

          See the CASM `Degrees of Freedom (DoF) and Properties`_
          documentation for the full list of supported properites and their
          definitions.

          .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/
      )pbdoc")
      .def("name", &xtal::Molecule::name,
           "The \"chemical name\" of the occupant")
      .def("is_vacancy", &xtal::Molecule::is_vacancy, "True if a vacancy.")
      .def("is_atomic", &xtal::Molecule::is_atomic,
           "True if a single isotropic atom or vacancy")
      .def("is_divisible", &xtal::Molecule::is_divisible,
           "True if is divisible in kinetic Monte Carlo calculations")
      .def("atoms", &xtal::Molecule::atoms,
           "Return the atomic components of the occupant")
      .def("atom", &xtal::Molecule::atom, py::arg("i"),
           "Return the `i`-th component atom")
      .def("properties", &get_molecule_properties,
           "Return the fixed properties of the occupant");

  m.def("make_vacancy", &xtal::Molecule::make_vacancy, R"pbdoc(
      Construct a Occupant object representing a vacancy

      This function is equivalent to ``casm.xtal.Occupant("Va")``.
      )pbdoc");
  m.def("make_atom", &xtal::Molecule::make_atom, py::arg("name"), R"pbdoc(
      Construct a Occupant object representing a single isotropic atom

      This function is equivalent to ``casm.xtal.Occupant(name)``.

      Parameters
      ----------
      name : str
          A \"chemical name\", which must be identical for occupants
          to be found symmetrically equivalent. The names are case
          sensitive, and "Va" is reserved for vacancies.
      )pbdoc");

  py::class_<DoFSetBasis>(m, "DoFSetBasis", R"pbdoc(
      The basis for a set of degrees of freedom (DoF)

      Degrees of freedom (DoF) are continuous-valued vectors having a
      standard basis that is related to the fixed reference frame of
      the crystal. CASM supports both site DoF, which are associated
      with a particular prim basis site, and global DoF, which are
      associated with the infinite crystal. Standard DoF types are
      implemented in CASM and a traits system allows developers to
      extend CASM to include additional types of DoF.

      In many cases, the standard basis is the appropriate choice, but
      CASM also allows for a user-specified basis in terms of the
      standard basis. A user-specified basis may fully span the
      standard basis or only a subspace. This allows:

      - restricting strain to a subspace, such as only volumetric or
        only shear strains
      - restricting displacements to a subspace, such as only within
        a particular plane
      - reorienting DoF, such as to use symmetry-adapted strain order
        parameters

      See the CASM `Degrees of Freedom (DoF) and Properties`_
      documentation for the full list of supported DoF types and their
      definitions. Some examples:

      - `"disp"`: Atomic displacement
      - `"EAstrain"`: Euler-Almansi strain metric
      - `"GLstrain"`: Green-Lagrange strain metric
      - `"Hstrain"`: Hencky strain metric
      - `"Cmagspin"`: Collinear magnetic spin
      - `"SOmagspin"`: Non-collinear magnetic spin, with spin-orbit coupling

      .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/
      )pbdoc")
      .def(py::init(&make_dofsetbasis), py::arg("dofname"),
           py::arg("axis_names") = std::vector<std::string>{},
           py::arg("basis") = Eigen::MatrixXd(0, 0), R"pbdoc(

      Parameters
      ----------
      dofname : str
          The type of DoF. Must be a CASM supported DoF type.
      basis : numpy.ndarray[numpy.float64[m, n]], default=numpy.ndarray[numpy.float64[1, 0]]
          User-specified DoF basis vectors, as columns of a matrix. The
          DoF values in this basis, `x_prim`, are related to the DoF
          values in the CASM standard basis, `x_standard`, according to
          `x_standard = basis * x_prim`. The number of rows in the basis
          matrix must match the standard dimension of the CASM DoF type.
          The number of columns must be less than or equal to the number
          of rows. The default value indicates the standard basis should
          be used.
      axis_names : List[str], default=[]
          Names for the DoF basis vectors (i.e. names for the basis matrix
          columns). Size must match number of columns in the basis matrix.
          The axis names should be appropriate for use in latex basis
          function formulas. Example, for ``dofname="disp"``:

              axis_names=["d_{1}", "d_{2}", "d_{3}"]

          The default value indicates the standard basis should be used.
      )pbdoc")
      .def("dofname", &get_dofsetbasis_dofname, "Return the DoF type name.")
      .def("axis_names", &get_dofsetbasis_axis_names, "Return the axis names.")
      .def("basis", &get_dofsetbasis_basis, "Return the basis matrix.");

  py::class_<xtal::BasicStructure>(m, "Prim", R"pbdoc(
      A primitive crystal structure and allowed degrees of freedom (DoF) (the `"Prim"`)

      The Prim specifies:

      - lattice vectors
      - crystal basis sites
      - occupation DoF,
      - continuous local (site) DoF
      - continuous global DoF.

      It is usually best practice for the Prim to be an actual primitive
      cell, but it is not forced to be. The actual primitive cell will
      have a factor group with the minimum number of symmetry operations,
      which will result in more efficient methods. Some methods may have
      unexpected results when using a non-primitive Prim.

      The :func:`~casm.xtal.Prim.make_primitive` method may be used to find
      the primitive equivalent, and the :func:`~casm.xtal.Prim.make_canonical`
      method may be used to find the equivalent with a Niggli cell lattice
      aligned in a CASM standard direction.
      )pbdoc")
      .def(py::init(&make_basicstructure), py::arg("lattice"),
           py::arg("coordinate_frac"), py::arg("occ_dof"),
           py::arg("local_dof") = std::vector<std::vector<DoFSetBasis>>{},
           py::arg("global_dof") = std::vector<DoFSetBasis>{},
           py::arg("occupants") = std::map<std::string, xtal::Molecule>{},
           py::arg("title") = std::string("prim"),
           R"pbdoc(

      Parameters
      ----------
      lattice : Lattice
          The primitive cell Lattice.
      coordinate_frac : numpy.ndarray[numpy.float64[3, n]]
          Basis site positions, as columns of a matrix, in fractional
          coordinates with respect to the lattice vectors.
      occ_dof : List[List[str]]
          Labels of occupants allowed on each basis site. The value
          occ_dof[b] is the list of occupants allowed on the `b`-th basis
          site. The values may either be (i) the name of an isotropic atom
          (i.e. "Mg") or vacancy ("Va"), or (ii) a key in the occupants
          dictionary (i.e. "H2O", or "H2_xx"). The names are case
          sensitive, and "Va" is reserved for vacancies.
      local_dof : List[List[DoFSetBasis]], default=[[]]
          Continuous DoF allowed on each basis site. No effect if empty.
          If not empty, the value local_dof[b] is a list of :class:`DoFSetBasis`
          objects describing the DoF allowed on the `b`-th basis site.
      global_dof : List[DoFSetBasis], default=[]
          Global continuous DoF allowed for the entire crystal.
      occupants : Dict[str, Occupant], default=[]
          :class:`Occupant` allowed in the crystal. The keys are labels
          used in the occ_dof parameter. This may include isotropic
          atoms, vacancies, atoms with fixed anisotropic properties, and
          molecular occupants. A seperate key and value is required for
          all species with distinct anisotropic properties (i.e. "H2_xy",
          "H2_xz", and "H2_yz" for distinct orientations, or "Aup", and
          "Adown" for distinct collinear magnetic spins, etc.).
      title : str, default="prim"
          A title for the prim. When the prim is used to construct a
          cluster expansion, this must consist of alphanumeric characters
          and underscores only. The first character may not be a number.
      )pbdoc")
      .def("lattice", &xtal::BasicStructure::lattice, "Return the lattice")
      .def("coordinate_frac", &get_basicstructure_coordinate_frac,
           "Return the basis site positions, as columns of a matrix, in "
           "fractional coordinates with respect to the lattice vectors")
      .def("coordinate_cart", &get_basicstructure_coordinate_cart,
           "Return the basis site positions, as columns of a matrix, in "
           "Cartesian coordinates")
      .def("occ_dof", &get_basicstructure_occ_dof,
           "Return the labels of occupants allowed on each basis site")
      .def("local_dof", &get_basicstructure_local_dof,
           "Return the continuous DoF allowed on each basis site")
      .def("global_dof", &get_basicstructure_global_dof,
           "Return the continuous DoF allowed for the entire crystal structure")
      .def("occupants", &get_basicstructure_molecules,
           "Return the :class:`Occupant` allowed in the crystal.")
      .def_static(
          "from_json", &basicstructure_from_json,
          "Construct a Prim from a JSON-formatted string. The `Prim reference "
          "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
          "crystallography/BasicStructure/>`_ documents the expected JSON "
          "format.",
          py::arg("prim_json_str"), py::arg("xtal_tol") = TOL)
      .def("to_json", &basicstructure_to_json,
           "Represent the Prim as a JSON-formatted string. The `Prim reference "
           "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
           "crystallography/BasicStructure/>`_ documents the expected JSON "
           "format.")
      .def("make_within", &make_within, R"pbdoc(
            Return an equivalent Prim with all basis sites within the unit cell
            )pbdoc")
      .def("make_primitive", &make_primitive, R"pbdoc(
            Return a primitive equivalent Prim

            A :class:`Prim` object is not forced to be the primitive equivalent
            cell at construction. This function finds and returns the primitive
            equivalent cell by checking for internal translations that map all
            basis sites onto equivalent basis sites, including allowed
            occupants and equivalent local degrees of freedom (DoF), if they
            exist.
            )pbdoc")
      .def("make_canonical", &make_canonical_basicstructure, R"pbdoc(
          Return an equivalent Prim with canonical lattice

          Finds the canonical right-handed Niggli cell of the lattice, applying
          lattice point group operations to find the equivalent lattice in a
          standardized orientation. The canonical orientation prefers lattice
          vectors that form symmetric matrices with large positive values on the
          diagonal and small values off the diagonal. See also `Lattice Canonical Form`_.

          .. _`Lattice Canonical Form`: https://prisms-center.github.io/CASMcode_docs/formats/lattice_canonical_form/

          )pbdoc")
      .def("asymmetric_unit_indices", asymmetric_unit_indices, R"pbdoc(
          Return the indices of equivalent basis sites

          Returns
          -------
          asymmetric_unit_indices : List[List[int]]
              One list of basis site indices for each set of symmetrically equivalent basis sites.
              In other words, the elements of asymmetric_unit_indices[i] are the indices of the
              i-th set of basis sites which are symmetrically equivalent to each other.

          )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
