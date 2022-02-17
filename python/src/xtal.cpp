#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/BasicStructureTools.hh"
#include "casm/crystallography/CanonicalForm.hh"
#include "casm/crystallography/LatticeIsEquivalent.hh"
#include "casm/crystallography/SuperlatticeEnumerator.hh"
#include "casm/crystallography/SymInfo.hh"
#include "casm/crystallography/SymTools.hh"
#include "casm/crystallography/io/BasicStructureIO.hh"
#include "casm/crystallography/io/SymInfo_json_io.hh"
#include "casm/crystallography/io/SymInfo_stream_io.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

// xtal

double default_tol() { return TOL; }

// Lattice

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

std::vector<xtal::SymOp> make_lattice_point_group(
    xtal::Lattice const &lattice) {
  return xtal::make_point_group(lattice);
}

std::vector<xtal::Lattice> enumerate_superlattices(
    xtal::Lattice const &unit_lattice,
    std::vector<xtal::SymOp> const &point_group, Index max_volume,
    Index min_volume = 1, std::string dirs = std::string("abc")) {
  xtal::ScelEnumProps enum_props{min_volume, max_volume + 1, dirs};
  xtal::SuperlatticeEnumerator enumerator{unit_lattice, point_group,
                                          enum_props};
  std::vector<xtal::Lattice> superlattices;
  for (auto const &superlat : enumerator) {
    superlattices.push_back(
        xtal::canonical::equivalent(superlat, point_group, unit_lattice.tol()));
  }
  return superlattices;
}

std::pair<bool, Eigen::Matrix3d> is_superlattice_of(
    xtal::Lattice const &superlattice, xtal::Lattice const &unit_lattice) {
  double tol = std::max(superlattice.tol(), unit_lattice.tol());
  return xtal::is_superlattice(superlattice, unit_lattice, tol);
}

Eigen::Matrix3l make_transformation_matrix_to_super(
    xtal::Lattice const &superlattice, xtal::Lattice const &unit_lattice) {
  double tol = std::max(superlattice.tol(), unit_lattice.tol());
  return xtal::make_transformation_matrix_to_super(unit_lattice, superlattice,
                                                   tol);
}

/// \brief Check if S = point_group[point_group_index] * L * T, with integer T
///
/// \returns (is_equivalent, T, point_group_index)
std::tuple<bool, Eigen::MatrixXd, Index> is_equivalent_superlattice_of(
    xtal::Lattice const &superlattice, xtal::Lattice const &unit_lattice,
    std::vector<xtal::SymOp> const &point_group = std::vector<xtal::SymOp>{}) {
  double tol = std::max(superlattice.tol(), unit_lattice.tol());
  auto result = is_equivalent_superlattice(
      superlattice, unit_lattice, point_group.begin(), point_group.end(), tol);
  bool is_equivalent = (result.first != point_group.end());
  Index point_group_index = -1;
  if (is_equivalent) {
    point_group_index = std::distance(point_group.begin(), result.first);
  }
  return std::tuple<bool, Eigen::MatrixXd, Index>(is_equivalent, result.second,
                                                  point_group_index);
}

xtal::Lattice make_superduperlattice(
    std::vector<xtal::Lattice> const &lattices,
    std::string mode = std::string("commensurate"),
    std::vector<xtal::SymOp> const &point_group = std::vector<xtal::SymOp>{}) {
  if (mode == "commensurate") {
    return xtal::make_commensurate_superduperlattice(lattices.begin(),
                                                     lattices.end());
  } else if (mode == "minimal_commensurate") {
    return xtal::make_minimal_commensurate_superduperlattice(
        lattices.begin(), lattices.end(), point_group.begin(),
        point_group.end());
  } else if (mode == "fully_commensurate") {
    return xtal::make_fully_commensurate_superduperlattice(
        lattices.begin(), lattices.end(), point_group.begin(),
        point_group.end());
  } else {
    std::stringstream msg;
    msg << "Error in make_superduperlattice: Unrecognized mode=" << mode;
    throw std::runtime_error(msg.str());
  }
}

// DoFSetBasis

struct DoFSetBasis {
  DoFSetBasis(
      std::string const &_dofname,
      std::vector<std::string> const &_axis_names = std::vector<std::string>{},
      Eigen::MatrixXd const &_basis = Eigen::MatrixXd(0, 0))
      : dofname(_dofname), axis_names(_axis_names), basis(_basis) {
    if (Index(axis_names.size()) != basis.cols()) {
      throw std::runtime_error(
          "Error in DoFSetBasis::DoFSetBasis(): axis_names.size() != "
          "basis.cols()");
    }
    if (axis_names.size() == 0) {
      axis_names = CASM::AnisoValTraits(dofname).standard_var_names();
      Index dim = axis_names.size();
      basis = Eigen::MatrixXd::Identity(dim, dim);
    }
  }

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

// SpeciesProperty -> properties

std::map<std::string, xtal::SpeciesProperty> make_species_properties(
    std::map<std::string, Eigen::MatrixXd> species_properties) {
  std::map<std::string, xtal::SpeciesProperty> result;
  for (auto const &pair : species_properties) {
    result.emplace(pair.first, xtal::SpeciesProperty{AnisoValTraits(pair.first),
                                                     pair.second});
  }
  return result;
}

// AtomComponent

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

// Occupant

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

// Prim

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
  for (auto const &orbit : asym_unit) {
    result.push_back(std::vector<Index>(orbit.begin(), orbit.end()));
  }
  return result;
}

std::vector<xtal::SymOp> make_factor_group(xtal::BasicStructure const &prim) {
  return xtal::make_factor_group(prim);
}

std::vector<xtal::SymOp> make_crystal_point_group(
    xtal::BasicStructure const &prim) {
  auto fg = xtal::make_factor_group(prim);
  return xtal::make_crystal_point_group(fg, prim.lattice().tol());
}

// SymOp

xtal::SymOp make_symop(Eigen::Matrix3d const &matrix,
                       Eigen::Vector3d const &translation, bool time_reversal) {
  return xtal::SymOp(matrix, translation, time_reversal);
}

std::string symop_to_json(xtal::SymOp const &op, xtal::Lattice const &lattice) {
  jsonParser json;
  to_json(op.matrix, json["matrix"]);
  to_json_array(op.translation, json["translation"]);
  to_json(op.is_time_reversal_active, json["time_reversal"]);

  std::stringstream ss;
  ss << json;
  return ss.str();
}

// SymInfo

xtal::SymInfo make_syminfo(xtal::SymOp const &op,
                           xtal::Lattice const &lattice) {
  return xtal::SymInfo(op, lattice);
}

std::string get_syminfo_type(xtal::SymInfo const &syminfo) {
  return to_string(syminfo.op_type);
}

Eigen::Vector3d get_syminfo_axis(xtal::SymInfo const &syminfo) {
  return syminfo.axis.const_cart();
}

double get_syminfo_angle(xtal::SymInfo const &syminfo) { return syminfo.angle; }

Eigen::Vector3d get_syminfo_screw_glide_shift(xtal::SymInfo const &syminfo) {
  return syminfo.screw_glide_shift.const_cart();
}

Eigen::Vector3d get_syminfo_location(xtal::SymInfo const &syminfo) {
  return syminfo.location.const_cart();
}

std::string get_syminfo_brief_cart(xtal::SymInfo const &syminfo) {
  return to_brief_unicode(syminfo, xtal::SymInfoOptions(CART));
}

std::string get_syminfo_brief_frac(xtal::SymInfo const &syminfo) {
  return to_brief_unicode(syminfo, xtal::SymInfoOptions(FRAC));
}

std::string syminfo_to_json(xtal::SymInfo const &syminfo) {
  jsonParser json;
  to_json(syminfo, json);

  to_json(to_brief_unicode(syminfo, xtal::SymInfoOptions(CART)),
          json["brief"]["CART"]);
  to_json(to_brief_unicode(syminfo, xtal::SymInfoOptions(FRAC)),
          json["brief"]["FRAC"]);

  std::stringstream ss;
  ss << json;
  return ss.str();
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

        - Data structures for representing lattices, crystal structures, and
          degrees of freedom (DoF).
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
      .def(py::self < py::self,
           "Sorts lattices by how canonical the lattice vectors are")
      .def(py::self <= py::self,
           "Sort lattices by how canonical the lattice vectors are")
      .def(py::self > py::self,
           "Sort lattices by how canonical the lattice vectors are")
      .def(py::self >= py::self,
           "Sort lattices by how canonical the lattice vectors are")
      .def(py::self == py::self,
           "True if lattice vectors are approximately equal")
      .def(py::self != py::self,
           "True if lattice vectors are not approximately equal");

  m.def("make_canonical_lattice", &make_canonical_lattice, py::arg("lattice"),
        R"pbdoc(
    Return the canonical equivalent lattice

    Finds the canonical right-handed Niggli cell of the lattice, applying
    lattice point group operations to find the equivalent lattice in a
    standardized orientation. The canonical orientation prefers lattice
    vectors that form symmetric matrices with large positive values on the
    diagonal and small values off the diagonal. See also `Lattice Canonical
    Form`_.

    Notes
    -----
    The returned lattice is not canonical in the context of Prim supercell
    lattices, in which case the crystal point group must be used in
    determining the canonical orientation of the supercell lattice.

    .. _`Lattice Canonical Form`:
    https://prisms-center.github.io/CASMcode_docs/formats/lattice_canonical_form/

    Parameters
    ----------
    init_lattice : casm.xtal.Lattice
        The initial lattice.

    Returns
    ----------
    lattice : casm.xtal.Lattice
        The canonical equivalent lattice, using the lattice point group.

  )pbdoc");

  m.def("make_canonical", &make_canonical_lattice, py::arg("init_lattice"),
        "Equivalent to :func:`~casm.xtal.make_canonical_lattice`");

  m.def("fractional_to_cartesian", &fractional_to_cartesian, py::arg("lattice"),
        py::arg("coordinate_frac"), R"pbdoc(
    Convert fractional coordinates to Cartesian coordinates

    The result is equal to:

    .. code-block:: Python

        lattice.column_vector_matrix() @ coordinate_frac

    Parameters
    ----------
    lattice : casm.xtal.Lattice
        The lattice.
    coordinate_frac : array_like, shape (3, n)
        Coordinates, as columns of a matrix, in fractional coordinates
        with respect to the lattice vectors.

    Returns
    -------
    coordinate_cart : numpy.ndarray[numpy.float64[3, n]]
        Coordinates, as columns of a matrix, in Cartesian coordinates.
    )pbdoc");

  m.def("cartesian_to_fractional", &cartesian_to_fractional, py::arg("lattice"),
        py::arg("coordinate_cart"), R"pbdoc(
    Convert Cartesian coordinates to fractional coordinates

    The result is equal to:

    .. code-block:: Python

        np.linalg.pinv(lattice.column_vector_matrix()) @ coordinate_cart

    Parameters
    ----------
    lattice : casm.xtal.Lattice
        The lattice.
    coordinate_cart : array_like, shape (3, n)
        Coordinates, as columns of a matrix, in Cartesian coordinates.

    Returns
    -------
    coordinate_frac : numpy.ndarray[numpy.float64[3, n]]
        Coordinates, as columns of a matrix, in fractional coordinates
        with respect to the lattice vectors.
    )pbdoc");

  m.def("fractional_within", &fractional_within, py::arg("lattice"),
        py::arg("init_coordinate_frac"), R"pbdoc(
    Translate fractional coordinates within the lattice unit cell

    Parameters
    ----------
    lattice : casm.xtal.Lattice
        The lattice.
    init_coordinate_frac : array_like, shape (3, n)
        Coordinates, as columns of a matrix, in fractional coordinates
        with respect to the lattice vectors.

    Returns
    -------
    coordinate_frac : numpy.ndarray[numpy.float64[3, n]]
        Coordinates, as columns of a matrix, in fractional coordinates
        with respect to the lattice vectors, translatd within the
        lattice unit cell.
    )pbdoc");

  m.def("make_point_group", &make_lattice_point_group, py::arg("lattice"),
        R"pbdoc(
      Return the lattice point group

      Parameters
      ----------
      lattice : casm.xtal.Lattice
          The lattice.

      Returns
      -------
      point_group : List[casm.xtal.SymOp]
          The set of rigid transformations that keep the origin fixed
          (i.e. have zero translation vector) and map the lattice (i.e.
          all points that are integer multiples of the lattice vectors)
          onto itself.
      )pbdoc");

  m.def("is_equivalent_to", &xtal::is_equivalent, py::arg("lattice1"),
        py::arg("lattice2"), R"pbdoc(
      Check if lattice1 is equivalent to lattice2

      Two lattices, L1 and L2, are equivalent (i.e. have the same
      lattice points) if there exists U such that:

      .. code-block:: Python

          L1 = L2 @ U,

      where L1 and L2 are the lattice vectors as matrix columns, and
      U is a unimodular matrix (integer matrix, with abs(det(U))==1).

      Parameters
      ----------
      lattice1 : casm.xtal.Lattice
          The first lattice.
      lattice2 : casm.xtal.Lattice
          The second lattice.

      Returns
      -------
      is_equivalent: bool
          True if lattice1 is equivalent to lattice2.
      )pbdoc");

  m.def("is_superlattice_of", &is_superlattice_of, py::arg("lattice1"),
        py::arg("lattice2"), R"pbdoc(
      Check if lattice1 is a superlattice of lattice2

      If lattice1 is a superlattice of lattice2, then

      .. code-block:: Python

          L1 = L2 @ T

      where p is the index of a point_group operation, T is an approximately
      integer tranformation matrix T, and L1 and L2 are the lattice vectors, as
      columns of a matrix, of lattice1 and lattice2, respectively.

      Parameters
      ----------
      lattice1 : casm.xtal.Lattice
          The first lattice.
      lattice2 : casm.xtal.Lattice
          The second lattice.

      Returns
      -------
      (is_superlattice_of, T): (bool, numpy.ndarray[numpy.float64[3, 3]])
          Returns tuple with a boolean that is True if lattice1 is a
          superlattice of lattice2, and the tranformation matrix T
          such that L1 = L2 @ T. Note: If is_superlattice_of==True,
          numpy.rint(T).astype(int) can be used to round array elements to
          the nearest integer.
      )pbdoc");

  m.def("is_equivalent_superlattice_of", &is_equivalent_superlattice_of,
        py::arg("lattice1"), py::arg("lattice2"),
        py::arg("point_group") = std::vector<xtal::SymOp>{}, R"pbdoc(
      Check if lattice1 is equivalent to a superlattice of lattice2

      If lattice1 is equivalent to a superlattice of lattice2, then

      .. code-block:: Python

          L1 = point_group[p].matrix() @ L2 @ T

      where p is the index of a point_group operation, T is an approximately
      integer tranformation matrix T, and L1 and L2 are the lattice vectors, as
      columns of a matrix, of lattice1 and lattice2, respectively.

      Parameters
      ----------
      lattice1 : casm.xtal.Lattice
          The first lattice.
      lattice2 : casm.xtal.Lattice
          The second lattice.
      point_group : List[casm.xtal.SymOp]
          The point group symmetry that generates equivalent lattices. Depending
          on the use case, this is often the prim crystal point group,
          :func:`~casm.xtal.make_crystal_point_group()`, or the lattice
          point group, :func:`~casm.xtal.make_point_group()`.

      Returns
      -------
      (is_equivalent_superlattice_of, T, p): (bool,
      numpy.ndarray[numpy.float64[3, 3]], int)
          Returns tuple with a boolean that is True if lattice1 is
          equivalent to a superlattice of lattice2, the
          tranformation matrix T, and point group index, p, such that L1 =
          point_group[p].matrix() @ L2 @ T. Note: If
          is_equivalent_superlattice_of==True, numpy.rint(T).astype(int) can
          be used to round array elements to the nearest integer.
      )pbdoc");

  m.def("make_transformation_matrix_to_super",
        &make_transformation_matrix_to_super, py::arg("superlattice"),
        py::arg("unit_lattice"),
        R"pbdoc(
     Return the integer transformation matrix for the superlattice relative a unit lattice.

     Parameters
     ----------
     superlattice : casm.xtal.Lattice
         The superlattice.
     unit_lattice : casm.xtal.Lattice
         The unit lattice.

     Returns
     -------
     T: numpy.ndarray[numpy.int64[3, 3]]
         Returns the integer tranformation matrix T such that S = L @ T, where S and L
         are the lattice vectors, as columns of a matrix, of the superlattice and
         unit_lattice, respectively.

     Raises
     ------
     RuntimeError:
         If superlattice is not a superlattice of unit_lattice.
     )pbdoc");

  m.def("enumerate_superlattices", &enumerate_superlattices,
        py::arg("unit_lattice"), py::arg("point_group"), py::arg("max_volume"),
        py::arg("min_volume") = Index(1), py::arg("dirs") = std::string("abc"),
        R"pbdoc(
      Enumerate symmetrically distinct superlattices

      Superlattices satify:

      .. code-block:: Python

          S = L @ T,

      where S and L are, respectively, the superlattice and unit lattice vectors as columns of
      (3x3) matrices, and T is an integer (3x3) transformation matrix.

      Superlattices S1 and S2 are symmetrically equivalent if there exists p and U such that:

      .. code-block:: Python

          S2 = p.matrix() @ S1 @ U,

      where p is an element in the point group, and U is a unimodular matrix (integer matrix, with
      abs(det(U))==1).

      Parameters
      ----------
      unit_lattice : casm.xtal.Lattice
          The unit lattice.
      point_group : List[casm.xtal.SymOp]
          The point group symmetry that determines if superlattices are equivalent. Depending on the use case, this is often the prim crystal point group, :func:`~casm.xtal.make_crystal_point_group()`, or the lattice point group, :func:`~casm.xtal.make_point_group()`.
      max_volume : int
          The maximum volume superlattice to enumerate, as a multiple of the volume of unit_lattice.
      min_volume : int, default=1
          The minimum volume superlattice to enumerate, as a multiple of the volume of unit_lattice.
      dirs : str, default="abc"
          A string indicating which lattice vectors to enumerate over. Some combination of 'a',
          'b', and 'c', where 'a' indicates the first lattice vector of the unit cell, 'b' the
          second, and 'c' the third.

      Returns
      -------
      superlattices : List[casm.xtal.Lattice]
          A list of superlattices of the unit lattice which are distinct under application of
          point_group. The resulting lattices will be in canonical form with respect to the
          point_group.
      )pbdoc");

  m.def("make_superduperlattice", &make_superduperlattice, py::arg("lattices"),
        py::arg("mode") = std::string("commensurate"),
        py::arg("point_group") = std::vector<xtal::SymOp>{}, R"pbdoc(
      Return the smallest lattice that is superlattice of the input lattices

      Parameters
      ----------
      lattices : List[casm.xtal.Lattice]
          List of lattices.
      mode : str, default="commensurate"
          One of:

          - "commensurate": Returns the smallest possible superlattice of all input lattices
          - "minimal_commensurate": Returns the lattice that is the smallest possible superlattice of an equivalent lattice to all input lattice
          - "fully_commensurate": Returns the lattice that is a superlattice of all equivalents of
            all input lattices
      point_group : List[casm.xtal.symop], default=[]
          Point group that generates the equivalent lattices for the the "minimal_commensurate" and
          "fully_commensurate" modes.

      Returns
      -------
      superduperlattice : casm.xtal.Lattice
          The superduperlattice
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
      .def("is_divisible", &xtal::Molecule::is_divisible,
           "True if is divisible in kinetic Monte Carlo calculations")
      .def("atoms", &xtal::Molecule::atoms,
           "Return the atomic components of the occupant")
      .def("properties", &get_molecule_properties,
           "Return the fixed properties of the occupant");

  m.def("is_vacancy", &xtal::Molecule::is_vacancy,
        "True if occupant is a vacancy.");

  m.def("is_atomic", &xtal::Molecule::is_atomic,
        "True if occupant is a single isotropic atom or vacancy");

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
      basis : array_like, shape (m, n), default=numpy.ndarray[numpy.float64[1, 0]]
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

      The :func:`~casm.xtal.make_primitive` method may be used to find
      the primitive equivalent, and the :func:`~casm.xtal.make_canonical_prim`
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

      .. _prim-init:

      Parameters
      ----------
      lattice : Lattice
          The primitive cell Lattice.
      coordinate_frac : array_like, shape (3, n)
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
          "H2_xz", and "H2_yz" for distinct orientations, or "A.up", and
          "A.down" for distinct collinear magnetic spins, etc.).
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
           "format.");

  m.def("make_within", &make_within, py::arg("init_prim"), R"pbdoc(
            Return an equivalent Prim with all basis site coordinates within the unit cell

            Parameters
            ----------
            init_prim : casm.xtal.Prim
                The initial prim.

            Returns
            ----------
            prim : Lattice
                The prim with all basis site coordinates within the unit cell.

            )pbdoc");

  m.def("make_primitive", &make_primitive, py::arg("init_prim"), R"pbdoc(
            Return a primitive equivalent Prim

            A :class:`Prim` object is not forced to be the primitive equivalent
            cell at construction. This function finds and returns the primitive
            equivalent cell by checking for internal translations that map all
            basis sites onto equivalent basis sites, including allowed
            occupants and equivalent local degrees of freedom (DoF), if they
            exist.

            Parameters
            ----------
            init_prim : casm.xtal.Prim
                The initial prim.

            Returns
            ----------
            prim : Lattice
                The primitive equivalent prim.
            )pbdoc");

  m.def("make_canonical_prim", &make_canonical_basicstructure,
        py::arg("init_prim"),
        R"pbdoc(
          Return an equivalent Prim with canonical lattice

          Finds the canonical right-handed Niggli cell of the lattice, applying
          lattice point group operations to find the equivalent lattice in a
          standardized orientation. The canonical orientation prefers lattice
          vectors that form symmetric matrices with large positive values on the
          diagonal and small values off the diagonal. See also `Lattice Canonical Form`_.

          .. _`Lattice Canonical Form`: https://prisms-center.github.io/CASMcode_docs/formats/lattice_canonical_form/

          Parameters
          ----------
          init_prim : casm.xtal.Prim
              The initial prim.

          Returns
          ----------
          prim : Lattice
              The prim with canonical lattice.

        )pbdoc");

  m.def("make_canonical", &make_canonical_basicstructure, py::arg("init_prim"),
        "Equivalent to :func:`~casm.xtal.make_canonical_prim`");

  m.def("asymmetric_unit_indices", &asymmetric_unit_indices, py::arg("prim"),
        R"pbdoc(
          Return the indices of equivalent basis sites

          Parameters
          ----------
          prim : casm.xtal.Prim
              The prim.

          Returns
          -------
          asymmetric_unit_indices : List[List[int]]
              One list of basis site indices for each set of symmetrically equivalent basis sites.
              In other words, the elements of asymmetric_unit_indices[i] are the indices of the
              i-th set of basis sites which are symmetrically equivalent to each other.

          )pbdoc");

  m.def("make_factor_group", &make_factor_group, py::arg("prim"), R"pbdoc(
          Return the factor group

          Parameters
          ----------
          prim : casm.xtal.Prim
              The prim.

          Returns
          -------
          factor_group : List[casm.xtal.SymOp]
              The the set of symmery operations, with translation lying within the primitive unit
              cell, that leave the lattice vectors, basis site coordinates, and all DoF invariant.

          )pbdoc");

  m.def("make_crystal_point_group", &make_crystal_point_group, py::arg("prim"),
        R"pbdoc(
          Return the crystal point group

          Parameters
          ----------
          prim : casm.xtal.Prim
              The prim.

          Returns
          -------
          crystal_point_group : List[casm.xtal.SymOp]
              The crystal point group is the group constructed from the prim factor group operations
              with translation vector set to zero.

          )pbdoc");

  py::class_<xtal::SymOp>(m, "SymOp", R"pbdoc(
      A symmetry operation representation that acts on Cartesian coordinates

      A SymOp, op, transforms a Cartesian coordinate according to:

      .. code-block:: Python

          r_after = op.matrix() @ r_before + op.translation()

      where r_before and r_after are shape=(3,) arrays with the Cartesian
      coordinates before and after transformation, respectively.

      Additionally, the sign of magnetic spins is flipped according to:

      .. code-block:: Python

          if op.time_reversal() is True:
              s_after = -s_before

      where s_before and s_after are the spins before and after
      transformation, respectively.

      )pbdoc")
      .def(py::init(&make_symop), py::arg("matrix"), py::arg("translation"),
           py::arg("time_reversal"),
           R"pbdoc(

          Parameters
          ----------
          matrix : array_like, shape (3, 3)
              The transformation matrix component of the symmetry operation.
          translation : array_like, shape (3,)
              Translation component of the symmetry operation.
          time_reversal : bool
              True if the symmetry operation includes time reversal (spin flip),
              False otherwise
          )pbdoc")
      .def("matrix", &xtal::get_matrix,
           "Return the transformation matrix value.")
      .def("translation", &xtal::get_translation,
           "Return the translation value.")
      .def("time_reversal", &xtal::get_time_reversal,
           "Return the time reversal value.");

  py::class_<xtal::SymInfo>(m, "SymInfo", R"pbdoc(
      Symmetry operation type, axis, invariant point, etc.

      )pbdoc")
      .def(py::init<xtal::SymOp const &, xtal::Lattice const &>(),
           py::arg("op"), py::arg("lattice"),
           R"pbdoc(

          Parameters
          ----------
          op : casm.xtal.SymOp
              The symmetry operation.
          lattice : casm.xtal.Lattice
              The lattice
          )pbdoc")
      .def("op_type", &get_syminfo_type, R"pbdoc(
          Return the symmetry operation type.

          Returns
          -------
          op_type: str
              One of:

              - "identity"
              - "mirror"
              - "glide"
              - "rotation"
              - "screw"
              - "inversion"
              - "rotoinversion"
              - "invalid"
          )pbdoc")
      .def("axis", get_syminfo_axis, R"pbdoc(
          Return the symmetry operation axis.

          Returns
          -------
          axis: numpy.ndarray[numpy.float64[3, 1]]
              This is:

              - the rotation axis, if the operation is a rotation or screw operation
              - the rotation axis of inversion * self, if this is an improper rotation (then the axis is a normal vector for a mirror plane)
              - zero vector, if the operation is identity or inversion

              The axis is in Cartesian coordinates and normalized to length 1.
          )pbdoc")
      .def("angle", &get_syminfo_angle, R"pbdoc(
          Return the symmetry operation angle.

          Returns
          -------
          angle: float
              This is:

              - the rotation angle, if the operation is a rotation or screw operation
              - the rotation angle of inversion * self, if this is an improper rotation (then the axis is a normal vector for a mirror plane)
              - zero, if the operation is identity or inversion

          )pbdoc")
      .def("screw_glide_shift", &get_syminfo_screw_glide_shift, R"pbdoc(
          Return the screw or glide translation component

          Returns
          -------
          screw_glide_shift: numpy.ndarray[numpy.float64[3, 1]]
              This is:

              - the component of translation parallel to `axis`, if the
                operation is a rotation
              - the component of translation perpendicular to `axis`, if
                the operation is a mirror

              The screw_glide_shift is in Cartesian coordinates.
          )pbdoc")
      .def("location", &get_syminfo_location, R"pbdoc(
          A Cartesian coordinate that is invariant to the operation (if one exists)

          Returns
          -------
          location: numpy.ndarray[numpy.float64[3, 1]]
              The location is in Cartesian coordinates. This does not exist for the identity
              operation.
          )pbdoc")
      .def("brief_cart", &get_syminfo_brief_cart, R"pbdoc(
          A brief description of the symmetry operation, in Cartesian coordinates

          Returns
          -------
          brief_cart: str
              A brief string description of the symmetry operation, in Cartesian coordinates,
              following the conventions of (International Tables for Crystallography (2015). Vol.
              A. ch. 1.4, pp. 50-59).
          )pbdoc")
      .def("brief_frac", &get_syminfo_brief_frac, R"pbdoc(
          A brief description of the symmetry operation, in fractional coordinates

          Returns
          -------
          brief_cart: str
              A brief string description of the symmetry operation, in fractional coordinates,
              following the conventions of (International Tables for Crystallography (2015). Vol.
              A. ch. 1.4, pp. 50-59).
          )pbdoc")
      .def("to_json", &syminfo_to_json, R"pbdoc(
          Represent the symmetry operation information as a JSON-formatted string.

          The `Symmetry Operation Information JSON Object reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/symmetry/SymGroup/#symmetry-operation-json-object/>`_ documents JSON format, except conjugacy class and inverse operation are not currently included.
          )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
