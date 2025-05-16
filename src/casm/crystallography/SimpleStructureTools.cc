#include "casm/crystallography/SimpleStructureTools.hh"

#include <stdexcept>

#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/IntegralCoordinateWithin.hh"
#include "casm/crystallography/LatticeIsEquivalent.hh"
#include "casm/crystallography/SimpleStructure.hh"
#include "casm/crystallography/Site.hh"
#include "casm/crystallography/UnitCellCoord.hh"
#include "casm/external/Eigen/Core"

namespace CASM {
namespace xtal {
namespace Local {

static SimpleStructure::Info _replicate(SimpleStructure::Info const &_info,
                                        Index mult) {
  SimpleStructure::Info result;
  result.resize(_info.size() * mult);

  for (Index i = 0; i < _info.size(); ++i)
    result.coords.block(0, i * mult, 3, mult) =
        _info.cart_coord(i).replicate(1, mult);

  for (auto const &p : _info.properties) {
    result.properties.emplace(
        p.first, Eigen::MatrixXd(p.second.rows(), mult * p.second.cols()));
    Eigen::MatrixXd &result_p = result.properties.at(p.first);
    for (Index i = 0; i < p.second.cols(); ++i)
      result_p.block(0, i * mult, p.second.rows(), mult) =
          p.second.col(i).replicate(1, mult);
  }

  Index l = 0;
  for (Index b = 0; b < _info.size(); ++b) {
    for (Index g = 0; g < mult; ++g) {
      result.names[l++] = _info.names[b];
    }
  }
  return result;
}
}  // namespace Local

//***************************************************************************

SimpleStructure make_superstructure(Eigen::Ref<const Eigen::Matrix3i> const &_T,
                                    SimpleStructure const &_sstruc) {
  SimpleStructure superstructure;
  superstructure.lat_column_mat = _sstruc.lat_column_mat * _T.cast<double>();
  superstructure.properties = _sstruc.properties;

  auto all_lattice_points = make_lattice_points(_T.cast<long>());

  Index Nvol = all_lattice_points.size();

  superstructure.mol_info = Local::_replicate(_sstruc.mol_info, Nvol);
  superstructure.atom_info = Local::_replicate(_sstruc.atom_info, Nvol);

  Index nm = _sstruc.mol_info.size();
  Index na = _sstruc.atom_info.size();

  for (Index g = 0; g < Nvol; ++g) {
    Eigen::Vector3d lattice_point_vector =
        _sstruc.lat_column_mat * all_lattice_points[g].cast<double>();

    for (Index m = 0; m < nm; ++m) {
      superstructure.mol_info.cart_coord(g + m * Nvol) += lattice_point_vector;
    }
    for (Index a = 0; a < na; ++a) {
      superstructure.atom_info.cart_coord(g + a * Nvol) += lattice_point_vector;
    }
  }

  return superstructure;
}

//***************************************************************************************************

/// Calculates the parent basis index of each site in a supercell that is
/// generated with the make_superstructure method
/// @param _T is the transformation matrix linking `_sstruc` and the supercell
/// @param _sstruc is a SimpleStructure
/// @return std::vector<Index> with the Index in the ith entry corresponding to
/// the index of site i in _sstruc
std::vector<Index> superstructure_basis_idx(
    Eigen::Ref<const Eigen::Matrix3i> const &_T,
    SimpleStructure const &_sstruc) {
  auto all_lattice_points = make_lattice_points(_T.cast<long>());
  Index Nvol = all_lattice_points.size();
  std::vector<Index> basis_idx(_sstruc.atom_info.size() * Nvol, -1);
  for (Index grid_idx = 0; grid_idx < Nvol; ++grid_idx) {
    for (Index atom_idx = 0; atom_idx < _sstruc.atom_info.size(); ++atom_idx)
      basis_idx[grid_idx + atom_idx * Nvol] = atom_idx;
  }
  return basis_idx;
}

//***************************************************************************

SimpleStructure make_simple_structure(BasicStructure const &_struc) {
  SimpleStructure result;
  Index N_site = _struc.basis().size();
  result.lat_column_mat = _struc.lattice().lat_column_mat();

  result.mol_info.coords.resize(3, N_site);
  result.mol_info.names.reserve(N_site);
  Eigen::VectorXi _mol_occ;
  // For now, default to first occupant. In future we may decide
  // to force user to pass mol_occ explicitly
  _mol_occ.setZero(N_site);
  for (Index b = 0; b < N_site; ++b) {
    auto const &site = _struc.basis()[b];
    auto const &mol = site.occupant_dof()[_mol_occ[b]];
    result.mol_info.cart_coord(b) = site.const_cart();
    result.mol_info.names.push_back(mol.name());

    // Initialize mol_info.properties for *molecule* properties
    for (auto const &attr : mol.properties()) {
      auto it = result.mol_info.properties.find(attr.first);
      if (it == result.mol_info.properties.end()) {
        it = result.mol_info.properties
                 .emplace(attr.first, Eigen::MatrixXd::Zero(
                                          attr.second.traits().dim(), N_site))
                 .first;
      }
      // Record properties of mol
      it->second.col(b) = attr.second.value();
    }
  }
  _atomize(result, _mol_occ, _struc);
  return result;
}

SimpleStructure make_simple_structure(std::istream &poscar_stream, double tol) {
  return make_simple_structure(
      BasicStructure::from_poscar_stream(poscar_stream, tol));
}

BasicStructure make_basic_structure(
    SimpleStructure const &_sstruc, std::vector<DoFKey> const &_all_dofs,
    SimpleStructure::SpeciesMode mode,
    std::vector<std::vector<Molecule>> _allowed_occupants) {
  std::map<DoFKey, DoFSet> global_dof;
  std::map<DoFKey, SiteDoFSet> local_dof;
  for (DoFKey const &dof : _all_dofs) {
    if (AnisoValTraits(dof).global()) {
      global_dof.emplace(dof, AnisoValTraits(dof));
    } else {
      local_dof.emplace(dof, AnisoValTraits(dof));
    }
  }

  auto const &info = _sstruc.info(mode);
  if (_allowed_occupants.empty()) _allowed_occupants.resize(info.size());
  for (Index i = 0; i < info.size(); ++i) {
    if (_allowed_occupants[i].empty()) {
      _allowed_occupants[i].push_back(Molecule::make_atom(info.names[i]));
    }
    if (_allowed_occupants[i].size() == 1) {
      std::map<std::string, SpeciesProperty> attr_map =
          _allowed_occupants[i][0].properties();
      for (auto it = attr_map.begin(); it != attr_map.end(); ++it) {
        if (local_dof.count(it->first)) {
          auto er_it = it++;
          attr_map.erase(er_it);
        }
      }

      for (auto const &prop : info.properties) {
        if (local_dof.count(prop.first)) continue;

        if (prop.first == "disp") continue;

        if (!almost_zero(prop.second.col(i)))
          attr_map.emplace(prop.first,
                           SpeciesProperty(prop.first, prop.second.col(i)));
      }
      _allowed_occupants[i][0].set_properties(attr_map);
    }
  }

  BasicStructure result(Lattice(_sstruc.lat_column_mat));
  result.set_global_dofs(global_dof);
  std::vector<Site> tbasis(info.size(), Site(result.lattice()));

  for (Index i = 0; i < info.size(); ++i) {
    tbasis[i].cart() = info.cart_coord(i);
    tbasis[i].set_allowed_occupants(std::move(_allowed_occupants[i]));
    tbasis[i].set_dofs(local_dof);
  }

  result.set_basis(tbasis);
  return result;
}

//***************************************************************************

/**
** Calculates the invariant shuffle modes in the primitive unit cell. They
*symmetry preserving distortions are found by applying the Reynolds operator to
*the basis of displacement vectors. The average of the resulting basis vectors
*is used to form an orthonormal basis.
*
* @param factor_group use make_factor_group(struc) to obtain this group
* @param permute_group use make_permute_group(struc,factor_group) to obtain this
*group
*
* @return the vector of shuffle modes that are invariant under symmetry. Each
* element has a size `3 x basis_size`.
*
*/
std::vector<Eigen::MatrixXd> generate_invariant_shuffle_modes(
    const std::vector<xtal::SymOp> &factor_group,
    const std::vector<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic,
                                               Index>> &permute_group) {
  if (factor_group.size() != permute_group.size()) {
    throw std::runtime_error(
        "error, the size of the symmetry operations in "
        "generate_invariant_shuffle_modes do not match");
  }
  int struc_basis_size = permute_group[0].indices().size();
  // Generate a basis consisting of individual shuffles of each atom in the
  // structure.
  std::vector<Eigen::MatrixXd> displacement_basis;
  for (int basis_idx = 0; basis_idx < struc_basis_size; ++basis_idx) {
    for (int dir_idx = 0; dir_idx < 3; ++dir_idx) {
      Eigen::MatrixXd single_shuffle =
          Eigen::MatrixXd::Zero(3, struc_basis_size);
      single_shuffle(dir_idx, basis_idx) = 1.0;
      displacement_basis.push_back(single_shuffle);
    }
  }
  std::vector<Eigen::VectorXd> displacement_aggregate(
      displacement_basis.size(),
      Eigen::VectorXd::Zero(displacement_basis[0].cols() *
                            displacement_basis[0].rows()));

  for (int idx = 0; idx < factor_group.size(); ++idx) {
    for (int disp_basis_idx = 0; disp_basis_idx < displacement_basis.size();
         ++disp_basis_idx) {
      Eigen::MatrixXd transformed_disp = factor_group[idx].matrix *
                                         displacement_basis[disp_basis_idx] *
                                         permute_group[idx];
      displacement_aggregate[disp_basis_idx] +=
          Eigen::VectorXd(Eigen::Map<Eigen::VectorXd>(
              transformed_disp.data(),
              transformed_disp.cols() * transformed_disp.rows()));
    }
  }
  Eigen::MatrixXd sym_disp_basis = Eigen::MatrixXd::Zero(
      displacement_aggregate.size(), displacement_aggregate[0].size());
  for (int disp_basis_idx = 0; disp_basis_idx < displacement_basis.size();
       ++disp_basis_idx) {
    displacement_aggregate[disp_basis_idx] =
        displacement_aggregate[disp_basis_idx] / double(factor_group.size());
    sym_disp_basis.row(disp_basis_idx) = displacement_aggregate[disp_basis_idx];
  }

  // Perform a SVD of the resulting aggregate matrix to obtain the rank and
  // space of the symmetry invariant basis vectors
  sym_disp_basis.transposeInPlace();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      sym_disp_basis, Eigen::ComputeThinU | Eigen::ComputeThinV);
  int matrix_rank = (svd.singularValues().array().abs() >= CASM::TOL).sum();
  Eigen::MatrixXd sym_preserving_mode_matrix =
      svd.matrixV()(Eigen::all, Eigen::seq(0, matrix_rank - 1));
  std::vector<Eigen::MatrixXd> sym_preserving_modes;

  for (int sym_mode_idx = 0; sym_mode_idx < matrix_rank; ++sym_mode_idx) {
    Eigen::Map<Eigen::MatrixXd> _tmp_mode(
        sym_preserving_mode_matrix.col(sym_mode_idx).data(), 3,
        struc_basis_size);
    sym_preserving_modes.push_back(_tmp_mode);
  }
  return sym_preserving_modes;
}

//***************************************************************************
void _atomize(SimpleStructure &_sstruc,
              Eigen::Ref<const Eigen::VectorXi> const &_mol_occ,
              BasicStructure const &_reference) {
  Index N_atoms(0);

  Index nb = _reference.basis().size();
  Index nv = _mol_occ.size() / nb;
  Index s = 0;
  for (Index b = 0; b < nb; ++b) {
    for (Index v = 0; v < nv; ++v, ++s) {
      N_atoms += _reference.basis()[b].occupant_dof()[_mol_occ[s]].size();
    }
  }
  _sstruc.atom_info.coords.resize(3, N_atoms);
  _sstruc.atom_info.names.resize(N_atoms);

  // s indexes site (i.e., molecule), a is index of atom within the entire
  // structure
  Index a = 0;
  s = 0;
  for (Index b = 0; b < nb; ++b) {
    for (Index v = 0; v < nv; ++v, ++s) {
      Molecule const &molref =
          _reference.basis()[b].occupant_dof()[_mol_occ[s]];

      // Initialize atom_info.properties for *molecule* properties
      for (auto const &property : _sstruc.mol_info.properties) {
        auto it = _sstruc.atom_info.properties.find(property.first);
        if (it == _sstruc.atom_info.properties.end()) {
          Index dim = AnisoValTraits(property.first).dim();
          _sstruc.atom_info.properties.emplace(
              property.first, Eigen::MatrixXd::Zero(dim, N_atoms));
        }
      }

      // ma is index of atom within individual molecule
      for (Index ma = 0; ma < molref.size(); ++ma, ++a) {
        // Record position of atom
        _sstruc.atom_info.cart_coord(a) =
            _sstruc.mol_info.cart_coord(s) + molref.atom(ma).cart();
        // Record name of atom
        _sstruc.atom_info.names[a] = molref.atom(ma).name();

        // Initialize atom_info.properties for *atom* properties
        for (auto const &attr : molref.atom(ma).properties()) {
          auto it = _sstruc.atom_info.properties.find(attr.first);
          if (it == _sstruc.atom_info.properties.end()) {
            it = _sstruc.atom_info.properties
                     .emplace(attr.first,
                              Eigen::MatrixXd::Zero(attr.second.traits().dim(),
                                                    N_atoms))
                     .first;
          }
          // Record properties of atom
          it->second.col(a) = attr.second.value();
        }

        // Split molecule properties into atom properties using appropriate
        // extensivity rules If an property is specified both at the atom and
        // molecule levels then the two are added
        for (auto const &property : _sstruc.mol_info.properties) {
          auto it = _sstruc.atom_info.properties.find(property.first);
          if (AnisoValTraits(property.first).extensive()) {
            it->second.col(a) += property.second.col(s) / double(molref.size());
          } else {
            it->second.col(a) += property.second.col(s);
          }
        }
      }
    }
  }
}

//***************************************************************************

std::vector<std::set<Index>> mol_site_compatibility(
    SimpleStructure const &sstruc, BasicStructure const &_prim) {
  std::vector<std::set<Index>> result;
  result.reserve(sstruc.mol_info.names.size());
  for (std::string const &sp : sstruc.mol_info.names) {
    result.push_back({});
    for (Index b = 0; b < _prim.basis().size(); ++b) {
      if (_prim.basis()[b].contains(sp)) {
        result.back().insert(b);
      }
    }
  }
  return result;
}

std::vector<std::set<Index>> atom_site_compatibility(
    SimpleStructure const &sstruc, BasicStructure const &_prim) {
  std::vector<std::set<Index>> result;
  result.reserve(sstruc.atom_info.names.size());
  for (std::string const &sp : sstruc.atom_info.names) {
    result.push_back({});
    for (Index b = 0; b < _prim.basis().size(); ++b) {
      for (Molecule const &mol : _prim.basis()[b].occupant_dof()) {
        if (mol.contains(sp)) {
          result.back().insert(b);
          break;
        }
      }
    }
  }
  return result;
}

/// \brief Transform local or global properties
std::map<std::string, Eigen::MatrixXd> &apply(
    xtal::SymOp const &op, std::map<std::string, Eigen::MatrixXd> &properties) {
  properties = copy_apply(op, properties);
  return properties;
}

/// \brief Copy and transform local or global properties
std::map<std::string, Eigen::MatrixXd> copy_apply(
    xtal::SymOp const &op,
    std::map<std::string, Eigen::MatrixXd> const &properties) {
  std::map<std::string, Eigen::MatrixXd> result;
  for (auto &pair : properties) {
    std::string key = pair.first;
    Eigen::MatrixXd const &value = pair.second;
    try {
      AnisoValTraits traits(key);
      Eigen::MatrixXd M = traits.symop_to_matrix(
          get_matrix(op), get_translation(op), get_time_reversal(op));
      result.emplace(key, M * value);
    } catch (std::exception &e) {
      std::stringstream msg;
      msg << "Error applying SymOp to properties: CASM does not know how to "
             "transform the property '"
          << key << "'.";
      throw std::runtime_error(msg.str());
    }
  }
  return result;
}

/// \brief Transform global properties
std::map<std::string, Eigen::VectorXd> &apply(
    xtal::SymOp const &op, std::map<std::string, Eigen::VectorXd> &properties) {
  properties = copy_apply(op, properties);
  return properties;
}

/// \brief Copy and transform global properties
std::map<std::string, Eigen::VectorXd> copy_apply(
    xtal::SymOp const &op,
    std::map<std::string, Eigen::VectorXd> const &properties) {
  std::map<std::string, Eigen::VectorXd> result;
  for (auto &pair : properties) {
    std::string key = pair.first;
    Eigen::MatrixXd const &value = pair.second;
    try {
      AnisoValTraits traits(key);
      Eigen::MatrixXd M = traits.symop_to_matrix(
          get_matrix(op), get_translation(op), get_time_reversal(op));
      result.emplace(key, M * value);
    } catch (std::exception &e) {
      std::stringstream msg;
      msg << "Error applying SymOp to properties: CASM does not know how to "
             "transform the property '"
          << key << "'.";
      throw std::runtime_error(msg.str());
    }
  }
  return result;
}

/// \brief Transform a SimpleStructure
SimpleStructure apply(xtal::SymOp const &op, SimpleStructure &sstruc) {
  sstruc.rotate_coords(op.matrix);

  // Transform local coords and properties
  auto _translate = [](xtal::SymOp const &op, SimpleStructure::Info &info) {
    for (Index l = 0; l < info.coords.cols(); ++l) {
      info.coords.col(l) += get_translation(op);
    }
  };
  _translate(op, sstruc.atom_info);
  _translate(op, sstruc.mol_info);
  sstruc.within();
  apply(op, sstruc.atom_info.properties);
  apply(op, sstruc.mol_info.properties);
  apply(op, sstruc.properties);
  return sstruc;
}

/// \brief Copy and transform a SimpleStructure
SimpleStructure copy_apply(xtal::SymOp const &op, SimpleStructure sstruc) {
  apply(op, sstruc);
  return sstruc;
}

/// \brief Check if two structures are equivalent
///
/// \param first The first structure
/// \param second The second structure
/// \param xtal_tol Tolerance for comparison of lattices and coordinates
/// \param properties_tol Tolerance for comparison of properties, by global or
/// local
///     property name. If not present, "default" will be used. If "default" is
///     not present, TOL will be used.
/// \return Returns true if the lattice points are equivalent, coordinates are
/// equivalent
///     after accounting for periodic boundary conditions, names are identical,
///     and all local and global properties of the two structures are equal to
///     the specified tolerance. Returns false otherwise.
///
/// \notes This does not check for rotations or less-then-lattice-vector
/// translations.
///     For structures that are equivalent up to a rotation, or after
///     translation of basis sites, this returns false. That type of equivalence
///     should be checked using the mapping methods.
bool is_equivalent(const SimpleStructure &first, const SimpleStructure &second,
                   double xtal_tol,
                   std::map<std::string, double> properties_tol) {
  Lattice first_lattice(first.lat_column_mat, xtal_tol);
  Lattice second_lattice(second.lat_column_mat, xtal_tol);
  if (!is_equivalent(first_lattice, second_lattice)) {
    return false;
  }

  auto get_property_tol = [&](std::string name) {
    auto it = properties_tol.find(name);
    if (it == properties_tol.end()) {
      it = properties_tol.find("default");
      if (it == properties_tol.end()) {
        return TOL;
      }
    }
    return it->second;
  };

  auto properties_are_equal =
      [&](std::map<std::string, Eigen::MatrixXd> const &first_properties,
          std::map<std::string, Eigen::MatrixXd> const &second_properties) {
        if (first_properties.size() != second_properties.size()) {
          return false;
        }
        for (auto const &pair : first_properties) {
          auto it = second_properties.find(pair.first);
          if (it == second_properties.end()) {
            return false;
          }
          double tol = get_property_tol(pair.first);
          if (!almost_equal(pair.second, it->second, tol)) {
            return false;
          }
        }
        return true;
      };

  // check global properties
  if (!properties_are_equal(first.properties, second.properties)) {
    return false;
  }

  auto site_properties_are_equal =
      [&](Index i_first, Index i_second,
          std::map<std::string, Eigen::MatrixXd> const &first_properties,
          std::map<std::string, Eigen::MatrixXd> const &second_properties) {
        if (first_properties.size() != second_properties.size()) {
          return false;
        }
        for (auto const &pair : first_properties) {
          auto it = second_properties.find(pair.first);
          double tol = get_property_tol(pair.first);
          if (it == second_properties.end()) {
            return false;
          }
          if (!almost_equal(pair.second.col(i_first), it->second.col(i_second),
                            tol)) {
            return false;
          }
        }
        return true;
      };

  auto find_info_index = [&](Index i_first,
                             SimpleStructure::Info const &first_info,
                             SimpleStructure::Info const &second_info) {
    Coordinate coord_ref{first_info.coords.col(i_first), first_lattice, CART};
    for (Index i_second = 0; i_second < second_info.size(); ++i_second) {
      Coordinate coord_other{second_info.coords.col(i_second), second_lattice,
                             CART};
      if (first_info.names[i_first] == second_info.names[i_second] &&
          coord_ref.min_dist(coord_other) < xtal_tol &&
          site_properties_are_equal(i_first, i_second, first_info.properties,
                                    second_info.properties)) {
        return i_second;
      }
    }
    return second_info.size();
  };

  auto find_atom_index = [&](Index i_first, SimpleStructure const &first,
                             SimpleStructure const &second) {
    SimpleStructure::Info const &first_info = first.atom_info;
    SimpleStructure::Info const &second_info = second.atom_info;
    return find_info_index(i_first, first_info, second_info);
  };

  auto find_mol_index = [&](Index i_first, SimpleStructure const &first,
                            SimpleStructure const &second) {
    SimpleStructure::Info const &first_info = first.mol_info;
    SimpleStructure::Info const &second_info = second.mol_info;
    return find_info_index(i_first, first_info, second_info);
  };

  // check atoms
  if (first.n_atom() != second.n_atom()) {
    return false;
  }
  for (Index i = 0; i < first.n_atom(); ++i) {
    if (find_atom_index(i, first, second) >= second.n_atom()) {
      return false;
    }
  }

  // check mol
  if (first.n_mol() != second.n_mol()) {
    return false;
  }
  for (Index i = 0; i < first.n_mol(); ++i) {
    if (find_mol_index(i, first, second) >= second.n_mol()) {
      return false;
    }
  }

  return true;
}

}  // namespace xtal
}  // namespace CASM
