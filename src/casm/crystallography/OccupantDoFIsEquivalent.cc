#include "casm/crystallography/OccupantDoFIsEquivalent.hh"

#include "casm/crystallography/Molecule.hh"
#include "casm/crystallography/SymType.hh"

namespace CASM {
namespace xtal {

/// returns true if m_dof and _other have matching labels, and m_dof =
/// P.permute(_other)
bool OccupantDoFIsEquivalent::operator()(
    std::vector<xtal::Molecule> const &_other) const {
  if (_other.size() != m_dof.size()) return false;
  Index j;
  m_atom_position_P.clear();
  for (Index i = 0; i < m_dof.size(); ++i) {
    for (j = 0; j < _other.size(); ++j) {
      Permutation atom_position_perm{m_dof[i].size()};
      if (m_dof[i].identical(_other[j], m_tol, atom_position_perm)) {
        m_P.set(i) = j;
        m_atom_position_P.push_back(atom_position_perm);
        break;
      }
    }
    if (j == _other.size()) {
      return false;
    }
  }
  return true;
}

/// returns true if copy_apply(_op,m_dof) = P.permute(m_dof)
bool OccupantDoFIsEquivalent::operator()(xtal::SymOp const &_op) const {
  Index j;
  m_atom_position_P.clear();
  for (Index i = 0; i < m_dof.size(); ++i) {
    xtal::Molecule t_occ = sym::copy_apply(_op, m_dof[i]);
    for (j = 0; j < m_dof.size(); ++j) {
      Permutation atom_position_perm{t_occ.size()};
      if (t_occ.identical(m_dof[j], m_tol, atom_position_perm)) {
        m_P.set(i) = j;
        m_atom_position_P.push_back(atom_position_perm);
        break;
      }
    }
    if (j == m_dof.size()) {
      return false;
    }
  }
  return true;
}

/// returns true if copy_apply(_op,m_dof) =  P.permute(_other)
bool OccupantDoFIsEquivalent::operator()(
    xtal::SymOp const &_op, std::vector<xtal::Molecule> const &_other) const {
  if (_other.size() != m_dof.size()) return false;
  Index j;
  m_atom_position_P.clear();
  for (Index i = 0; i < m_dof.size(); ++i) {
    xtal::Molecule t_occ = sym::copy_apply(_op, m_dof[i]);
    for (j = 0; j < _other.size(); ++j) {
      Permutation atom_position_perm{t_occ.size()};
      if (t_occ.identical(_other[j], m_tol, atom_position_perm)) {
        m_P.set(i) = j;
        m_atom_position_P.push_back(atom_position_perm);
        break;
      }
    }
    if (j == _other.size()) {
      return false;
    }
  }
  return true;
}

}  // namespace xtal
}  // namespace CASM
