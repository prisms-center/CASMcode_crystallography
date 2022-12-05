#ifndef CASM_xtal_StrainConverter
#define CASM_xtal_StrainConverter

#include "casm/global/definitions.hh"
#include "casm/global/eigen.hh"

namespace CASM {
namespace xtal {

/// \brief Holds strain metric and basis to facilitate conversions
struct StrainConverter {
  StrainConverter(std::string _metric, Eigen::MatrixXd const &_basis);

  /// \brief Decompose deformation tensor, F, as Q*U
  static std::pair<Eigen::Matrix3d, Eigen::Matrix3d> F_to_QU(
      Eigen::Matrix3d const &F);

  /// \brief Decompose deformation tensor, F, as V*Q
  static std::pair<Eigen::Matrix3d, Eigen::Matrix3d> F_to_VQ(
      Eigen::Matrix3d const &F);

  /// \brief Name of strain metric (i.e. 'Hstrain', etc.)
  std::string metric() const;

  /// \brief Strain metric basis, such that E_standard = basis * E_basis
  Eigen::MatrixXd basis() const;

  /// \brief Returns the strain space dimension (number of columns of basis())
  Index dim() const;

  /// \brief Pseudoinverse of basis
  Eigen::MatrixXd basis_pinv() const;

  /// \brief Returns strain metric vector value in standard basis
  Eigen::VectorXd to_standard_basis(Eigen::VectorXd const &E_vector) const;

  /// \brief Returns strain metric vector value in converter basis
  Eigen::VectorXd from_standard_basis(
      Eigen::VectorXd const &E_vector_in_standard_basis) const;

  /// \brief Convert strain metric vector value to matrix value
  Eigen::Matrix3d to_E_matrix(Eigen::VectorXd const &E_vector) const;

  /// \brief Convert strain metric matrix value to vector value
  Eigen::VectorXd from_E_matrix(Eigen::Matrix3d const &E_matrix) const;

  /// \brief Convert strain metric value to deformation tensor
  Eigen::Matrix3d to_F(Eigen::VectorXd const &E_vector) const;

  /// \brief Convert strain metric value to deformation tensor
  Eigen::VectorXd from_F(Eigen::Matrix3d const &F) const;

 private:
  /// \brief Name of strain metric (i.e. 'Hstrain', etc.)
  std::string m_metric;

  /// \brief Strain metric basis, such that E_standard = basis * E_basis
  Eigen::MatrixXd m_basis;

  /// \brief Pseudoinverse of basis
  Eigen::MatrixXd m_basis_pinv;
};

Eigen::MatrixXd make_symmetry_adapted_strain_basis();

}  // namespace xtal
}  // namespace CASM

#endif
