#include "casm/crystallography/StrainConverter.hh"

#include "casm/crystallography/Strain.hh"
#include "casm/global/eigen.hh"
#include "casm/misc/CASM_Eigen_math.hh"

// debug
#include <iostream>

namespace CASM {
namespace xtal {

/// \brief Holds strain metric and basis to facilitate conversions
StrainConverter::StrainConverter(std::string _metric,
                                 Eigen::MatrixXd const &_basis)
    : m_metric(_metric), m_basis(_basis) {
  m_basis_pinv = m_basis.completeOrthogonalDecomposition().pseudoInverse();
}

/// \brief Decompose deformation tensor, F, as Q*U
///
/// \param F A deformation tensor
/// \returns {Q, U}
std::pair<Eigen::Matrix3d, Eigen::Matrix3d> StrainConverter::F_to_QU(
    Eigen::Matrix3d const &F) {
  Eigen::Matrix3d right_stretch = polar_decomposition(F);
  Eigen::Matrix3d isometry = F * right_stretch.inverse();
  return std::make_pair(isometry, right_stretch);
}

/// \brief Decompose deformation tensor, F, as V*Q
///
/// \param F A deformation tensor
/// \returns {Q, V} Note the isometry matrix, Q, is returned first.
std::pair<Eigen::Matrix3d, Eigen::Matrix3d> StrainConverter::F_to_VQ(
    Eigen::Matrix3d const &F) {
  auto result = F_to_QU(F);
  result.second = F * result.first.transpose();
  return result;
}

/// \brief Name of strain metric (i.e. 'Hstrain', etc.)
std::string StrainConverter::metric() const { return m_metric; }

/// \brief Strain metric basis, such that E_standard = basis * E_basis
Eigen::MatrixXd StrainConverter::basis() const { return m_basis; }

/// \brief Returns the strain space dimension (number of columns of basis())
Index StrainConverter::dim() const { return m_basis.cols(); }

/// \brief Pseudoinverse of basis
Eigen::MatrixXd StrainConverter::basis_pinv() const { return m_basis_pinv; }

/// \brief Returns strain metric vector value in standard basis
Eigen::VectorXd StrainConverter::to_standard_basis(
    Eigen::VectorXd const &E_vector) const {
  return m_basis * E_vector;
}

/// \brief Returns strain metric vector value in converter basis
Eigen::VectorXd StrainConverter::from_standard_basis(
    Eigen::VectorXd const &E_vector_in_standard_basis) const {
  return m_basis_pinv * E_vector_in_standard_basis;
}

/// \brief Convert strain metric vector value to matrix value
///
/// Strain metric vector value is:
///   [Exx, Eyy, Ezz, sqrt(2)*Eyz, sqrt(2)*Exz, sqrt(2)*Exy]
///
/// \param E_vector, strain metric vector value in basis this->basis()
/// \returns E_matrix Strain metric matrix
///
Eigen::Matrix3d StrainConverter::to_E_matrix(
    Eigen::VectorXd const &E_vector) const {
  Eigen::VectorXd e = m_basis * E_vector;
  double w = sqrt(2.);
  Eigen::Matrix3d E_matrix;
  E_matrix <<  //
      e(0),
      e(5) / w, e(4) / w,        //
      e(5) / w, e(1), e(3) / w,  //
      e(4) / w, e(3) / w, e(2);  //
  return E_matrix;
}

/// \brief Convert strain metric matrix value to vector value
///
/// Strain metric vector value is:
///   [Exx, Eyy, Ezz, sqrt(2)*Eyz, sqrt(2)*Exz, sqrt(2)*Exy]
///
/// \param E_matrix Strain metric matrix
/// \return E_vector, strain metric vector value in this->basis()
///
Eigen::VectorXd StrainConverter::from_E_matrix(
    Eigen::Matrix3d const &E_matrix) const {
  Eigen::Matrix3d const &e = E_matrix;
  Eigen::VectorXd E_vector = Eigen::VectorXd::Zero(6);
  double w = std::sqrt(2.);
  E_vector << e(0, 0), e(1, 1), e(2, 2), w * e(1, 2), w * e(0, 2), w * e(0, 1);
  return m_basis_pinv * E_vector;
}

/// \brief Convert strain metric value to deformation tensor
///
/// \param E_vector Unrolled strain metric value, in this->basis(),
///     such that E_standard = this->basis() * E_vector
/// \returns F, the deformation tensor
Eigen::Matrix3d StrainConverter::to_F(Eigen::VectorXd const &E_vector) const {
  using namespace strain;
  Eigen::Matrix3d E_matrix = this->to_E_matrix(E_vector);

  if (m_metric == "Hstrain") {
    return metric_to_deformation_tensor<METRIC::HENCKY>(E_matrix);
  } else if (m_metric == "EAstrain") {
    return metric_to_deformation_tensor<METRIC::EULER_ALMANSI>(E_matrix);
  } else if (m_metric == "GLstrain") {
    return metric_to_deformation_tensor<METRIC::GREEN_LAGRANGE>(E_matrix);
  } else if (m_metric == "Bstrain") {
    return metric_to_deformation_tensor<METRIC::BIOT>(E_matrix);
  } else if (m_metric == "Ustrain") {
    return E_matrix;
  } else {
    std::stringstream ss;
    ss << "StrainConverter error: Unexpected metric: " << m_metric;
    throw std::runtime_error(ss.str());
  }
}

/// \brief Convert strain metric value to deformation tensor
///
/// \param F Deformation gradient tensor
/// \returns Unrolled strain metric value, in this->basis(),
///     such that E_standard = this->basis() * E_vector
Eigen::VectorXd StrainConverter::from_F(Eigen::Matrix3d const &F) const {
  using namespace strain;
  Eigen::Matrix3d E_matrix;

  if (m_metric == "Hstrain") {
    E_matrix = deformation_tensor_to_metric<METRIC::HENCKY>(F);
  } else if (m_metric == "EAstrain") {
    E_matrix = deformation_tensor_to_metric<METRIC::EULER_ALMANSI>(F);
  } else if (m_metric == "GLstrain") {
    E_matrix = deformation_tensor_to_metric<METRIC::GREEN_LAGRANGE>(F);
  } else if (m_metric == "Bstrain") {
    E_matrix = deformation_tensor_to_metric<METRIC::BIOT>(F);
  } else if (m_metric == "Ustrain") {
    E_matrix = right_stretch_tensor(F);
  } else {
    std::stringstream ss;
    ss << "StrainConverter error: Unexpected metric: " << m_metric;
    throw std::runtime_error(ss.str());
  }
  return this->from_E_matrix(E_matrix);
}

Eigen::MatrixXd make_symmetry_adapted_strain_basis() {
  Eigen::MatrixXd B(6, 6);
  B.col(0) << 1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3), 0, 0, 0;    // e1
  B.col(1) << 1 / sqrt(2), -1 / sqrt(2), 0.0, 0, 0, 0;           // e2
  B.col(2) << -1 / sqrt(6), -1 / sqrt(6), 2 / sqrt(6), 0, 0, 0;  // e3
  B.col(3) << 0, 0, 0, 1., 0, 0;                                 // e4
  B.col(4) << 0, 0, 0, 0, 1., 0;                                 // e5
  B.col(5) << 0, 0, 0, 0, 0, 1.;                                 // e6

  return B;
}

}  // namespace xtal
}  // namespace CASM
