#include "casm/crystallography/SymInfo.hh"

#include "casm/misc/CASM_Eigen_math.hh"

namespace CASM {
namespace xtal {

SymInfo::SymInfo(SymOp const &op, xtal::Lattice const &lat)
    : lattice(lat),
      axis(lattice),
      screw_glide_shift(lattice),
      location(lattice),
      time_reversal(op.is_time_reversal_active) {
  auto matrix = op.matrix;
  auto tau = op.translation;

  Eigen::Vector3d _axis;
  Eigen::Vector3d _screw_glide_shift;
  Eigen::Vector3d _location;

  // Simplest case is identity: has no axis and no location
  if (almost_equal(matrix.trace(), 3.)) {
    angle = 0;
    op_type = symmetry_type::identity_op;
    _axis = Eigen::Vector3d::Zero();
    _location = Eigen::Vector3d::Zero();
    _set(_axis, _screw_glide_shift, _location, lattice);
    return;
  }

  // second simplest case is inversion: has no axis and location is tau()/2
  if (almost_equal(matrix.trace(), -3.)) {
    angle = 0;
    op_type = symmetry_type::inversion_op;
    _axis = Eigen::Vector3d::Zero();
    _location = tau / 2.;
    _set(_axis, _screw_glide_shift, _location, lattice);
    return;
  }

  // det is -1 if improper and +1 if proper
  int det = round(matrix.determinant());

  // Find eigen decomposition of proper operation (by multiplying by
  // determinant)
  Eigen::EigenSolver<Eigen::Matrix3d> t_eig(det * matrix);

  // 'axis' is eigenvector whose eigenvalue is +1
  for (Index i = 0; i < 3; i++) {
    if (almost_equal(t_eig.eigenvalues()(i), std::complex<double>(1, 0))) {
      _axis = t_eig.eigenvectors().col(i).real();
      break;
    }
  }

  // Sign convention for 'axis': first non-zero element is positive
  for (Index i = 0; i < 3; i++) {
    if (!almost_zero(_axis[i])) {
      _axis *= float_sgn(_axis[i]);
      break;
    }
  }

  // get vector orthogonal to axis: ortho,
  // apply matrix: rot
  // and check angle between ortho and det*rot,
  // using determinant to get the correct angle for improper
  // (i.e. want angle before inversion for rotoinversion)
  Eigen::Vector3d ortho = _axis.unitOrthogonal();
  Eigen::Vector3d rot = det * (matrix * ortho);
  angle = fmod(
      (180. / M_PI) * atan2(_axis.dot(ortho.cross(rot)), ortho.dot(rot)) + 360.,
      360.);

  /*
  std::cout << "det: " << det << "\n";
  std::cout << "y: " << _axis.dot(ortho.cross(rot)) << "\n";
  std::cout << "x: " << ortho.dot(rot) << "\n";
  std::cout << "angle: " << angle << std::endl;
  */

  if (det < 0) {
    if (almost_equal(angle, 180.)) {
      // shift is component of tau perpendicular to axis
      xtal::Coordinate coord(tau - tau.dot(_axis) * _axis, lattice, CART);
      _screw_glide_shift = coord.cart();

      // location is 1/2 of component of tau parallel to axis:
      //   matrix*location+tau = -location+tau = location
      _location = tau.dot(_axis) * _axis / 2.;

      op_type = coord.is_lattice_shift() ? symmetry_type::mirror_op
                                         : symmetry_type::glide_op;
    } else {
      // shift is component of tau parallel to axis
      _screw_glide_shift = tau.dot(_axis) * _axis;

      // rotoinversion is point symmetry, so we can solve matrix*p+tau=p for
      // invariant point p
      _location = (Eigen::Matrix3d::Identity() - matrix).inverse() * tau;

      op_type = symmetry_type::rotoinversion_op;
    }
  } else {
    // shift is component of tau parallel to axis
    xtal::Coordinate coord(tau.dot(_axis) * _axis, lattice, CART);
    _screw_glide_shift = coord.cart();

    // Can only solve 2d location problem
    Eigen::MatrixXd tmat(3, 2);
    tmat << ortho, ortho.cross(_axis);

    // if A = tmat.transpose()*matrix()*tmat and s=tmat.transpose()*tau()
    // then 2d invariant point 'v' is solution to A*v+s=v
    // implies 3d invariant point 'p' is p=tmat*(eye(2)-A).inverse()*s
    _location =
        tmat *
        (Eigen::MatrixXd::Identity(2, 2) - tmat.transpose() * matrix * tmat)
            .inverse() *
        tmat.transpose() * tau;

    op_type = coord.is_lattice_shift() ? symmetry_type::rotation_op
                                       : symmetry_type::screw_op;
  }
  _set(_axis, _screw_glide_shift, _location, lattice);
  return;
}

SymInfo::SymInfo(SymInfo const &other)
    : lattice(other.lattice),
      op_type(other.op_type),
      axis(other.axis),
      angle(other.angle),
      screw_glide_shift(other.screw_glide_shift),
      location(other.location),
      time_reversal(other.time_reversal) {
  axis.set_lattice(lattice, CART);
  screw_glide_shift.set_lattice(lattice, CART);
  location.set_lattice(lattice, CART);
}

void SymInfo::_set(Eigen::Vector3d const &_axis,
                   Eigen::Vector3d const &_screw_glide_shift,
                   Eigen::Vector3d const &_location, xtal::Lattice const &lat) {
  axis = xtal::Coordinate(_axis, lat, CART);
  screw_glide_shift = xtal::Coordinate(_screw_glide_shift, lat, CART);
  location = xtal::Coordinate(_location, lat, CART);
}

}  // namespace xtal
}  // namespace CASM
