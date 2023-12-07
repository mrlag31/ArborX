/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_INTERP_DETAILS_MOVING_LEAST_SQUARES_COEFFICIENTS_HPP
#define ARBORX_INTERP_DETAILS_MOVING_LEAST_SQUARES_COEFFICIENTS_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_HyperPoint.hpp>
#include <ArborX_InterpDetailsPolynomialBasis.hpp>
#include <ArborX_InterpDetailsSymmetricPseudoInverseSVD.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Interpolation::Details
{

template <typename CRBF, typename PolynomialDegree, typename CoefficientsType,
          typename MemorySpace, typename TargetPoints, typename SourcePoints>
class MovingLeastSquaresCoefficientsKernel
{
private:
  using Phi = Kokkos::View<CoefficientsType **, MemorySpace>;
  using Vandermonde = Kokkos::View<CoefficientsType ***, MemorySpace>;
  using Moment = Kokkos::View<CoefficientsType ***, MemorySpace>;
  using SVDDiag = Kokkos::View<CoefficientsType ***, MemorySpace>;
  using SVDUnit = Kokkos::View<CoefficientsType ***, MemorySpace>;
  using Coefficients = Kokkos::View<CoefficientsType **, MemorySpace>;

  using SourcePoint = typename SourcePoints::non_const_value_type;
  using TargetAccess = AccessTraits<TargetPoints, PrimitivesTag>;
  using TargetPoint =
      typename ArborX::Details::AccessTraitsHelper<TargetAccess>::type;

  using LocalSourcePoints = Kokkos::Subview<SourcePoints, int, Kokkos::ALL_t>;
  using LocalPhi = Kokkos::Subview<Phi, int, Kokkos::ALL_t>;
  using LocalVandermonde =
      Kokkos::Subview<Vandermonde, int, Kokkos::ALL_t, Kokkos::ALL_t>;
  using LocalMoment =
      Kokkos::Subview<Moment, int, Kokkos::ALL_t, Kokkos::ALL_t>;
  using LocalCoefficients = Kokkos::Subview<Coefficients, int, Kokkos::ALL_t>;

public:
  template <typename ExecutionSpace>
  MovingLeastSquaresCoefficientsKernel(ExecutionSpace const &space,
                                       TargetPoints const &target_points,
                                       SourcePoints &source_points)
      : _target_points(target_points)
      , _source_points(source_points)
  {
    int const num_targets = source_points.extent_int(0);
    int const num_neighbors = source_points.extent_int(1);
    static constexpr int dimension = GeometryTraits::dimension_v<SourcePoint>;
    static constexpr int degree = PolynomialDegree::value;
    static constexpr int poly_size = polynomialBasisSize<dimension, degree>();

    _phi = Phi(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::MovingLeastSquaresCoefficientsKernel::phi"),
        num_targets, num_neighbors);

    _vandermonde = Vandermonde(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::MovingLeastSquaresCoefficientsKernel::vandermonde"),
        num_targets, num_neighbors, poly_size);

    _moment =
        Moment(Kokkos::view_alloc(
                   space, Kokkos::WithoutInitializing,
                   "ArborX::MovingLeastSquaresCoefficientsKernel::moment"),
               num_targets, poly_size, poly_size);

    _svd_diag =
        SVDDiag(Kokkos::view_alloc(
                    space, Kokkos::WithoutInitializing,
                    "ArborX::MovingLeastSquaresCoefficientsKernel::svd_diag"),
                num_targets, poly_size, poly_size);

    _svd_unit =
        SVDUnit(Kokkos::view_alloc(
                    space, Kokkos::WithoutInitializing,
                    "ArborX::MovingLeastSquaresCoefficientsKernel::svd_unit"),
                num_targets, poly_size, poly_size);

    _coefficients = Coefficients(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::MovingLeastSquaresCoefficientsKernel::coefficients"),
        num_targets, num_neighbors);
  }

private:
  static KOKKOS_FUNCTION void
  sourcePointsRecentering(int const neighbor, TargetPoint const &target_point,
                          LocalSourcePoints &source_points)
  {
    static constexpr int dimension = GeometryTraits::dimension_v<SourcePoint>;

    for (int k = 0; k < dimension; k++)
      source_points(neighbor)[k] -= target_point[k];
  }

  static KOKKOS_FUNCTION CoefficientsType
  radiusComputation(LocalSourcePoints const &source_points)
  {
    static constexpr SourcePoint origin = {};

    CoefficientsType radius = Kokkos::Experimental::epsilon_v<CoefficientsType>;
    for (int neighbor = 0; neighbor < source_points.extent_int(0); neighbor++)
    {
      CoefficientsType norm =
          ArborX::Details::distance(source_points(neighbor), origin);
      radius = Kokkos::max(radius, norm);
    }

    // The one at the limit would be 0 due to how CRBFs work
    return radius * CoefficientsType(1.1);
  }

  static KOKKOS_FUNCTION void
  phiComputation(int const neighbor, LocalSourcePoints const &source_points,
                 CoefficientsType const radius, LocalPhi &phi)
  {
    static constexpr SourcePoint origin = {};

    CoefficientsType norm =
        ArborX::Details::distance(source_points(neighbor), origin);
    phi(neighbor) = CRBF::evaluate(norm / radius);
  }

  static KOKKOS_FUNCTION void
  vandermondeComputation(int const neighbor,
                         LocalSourcePoints const &source_points,
                         LocalVandermonde &vandermonde)
  {
    static constexpr int degree = PolynomialDegree::value;

    auto local_vandermonde =
        Kokkos::subview(vandermonde, neighbor, Kokkos::ALL);
    auto basis = evaluatePolynomialBasis<degree>(source_points(neighbor));
    for (int k = 0; k < local_vandermonde.extent_int(0); k++)
      local_vandermonde(k) = basis[k];
  }

  static KOKKOS_FUNCTION void
  momentComputation(int const i, int const j, LocalPhi const &phi,
                    LocalVandermonde const &vandermonde, LocalMoment &moment)
  {
    moment(i, j) = 0;
    for (int k = 0; k < phi.extent_int(0); k++)
      moment(i, j) += vandermonde(k, i) * vandermonde(k, j) * phi(k);
  }

  static KOKKOS_FUNCTION void
  coefficientsComputation(int const neighbor, LocalPhi const &phi,
                          LocalVandermonde const &vandermonde,
                          LocalMoment const &moment,
                          LocalCoefficients &coefficients)
  {
    coefficients(neighbor) = 0;
    for (int i = 0; i < moment.extent_int(0); i++)
      coefficients(neighbor) +=
          moment(0, i) * vandermonde(neighbor, i) * phi(neighbor);
  }

public:
  auto size() const { return _source_points.extent(0); }
  auto coefficients() { return _coefficients; }

  KOKKOS_FUNCTION void operator()(int const target) const
  {
    auto target_point = TargetAccess::get(_target_points, target);
    auto source_points = Kokkos::subview(_source_points, target, Kokkos::ALL);
    auto phi = Kokkos::subview(_phi, target, Kokkos::ALL);
    auto vandermonde =
        Kokkos::subview(_vandermonde, target, Kokkos::ALL, Kokkos::ALL);
    auto moment = Kokkos::subview(_moment, target, Kokkos::ALL, Kokkos::ALL);
    auto svd_diag =
        Kokkos::subview(_svd_diag, target, Kokkos::ALL, Kokkos::ALL);
    auto svd_unit =
        Kokkos::subview(_svd_unit, target, Kokkos::ALL, Kokkos::ALL);
    auto coefficients = Kokkos::subview(_coefficients, target, Kokkos::ALL);

    int const poly_size = moment.extent_int(0);
    int const num_neighbors = source_points.extent_int(0);

    // The goal is to compute the following line vector for each target point:
    // p(x).[P^T.PHI.P]^-1.P^T.PHI
    // Where:
    // - p(x) is the polynomial basis of point x (line vector).
    // - P is the multidimensional Vandermonde matrix built from the source
    //   points, i.e., each line is the polynomial basis of a source point.
    // - PHI is the diagonal weight matrix / CRBF evaluated at each source
    // point.

    // We first change the origin of the evaluation to be at the target point.
    // This lets us use p(0) which is [1 0 ... 0].
    for (int neighbor = 0; neighbor < num_neighbors; neighbor++)
      sourcePointsRecentering(neighbor, target_point, source_points);

    // We then compute the radius for each target that will be used in
    // evaluating the weight for each source point.
    auto radius = radiusComputation(source_points);

    // This computes PHI given the source points as well as the radius
    for (int neighbor = 0; neighbor < num_neighbors; neighbor++)
      phiComputation(neighbor, source_points, radius, phi);

    // This builds the Vandermonde (P) matrix
    for (int neighbor = 0; neighbor < num_neighbors; neighbor++)
      vandermondeComputation(neighbor, source_points, vandermonde);

    // We then create what is called the moment matrix, which is P^T.PHI.P. By
    // construction, it is symmetric.
    for (int i = 0; i < poly_size; i++)
      for (int j = 0; j < poly_size; j++)
        momentComputation(i, j, phi, vandermonde, moment);

    // We need the inverse of P^T.PHI.P, and because it is symmetric, we can use
    // the symmetric SVD algorithm to get it.
    symmetricPseudoInverseSVDKernel(moment, svd_diag, svd_unit);
    // Now, the moment has [P^T.PHI.P]^-1

    // Finally, the result is produced by computing p(0).[P^T.PHI.P]^-1.P^T.PHI
    for (int neighbor = 0; neighbor < num_neighbors; neighbor++)
      coefficientsComputation(neighbor, phi, vandermonde, moment, coefficients);
  }

private:
  TargetPoints _target_points;
  SourcePoints _source_points;
  Phi _phi;
  Vandermonde _vandermonde;
  Moment _moment;
  SVDDiag _svd_diag;
  SVDUnit _svd_unit;
  Coefficients _coefficients;
};

template <typename CRBF, typename PolynomialDegree, typename CoefficientsType,
          typename MemorySpace, typename ExecutionSpace, typename TargetPoints,
          typename SourcePoints>
Kokkos::View<CoefficientsType **, MemorySpace>
movingLeastSquaresCoefficients(ExecutionSpace const &space,
                               TargetPoints const &target_points,
                               SourcePoints &source_points)
{
  KokkosExt::ScopedProfileRegion guard(
      "ArborX::MovingLeastSquaresCoefficients");

  MovingLeastSquaresCoefficientsKernel<CRBF, PolynomialDegree, CoefficientsType,
                                       MemorySpace, TargetPoints, SourcePoints>
      kernel(space, target_points, source_points);

  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::operation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, kernel.size()), kernel);

  return kernel.coefficients();
}

} // namespace ArborX::Interpolation::Details

#endif