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

template <typename TargetPoint, typename SourcePoints>
KOKKOS_FUNCTION void sourcePointsRecentering(int const neighbor,
                                             TargetPoint const &target_point,
                                             SourcePoints &source_points)
{
  static constexpr int dimension =
      GeometryTraits::dimension_v<typename SourcePoints::value_type>;

  for (int k = 0; k < dimension; k++)
    source_points(neighbor)[k] -= target_point[k];
}

template <typename WorkType, typename SourcePoints>
KOKKOS_FUNCTION WorkType radiusComputation(SourcePoints const &source_points)
{
  static constexpr typename SourcePoints::non_const_value_type origin = {};

  WorkType radius = Kokkos::Experimental::epsilon_v<WorkType>;
  for (int neighbor = 0; neighbor < source_points.extent_int(0); neighbor++)
  {
    WorkType norm = ArborX::Details::distance(source_points(neighbor), origin);
    radius = Kokkos::max(radius, norm);
  }

  // The one at the limit would be 0 due to how CRBFs work
  return radius * WorkType(1.1);
}

template <typename CRBF, typename SourcePoints, typename WorkType, typename Phi>
KOKKOS_FUNCTION void phiComputation(int const neighbor,
                                    SourcePoints const &source_points,
                                    WorkType const radius, Phi &phi)
{
  static constexpr typename SourcePoints::non_const_value_type origin = {};

  WorkType norm = ArborX::Details::distance(source_points(neighbor), origin);
  phi(neighbor) = CRBF::evaluate(norm / radius);
}

template <typename PolynomialDegree, typename SourcePoints,
          typename Vandermonde>
KOKKOS_FUNCTION void vandermondeComputation(int const neighbor,
                                            SourcePoints const &source_points,
                                            Vandermonde &vandermonde)
{
  static constexpr int degree = PolynomialDegree::value;

  auto local_vandermonde = Kokkos::subview(vandermonde, neighbor, Kokkos::ALL);
  auto basis = evaluatePolynomialBasis<degree>(source_points(neighbor));
  for (int k = 0; k < local_vandermonde.extent_int(0); k++)
    local_vandermonde(k) = basis[k];
}

template <typename Phi, typename Vandermonde, typename Moment>
KOKKOS_FUNCTION void momentComputation(int const i, int const j, Phi const &phi,
                                       Vandermonde const &vandermonde,
                                       Moment &moment)
{
  moment(i, j) = 0;
  for (int k = 0; k < phi.extent_int(0); k++)
    moment(i, j) += vandermonde(k, i) * vandermonde(k, j) * phi(k);
}

template <typename Phi, typename Vandermonde, typename MomentInverse,
          typename Coefficients>
KOKKOS_FUNCTION void coefficientsComputation(
    int const neighbor, Phi const &phi, Vandermonde const &vandermonde,
    MomentInverse const &moment_inverse, Coefficients &coefficients)
{
  coefficients(neighbor) = 0;
  for (int i = 0; i < moment_inverse.extent_int(0); i++)
    coefficients(neighbor) +=
        moment_inverse(0, i) * vandermonde(neighbor, i) * phi(neighbor);
}

template <typename CRBF, typename PolynomialDegree, typename TargetPoint,
          typename SourcePoints, typename Phi, typename Vandermonde,
          typename Moment, typename SVDDiag, typename SVDUnit,
          typename Coefficients>
KOKKOS_FUNCTION void movingLeastSquaresCoefficientsKernel(
    TargetPoint const &target_point, SourcePoints &source_points, Phi &phi,
    Vandermonde &vandermonde, Moment &moment, SVDDiag &svd_diag,
    SVDUnit &svd_unit, Coefficients &coefficients)
{
  using coefficients_t = typename Coefficients::non_const_value_type;
  int const poly_size = moment.extent_int(0);
  int const num_neighbors = source_points.extent_int(0);

  // The goal is to compute the following line vector for each target point:
  // p(x).[P^T.PHI.P]^-1.P^T.PHI
  // Where:
  // - p(x) is the polynomial basis of point x (line vector).
  // - P is the multidimensional Vandermonde matrix built from the source
  //   points, i.e., each line is the polynomial basis of a source point.
  // - PHI is the diagonal weight matrix / CRBF evaluated at each source point.

  // We first change the origin of the evaluation to be at the target point.
  // This lets us use p(0) which is [1 0 ... 0].
  for (int neighbor = 0; neighbor < num_neighbors; neighbor++)
    sourcePointsRecentering(neighbor, target_point, source_points);

  // We then compute the radius for each target that will be used in evaluating
  // the weight for each source point.
  auto radius = radiusComputation<coefficients_t>(source_points);

  // This computes PHI given the source points as well as the radius
  for (int neighbor = 0; neighbor < num_neighbors; neighbor++)
    phiComputation<CRBF>(neighbor, source_points, radius, phi);

  // This builds the Vandermonde (P) matrix
  for (int neighbor = 0; neighbor < num_neighbors; neighbor++)
    vandermondeComputation<PolynomialDegree>(neighbor, source_points,
                                             vandermonde);

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

  using tgt_acc = AccessTraits<TargetPoints, PrimitivesTag>;
  using tgt_point = typename ArborX::Details::AccessTraitsHelper<tgt_acc>::type;
  int const num_targets = source_points.extent_int(0);
  int const num_neighbors = source_points.extent_int(1);
  static constexpr int dimension = GeometryTraits::dimension_v<tgt_point>;
  static constexpr int degree = PolynomialDegree::value;
  static constexpr int poly_size = polynomialBasisSize<dimension, degree>();

  Kokkos::View<CoefficientsType **, MemorySpace> phi(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::phi"),
      num_targets, num_neighbors);
  Kokkos::View<CoefficientsType ***, MemorySpace> vandermonde(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::vandermonde"),
      num_targets, num_neighbors, poly_size);
  Kokkos::View<CoefficientsType ***, MemorySpace> moment(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::moment"),
      num_targets, poly_size, poly_size);
  Kokkos::View<CoefficientsType ***, MemorySpace> svd_diag(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::svd_diag"),
      num_targets, poly_size, poly_size);
  Kokkos::View<CoefficientsType ***, MemorySpace> svd_unit(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::svd_unit"),
      num_targets, poly_size, poly_size);
  Kokkos::View<CoefficientsType **, MemorySpace> coefficients(
      Kokkos::view_alloc(
          space, Kokkos::WithoutInitializing,
          "ArborX::MovingLeastSquaresCoefficients::coefficients"),
      num_targets, num_neighbors);

  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::operation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_targets),
      KOKKOS_LAMBDA(int const target) {
        auto target_point = tgt_acc::get(target_points, target);
        auto local_source_points =
            Kokkos::subview(source_points, target, Kokkos::ALL);
        auto local_phi = Kokkos::subview(phi, target, Kokkos::ALL);
        auto local_vandermonde =
            Kokkos::subview(vandermonde, target, Kokkos::ALL, Kokkos::ALL);
        auto local_moment =
            Kokkos::subview(moment, target, Kokkos::ALL, Kokkos::ALL);
        auto local_svd_diag =
            Kokkos::subview(svd_diag, target, Kokkos::ALL, Kokkos::ALL);
        auto local_svd_unit =
            Kokkos::subview(svd_unit, target, Kokkos::ALL, Kokkos::ALL);
        auto local_coefficients =
            Kokkos::subview(coefficients, target, Kokkos::ALL);

        movingLeastSquaresCoefficientsKernel<CRBF, PolynomialDegree>(
            target_point, local_source_points, local_phi, local_vandermonde,
            local_moment, local_svd_diag, local_svd_unit, local_coefficients);
      });

  return coefficients;
}

} // namespace ArborX::Interpolation::Details

#endif