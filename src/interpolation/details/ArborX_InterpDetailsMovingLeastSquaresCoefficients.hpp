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

template <typename SourcePoints, typename CenteredSourcePoints,
          typename TargetPoint>
KOKKOS_FUNCTION void
sourcePointsRecentering(int const neighbor, SourcePoints const &source_points,
                        CenteredSourcePoints &centered_source_points,
                        TargetPoint const &target_point)
{
  // SourcePoints must be a 1D view of points
  // CenteredSourcePoints must be a 1D view of points (same size as
  // SourcePoints)
  // TargetPoint is a point
  // all point dimensions must be the same
  static constexpr int dimension =
      GeometryTraits::dimension_v<typename SourcePoints::value_type>;

  auto source_point = source_points(neighbor);
  for (int k = 0; k < dimension; k++)
    source_point[k] -= target_point[k];
  centered_source_points(neighbor) = source_point;
}

template <typename SourcePoints, typename WorkType>
KOKKOS_FUNCTION void radiusComputation(SourcePoints const &source_points,
                                       WorkType &radius)
{
  // SourcePoints must be a 1D view of points
  static constexpr typename SourcePoints::value_type origin = {};

  radius = Kokkos::Experimental::epsilon_v<WorkType>;
  for (int neighbor = 0; neighbor < source_points.extent_int(0); neighbor++)
  {
    WorkType norm = ArborX::Details::distance(source_points(neighbor), origin);
    radius = Kokkos::max(radius, norm);
  }
  radius *= 1.1; // The one at the limit would be 0 due to how CRBFs work
}

template <typename CRBF, typename SourcePoints, typename WorkType, typename Phi>
KOKKOS_FUNCTION void phiComputation(int const neighbor,
                                    SourcePoints const &source_points,
                                    WorkType const radius, Phi &phi)
{
  // SourcePoints must be a 1D view of points
  // Phi must be a 1D view of values (same size as SourcePoints)
  static constexpr typename SourcePoints::value_type origin = {};

  WorkType norm = ArborX::Details::distance(source_points(neighbor), origin);
  phi(neighbor) = CRBF::evaluate(norm / radius);
}

template <typename PolynomialDegree, typename SourcePoints,
          typename Vandermonde>
KOKKOS_FUNCTION void vandermondeComputation(int const neighbor,
                                            SourcePoints const &source_points,
                                            Vandermonde &vandermonde)
{
  // SourcePoints must be a 1D view of points
  // Vandermonde must be a 2D view of values (neighbors x poly basis size)
  static constexpr int degree = PolynomialDegree::value;

  auto local_vandermonde = Kokkos::subview(vandermonde, neighbor, Kokkos::ALL);
  auto basis = evaluatePolynomialBasis<degree>(source_points(neighbor));
  for (int k = 0; k < int(basis.size()); k++)
    local_vandermonde(k) = basis[k];
}

template <typename Phi, typename Vandermonde, typename Moment>
KOKKOS_FUNCTION void momentComputation(int const i, int const j, Phi const &phi,
                                       Vandermonde const &vandermonde,
                                       Moment &moment)
{
  // Phi must be a 1D view of values
  // Vandermonde must be a 2D view of values (num lines == size of Phi)
  // Moment must be a 2D view of values (num lines == num cols == num vols
  // Vandermonde)

  moment(i, j) = 0;
  for (int k = 0; k < phi.extent_int(0); k++)
  {
    moment(i, j) += vandermonde(k, i) * vandermonde(k, j) * phi(k);
  }
}

template <typename CRBF, typename PolynomialDegree, typename CoefficientsType,
          typename MemorySpace, typename ExecutionSpace, typename SourcePoints,
          typename TargetPoints>
Kokkos::View<CoefficientsType **, MemorySpace>
movingLeastSquaresCoefficients(ExecutionSpace const &space,
                               SourcePoints const &source_points,
                               TargetPoints const &target_points)
{
  KokkosExt::ScopedProfileRegion guard(
      "ArborX::MovingLeastSquaresCoefficients");

  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
      "Memory space must be accessible from the execution space");

  // SourcePoints is a 2D view of points
  static_assert(Kokkos::is_view_v<SourcePoints> && SourcePoints::rank == 2,
                "source points must be a 2D view of points");
  static_assert(
      KokkosExt::is_accessible_from<typename SourcePoints::memory_space,
                                    ExecutionSpace>::value,
      "source points must be accessible from the execution space");
  using src_point = typename SourcePoints::non_const_value_type;
  GeometryTraits::check_valid_geometry_traits(src_point{});
  static_assert(GeometryTraits::is_point<src_point>::value,
                "source points elements must be points");
  static constexpr int dimension = GeometryTraits::dimension_v<src_point>;

  // TargetPoints is an access trait of points
  ArborX::Details::check_valid_access_traits(PrimitivesTag{}, target_points);
  using tgt_acc = AccessTraits<TargetPoints, PrimitivesTag>;
  static_assert(KokkosExt::is_accessible_from<typename tgt_acc::memory_space,
                                              ExecutionSpace>::value,
                "target points must be accessible from the execution space");
  using tgt_point = typename ArborX::Details::AccessTraitsHelper<tgt_acc>::type;
  GeometryTraits::check_valid_geometry_traits(tgt_point{});
  static_assert(GeometryTraits::is_point<tgt_point>::value,
                "target points elements must be points");
  static_assert(dimension == GeometryTraits::dimension_v<tgt_point>,
                "target and source points must have the same dimension");

  int const num_targets = tgt_acc::size(target_points);
  int const num_neighbors = source_points.extent(1);

  // There must be a set of neighbors for each target
  KOKKOS_ASSERT(num_targets == source_points.extent_int(0));

  using point_t = ExperimentalHyperGeometry::Point<dimension, CoefficientsType>;
  static constexpr int degree = PolynomialDegree::value;
  static constexpr int poly_size = polynomialBasisSize<dimension, degree>();

  Kokkos::Profiling::pushRegion(
      "ArborX::MovingLeastSquaresCoefficients::source_ref_target_fill");

  // The goal is to compute the following line vector for each target point:
  // p(x).[P^T.PHI.P]^-1.P^T.PHI
  // Where:
  // - p(x) is the polynomial basis of point x (line vector).
  // - P is the multidimensional Vandermonde matrix built from the source
  //   points, i.e., each line is the polynomial basis of a source point.
  // - PHI is the diagonal weight matrix / CRBF evaluated at each source point.

  // We first change the origin of the evaluation to be at the target point.
  // This lets us use p(0) which is [1 0 ... 0].
  Kokkos::View<point_t **, MemorySpace> source_ref_target(
      Kokkos::view_alloc(
          space, Kokkos::WithoutInitializing,
          "ArborX::MovingLeastSquaresCoefficients::source_ref_target"),
      num_targets, num_neighbors);
  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::source_ref_target_fill",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
          space, {0, 0}, {num_targets, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        auto local_source_points =
            Kokkos::subview(source_points, i, Kokkos::ALL);
        auto centered_source_points =
            Kokkos::subview(source_ref_target, i, Kokkos::ALL);
        auto target_point = tgt_acc::get(target_points, i);
        sourcePointsRecentering(j, local_source_points, centered_source_points,
                                target_point);
      });

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::MovingLeastSquaresCoefficients::radii_computation");

  // We then compute the radius for each target that will be used in evaluating
  // the weight for each source point.
  Kokkos::View<CoefficientsType *, MemorySpace> radii(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::radii"),
      num_targets);
  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::radii_computation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_targets),
      KOKKOS_LAMBDA(int const i) {
        auto local_source_points =
            Kokkos::subview(source_ref_target, i, Kokkos::ALL);
        auto &radius = radii(i);
        radiusComputation(local_source_points, radius);
      });

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::MovingLeastSquaresCoefficients::phi_computation");

  // This computes PHI given the source points as well as the radius
  Kokkos::View<CoefficientsType **, MemorySpace> phi(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::phi"),
      num_targets, num_neighbors);
  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::phi_computation",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
          space, {0, 0}, {num_targets, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        auto local_source_points =
            Kokkos::subview(source_ref_target, i, Kokkos::ALL);
        auto radius = radii(i);
        auto local_phi = Kokkos::subview(phi, i, Kokkos::ALL);
        phiComputation<CRBF>(j, local_source_points, radius, local_phi);
      });

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::MovingLeastSquaresCoefficients::vandermonde");

  // This builds the Vandermonde (P) matrix
  Kokkos::View<CoefficientsType ***, MemorySpace> p(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::vandermonde"),
      num_targets, num_neighbors, poly_size);
  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::vandermonde_computation",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
          space, {0, 0}, {num_targets, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        auto local_source_points =
            Kokkos::subview(source_ref_target, i, Kokkos::ALL);
        auto local_p = Kokkos::subview(p, i, Kokkos::ALL, Kokkos::ALL);
        vandermondeComputation<PolynomialDegree>(j, local_source_points,
                                                 local_p);
      });

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::MovingLeastSquaresCoefficients::moment");

  // We then create what is called the moment matrix, which is A = P^T.PHI.P. By
  // construction, A is symmetric.
  Kokkos::View<CoefficientsType ***, MemorySpace> a(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::moment"),
      num_targets, poly_size, poly_size);
  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::moment_computation",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(
          space, {0, 0, 0}, {num_targets, poly_size, poly_size}),
      KOKKOS_LAMBDA(int const i, int const j, int const k) {
        auto local_phi = Kokkos::subview(phi, i, Kokkos::ALL);
        auto local_p = Kokkos::subview(p, i, Kokkos::ALL, Kokkos::ALL);
        auto local_a = Kokkos::subview(a, i, Kokkos::ALL, Kokkos::ALL);
        momentComputation(j, k, local_phi, local_p, local_a);
      });

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::MovingLeastSquaresCoefficients::pseudo_inverse_svd");

  // We need the inverse of A = P^T.PHI.P, and because A is symmetric, we can
  // use the symmetric SVD algorithm to get it.
  symmetricPseudoInverseSVD(space, a);
  // Now, A = [P^T.PHI.P]^-1

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::MovingLeastSquaresCoefficients::coefficients_computation");

  // Finally, the result is produced by computing p(0).A.P^T.PHI
  Kokkos::View<CoefficientsType **, MemorySpace> coeffs(
      Kokkos::view_alloc(
          space, Kokkos::WithoutInitializing,
          "ArborX::MovingLeastSquaresCoefficients::coefficients"),
      num_targets, num_neighbors);
  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::coefficients_computation",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
          space, {0, 0}, {num_targets, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        CoefficientsType tmp = 0;
        for (int k = 0; k < poly_size; k++)
          tmp += a(i, 0, k) * p(i, j, k) * phi(i, j);
        coeffs(i, j) = tmp;
      });

  Kokkos::Profiling::popRegion();
  return coeffs;
}

} // namespace ArborX::Interpolation::Details

#endif