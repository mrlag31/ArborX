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
          typename MemorySpace, typename ExecutionSpace, typename TargetPoints,
          typename SourcePoints>
class MovingLeastSquaresCoefficientsKernel
{
private:
  using ScratchMemorySpace = typename ExecutionSpace::scratch_memory_space;

  using SourcePoint = typename SourcePoints::non_const_value_type;
  using TargetAccess = AccessTraits<TargetPoints, PrimitivesTag>;
  using TargetPoint =
      typename ArborX::Details::AccessTraitsHelper<TargetAccess>::type;

  static constexpr int DIMENSION = GeometryTraits::dimension_v<SourcePoint>;
  static constexpr int DEGREE = PolynomialDegree::value;
  static constexpr int POLY_SIZE = polynomialBasisSize<DIMENSION, DEGREE>();
  static constexpr SourcePoint ORIGIN = {};

  template <typename T>
  using UnmanagedView =
      Kokkos::View<T, ScratchMemorySpace, Kokkos::MemoryUnmanaged>;

  using Coefficients = Kokkos::View<CoefficientsType **, MemorySpace>;

  using LocalSourcePoints = Kokkos::Subview<SourcePoints, int, Kokkos::ALL_t>;
  using LocalPhi = UnmanagedView<CoefficientsType *>;
  using LocalVandermonde = UnmanagedView<CoefficientsType **>;
  using LocalMoment = UnmanagedView<CoefficientsType **>;
  using LocalSVDDiag = UnmanagedView<CoefficientsType *>;
  using LocalSVDUnit = UnmanagedView<CoefficientsType **>;
  using LocalCoefficients = Kokkos::Subview<Coefficients, int, Kokkos::ALL_t>;

public:
  MovingLeastSquaresCoefficientsKernel(ExecutionSpace const &space,
                                       TargetPoints const &target_points,
                                       SourcePoints &source_points)
      : _target_points(target_points)
      , _source_points(source_points)
      , _num_targets(source_points.extent_int(0))
      , _num_neighbors(source_points.extent_int(1))
  {
    _coefficients = Coefficients(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::MovingLeastSquaresCoefficientsKernel::coefficients"),
        _num_targets, _num_neighbors);
  }

private:
  KOKKOS_FUNCTION void
  sourcePointsRecentering(TargetPoint const &target_point,
                          LocalSourcePoints &source_points) const
  {
    for (int neighbor = 0; neighbor < _num_neighbors; neighbor++)
      for (int k = 0; k < DIMENSION; k++)
        source_points(neighbor)[k] -= target_point[k];
  }

  KOKKOS_FUNCTION void phiComputation(LocalSourcePoints const &source_points,
                                      LocalPhi &phi) const
  {
    CoefficientsType radius = Kokkos::Experimental::epsilon_v<CoefficientsType>;
    for (int neighbor = 0; neighbor < _num_neighbors; neighbor++)
    {
      phi(neighbor) =
          ArborX::Details::distance(source_points(neighbor), ORIGIN);
      radius = Kokkos::max(radius, phi(neighbor));
    }

    // The one at the limit would be 0 due to how CRBFs work
    radius *= CoefficientsType(1.1);

    for (int neighbor = 0; neighbor < _num_neighbors; neighbor++)
      phi(neighbor) = CRBF::evaluate(phi(neighbor) / radius);
  }

  KOKKOS_FUNCTION void
  vandermondeComputation(LocalSourcePoints const &source_points,
                         LocalVandermonde &vandermonde) const
  {
    for (int neighbor = 0; neighbor < _num_neighbors; neighbor++)
    {
      auto basis = evaluatePolynomialBasis<DEGREE>(source_points(neighbor));
      for (int k = 0; k < POLY_SIZE; k++)
        vandermonde(neighbor, k) = basis[k];
    }
  }

  KOKKOS_FUNCTION void momentComputation(LocalPhi const &phi,
                                         LocalVandermonde const &vandermonde,
                                         LocalMoment &moment) const
  {
    for (int i = 0; i < POLY_SIZE; i++)
      for (int j = 0; j < POLY_SIZE; j++)
      {
        moment(i, j) = 0;
        for (int neighbor = 0; neighbor < _num_neighbors; neighbor++)
          moment(i, j) += vandermonde(neighbor, i) * vandermonde(neighbor, j) *
                          phi(neighbor);
      }
  }

  KOKKOS_FUNCTION void coefficientsComputation(
      LocalPhi const &phi, LocalVandermonde const &vandermonde,
      LocalMoment const &moment, LocalCoefficients &coefficients) const
  {
    for (int neighbor = 0; neighbor < _num_neighbors; neighbor++)
    {
      coefficients(neighbor) = 0;
      for (int i = 0; i < POLY_SIZE; i++)
        coefficients(neighbor) +=
            moment(0, i) * vandermonde(neighbor, i) * phi(neighbor);
    }
  }

public:
  auto coefficients() const { return _coefficients; }

  std::size_t team_shmem_size(int const team_size) const
  {
    std::size_t val = 0;
    val += LocalPhi::shmem_size(_num_neighbors);
    val += LocalVandermonde::shmem_size(_num_neighbors, POLY_SIZE);
    val += LocalMoment::shmem_size(POLY_SIZE, POLY_SIZE);
    val += LocalSVDDiag::shmem_size(POLY_SIZE);
    val += LocalSVDUnit::shmem_size(POLY_SIZE, POLY_SIZE);
    return team_size * val;
  }

  template <typename TeamMember>
  KOKKOS_FUNCTION void operator()(TeamMember member) const
  {
    int const rank = member.team_rank();
    int const size = member.team_size();
    auto const &scratch = member.thread_scratch(0);

    int target = member.league_rank() * size + rank;
    if (target >= _num_targets)
      return;

    auto target_point = TargetAccess::get(_target_points, target);
    auto source_points = Kokkos::subview(_source_points, target, Kokkos::ALL);
    LocalPhi phi(scratch, _num_neighbors);
    LocalVandermonde vandermonde(scratch, _num_neighbors, POLY_SIZE);
    LocalMoment moment(scratch, POLY_SIZE, POLY_SIZE);
    LocalSVDDiag svd_diag(scratch, POLY_SIZE);
    LocalSVDUnit svd_unit(scratch, POLY_SIZE, POLY_SIZE);
    auto coefficients = Kokkos::subview(_coefficients, target, Kokkos::ALL);

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
    sourcePointsRecentering(target_point, source_points);

    // This computes PHI given the source points (radius is computed inside)
    phiComputation(source_points, phi);

    // This builds the Vandermonde (P) matrix
    vandermondeComputation(source_points, vandermonde);

    // We then create what is called the moment matrix, which is P^T.PHI.P. By
    // construction, it is symmetric.
    momentComputation(phi, vandermonde, moment);

    // We need the inverse of P^T.PHI.P, and because it is symmetric, we can use
    // the symmetric SVD algorithm to get it.
    symmetricPseudoInverseSVDKernel(moment, svd_diag, svd_unit);
    // Now, the moment has [P^T.PHI.P]^-1

    // Finally, the result is produced by computing p(0).[P^T.PHI.P]^-1.P^T.PHI
    coefficientsComputation(phi, vandermonde, moment, coefficients);
  }

  Kokkos::TeamPolicy<ExecutionSpace>
  make_policy(ExecutionSpace const &space) const
  {
    Kokkos::TeamPolicy<ExecutionSpace> policy(space, 1, Kokkos::AUTO);
    int team_rec =
        policy.team_size_recommended(*this, Kokkos::ParallelForTag{});
    int div = _num_targets / team_rec;
    int mod = _num_targets % team_rec;
    int league_rec = div + ((mod == 0) ? 0 : 1);
    printf("Memory per block: %ld\n", team_shmem_size(team_rec));
    return Kokkos::TeamPolicy<ExecutionSpace>(space, league_rec, team_rec);
  }

private:
  TargetPoints _target_points;
  SourcePoints _source_points;
  Coefficients _coefficients;
  int _num_targets;
  int _num_neighbors;
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
                                       MemorySpace, ExecutionSpace,
                                       TargetPoints, SourcePoints>
      kernel(space, target_points, source_points);

  Kokkos::parallel_for("ArborX::MovingLeastSquaresCoefficients::operation",
                       kernel.make_policy(space), kernel);

  return kernel.coefficients();
}

} // namespace ArborX::Interpolation::Details

#endif