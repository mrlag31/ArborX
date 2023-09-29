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

#ifndef ARBORX_INTERP_DISTRIBUTED_MOVING_LEAST_SQUARES_HPP
#define ARBORX_INTERP_DISTRIBUTED_MOVING_LEAST_SQUARES_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DistributedTree.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_InterpDetailsCompactRadialBasisFunction.hpp>
#include <ArborX_InterpDetailsDistributedTreePostQueryComms.hpp>
#include <ArborX_InterpDetailsMovingLeastSquaresCoefficients.hpp>
#include <ArborX_InterpDetailsMovingLeastSquaresPredicates.hpp>
#include <ArborX_PairIndexRank.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

namespace ArborX::Interpolation
{

template <typename MemorySpace, typename FloatingCalculationType>
class DistributedMovingLeastSquares
{
public:
  // If num_neighbors is 0 or negative, it will instead be a default value
  template <typename ExecutionSpace, typename SourcePoints,
            typename TargetPoints, typename CRBF, typename PolynomialDegree>
  DistributedMovingLeastSquares(MPI_Comm comm, ExecutionSpace const &space,
                                SourcePoints const &source_points,
                                TargetPoints const &target_points,
                                int num_neighbors, CRBF, PolynomialDegree)
  {
    static_assert(
        KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
        "Memory space must be accessible from the execution space");

    // SourcePoints is an access trait of points
    ArborX::Details::check_valid_access_traits(PrimitivesTag{}, source_points);
    using src_acc = AccessTraits<SourcePoints, PrimitivesTag>;
    static_assert(KokkosExt::is_accessible_from<typename src_acc::memory_space,
                                                ExecutionSpace>::value,
                  "Source points must be accessible from the execution space");
    using src_point =
        typename ArborX::Details::AccessTraitsHelper<src_acc>::type;
    GeometryTraits::check_valid_geometry_traits(src_point{});
    static_assert(GeometryTraits::is_point<src_point>::value,
                  "Source points elements must be points");
    static constexpr int dimension = GeometryTraits::dimension_v<src_point>;

    // TargetPoints is an access trait of points
    ArborX::Details::check_valid_access_traits(PrimitivesTag{}, target_points);
    using tgt_acc = AccessTraits<TargetPoints, PrimitivesTag>;
    static_assert(KokkosExt::is_accessible_from<typename tgt_acc::memory_space,
                                                ExecutionSpace>::value,
                  "Target points must be accessible from the execution space");
    using tgt_point =
        typename ArborX::Details::AccessTraitsHelper<tgt_acc>::type;
    GeometryTraits::check_valid_geometry_traits(tgt_point{});
    static_assert(GeometryTraits::is_point<tgt_point>::value,
                  "Target points elements must be points");
    static_assert(dimension == GeometryTraits::dimension_v<tgt_point>,
                  "Target and source points must have the same dimension");

    _num_neighbors =
        (num_neighbors <= 0)
            ? Details::polynomialBasisSize<dimension, PolynomialDegree::value>()
            : num_neighbors;
    // num_neighbors must be common to ALL processes

    _num_targets = tgt_acc::size(target_points);
    _source_size = source_points.extent(0);
    // There must be enough source points but it cannot be checked right now
    // as we need to know the total number of source points
    // ARBORX_ASSERT(num_neighbors <= _source_size);

    // Organize the source points as a tree
    using distributed_tree = DistributedTree<MemorySpace>;
    distributed_tree source_tree(comm, space, source_points);

    // Create the predicates
    Details::MLSPointsPredicateWrapper<TargetPoints> predicates{target_points,
                                                                _num_neighbors};

    // Query the source
    Kokkos::View<PairIndexRank *, MemorySpace> indices_and_ranks(
        "ArborX::DistributedMovingLeastSquares::indices_and_ranks", 0);
    Kokkos::View<int *, MemorySpace> offsets(
        "ArborX::DistributedMovingLeastSquares::offsets", 0);
    source_tree.query(space, predicates, indices_and_ranks, offsets);

    // Prepare the comms so that data can be distributed, and distribute the
    // source points
    auto const source_view =
        setCommsAndGetSourceView(comm, space, indices_and_ranks, _num_targets,
                                 _num_neighbors, source_points);

    // Compute the Moving Least Squares
    _coeffs = Kokkos::View<FloatingCalculationType **, MemorySpace>(
        "ArborX::DistributedMovingLeastSquares::coefficients", 0, 0);
    Details::movingLeastSquaresCoefficients<CRBF, PolynomialDegree>(
        space, source_view, target_points, _coeffs);
  }

  template <typename ExecutionSpace, typename SourcePoints>
  Kokkos::View<typename ArborX::Details::AccessTraitsHelper<
                   AccessTraits<SourcePoints, PrimitivesTag>>::type **,
               MemorySpace>
  setCommsAndGetSourceView(
      MPI_Comm comm, ExecutionSpace const &space,
      Kokkos::View<PairIndexRank *, MemorySpace> const &indices_and_ranks,
      int const num_targets, int const num_neighbors,
      SourcePoints const &source_points)
  {
    using src_acc = AccessTraits<SourcePoints, PrimitivesTag>;
    using src_point =
        typename ArborX::Details::AccessTraitsHelper<src_acc>::type;

    // Set up comms
    _comms = Details::DistributedTreePostQueryComms<MemorySpace>(
        comm, space, indices_and_ranks);

    // Transform data as view
    auto const num_local_points = src_acc::size(source_points);
    Kokkos::View<src_point *, MemorySpace> local_source_points(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedMovingLeastSquares::local_source_points"),
        num_local_points);
    Kokkos::parallel_for(
        "ArborX::DistributedMovingLeastSquares::local_source_points_fill",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_local_points),
        KOKKOS_LAMBDA(int const i) {
          local_source_points(i) = src_acc::get(source_points, i);
        });

    // Distribute points data (and must have the correct size)
    _comms.distribute(space, local_source_points);
    ARBORX_ASSERT(local_source_points.extent_int(0) ==
                  num_targets * num_neighbors);

    // Properly build the source view
    Kokkos::View<src_point **, MemorySpace> source_view(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedMovingLeastSquares::source_view"),
        num_targets, num_neighbors);
    Kokkos::parallel_for(
        "ArborX::DistributedMovingLeastSquares::source_view_fill",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
            space, {0, 0}, {num_targets, num_neighbors}),
        KOKKOS_LAMBDA(int const i, int const j) {
          auto index = i * num_neighbors + j;
          source_view(i, j) = local_source_points(index);
        });

    return source_view;
  }

  template <typename ExecutionSpace, typename SourceValues>
  Kokkos::View<typename SourceValues::non_const_value_type *,
               typename SourceValues::memory_space>
  apply(ExecutionSpace const &space, SourceValues const &source_values) const
  {
    static_assert(
        KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
        "Memory space must be accessible from the execution space");

    // SourceValues is a 1D view of all source values
    static_assert(Kokkos::is_view_v<SourceValues> && SourceValues::rank == 1,
                  "Source values must be a 1D view of values");
    static_assert(
        KokkosExt::is_accessible_from<typename SourceValues::memory_space,
                                      ExecutionSpace>::value,
        "Source values must be accessible from the execution space");

    // Source values must be a valuation on the points so must be as big as the
    // original input
    ARBORX_ASSERT(_source_size == source_values.extent_int(0));

    using value_t = typename SourceValues::non_const_value_type;
    using view_t = Kokkos::View<value_t *, typename SourceValues::memory_space>;

    // We distribute the source values so that each target has the correct
    // source values
    view_t local_source_values(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedMovingLeastSquares::local_source_values"),
        source_values.extent(0));
    Kokkos::deep_copy(space, local_source_values, source_values);
    _comms.distribute(space, local_source_values);

    int const num_targets = _num_targets;
    int const num_neighbors = _num_neighbors;
    auto const coeffs = _coeffs;

    view_t target_values(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedMovingLeastSquares::target_values"),
        num_targets);
    Kokkos::parallel_for(
        "ArborX::DistributedMovingLeastSquares::target_interpolation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_targets),
        KOKKOS_LAMBDA(int const i) {
          value_t tmp = 0;
          for (int j = 0; j < num_neighbors; j++)
            tmp += coeffs(i, j) * local_source_values(i * num_neighbors + j);
          target_values(i) = tmp;
        });

    return target_values;
  }

private:
  Kokkos::View<FloatingCalculationType **, MemorySpace> _coeffs;
  Details::DistributedTreePostQueryComms<MemorySpace> _comms;
  int _source_size;
  int _num_targets;
  int _num_neighbors;
};

} // namespace ArborX::Interpolation

#endif