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
#include <ArborX_HyperBox.hpp>
#include <ArborX_InterpDetailsCompactRadialBasisFunction.hpp>
#include <ArborX_InterpDetailsDistributedValuesDistributor.hpp>
#include <ArborX_InterpDetailsMovingLeastSquaresCoefficients.hpp>
#include <ArborX_InterpDetailsPolynomialBasis.hpp>
#include <ArborX_PairIndexRank.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <optional>

#include <mpi.h>

namespace ArborX::Interpolation::Details
{

// This is done to avoid a clash with another predicates access trait
template <typename TargetAccess>
struct DMLSPredicateWrapper
{
  TargetAccess target_access;
  int num_neighbors;
};

template <typename SourceAccess, typename SourceLocalView>
struct DMLSLocalPointsKernel
{
  SourceAccess source_access;
  SourceLocalView source_local;

  KOKKOS_FUNCTION void operator()(int const i) const
  {
    source_local(i) = source_access(i);
  }
};

} // namespace ArborX::Interpolation::Details

namespace ArborX
{

template <typename TargetAccess>
struct AccessTraits<Interpolation::Details::DMLSPredicateWrapper<TargetAccess>,
                    PredicatesTag>
{
  using Self = Interpolation::Details::DMLSPredicateWrapper<TargetAccess>;

  KOKKOS_FUNCTION static auto size(Self const &tp)
  {
    return tp.target_access.size();
  }

  KOKKOS_FUNCTION static auto get(Self const &tp, int const i)
  {
    return nearest(tp.target_access(i), tp.num_neighbors);
  }

  using memory_space = typename TargetAccess::memory_space;
};

} // namespace ArborX

namespace ArborX::Interpolation
{

template <typename MemorySpace, typename FloatingCalculationType = double>
class DistributedMovingLeastSquares
{
public:
  template <typename ExecutionSpace, typename SourcePoints,
            typename TargetPoints, typename CRBFunc = CRBF::Wendland<0>,
            typename PolynomialDegree = PolynomialDegree<2>>
  DistributedMovingLeastSquares(MPI_Comm comm, ExecutionSpace const &space,
                                SourcePoints const &source_points,
                                TargetPoints const &target_points, CRBFunc = {},
                                PolynomialDegree = {},
                                std::optional<int> num_neighbors = std::nullopt)
  {
    auto guard = Kokkos::Profiling::ScopedRegion(
        "ArborX::DistributedMovingLeastSquares");

    namespace KokkosExt = ArborX::Details::KokkosExt;

    static_assert(
        KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
        "Memory space must be accessible from the execution space");

    // SourcePoints is an access trait of points
    ArborX::Details::check_valid_access_traits(PrimitivesTag{}, source_points);
    using SourceAccess =
        ArborX::Details::AccessValues<SourcePoints, PrimitivesTag>;
    static_assert(
        KokkosExt::is_accessible_from<typename SourceAccess::memory_space,
                                      ExecutionSpace>::value,
        "Source points must be accessible from the execution space");
    using SourcePoint = typename SourceAccess::value_type;
    GeometryTraits::check_valid_geometry_traits(SourcePoint{});
    static_assert(GeometryTraits::is_point<SourcePoint>::value,
                  "Source points elements must be points");
    static constexpr int dimension = GeometryTraits::dimension_v<SourcePoint>;

    // TargetPoints is an access trait of points
    ArborX::Details::check_valid_access_traits(PrimitivesTag{}, target_points);
    using TargetAccess =
        ArborX::Details::AccessValues<TargetPoints, PrimitivesTag>;
    static_assert(
        KokkosExt::is_accessible_from<typename TargetAccess::memory_space,
                                      ExecutionSpace>::value,
        "Target points must be accessible from the execution space");
    using TargetPoint = typename TargetAccess::value_type;
    GeometryTraits::check_valid_geometry_traits(TargetPoint{});
    static_assert(GeometryTraits::is_point<TargetPoint>::value,
                  "Target points elements must be points");
    static_assert(dimension == GeometryTraits::dimension_v<TargetPoint>,
                  "Target and source points must have the same dimension");

    // This must be the same through all processes
    _num_neighbors =
        num_neighbors ? *num_neighbors
                      : Details::polynomialBasisSize<dimension,
                                                     PolynomialDegree::value>();

    TargetAccess target_access{target_points};
    SourceAccess source_access{source_points};

    _num_targets = target_access.size();
    _source_size = source_access.size();
    // There must be enough source points on all processes
    // KOKKOS_ASSERT(0 < _num_neighbors && _num_neighbors <= _source_size);

    // Search for neighbors and get the arranged source points
    auto source_local =
        searchNeighbors(comm, space, source_access, target_access);

    Kokkos::View<SourcePoint **, Kokkos::LayoutRight, MemorySpace,
                 Kokkos::MemoryUnmanaged>
        source_view(source_local.data(), _num_targets, _num_neighbors);

    // Compute the moving least squares coefficients
    _coeffs = Details::movingLeastSquaresCoefficients<CRBFunc, PolynomialDegree,
                                                      FloatingCalculationType>(
        space, source_view, target_access._values);
  }

  template <typename ExecutionSpace, typename SourceValues,
            typename ApproxValues>
  void interpolate(ExecutionSpace const &space,
                   SourceValues const &source_values,
                   ApproxValues &approx_values) const
  {
    auto guard = Kokkos::Profiling::ScopedRegion(
        "ArborX::DistributedMovingLeastSquares::interpolate");

    namespace KokkosExt = ArborX::Details::KokkosExt;

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

    // ApproxValues is a 1D view for approximated values
    static_assert(Kokkos::is_view_v<ApproxValues> && ApproxValues::rank == 1,
                  "Approx values must be a 1D view");
    static_assert(
        KokkosExt::is_accessible_from<typename ApproxValues::memory_space,
                                      ExecutionSpace>::value,
        "Approx values must be accessible from the execution space");
    static_assert(!std::is_const_v<typename ApproxValues::value_type>,
                  "Approx values must be writable");

    // Source values must be a valuation on the points so must be as big as the
    // original input
    KOKKOS_ASSERT(_source_size == source_values.extent_int(0));

    // Approx values must have the correct size
    KOKKOS_ASSERT(approx_values.extent_int(0) == _num_targets);

    using Value = typename ApproxValues::non_const_value_type;
    Kokkos::View<Value *, MemorySpace> local_source_values(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedMovingLeastSquares::local_source_values"),
        _source_size);

    Kokkos::parallel_for(
        "ArborX::DistributedMovingLeastSquares::source_values_copy",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _source_size),
        KOKKOS_CLASS_LAMBDA(int const i) {
          local_source_values(i) = source_values(i);
        });

    _distributor.distribute(space, local_source_values);

    Kokkos::parallel_for(
        "ArborX::DistributedMovingLeastSquares::target_interpolation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_targets),
        KOKKOS_CLASS_LAMBDA(int const i) {
          Value tmp = 0;
          for (int j = 0; j < _num_neighbors; j++)
            tmp += _coeffs(i, j) * local_source_values(i * _num_neighbors + j);
          approx_values(i) = tmp;
        });
  }

private:
  template <typename ExecutionSpace, typename SourceAccess,
            typename TargetAccess>
  auto searchNeighbors(MPI_Comm comm, ExecutionSpace const &space,
                       SourceAccess const &source_access,
                       TargetAccess const &target_access)
  {
    auto guard = Kokkos::Profiling::ScopedRegion(
        "ArborX::DistributedMovingLeastSquares::searchNeighbors");

    // Organize the source points as a tree
    using SourcePoint = typename SourceAccess::value_type;
    static_assert(std::is_same_v<SourcePoint, Point>,
                  "Source points must be regular ArborX::Point");
    DistributedTree<MemorySpace> source_tree(comm, space, source_access);

    // Create the predicates
    Details::DMLSPredicateWrapper<TargetAccess> predicates{target_access,
                                                           _num_neighbors};

    // Create the data
    Kokkos::View<PairIndexRank *, MemorySpace> indices_and_ranks(
        "ArborX::DistributedMovingLeastSquares::indices_and_ranks", 0);
    Kokkos::View<int *, MemorySpace> offsets(
        "ArborX::DistributedMovingLeastSquares::offsets", 0);

    // Query the tree
    source_tree.query(space, predicates, indices_and_ranks, offsets);

    // Make the distributor
    _distributor = Details::DistributedValuesDistributor<MemorySpace>(
        comm, space, indices_and_ranks);

    // Make the temporary views
    Kokkos::View<SourcePoint *, MemorySpace> source_local(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedMovingLeastSquares::source_local"),
        _num_targets * _num_neighbors);

    // Prepare kernels
    Details::DMLSLocalPointsKernel<SourceAccess, decltype(source_local)>
        local_kernel{source_access, source_local};

    Kokkos::parallel_for("ArborX::DistributedMovingLeastSquares::local_kernel",
                         Kokkos::RangePolicy<ExecutionSpace>(
                             space, 0, _num_targets * _num_neighbors),
                         local_kernel);

    _distributor.distribute(space, source_local);

    return source_local;
  }

  Kokkos::View<FloatingCalculationType **, MemorySpace> _coeffs;
  Details::DistributedValuesDistributor<MemorySpace> _distributor;
  int _num_targets;
  int _num_neighbors;
  int _source_size;
};

} // namespace ArborX::Interpolation

#endif
