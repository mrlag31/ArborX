/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>
#include <ArborX_Interp.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

template <typename T>
KOKKOS_INLINE_FUNCTION double step(T const &p)
{
  return Kokkos::signbit(p[0]) ? 0 : 1;
}

using Point = ArborX::Point;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = typename ExecutionSpace::memory_space;

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);
  MPI_Init(&argc, &argv);
  ExecutionSpace space{};

  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  int mpi_rank;
  int mpi_size;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  static constexpr std::size_t local_num_points = 1000;
  std::size_t num_points = local_num_points * mpi_size;
  std::size_t local_offset = mpi_rank * local_num_points;

  Kokkos::View<Point *, MemorySpace> source_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::source_points"),
      local_num_points);
  Kokkos::View<double *, MemorySpace> source_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::source_values"),
      local_num_points);
  Kokkos::View<Point *, MemorySpace> target_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::target_points"),
      local_num_points);
  Kokkos::View<double *, MemorySpace> target_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::target_values"),
      local_num_points);
  Kokkos::parallel_for(
      "Example::fill_views",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, local_num_points),
      KOKKOS_LAMBDA(int const i) {
        float loc = (local_offset + i) / (num_points - 1.);
        float off = .5 / (num_points - 1.);

        source_points(i) = {2 * loc - 1, 0, 0};
        target_points(i) = {2 * (1 - loc) - 1 + off, 0, 0};

        source_values(i) = step(source_points(i));
        target_values(i) = step(target_points(i));
      });

  ArborX::Interpolation::DistributedMovingLeastSquares<MemorySpace, double> mls(
      mpi_comm, space, source_points, target_points);

  auto approx_values = mls.apply(space, source_values);

  double max_loc_error = 0.;
  Kokkos::parallel_reduce(
      "Example::reduce_error",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, local_num_points),
      KOKKOS_LAMBDA(int const i, double &loc_error) {
        loc_error = Kokkos::max(
            loc_error, Kokkos::abs(target_values(i) - approx_values(i)));
      },
      Kokkos::Max<double>(max_loc_error));

  double max_error = 0;
  MPI_Reduce(&max_loc_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_comm);

  if (mpi_rank == 0)
    std::cout << "Error: " << max_error << '\n';

  MPI_Finalize();
}