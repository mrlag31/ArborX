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

#include <ArborX.hpp>
#include <ArborX_InterpDistributedMovingLeastSquares.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

#include <mpi.h>

using Point = ArborX::Point;

KOKKOS_FUNCTION double functionToApproximate(Point const &p)
{
  return Kokkos::cos(p[0] + p[1] / 4);
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);
  MPI_Init(&argc, &argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  ExecutionSpace space{};

  {
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    int mpi_rank;
    int mpi_size;
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    bool const is_zero = mpi_rank == 0;
    bool const is_last = mpi_rank == mpi_size - 1;

    // Source space is a 3x3 grid
    // Target space is 3 off-grid points
    //
    // Rank 0:
    //  ^
    //  |
    //  |           S
    //  |
    //  |
    //  |           S
    //  |
    //  | T
    // -+-----------S->
    //  |
    //
    // Rank size - 1:
    //  ^
    //  |
    //  S     S
    //  |
    //  |   T
    //  S     S  T
    //  |
    //  |
    // -S-----S------->
    //  |

    int const num_sources = is_zero ? 3 : is_last ? 6 : 0;
    int const num_targets = is_zero ? 1 : is_last ? 2 : 0;

    // Set up points
    Kokkos::View<Point *, MemorySpace> src_points("Example::src_points",
                                                  num_sources);
    Kokkos::View<Point *, MemorySpace> tgt_points("Example::tgt_points",
                                                  num_targets);
    Kokkos::parallel_for(
        "Example::make_points",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
        KOKKOS_LAMBDA(int const) {
          if (is_zero)
          {
            src_points(0) = {0., 2., 0.};
            src_points(1) = {1., 2., 0.};
            src_points(2) = {2., 2., 0.};
            tgt_points(0) = {2. / 6., 1. / 3., 0.};
          }
          else if (is_last)
          {
            src_points(0) = {0., 0., 0.};
            src_points(1) = {1., 0., 0.};
            src_points(2) = {2., 0., 0.};
            src_points(3) = {0., 1., 0.};
            src_points(4) = {1., 1., 0.};
            src_points(5) = {2., 1., 0.};
            tgt_points(0) = {4. / 6., 4. / 3., 0.};
            tgt_points(1) = {9. / 6., 3. / 3., 0.};
          }
        });

    // Set up values
    Kokkos::View<double *, MemorySpace> src_values("Example::src_values",
                                                   num_sources);
    Kokkos::View<double *, MemorySpace> app_values("Example::app_values",
                                                   num_targets);
    Kokkos::View<double *, MemorySpace> ref_values("Example::ref_values",
                                                   num_targets);
    Kokkos::parallel_for(
        "Example::make_values",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_sources),
        KOKKOS_LAMBDA(int const i) {
          src_values(i) = functionToApproximate(src_points(i));
          if (i < num_targets)
            ref_values(i) = functionToApproximate(tgt_points(i));
        });

    // Build the moving least squares coefficients
    ArborX::Interpolation::DistributedMovingLeastSquares<MemorySpace> dmls(
        mpi_comm, space, src_points, tgt_points);

    // Interpolate
    dmls.interpolate(space, src_values, app_values);

    if (is_zero)
    {
      double app_values_host[3];
      double ref_values_host[3];

      MPI_Recv(&app_values_host[0], 2, MPI_DOUBLE, mpi_size - 1, 0, mpi_comm,
               nullptr);
      MPI_Recv(&ref_values_host[0], 2, MPI_DOUBLE, mpi_size - 1, 0, mpi_comm,
               nullptr);

      app_values_host[2] = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, app_values)(0);
      ref_values_host[2] = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, ref_values)(0);
      auto diff = [=](int const i) {
        return Kokkos::abs(app_values_host[i] - ref_values_host[i]);
      };

      std::cout << "Approximated values: " << app_values_host[0] << ' '
                << app_values_host[1] << ' ' << app_values_host[2] << '\n';
      std::cout << "Real values        : " << ref_values_host[0] << ' '
                << ref_values_host[1] << ' ' << ref_values_host[2] << '\n';
      std::cout << "Differences        : " << diff(0) << ' ' << diff(1) << ' '
                << diff(2) << '\n';
    }
    else if (is_last)
    {
      auto app_values_host =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, app_values);
      auto ref_values_host =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref_values);

      MPI_Send(app_values_host.data(), 2, MPI_DOUBLE, 0, 0, mpi_comm);
      MPI_Send(ref_values_host.data(), 2, MPI_DOUBLE, 0, 0, mpi_comm);
    }
  }

  MPI_Finalize();
  return 0;
}
