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

#include "ArborX_EnableDeviceTypes.hpp"
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_InterpDistributedMovingLeastSquares.hpp>

#include <Kokkos_Core.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE_TEMPLATE(distributed_moving_least_squares, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace space{};

  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  int mpi_rank;
  int mpi_size;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  // Case 1: f(x) = 3, 2 neighbors, linear
  //      -------0--------------->
  // SRC:        0   2   4   6 ...
  // TGT:          1   3   5   ...
  int local_source_id_0 = mpi_rank;
  int local_target_id_0 = mpi_rank % (mpi_size - 1);
  using point0 = ArborX::Point;
  Kokkos::View<point0 *, MemorySpace> srcp0("Testing::srcp0", 1);
  Kokkos::View<point0 *, MemorySpace> tgtp0("Testing::tgtp0", 1);
  Kokkos::View<double *, MemorySpace> srcv0("Testing::srcv0", 1);
  Kokkos::View<double *, MemorySpace> tgtv0("Testing::tgtv0", 1);
  Kokkos::View<double *, MemorySpace> eval0("Testing::eval0", 1);
  Kokkos::parallel_for(
      "for", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
      KOKKOS_LAMBDA(int const) {
        auto f = [](const point0 &) { return 3.; };

        srcp0(0) = {{2. * local_source_id_0, 0, 0}};
        srcv0(0) = f(srcp0(0));

        tgtp0(0) = {{2. * local_target_id_0 + 1, 0, 0}};
        tgtv0(0) = f(tgtp0(0));
      });
  ArborX::Interpolation::DistributedMovingLeastSquares<MemorySpace> dmls0(
      mpi_comm, space, srcp0, tgtp0, ArborX::Interpolation::CRBF::Wendland<0>{},
      ArborX::Interpolation::PolynomialDegree<1>{}, 2);
  dmls0.interpolate(space, srcv0, eval0);
  ARBORX_MDVIEW_TEST_TOL(eval0, tgtv0, Kokkos::Experimental::epsilon_v<float>);

  // Case 2: f(x, y) = xy + x, 8 neighbors, quad
  //  ^
  //  |
  //  S   S   S   ...
  //  | T   T     ...
  // -S---S---S-> ...
  //  | T   T     ...
  //  S   S   S   ...
  //  |
  int local_source_id_1 = mpi_rank;
  int local_target_id_1 = mpi_rank % (mpi_size - 1);
  using point1 = ArborX::Point;
  Kokkos::View<point1 *, MemorySpace> srcp1("Testing::srcp1", 3);
  Kokkos::View<point1 *, MemorySpace> tgtp1("Testing::tgtp1", 2);
  Kokkos::View<double *, MemorySpace> srcv1("Testing::srcv1", 3);
  Kokkos::View<double *, MemorySpace> tgtv1("Testing::tgtv1", 2);
  Kokkos::View<double *, MemorySpace> eval1("Testing::eval1", 2);
  Kokkos::parallel_for(
      "for", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
      KOKKOS_LAMBDA(int const) {
        auto f = [](const point1 &p) { return p[0] * p[1] + p[0]; };

        srcp1(0) = {{2. * local_source_id_1, 2, 0}};
        srcp1(1) = {{2. * local_source_id_1, 0, 0}};
        srcp1(2) = {{2. * local_source_id_1, -2, 0}};
        srcv1(0) = f(srcp1(0));
        srcv1(1) = f(srcp1(1));
        srcv1(2) = f(srcp1(2));

        tgtp1(0) = {{2. * local_target_id_1 + 1, 1, 0}};
        tgtp1(1) = {{2. * local_target_id_1 + 1, -1, 0}};
        tgtv1(0) = f(tgtp1(0));
        tgtv1(1) = f(tgtp1(1));
      });
  ArborX::Interpolation::DistributedMovingLeastSquares<MemorySpace> dmls1(
      mpi_comm, space, srcp1, tgtp1, ArborX::Interpolation::CRBF::Wendland<2>{},
      ArborX::Interpolation::PolynomialDegree<2>{}, 8);
  dmls1.interpolate(space, srcv1, eval1);
  ARBORX_MDVIEW_TEST_TOL(eval1, tgtv1, Kokkos::Experimental::epsilon_v<float>);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(distributed_moving_least_squares_edge_cases,
                              DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace space{};

  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  int mpi_rank;
  int mpi_size;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  // Case 1: Same as previous case 1, but points are 2D and locked on y=0
  // This has no effect as DistributedTree can only take 3D objects
  int local_source_id_0 = mpi_rank;
  int local_target_id_0 = mpi_rank % (mpi_size - 1);
  using point0 = ArborX::Point;
  Kokkos::View<point0 *, MemorySpace> srcp0("Testing::srcp0", 1);
  Kokkos::View<point0 *, MemorySpace> tgtp0("Testing::tgtp0", 1);
  Kokkos::View<double *, MemorySpace> srcv0("Testing::srcv0", 1);
  Kokkos::View<double *, MemorySpace> tgtv0("Testing::tgtv0", 1);
  Kokkos::View<double *, MemorySpace> eval0("Testing::eval0", 1);
  Kokkos::parallel_for(
      "for", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
      KOKKOS_LAMBDA(int const) {
        auto f = [](const point0 &) { return 3.; };

        srcp0(0) = {{2. * local_source_id_0, 0, 0}};
        srcv0(0) = f(srcp0(0));

        tgtp0(0) = {{2. * local_target_id_0 + 1, 0, 0}};
        tgtv0(0) = f(tgtp0(0));
      });
  ArborX::Interpolation::DistributedMovingLeastSquares<MemorySpace> dmls0(
      mpi_comm, space, srcp0, tgtp0, ArborX::Interpolation::CRBF::Wendland<0>{},
      ArborX::Interpolation::PolynomialDegree<1>{}, 2);
  dmls0.interpolate(space, srcv0, eval0);
  ARBORX_MDVIEW_TEST_TOL(eval0, tgtv0, Kokkos::Experimental::epsilon_v<float>);

  // Case 2: Same but corner source points are also targets
  int local_source_id_1 = mpi_rank;
  int local_target_id_1 = mpi_rank % (mpi_size - 1);
  using point1 = ArborX::Point;
  Kokkos::View<point1 *, MemorySpace> srcp1("Testing::srcp1", 3);
  Kokkos::View<point1 *, MemorySpace> tgtp1("Testing::tgtp1", 2);
  Kokkos::View<double *, MemorySpace> srcv1("Testing::srcv1", 3);
  Kokkos::View<double *, MemorySpace> tgtv1("Testing::tgtv1", 2);
  Kokkos::View<double *, MemorySpace> eval1("Testing::eval1", 2);
  Kokkos::parallel_for(
      "for", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
      KOKKOS_LAMBDA(int const) {
        auto f = [](const point1 &p) { return p[0] * p[1] + p[0]; };

        srcp1(0) = {{2. * local_source_id_1, 2, 0}};
        srcp1(1) = {{2. * local_source_id_1, 0, 0}};
        srcp1(2) = {{2. * local_source_id_1, -2, 0}};
        srcv1(0) = f(srcp1(0));
        srcv1(1) = f(srcp1(1));
        srcv1(2) = f(srcp1(2));

        tgtp1(0) = {{2. * local_target_id_1 + 2, 2, 0}};
        tgtp1(1) = {{2. * local_target_id_1, -2, 0}};
        tgtv1(0) = f(tgtp1(0));
        tgtv1(1) = f(tgtp1(1));
      });
  ArborX::Interpolation::DistributedMovingLeastSquares<MemorySpace> dmls1(
      mpi_comm, space, srcp1, tgtp1, ArborX::Interpolation::CRBF::Wendland<2>{},
      ArborX::Interpolation::PolynomialDegree<2>{}, 8);
  dmls1.interpolate(space, srcv1, eval1);
  ARBORX_MDVIEW_TEST_TOL(eval1, tgtv1, Kokkos::Experimental::epsilon_v<float>);
}
