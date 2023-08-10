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

// Example taken from DataTransferKit
// (https://github.com/ORNL-CEES/DataTransferKit)
// with MLS resolution from
// (http://dx.doi.org/10.1016/j.jcp.2015.11.055)

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>
#include <sstream>

#include "mpi_comms.hpp"
#include "symmetric_pseudoinverse_svd.hpp"
#include <mpi.h>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;
using DeviceSpace = Kokkos::Device<ExecutionSpace, MemorySpace>;

struct RBFWendland_0
{
  KOKKOS_INLINE_FUNCTION float operator()(float x)
  {
    x /= _radius;
    return (1.f - x) * (1.f - x);
  }

  float _radius;
};

struct MVPolynomialBasis_3D
{
  static constexpr std::size_t size = 10;

  KOKKOS_INLINE_FUNCTION Kokkos::Array<float, size>
  operator()(ArborX::Point const &p) const
  {
    return {{1.f, p[0], p[1], p[2], p[0] * p[0], p[0] * p[1], p[0] * p[2],
             p[1] * p[1], p[1] * p[2], p[2] * p[2]}};
  }
};

struct TargetPoints
{
  Kokkos::View<ArborX::Point *, MemorySpace> target_points;
  std::size_t num_neighbors;
};

template <>
struct ArborX::AccessTraits<TargetPoints, ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t size(TargetPoints const &tp)
  {
    return tp.target_points.extent(0);
  }

  static KOKKOS_FUNCTION auto get(TargetPoints const &tp, std::size_t i)
  {
    return ArborX::nearest(tp.target_points(i), tp.num_neighbors);
  }

  using memory_space = MemorySpace;
};

// Function to approximate
KOKKOS_INLINE_FUNCTION float manufactured_solution(ArborX::Point const &p)
{
  return Kokkos::sin(p[0]) * p[2] + p[1];
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::ScopeGuard guard(argc, argv);

  constexpr float epsilon = std::numeric_limits<float>::epsilon();
  constexpr std::size_t num_neighbors = MVPolynomialBasis_3D::size;
  constexpr std::size_t cube_side = 20;
  constexpr std::size_t source_points_num = cube_side * cube_side * cube_side;
  constexpr std::size_t target_points_num = 4;

  ExecutionSpace space{};
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  int mpi_size, mpi_rank;
  MPI_Comm_size(mpi_comm, &mpi_size);
  MPI_Comm_rank(mpi_comm, &mpi_rank);

  std::size_t local_source_points_num = source_points_num / mpi_size;

  Kokkos::View<ArborX::Point *, MemorySpace> source_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::source_points"),
      local_source_points_num);
  Kokkos::View<ArborX::Point *, MemorySpace> target_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::target_points"),
      target_points_num);
  auto target_points_host = Kokkos::create_mirror_view(target_points);

  // Generate source points (Organized within a [-10, 10]^3 cube)
  std::size_t thickness = cube_side / mpi_size;
  Kokkos::parallel_for(
      "Example::source_points_init",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(space, {0, 0, 0},
                                             {cube_side, cube_side, thickness}),
      KOKKOS_LAMBDA(int const i, int const j, int const k) {
        source_points(i * cube_side * thickness + j * thickness +
                      k) = ArborX::Point{
            20.f * (float(i) / (cube_side - 1) - .5f),
            20.f * (float(j) / (cube_side - 1) - .5f),
            20.f * (float(k + thickness * mpi_rank) / (cube_side - 1) - .5f)};
      });

  // Generate target points
  target_points_host(0) = ArborX::Point{1.f, 0.f, 1.f};
  target_points_host(1) = ArborX::Point{5.f, 5.f, 5.f};
  target_points_host(2) = ArborX::Point{-5.f, 5.f, 3.f};
  target_points_host(3) = ArborX::Point{1.f, -3.3f, 7.f};
  Kokkos::deep_copy(space, target_points, target_points_host);

  // Organize source points as tree
  ArborX::DistributedTree<MemorySpace> source_tree(mpi_comm, space,
                                                   source_points);

  // Perform the query and split the indices/ranks
  Kokkos::View<Kokkos::pair<int, int> *, MemorySpace> index_ranks(
      "Example::index_ranks", 0);
  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
  source_tree.query(space, TargetPoints{target_points, num_neighbors},
                    index_ranks, offsets);
  Kokkos::View<int *, MemorySpace> local_indices(
      "Example::local_indices", target_points_num * num_neighbors);
  Kokkos::View<int *, MemorySpace> local_ranks(
      "Example::local_ranks", target_points_num * num_neighbors);
  Kokkos::parallel_for(
      "Example::index_ranks_split",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0,
                                          target_points_num * num_neighbors),
      KOKKOS_LAMBDA(int const i) {
        local_indices(i) = index_ranks(i).first;
        local_ranks(i) = index_ranks(i).second;
      });

  MPIComms<ExecutionSpace, MemorySpace> comms(space, mpi_comm, local_indices,
                                              local_ranks);
  auto local_source_points = comms.distribute(space, source_points);

  // Now that we have the neighbors, we recompute their position using
  // their target point as the origin.
  // This is used as an optimisation later in the algorithm
  Kokkos::View<ArborX::Point **, MemorySpace> tr_source_points(
      "Example::tr_source_points", target_points_num, num_neighbors);
  Kokkos::parallel_for(
      "Example::transform_source_points",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        for (int j = offsets(i); j < offsets(i + 1); j++)
        {
          tr_source_points(i, j - offsets(i)) = ArborX::Point{
              local_source_points(j)[0] - target_points(i)[0],
              local_source_points(j)[1] - target_points(i)[1],
              local_source_points(j)[2] - target_points(i)[2],
          };
        }
      });

  // Compute the radii for the weight (phi) vector
  Kokkos::View<float *, MemorySpace> radii("Example::radii", target_points_num);
  Kokkos::parallel_for(
      "Example::radii_computation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        float radius = 10.f * epsilon;

        for (int j = 0; j < num_neighbors; j++)
        {
          float norm = ArborX::Details::distance(tr_source_points(i, j),
                                                 ArborX::Point{0.f, 0.f, 0.f});
          radius = (radius < norm) ? norm : radius;
        }

        radii(i) = 1.1f * radius;
      });

  // Compute the weight (phi) vector
  Kokkos::View<float **, MemorySpace> phi("Example::phi", target_points_num,
                                          num_neighbors);
  Kokkos::parallel_for(
      "Example::phi_computation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        RBFWendland_0 rbf{radii(i)};

        for (int j = 0; j < num_neighbors; j++)
        {
          float norm = ArborX::Details::distance(tr_source_points(i, j),
                                                 ArborX::Point{0.f, 0.f, 0.f});
          phi(i, j) = rbf(norm);
        }
      });

  // Compute multivariable Vandermonde (P) matrix
  Kokkos::View<float ***, MemorySpace> p("Example::vandermonde",
                                         target_points_num, num_neighbors,
                                         MVPolynomialBasis_3D::size);
  Kokkos::parallel_for(
      "Example::vandermonde_computation",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
          space, {0, 0}, {target_points_num, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        auto basis = MVPolynomialBasis_3D{}(tr_source_points(i, j));

        for (int k = 0; k < MVPolynomialBasis_3D::size; k++)
        {
          p(i, j, k) = basis[k];
        }
      });

  // Compute moment (A) matrix
  Kokkos::View<float ***, MemorySpace> a("Example::A", target_points_num,
                                         MVPolynomialBasis_3D::size,
                                         MVPolynomialBasis_3D::size);
  Kokkos::parallel_for(
      "Example::A_computation",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(space, {0, 0, 0},
                                             {target_points_num,
                                              MVPolynomialBasis_3D::size,
                                              MVPolynomialBasis_3D::size}),
      KOKKOS_LAMBDA(int const i, int const j, int const k) {
        float tmp = 0;
        for (int l = 0; l < num_neighbors; l++)
        {
          tmp += p(i, l, j) * p(i, l, k) * phi(i, l);
        }

        a(i, j, k) = tmp;
      });

  // Compute the pseudo inverse
  auto a_inv =
      SymmPseudoInverseSVD<float, ExecutionSpace,
                           MemorySpace>::compute_pseudo_inverses(space, a);

  // Compute the coefficients
  Kokkos::View<float **, MemorySpace> coeffs("Example::coefficients",
                                             target_points_num, num_neighbors);
  Kokkos::parallel_for(
      "Example::coefficients_computation",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
          space, {0, 0}, {target_points_num, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        float tmp = 0.f;

        for (int k = 0; k < MVPolynomialBasis_3D::size; k++)
        {
          tmp += a_inv(i, 0, k) * p(i, j, k) * phi(i, j);
        }

        coeffs(i, j) = tmp;
      });

  // Compute source values
  Kokkos::View<float *, MemorySpace> source_values("Example::source_values",
                                                   local_source_points_num);
  Kokkos::parallel_for(
      "Example::source_evaluation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, local_source_points_num),
      KOKKOS_LAMBDA(int const i) {
        source_values(i) = manufactured_solution(source_points(i));
      });

  auto local_source_values = comms.distribute(space, source_values);

  // Compute target values via interpolation
  Kokkos::View<float *, MemorySpace> target_values("Example::target_values",
                                                   target_points_num);
  Kokkos::parallel_for(
      "Example::target_interpolation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        float tmp = 0;
        for (int j = offsets(i); j < offsets(i + 1); j++)
        {
          tmp += coeffs(i, j - offsets(i)) * local_source_values(j);
        }
        target_values(i) = tmp;
      });

  // Compute target values via evaluation
  Kokkos::View<float *, MemorySpace> target_values_exact(
      "Example::target_values_exact", target_points_num);
  Kokkos::parallel_for(
      "Example::target_evaluation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        target_values_exact(i) = manufactured_solution(target_points(i));
      });

  // Show difference
  auto target_values_host = Kokkos::create_mirror_view(target_values);
  Kokkos::deep_copy(space, target_values_host, target_values);
  auto target_values_exact_host =
      Kokkos::create_mirror_view(target_values_exact);
  Kokkos::deep_copy(space, target_values_exact_host, target_values_exact);

  std::stringstream ss{};
  float error = 0.f;
  for (int i = 0; i < target_points_num; i++)
  {
    error = Kokkos::max(
        Kokkos::abs(target_values_host(i) - target_values_exact_host(i)) /
            Kokkos::abs(target_values_exact_host(i)),
        error);

    ss << mpi_rank << ": ==== Target " << i << '\n'
       << mpi_rank << ": Interpolation: " << target_values_host(i) << '\n'
       << mpi_rank << ": Real value   : " << target_values_exact_host(i)
       << '\n';
  }
  ss << mpi_rank << ": Maximum relative error: " << error << std::endl;

  std::cout << ss.str();

  MPI_Finalize();
  return 0;
}
