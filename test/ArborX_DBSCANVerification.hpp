/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILSDBSCANVERIFICATION_HPP
#define ARBORX_DETAILSDBSCANVERIFICATION_HPP

#include <ArborX_DetailsUtils.hpp>
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_View.hpp>

#include <set>
#include <stack>

namespace ArborX
{
namespace Details
{

// Check that connected core points have same cluster indices
// NOTE: if core_min_size = 2, all points are core points
template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename LabelsView>
bool verifyConnectedCorePointsShareIndex(ExecutionSpace const &exec_space,
                                         IndicesView indices, OffsetView offset,
                                         LabelsView labels, int core_min_size)
{
  int n = labels.size();

  int num_incorrect = 0;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::verify_connected_core_points",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update) {
        bool self_is_core_point = (offset(i + 1) - offset(i) >= core_min_size);
        if (self_is_core_point)
        {
          for (int jj = offset(i); jj < offset(i + 1); ++jj)
          {
            int j = indices(jj);
            bool neigh_is_core_point =
                (offset(j + 1) - offset(j) >= core_min_size);

            if (neigh_is_core_point && labels(i) != labels(j))
            {
#ifndef __SYCL_DEVICE_ONLY__
              printf("Connected cores do not belong to the same cluster: "
                     "%d [%d] -> %d [%d]\n",
                     i, labels(i), j, labels(j));
#endif
              update++;
            }
          }
        }
      },
      num_incorrect);
  return (num_incorrect == 0);
}

// Check that boundary points share index with at least one core point, and
// that noise points have index -1
template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename LabelsView>
bool verifyBoundaryAndNoisePoints(ExecutionSpace const &exec_space,
                                  IndicesView indices, OffsetView offset,
                                  LabelsView labels, int core_min_size)
{
  int n = labels.size();

  int num_incorrect = 0;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::verify_connected_boundary_points",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update) {
        bool self_is_core_point = (offset(i + 1) - offset(i) >= core_min_size);
        if (!self_is_core_point)
        {
          bool is_boundary = false;
          bool have_shared_core = false;
          for (int jj = offset(i); jj < offset(i + 1); ++jj)
          {
            int j = indices(jj);
            bool neigh_is_core_point =
                (offset(j + 1) - offset(j) >= core_min_size);

            if (neigh_is_core_point)
            {
              is_boundary = true;
              if (labels(i) == labels(j))
              {
                have_shared_core = true;
                break;
              }
            }
          }

          // Boundary point must be connected to a core point
          if (is_boundary && !have_shared_core)
          {
#ifndef __SYCL_DEVICE_ONLY__
            printf("Boundary point does not belong to a cluster: %d [%d]\n", i,
                   labels(i));
#endif
            update++;
          }
          // Noise points must have index -1
          if (!is_boundary && labels(i) != -1)
          {
#ifndef __SYCL_DEVICE_ONLY__
            printf("Noise point does not have index -1: %d [%d]\n", i,
                   labels(i));
#endif
            update++;
          }
        }
      },
      num_incorrect);
  return (num_incorrect == 0);
}

// Check that cluster indices are unique
template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename LabelsView>
bool verifyClustersAreUnique(ExecutionSpace const &exec_space,
                             IndicesView indices, OffsetView offset,
                             LabelsView labels, int core_min_size)
{
  int n = labels.size();

  // FIXME we don't want to modify the labels view in this check. What we
  // want here is to create a view on the host, and deep_copy into it.
  // create_mirror_view_and_copy won't work, because it is a no-op if labels
  // is already on the host.
  decltype(Kokkos::create_mirror_view(Kokkos::HostSpace{},
                                      std::declval<LabelsView>()))
      labels_host(Kokkos::ViewAllocateWithoutInitializing(
                      "ArborX::DBSCAN::labels_host"),
                  labels.size());
  Kokkos::deep_copy(exec_space, labels_host, labels);
  auto offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);
  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);

  auto is_core_point = [&](int i) {
    return offset_host(i + 1) - offset_host(i) >= core_min_size;
  };

  // Remove all boundary points from consideration (noise points are already -1)
  // The idea is that this way if labels were bridged through a boundary
  // point, we will count them as separate labels but with a shared cluster
  // index, which will fail the unique labels check
  for (int i = 0; i < n; ++i)
  {
    if (!is_core_point(i))
    {
      for (int jj = offset_host(i); jj < offset_host(i + 1); ++jj)
      {
        int j = indices_host(jj);
        if (is_core_point(j))
        {
          // The point is a boundary point
          labels_host(i) = -1;
          break;
        }
      }
    }
  }

  // Record all unique cluster indices
  std::set<int> unique_cluster_indices;
  for (int i = 0; i < n; ++i)
    if (labels_host(i) != -1)
      unique_cluster_indices.insert(labels_host(i));
  auto num_unique_cluster_indices = unique_cluster_indices.size();

  // Record all cluster indices, assigning a unique index to each (which is
  // different from the original cluster index). This will only use noise and
  // core points (see above).
  unsigned int num_clusters = 0;
  std::set<int> cluster_sets;
  for (int i = 0; i < n; ++i)
  {
    if (labels_host(i) >= 0)
    {
      auto id = labels_host(i);
      cluster_sets.insert(id);
      num_clusters++;

      // DFS search
      std::stack<int> stack;
      stack.push(i);
      while (!stack.empty())
      {
        auto k = stack.top();
        stack.pop();
        if (labels_host(k) >= 0)
        {
          labels_host(k) = -1;
          for (int jj = offset_host(k); jj < offset_host(k + 1); ++jj)
          {
            int j = indices_host(jj);
            if (is_core_point(j) || (labels_host(j) == id))
              stack.push(j);
          }
        }
      }
    }
  }
  if (cluster_sets.size() != num_unique_cluster_indices)
  {
    std::cerr << "Number of components does not match" << std::endl;
    return false;
  }
  if (num_clusters != num_unique_cluster_indices)
  {
    std::cerr << "Cluster IDs are not unique" << std::endl;
    return false;
  }

  return true;
}

template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename LabelsView>
bool verifyClusters(ExecutionSpace const &exec_space, IndicesView indices,
                    OffsetView offset, LabelsView labels, int core_min_size)
{
  int n = labels.size();
  if ((int)offset.size() != n + 1 ||
      ArborX::lastElement(offset) != (int)indices.size())
    return false;

  using Verify = bool (*)(ExecutionSpace const &, IndicesView, OffsetView,
                          LabelsView, int);

  for (auto verify : {static_cast<Verify>(verifyConnectedCorePointsShareIndex),
                      static_cast<Verify>(verifyBoundaryAndNoisePoints),
                      static_cast<Verify>(verifyClustersAreUnique)})
  {
    if (!verify(exec_space, indices, offset, labels, core_min_size))
      return false;
  }

  return true;
}

template <typename ExecutionSpace, typename Primitives, typename LabelsView>
bool verifyDBSCAN(ExecutionSpace exec_space, Primitives const &primitives,
                  float eps, int core_min_size, LabelsView const &labels)
{
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::verify");

  static_assert(Kokkos::is_view<LabelsView>{}, "");

  using MemorySpace = typename Primitives::memory_space;

  static_assert(std::is_same<typename LabelsView::value_type, int>{}, "");
  static_assert(std::is_same<typename LabelsView::memory_space, MemorySpace>{},
                "");

  ARBORX_ASSERT(eps > 0);
  ARBORX_ASSERT(core_min_size >= 2);

  ArborX::BVH<MemorySpace> bvh(exec_space, primitives);

  auto const predicates = buildPredicates(primitives, eps);

  Kokkos::View<int *, MemorySpace> indices("ArborX::DBSCAN::indices", 0);
  Kokkos::View<int *, MemorySpace> offset("ArborX::DBSCAN::offset", 0);
  ArborX::query(bvh, exec_space, predicates, indices, offset);

  auto passed = Details::verifyClusters(exec_space, indices, offset, labels,
                                        core_min_size);
  Kokkos::Profiling::popRegion();

  return passed;
}
} // namespace Details
} // namespace ArborX

#endif
