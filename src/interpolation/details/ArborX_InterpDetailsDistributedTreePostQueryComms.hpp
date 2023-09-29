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

#ifndef ARBORX_INTERP_DETAILS_DISTRIBUTED_TREE_POST_QUERY_COMMS_HPP
#define ARBORX_INTERP_DETAILS_DISTRIBUTED_TREE_POST_QUERY_COMMS_HPP

#include <ArborX_DetailsDistributor.hpp>
#include <ArborX_DistributedTree.hpp>
#include <ArborX_PairIndexRank.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>

#include <mpi.h>

namespace ArborX::Interpolation::Details
{

template <typename MemorySpace>
class DistributedTreePostQueryComms
{
public:
  template <typename ExecutionSpace, typename IndicesAndRanks>
  DistributedTreePostQueryComms(MPI_Comm comm, ExecutionSpace const &space,
                                IndicesAndRanks const &indices_and_ranks)
      : _distributor(nullptr)
  {
    static_assert(
        KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
        "Memory space must be accessible from the execution space");

    // IndicesAndRanks must be a 1D view of ArborX::PairIndexRank
    static_assert(
        Kokkos::is_view_v<IndicesAndRanks> && IndicesAndRanks::rank == 1,
        "indices and ranks must be a 1D view of ArborX::PairIndexRank");
    static_assert(
        KokkosExt::is_accessible_from<typename IndicesAndRanks::memory_space,
                                      ExecutionSpace>::value,
        "indices and ranks must be accessible from the execution space");
    static_assert(std::is_same_v<typename IndicesAndRanks::non_const_value_type,
                                 PairIndexRank>,
                  "indices and ranks elements must be ArborX::PairIndexRank");

    _comm.reset(
        [comm]() {
          auto p = new MPI_Comm;
          MPI_Comm_dup(comm, p);
          return p;
        }(),
        [](MPI_Comm *p) {
          // Avoid freeing if MPI has already exited
          int mpi_finalized;
          MPI_Finalized(&mpi_finalized);
          if (!mpi_finalized)
            MPI_Comm_free(p);
          delete p;
        });

    int const data_len = indices_and_ranks.extent(0);
    int rank;
    MPI_Comm_rank(*_comm, &rank);

    Kokkos::View<int *, MemorySpace> mpi_tmp(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::DistributedTreePostQueryComms::mpi_tmp"),
        data_len);

    // Split indices/ranks
    auto [indices, ranks] =
        indicesAndRanksSplit(space, indices_and_ranks, data_len);

    // Computes what will be common to every exchange. Every time
    // someone wants to get the value from the same set of elements,
    // they will use the same list of recv and send indices.
    // The rank data will be saved inside the back distributor,
    // as the front one is not relevant once the recv indices
    // are computed.

    // This builds for each process a local array indicating how much
    // informatiom will be gathered
    _distributor = ArborX::Details::Distributor<MemorySpace>(*_comm);
    _num_requests = _distributor.createFromSends(space, ranks);

    // This creates the temporary buffer that will help when producing the
    // array that rebuilds the output
    Kokkos::View<int *, MemorySpace> mpi_rev_indices(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedTreePostQueryComms:::mpi_rev_indices"),
        _num_requests);
    ArborX::iota(space, mpi_tmp);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, _distributor, mpi_tmp, mpi_rev_indices);

    // This retrieves which source index a process wants and gives it to
    // the process owning the source
    _mpi_send_indices = Kokkos::View<int *, MemorySpace>(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedTreePostQueryComms::mpi_send_indices"),
        _num_requests);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, _distributor, indices, _mpi_send_indices);

    // This builds the temporary buffer that will create the reverse
    // distributor to dispatch the values
    Kokkos::View<int *, MemorySpace> mpi_rev_ranks(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedTreePostQueryComms::mpi_rev_ranks"),
        _num_requests);
    Kokkos::deep_copy(space, mpi_tmp, rank);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, _distributor, mpi_tmp, mpi_rev_ranks);

    // This will create the reverse of the previous distributor
    _distributor = ArborX::Details::Distributor<MemorySpace>(*_comm);
    _num_responses = _distributor.createFromSends(space, mpi_rev_ranks);

    // There should be enough responses to perfectly fill what was requested
    ARBORX_ASSERT(_num_responses == data_len);

    // The we send back the requested indices so that each process can rebuild
    // their output
    _mpi_recv_indices = Kokkos::View<int *, MemorySpace>(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedTreePostQueryComms::mpi_recv_indices"),
        _num_responses);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, _distributor, mpi_rev_indices, _mpi_recv_indices);
  }

  template <typename ExecutionSpace, typename IndicesAndRanks>
  static std::array<Kokkos::View<int *, MemorySpace>, 2>
  indicesAndRanksSplit(ExecutionSpace const &space,
                       IndicesAndRanks const &indices_and_ranks,
                       int const data_len)
  {
    Kokkos::View<int *, MemorySpace> indices(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::DistributedTreePostQueryComms::indices"),
        data_len);
    Kokkos::View<int *, MemorySpace> ranks(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::DistributedTreePostQueryComms::ranks"),
        data_len);

    Kokkos::parallel_for(
        "ArborX::DistributedTreePostQueryComms::indices_and_ranks_split",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, data_len),
        KOKKOS_LAMBDA(int const i) {
          indices(i) = indices_and_ranks(i).index;
          ranks(i) = indices_and_ranks(i).rank;
        });

    return {{indices, ranks}};
  }

  template <typename ExecutionSpace, typename Values>
  void distribute(ExecutionSpace const &space, Values &values) const
  {
    // Values is a 1D view of values
    static_assert(Kokkos::is_view_v<Values> && Values::rank == 1,
                  "values must be a 1D view");
    static_assert(KokkosExt::is_accessible_from<typename Values::memory_space,
                                                ExecutionSpace>::value,
                  "values must be accessible from the execution space");
    static_assert(!std::is_const_v<typename Values::value_type>,
                  "values must be writable");

    using value_t = typename Values::non_const_value_type;
    using memory_space = typename Values::memory_space;

    auto const mpi_send_indices = _mpi_send_indices;
    auto const mpi_recv_indices = _mpi_recv_indices;

    // We know what each process want so we prepare the data to be sent
    Kokkos::View<value_t *, memory_space> data_to_send(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedTreePostQueryComms::data_to_send"),
        _num_requests);
    Kokkos::parallel_for(
        "ArborX::DistributedTreePostQueryComms::data_to_send_fill",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_requests),
        KOKKOS_CLASS_LAMBDA(int const i) {
          data_to_send(i) = values(mpi_send_indices(i));
        });

    // We properly send the data, and each process has what it wants, but in the
    // wrong order
    Kokkos::View<value_t *, memory_space> data_to_recv(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedTreePostQueryComms::data_to_recv"),
        _num_responses);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, _distributor, data_to_send, data_to_recv);

    // So we fix this by moving everything
    Kokkos::resize(space, values, _num_responses);
    Kokkos::parallel_for(
        "ArborX::DistributedTreePostQueryComms::output_fill",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_responses),
        KOKKOS_CLASS_LAMBDA(int const i) {
          values(mpi_recv_indices(i)) = data_to_recv(i);
        });
  }

  DistributedTreePostQueryComms()
      : _distributor(nullptr)
  {}

private:
  std::shared_ptr<MPI_Comm> _comm;
  Kokkos::View<int *, MemorySpace> _mpi_send_indices;
  Kokkos::View<int *, MemorySpace> _mpi_recv_indices;
  ArborX::Details::Distributor<MemorySpace> _distributor;
  int _num_requests;
  int _num_responses;
};

} // namespace ArborX::Interpolation::Details

#endif