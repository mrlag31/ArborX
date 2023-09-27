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

#ifndef ARBORX_INTERP_DETAILS_MOVING_LEAST_SQUARES_PREDICATES_HPP
#define ARBORX_INTERP_DETAILS_MOVING_LEAST_SQUARES_PREDICATES_HPP

#include <ArborX_AccessTraits.hpp>

namespace ArborX::Interpolation::Details
{

// This is done to avoid a clash with another predicates access trait
// Points must be an access traits of points
template <typename Points>
struct MLSPointsPredicateWrapper
{
  Points target_points;
  int num_neighbors;
};

} // namespace ArborX::Interpolation::Details

namespace ArborX
{

template <typename Points>
struct AccessTraits<Interpolation::Details::MLSPointsPredicateWrapper<Points>,
                    PredicatesTag>
{
  KOKKOS_INLINE_FUNCTION static auto
  size(Interpolation::Details::MLSPointsPredicateWrapper<Points> const &tp)
  {
    return AccessTraits<Points, PrimitivesTag>::size(tp.target_points);
  }

  KOKKOS_INLINE_FUNCTION static auto
  get(Interpolation::Details::MLSPointsPredicateWrapper<Points> const &tp,
      int const i)
  {
    return nearest(
        AccessTraits<Points, PrimitivesTag>::get(tp.target_points, i),
        tp.num_neighbors);
  }

  using memory_space =
      typename AccessTraits<Points, PrimitivesTag>::memory_space;
};

} // namespace ArborX

#endif