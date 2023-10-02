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

#ifndef ARBORX_INTERP_DETAILS_COMPACT_RADIAL_BASIS_FUNCTION_HPP
#define ARBORX_INTERP_DETAILS_COMPACT_RADIAL_BASIS_FUNCTION_HPP

#include <Kokkos_Core.hpp>

namespace ArborX::Interpolation
{

namespace Details
{

template <typename T, std::size_t N>
KOKKOS_INLINE_FUNCTION T evaluatePolynomial(T const x, T const (&coeffs)[N])
{
  T eval = 0;
  for (std::size_t i = 0; i < N; i++)
    eval = x * eval + coeffs[i];
  return eval;
}

} // namespace Details

namespace CRBF
{

#define CRBF_DECL(NAME)                                                        \
  template <std::size_t>                                                       \
  struct NAME;

#define CRBF_DEF(NAME, N, FUNC)                                                \
  template <>                                                                  \
  struct NAME<N>                                                               \
  {                                                                            \
    template <typename T>                                                      \
    KOKKOS_INLINE_FUNCTION static constexpr T evaluate(T const y)              \
    {                                                                          \
      T const x = Kokkos::min(Kokkos::abs(y), T(1));                           \
      return Kokkos::abs(FUNC);                                                \
    }                                                                          \
  };

#define CRBF_POLY(...) Details::evaluatePolynomial<T>(x, {__VA_ARGS__})
#define CRBF_POW(X, N) Kokkos::pow(X, N)

CRBF_DECL(Wendland)
CRBF_DEF(Wendland, 0, CRBF_POW(1 - x, 2))
CRBF_DEF(Wendland, 2, CRBF_POW(1 - x, 4) * CRBF_POLY(4, 1))
CRBF_DEF(Wendland, 4, CRBF_POW(1 - x, 6) * CRBF_POLY(35, 18, 3))
CRBF_DEF(Wendland, 6, CRBF_POW(1 - x, 6) * CRBF_POLY(32, 25, 8, 1))

CRBF_DECL(Wu)
CRBF_DEF(Wu, 2, CRBF_POW(1 - x, 4) * CRBF_POLY(3, 12, 16, 4))
CRBF_DEF(Wu, 4, CRBF_POW(1 - x, 6) * CRBF_POLY(5, 30, 72, 82, 36, 6))

CRBF_DECL(Buhmann)
CRBF_DEF(Buhmann, 2,
         (x == T(0)) ? T(1) / 6
                     : CRBF_POLY(12 * Kokkos::log(x) - 21, 32, -12, 0, 1) / 6)
CRBF_DEF(Buhmann, 3,
         CRBF_POLY(5, 0, -84, 0, 1024 * Kokkos::sqrt(x) - 1890,
                   1024 * Kokkos::sqrt(x), -84, 0, 5) /
             5)
CRBF_DEF(Buhmann, 4,
         CRBF_POLY(99, 0, -4620, 9216 * Kokkos::sqrt(x),
                   -11264 * Kokkos::sqrt(x) + 6930, 0, -396, 0, 35) /
             35)

#undef CRBF_POW
#undef CRBF_POLY
#undef CRBF_DEF
#undef CRBF_DECL

} // namespace CRBF

} // namespace ArborX::Interpolation

#endif