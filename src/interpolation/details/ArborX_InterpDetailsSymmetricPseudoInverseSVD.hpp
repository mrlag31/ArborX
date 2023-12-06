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

#ifndef ARBORX_INTERP_DETAILS_SYMMETRIC_PSEUDO_INVERSE_SVD_HPP
#define ARBORX_INTERP_DETAILS_SYMMETRIC_PSEUDO_INVERSE_SVD_HPP

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Interpolation::Details
{

// Gets the argmax from the upper triangle part of a matrix
template <typename Matrix>
KOKKOS_FUNCTION auto argmaxUpperTriangle(Matrix const &mat)
{
  using value_t = typename Matrix::non_const_value_type;

  struct
  {
    value_t max = 0;
    int row = 0;
    int col = 0;
  } result;

  int const size = mat.extent(0);
  for (int i = 0; i < size; i++)
    for (int j = i + 1; j < size; j++)
    {
      value_t val = Kokkos::abs(mat(i, j));
      if (result.max < val)
      {
        result.max = val;
        result.row = i;
        result.col = j;
      }
    }

  return result;
}

// Pseudo-inverse of symmetric matrices using SVD
// We must find U, E (diagonal and positive) and V such that A = U.E.V^T
// We also suppose, as the input, that A is symmetric, so U = SV where S is
// a sign matrix (only 1 or -1 on the diagonal, 0 elsewhere).
// Thus A = U.ES.U^T and A^-1 = U.[ ES^-1 ].U^T
//
// mat <=> A
// diag <=> ES
// unit <=> U
template <typename Matrix, typename Diag, typename Unit>
KOKKOS_FUNCTION void symmetricPseudoInverseSVDKernel(Matrix &mat, Diag &diag,
                                                     Unit &unit)
{
  using value_t = typename Matrix::non_const_value_type;
  int const size = mat.extent(0);

  // We first initialize U as the identity matrix and copy A to ES
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
    {
      unit(i, j) = value_t(i == j);
      diag(i, j) = mat(i, j);
    }

  static constexpr value_t epsilon = Kokkos::Experimental::epsilon_v<float>;
  while (true)
  {
    // We have a guarantee that p < q
    auto const [max_val, p, q] = argmaxUpperTriangle(diag);
    if (max_val <= epsilon)
      break;

    auto const a = diag(p, p);
    auto const b = diag(p, q);
    auto const c = diag(q, q);

    // Our submatrix is now
    // +----------+----------+   +---+---+
    // | ES(p, p) | ES(p, q) |   | a | b |
    // +----------+----------+ = +---+---+
    // | ES(q, p) | ES(q, q) |   | b | c |
    // +----------+----------+   +---+---+

    // Let's compute x, y and theta such that
    // +---+---+              +---+---+
    // | a | b |              | x | 0 |
    // +---+---+ = R(theta) * +---+---+ * R(theta)^T
    // | b | c |              | 0 | y |
    // +---+---+              +---+---+

    value_t cos_theta;
    value_t sin_theta;
    value_t x;
    value_t y;
    if (a == c)
    {
      cos_theta = Kokkos::sqrt(value_t(2)) / 2;
      sin_theta = cos_theta;
      x = a + b;
      y = a - b;
    }
    else
    {
      auto const u = (2 * b) / (a - c);
      auto const v = 1 / Kokkos::sqrt(u * u + 1);
      cos_theta = Kokkos::sqrt((1 + v) / 2);
      sin_theta = Kokkos::copysign(Kokkos::sqrt((1 - v) / 2), u);
      x = (a + c + (a - c) / v) / 2;
      y = a + c - x;
    }

    // Now let's compute the following new values for U and ES
    // ES <- R'(theta)^T . ES . R'(theta)
    // U  <- U . R'(theta)

    // R'(theta)^T . ES . R'(theta)
    for (int i = 0; i < p; i++)
    {
      auto const es_ip = diag(i, p);
      auto const es_iq = diag(i, q);
      diag(i, p) = cos_theta * es_ip + sin_theta * es_iq;
      diag(i, q) = -sin_theta * es_ip + cos_theta * es_iq;
    }
    diag(p, p) = x;
    for (int i = p + 1; i < q; i++)
    {
      auto const es_pi = diag(p, i);
      auto const es_iq = diag(i, q);
      diag(p, i) = cos_theta * es_pi + sin_theta * es_iq;
      diag(i, q) = -sin_theta * es_pi + cos_theta * es_iq;
    }
    diag(q, q) = y;
    for (int i = q + 1; i < size; i++)
    {
      auto const es_pi = diag(p, i);
      auto const es_qi = diag(q, i);
      diag(p, i) = cos_theta * es_pi + sin_theta * es_qi;
      diag(q, i) = -sin_theta * es_pi + cos_theta * es_qi;
    }
    diag(p, q) = 0;

    // U . R'(theta)
    for (int i = 0; i < size; i++)
    {
      auto const u_ip = unit(i, p);
      auto const u_iq = unit(i, q);
      unit(i, p) = cos_theta * u_ip + sin_theta * u_iq;
      unit(i, q) = -sin_theta * u_ip + cos_theta * u_iq;
    }
  }

  // We compute the max to get a range of the invertible eigenvalues
  auto max_eigen = epsilon;
  for (int i = 0; i < size; i++)
    max_eigen = Kokkos::max(Kokkos::abs(diag(i, i)), max_eigen);

  // We invert the diagonal of ES, except if "0" is found
  for (int i = 0; i < size; i++)
    diag(i, i) =
        (Kokkos::abs(diag(i, i)) < max_eigen * epsilon) ? 0 : 1 / diag(i, i);

  // Then we fill out A as the pseudo inverse
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
    {
      value_t tmp = 0;
      for (int k = 0; k < size; k++)
        tmp += diag(k, k) * unit(i, k) * unit(j, k);
      mat(i, j) = tmp;
    }
}

} // namespace ArborX::Interpolation::Details

#endif