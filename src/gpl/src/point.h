// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2018-2025, The OpenROAD Authors

#pragma once
#include <cmath>
#include <utility>

namespace gpl {

class FloatPoint
{
 public:
  float x = 0;
  float y = 0;
  FloatPoint() = default;
   FloatPoint(float x, float y);
   FloatPoint(int x, int y);
   FloatPoint(std::pair<int, int> coords);

  FloatPoint operator+(const FloatPoint& other) const;
  FloatPoint operator-(const FloatPoint& other) const;
   FloatPoint operator*(const float& w) const;
   float distance(const FloatPoint& other) const;
   float magnitude() const;
};

float floatPointDotProduct(FloatPoint a, FloatPoint b);

FloatPoint proj_vector(FloatPoint src, FloatPoint a, FloatPoint b);

}  // namespace gpl
