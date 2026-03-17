// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2018-2025, The OpenROAD Authors

#pragma once

namespace gpl {

class FloatPoint
{
 public:
  float x = 0;
  float y = 0;
  FloatPoint() = default;
  FloatPoint(float x, float y) : x(x), y(y) {}
  FloatPoint operator+(const FloatPoint& other) const{
    return FloatPoint(this->x + other.x, this->y + other.y);
  }
};

}  // namespace gpl
