// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2018-2025, The OpenROAD Authors

#pragma once
#include <cmath>

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

  // Multiply by scalar.
  // TODO: Figure out if this is a dangerous overload to do.
  FloatPoint operator*(const float& w) const{
    return FloatPoint(this->x * w, this->y * w);
  }
};

// class FloatVec
// {
// public:
//
// };

float floatPointDotProduct(FloatPoint a, FloatPoint b){
  return (a.x * b.x) + (a.y * b.y);
}

// Get Projection of src onto the vector A->B
FloatPoint proj_vector(FloatPoint src, FloatPoint a, FloatPoint b){
  FloatPoint ab = FloatPoint(b.x - a.x, b.y - a.y); // origin considered at A for this
  float ab_mag = sqrt(pow(ab.x, 2) + pow(ab.y, 2));

  FloatPoint as = FloatPoint(src.x - a.x, src.y - a.y); // vector from A to src
  float dot = floatPointDotProduct(as, ab);

  float proj_w = dot / (ab_mag * ab_mag);                // scalar projection factor

  return ab * proj_w;                                    // projection vector along ab
}

}  // namespace gpl
