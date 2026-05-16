// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2018-2025, The OpenROAD Authors

#include "point.h"

namespace gpl {

FloatPoint::FloatPoint(float x, float y) : x(x), y(y)
{
}

FloatPoint::FloatPoint(int x, int y) : x(static_cast<float>(x)), y(static_cast<float>(y))
{
}

FloatPoint::FloatPoint(std::pair<int, int> coords) : x(static_cast<float>(coords.first)), y(static_cast<float>(coords.second))
{
}

FloatPoint FloatPoint::operator+(const FloatPoint& other) const
{
  return FloatPoint(this->x + other.x, this->y + other.y);
}

FloatPoint FloatPoint::operator-(const FloatPoint& other) const
{
  return FloatPoint(this->x - other.x, this->y - other.y);
}

FloatPoint FloatPoint::operator*(const float& w) const
{
  return FloatPoint(this->x * w, this->y * w);
}

float FloatPoint::distance(const FloatPoint& other) const
{
  return std::sqrt(std::pow(this->x - other.x, 2) + std::pow(this->y - other.y, 2));
}

float FloatPoint::magnitude() const
{
  return std::sqrt(this->x * this->x + this->y * this->y);
}

float floatPointDotProduct(FloatPoint a, FloatPoint b)
{
  return (a.x * b.x) + (a.y * b.y);
}

FloatPoint proj_vector(FloatPoint src, FloatPoint a, FloatPoint b)
{
  FloatPoint ab = FloatPoint(b.x - a.x, b.y - a.y);
  float ab_mag = sqrt(pow(ab.x, 2) + pow(ab.y, 2));

  FloatPoint as = FloatPoint(src.x - a.x, src.y - a.y);
  float dot = floatPointDotProduct(as, ab);

  float proj_w = dot / (ab_mag * ab_mag);

  return ab * proj_w;
}

}  // namespace gpl
