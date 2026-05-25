#include <cmath>

namespace gpl {
// Pure normalized softplus: smooth approximation of max(0, x).
// sharpness controls the knee only; offset, negation, and slope
// scaling are all done by the caller on the input and output.
//
//   softplus_exact(x)  ≈  max(0, x)  for large |sharpness·x|
//
// sharpness should be normalized by the caller (e.g. multiplied by
// a slope parameter) so that it becomes a dimensionless knee-width
// factor rather than depending on the absolute unit of x.
template <typename T>
T softplus_exact(T x, T sharpness)
{
  T kx = sharpness * x;
  T retval;

  // Gated to avoid taking the exponential of a large number and getting
  // infinity.
  if (kx > T(0)) {
    retval = kx / sharpness + std::log1p(std::exp(-kx)) / sharpness;
  } else {
    retval = std::log1p(std::exp(kx)) / sharpness;
  }
  return retval;
}

};  // namespace gpl
