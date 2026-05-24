#include <cmath>

namespace gpl{
// TODO: There are numerous ways of piecewise cheesing this that are probably significantly more efficient in the common case despite having branches
// Negate slope for the negative slack usecase
template <typename T>
T softplus_exact(T x, T x0, T slope, T sharpness){
    return std::log<T>(
        T(1) +
        std::exp<T>(slope * sharpness * (x - x0))
    )/sharpness;
}
}; // namespace gpl
