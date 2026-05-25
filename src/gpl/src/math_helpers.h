#include <cmath>

namespace gpl{
// TODO: There are numerous ways of piecewise cheesing this that are probably significantly more efficient in the common case despite having branches
// Negate slope for the negative slack usecase
template <typename T>
T softplus_exact(T x, T x0, T slope, T sharpness){
    T kx = slope * sharpness * (x - x0);
    T retval;

    // Gated to avoid taking the exponential of a large number and getting infinity
    if (kx > T(0)){
        retval =  kx / sharpness + std::log1p(std::exp(-kx)) / sharpness;
    }
    else{
        retval = std::log1p(std::exp(kx)) / sharpness;
    }
    return retval;
}


}; // namespace gpl
