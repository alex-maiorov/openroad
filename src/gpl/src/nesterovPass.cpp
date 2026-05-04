// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2018-2025, The OpenROAD Authors
#include "nesterovPass.h"
#include "nesterovBase.h"

namespace gpl {

void WirelengthGradientPass::gradientPass(NesterovBaseCommon& nbc, NesterovBaseVars& nbv, std::vector<FloatPoint>& grad) {
    for (size_t i = 0; i < nbc.getGCells().size(); i++) {
        const GCellHandle& handle = nbc.getGCells()[i];
        grad[i] = nbc.getWireLengthGradientWA(handle, wlX_, wlY_);
    }
}

void DensityGradientPass::gradientPass(NesterovBaseCommon& nbc, NesterovBaseVars& nbv, std::vector<FloatPoint>& grad) {
    for (size_t i = 0; i < nbc.getGCells().size(); i++) {
        const GCellHandle& handle = nbc.getGCells()[i];
        grad[i] = nb_->getDensityGradient(handle);
    }
}

} // namespace gpl
