// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2018-2025, The OpenROAD Authors
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#include <set>

#include "boost/unordered/unordered_flat_map.hpp"
#include "gpl/Replace.h"
#include "odb/db.h"
#include "placerBase.h"
#include "point.h"
#include "routeBase.h"
#include "utl/Logger.h"
#include "nesterovBase.h"

namespace gpl {
    // A pass on all cells, all other pass classes should inherit from it.
    // Any pass class should expose any arguments or configuration data(such as weights) through bespoke means.
    class NesterovPassBase
    {
    public:

        // Takes the nbc/nbv, and accumulates its gradient to the vector passed in as grad. Is allowed to silently fail if the size of the vector does not match what is found in nbc.
        virtual void gradientPass(NesterovBaseCommon& nbc, NesterovBaseVars& nbv, std::vector<FloatPoint>& grad) = 0;
        // Optional: Get preconditioning factor on a per-gcell basis.
        virtual float getPrecondi(size_t GCellIndex) {
            return 1.0;
        }
        const std::string pass_name = "UndefinedPass"; 
    };

    class WirelengthGradientPass : public NesterovPassBase
    {
    public:
        WirelengthGradientPass(float wlX, float wlY) : wlX_(wlX), wlY_(wlY) {}
        void gradientPass(NesterovBaseCommon& nbc, NesterovBaseVars& nbv, std::vector<FloatPoint>& grad) override;

    private:
        float wlX_;
        float wlY_;
    };

    class DensityGradientPass : public NesterovPassBase
    {
    public:
        DensityGradientPass(NesterovBase* nb) : nb_(nb) {}
        void gradientPass(NesterovBaseCommon& nbc, NesterovBaseVars& nbv, std::vector<FloatPoint>& grad) override;

    private:
        NesterovBase* nb_;
    };




}; //namespace gpl
