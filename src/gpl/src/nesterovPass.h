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
        // Theoretically can overwrite grad_acc with anything, in practice most passes will be using grad as an accumulator of some sort.
        // TODO: Not sure if void is the best return type but it stays for now.
        virtual void gradientPass(NesterovBaseCommon& nbc, NesterovBaseVars& nbv, std::vector<FloatPoint>& grad) = 0;

        const std::string pass_name = "UndefinedPass"; // This may not be the optimal way to communicate debug info, discuss with advisors
    };


}; //namespace gpl
