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
        virtual void gradientPass(NesterovBaseCommon& nbc, NesterovBaseVars& nbv, const std::vector<FloatPoint>& grad) = 0;
        const std::string pass_name = "UndefinedPass"; // This may not be the optimal way to communicate debug info, discuss with advisors
    };


    // Pass with cell-to-cell force of some kind. Do not use if you have some intelligent way of caching results or need flexible accumulation as the call implementation is designed to be parallelizable.
    class NesterovPassCellPairwiseBase : public NesterovPassBase
    {
    public:
        // Takes source cell ID, returns nonzero-force destination indeces for cell-to-cell force calculations.
        virtual std::set<size_t> nonZeroCellPairings(NesterovBaseCommon& nbc, NesterovBaseVars& nbv, size_t sourceGCellIndex){
            return getConnectedCells(nbc, nbv, sourceGCellIndex);
        }

        //Gives all cells connected via nets to this cell.
        std::set<size_t> getConnectedCells(NesterovBaseCommon& nbc,
                                            NesterovBaseVars& nbv,
                                            size_t sourceGCellIndex);

        // Calculates force on source given destination. Returns by value because floatpoint is the same size as a 64 bit pointer.
        virtual FloatPoint calculateCellPairwiseGradient(NesterovBaseCommon& nbc, NesterovBaseVars& nbv, size_t sourceGCellIndex, size_t destGCellIndex)=0;

        // Please read through what this does before using this class.
        void gradientPass(NesterovBaseCommon& nbc,
                        NesterovBaseVars& nbv,
                        const std::vector<FloatPoint>& grad) override;

    };

    // Force from each cell to the nets that it is connected to.
    class NesterovPassCellToNetBase : public NesterovPassBase
    {

        virtual FloatPoint calculatePinToNetGradient(NesterovBaseCommon& nbc, NesterovBaseVars& nbv, size_t GCellIndex, size_t GNetIndex)=0;


        // Default implementation, in case someone wants to add pairings without removing default ones.
        std::set<size_t> getConnectedNetsToPin(NesterovBaseCommon& nbc,
                                            NesterovBaseVars& nbv,
                                            size_t sourceGCellIndex);

        // Defaults to the nets the cell is connected to
        virtual std::set<size_t> nonZeroPinToNet(NesterovBaseCommon& nbc,
                                                NesterovBaseVars& nbv,
                                                size_t sourceGCellIndex)
        {
            return getConnectedNetsToPin(nbc, nbv, sourceGCellIndex);
        }

        void gradientPass(NesterovBaseCommon& nbc,
                        NesterovBaseVars& nbv,
                        const std::vector<FloatPoint>& grad) override;
    };
    // // Force
    // class NesterovPinPairwiseBase : public NesterovPassBase
    // {
    //
    //     virtual FloatPoint calculatePinPairwiseGradient(NesterovBaseCommon& nbc, NesterovBaseVars& nbv, size_t srcGPinIndex, size_t destGPinIndex)=0;
    //
    //     // Pin-based
    //     virtual std::set<std::pair<size_t, size_t>> nonZeroPinPairings(NesterovBaseCommon& nbc,
    //                                                                     NesterovBaseVars& nbv,
    //                                                                     size_t sourceGPinIndex) = 0;
    //
    //                                                 void gradientPass(NesterovBaseCommon& nbc,
    //                                                                 NesterovBaseVars& nbv,
    //                                                                 const std::vector<FloatPoint>& grad) override;
    // };



}; //namespace gpl
