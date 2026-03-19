// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2018-2025, The OpenROAD Authors

#include "nesterovPass.h"
#include "nesterovBase.h"

// A pass on all cells, all other pass classes should inherit from it.
// Any pass class should expose any arguments or configuration data(such as weights) through bespoke means.
namespace gpl{

    void NesterovPassCellPairwiseBase::gradientPass(NesterovBaseCommon& nbc,
                                                    NesterovBaseVars& nbv,
                                                    const std::vector<FloatPoint>& grad)
    {

        auto nbc_gcells_ = nbc.getGCells();
        size_t n_cells = nbc_gcells_.size();

        // Found this comment in NesterovBase::updateGradients where I copied some code from. Not sure if this will apply to this new code.
        // FIXME: Figure out if this is actually true
        // ==================PASTED COMMENT BEGIN===================
        // TODO: This OpenMP parallel section is causing non-determinism. Consider
        // revisiting this in the future to restore determinism.
        // #pragma omp parallel for num_threads(nbc_->getNumThreads()) reduction(+ :
        // wireLengthGradSum_, densityGradSum_, gradSum)
        // ================== PASTED COMMENT END ===================


        // FIXME: The inflexibility of the accumulation is not ideal. Look into equally performant but more flexible methods to support different weights. Alternatively rewrite this class to group calculations by sets of cells rather than pairs.
        for (size_t src_idx = 0; src_idx < n_cells; src_idx++) {
            auto dest_idxs = nonZeroCellPairings(nbc, nbv, src_idx);
            for (auto dest_idx : dest_idxs){
                grad[src_idx] += calculatePairwiseGradient(nbc, nbv, src_idx, dest_idx);
            }
        }
    }

    std::set<size_t>& NesterovPassCellPairwiseBase::getConnectedCells(
        NesterovBaseCommon& nbc,
        NesterovBaseVars& nbv,
        size_t sourceGCellIndex)
    {
        GCell gCell = nbc.getGCell(sourceGCellIndex);
        auto pins = gCell.gPins();

        std::set<size_t> connectedGCells;

        // FIXME: Figure out a more efficient way to do this
        for (auto pin : pins) {
            GNet* net_ptr = pin->getGNet();
            net_ptr->

        }
    }
    //   std::set<size_t> NesterovPassCellToNetBase::nonZeroPinToNet(
    //     NesterovBaseCommon& nbc,
    //     NesterovBaseVars& nbv,
    //     size_t sourceGCellIndex)
    //   {
    //
    //   }

    std::set<size_t> NesterovPassCellToNetBase::getConnectedNetsToPin(
        NesterovBaseCommon& nbc,
        NesterovBaseVars& nbv,
        size_t sourceGCellIndex)
    {
        auto cell = nbc.getGCell(sourceGCellIndex);
        auto pins = cell.gPins();

        std::set<size_t> nets;

        for (auto pin : pins) {
            auto net_ptr = pin->getGNet();
            nets.insert(nbc.getGNetIndex(net_ptr));
        }

        return nets;
    }

    void NesterovPassCellToNetBase::gradientPass(NesterovBaseCommon& nbc,
                                             NesterovBaseVars& nbv,
                                             const std::vector<FloatPoint>& grad)
    {
        auto nbc_gcells_ = nbc.getGCells();
        size_t n_cells = nbc_gcells_.size();
        for (size_t cell_idx = 0; cell_idx < n_cells; cell_idx++) {
            auto net_idxs = nonZeroPinToNet(nbc, nbv, cell_idx);
            for (auto net_idx : net_idxs){
                grad[cell_idx] += calculatePinToNetGradient(nbc, nbv, cell_idx, net_idx);
            }
        }

    }



}  // namespace gpl
