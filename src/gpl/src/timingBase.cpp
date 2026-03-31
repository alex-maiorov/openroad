// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2018-2025, The OpenROAD Authors

#include "timingBase.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "db_sta/dbNetwork.hh"
#include "grt/GlobalRouter.h"
#include "nesterovBase.h"
#include "placerBase.h"
#include "rsz/Resizer.hh"
#include "sta/Fuzzy.hh"
#include "sta/NetworkClass.hh"
#include "sta/PathEnd.hh"
#include "sta/Search.hh"
#include "sta/Sta.hh"
#include "utl/Logger.h"

namespace gpl {

using utl::GPL;

TimingPass::TimingPass(sta::Sta* sta,
                       utl::Logger* log,
                       size_t top_n,
                       float proj_weight,
                       float end_to_end_weight,
                       float slack_sharpness,
                       float slack_offset)
    : sta_(sta),
      log_(log),
      top_n(top_n),
      proj_weight(proj_weight),
      end_to_end_weight(end_to_end_weight),
      slack_sharpness(slack_sharpness),
      slack_offset(slack_offset)
{
}

// TimingBase
TimingBase::TimingBase() = default;

TimingBase::TimingBase(std::shared_ptr<NesterovBaseCommon> nbc,
                       grt::GlobalRouter* grt,
                       rsz::Resizer* rs,
                       utl::Logger* log)
    : TimingBase()
{
  grt_ = grt;
  rs_ = rs;
  nbc_ = std::move(nbc);
  log_ = log;
}

void TimingBase::initTimingOverflowChk()
{
  timingOverflowChk_.clear();
  timingOverflowChk_.resize(timingNetWeightOverflow_.size(), false);
}

bool TimingBase::isTimingNetWeightOverflow(float overflow)
{
  int intOverflow = std::round(overflow * 100);
  // exception case handling
  if (timingNetWeightOverflow_.empty()
      || intOverflow > timingNetWeightOverflow_[0]) {
    return false;
  }

  bool needTdRun = false;
  for (int i = 0; i < timingNetWeightOverflow_.size(); i++) {
    if (timingNetWeightOverflow_[i] > intOverflow) {
      if (!timingOverflowChk_[i]) {
        timingOverflowChk_[i] = true;
        needTdRun = true;
      }
      continue;
    }
    return needTdRun;
  }
  return needTdRun;
}

void TimingBase::addTimingNetWeightOverflow(int overflow)
{
  std::vector<int>::iterator it
      = std::ranges::find(timingNetWeightOverflow_, overflow);

  // only push overflow when the overflow is not in vector.
  if (it == timingNetWeightOverflow_.end()) {
    timingNetWeightOverflow_.push_back(overflow);
  }

  // do sort in reverse order
  std::ranges::sort(timingNetWeightOverflow_, std::greater<int>());
}

void TimingBase::setTimingNetWeightOverflows(const std::vector<int>& overflows)
{
  // sort by decreasing order
  auto sorted = overflows;
  std::ranges::sort(sorted, std::greater<int>());
  for (auto& overflow : sorted) {
    addTimingNetWeightOverflow(overflow);
  }
  initTimingOverflowChk();
}

void TimingBase::deleteTimingNetWeightOverflow(int overflow)
{
  std::vector<int>::iterator it
      = std::ranges::find(timingNetWeightOverflow_, overflow);
  // only erase overflow when the overflow is in vector.
  if (it != timingNetWeightOverflow_.end()) {
    timingNetWeightOverflow_.erase(it);
  }
}

void TimingBase::clearTimingNetWeightOverflow()
{
  timingNetWeightOverflow_.clear();
}

size_t TimingBase::getTimingNetWeightOverflowSize() const
{
  return timingNetWeightOverflow_.size();
}

void TimingBase::setTimingNetWeightMax(float max)
{
  net_weight_max_ = max;
}

bool TimingBase::executeTimingDriven(bool run_journal_restore)
{
  rs_->findResizeSlacks(run_journal_restore);

  if (!run_journal_restore) {
    nbc_->fixPointers();
  }

  // get worst resize nets
  sta::NetSeq worst_slack_nets = rs_->resizeWorstSlackNets();

  if (worst_slack_nets.empty()) {
    log_->warn(
        GPL,
        105,
        "Timing-driven: no net slacks found. Timing-driven mode disabled.");
    return false;
  }

  // min/max slack for worst nets
  auto slack_min = rs_->resizeNetSlack(worst_slack_nets[0]).value();
  auto slack_max
      = rs_->resizeNetSlack(worst_slack_nets[worst_slack_nets.size() - 1])
            .value();

  log_->info(GPL, 106, "Timing-driven: worst slack {}", slack_min);

  if (sta::fuzzyInf(slack_min)) {
    log_->warn(GPL,
               102,
               "Timing-driven: no slacks found. Timing-driven mode disabled.");
    return false;
  }

  int weighted_net_count = 0;
  for (auto& gNet : nbc_->getGNets()) {
    // default weight
    gNet->setTimingWeight(1.0);
    if (gNet->getGPins().size() > 1) {
      auto net_slack_opt = rs_->resizeNetSlack(gNet->getPbNet()->getDbNet());
      if (!net_slack_opt) {
        continue;
      }
      auto net_slack = net_slack_opt.value();
      if (net_slack < slack_max) {
        if (slack_max == slack_min) {
          gNet->setTimingWeight(1.0);
        } else {
          // weight(min_slack) = net_weight_max_
          // weight(max_slack) = 1
          const float weight = 1
                               + (net_weight_max_ - 1) * (slack_max - net_slack)
                                     / (slack_max - slack_min);
          gNet->setTimingWeight(weight);
        }
        weighted_net_count++;
      }
      debugPrint(log_,
                 GPL,
                 "timing",
                 1,
                 "net:{} slack:{} weight:{}",
                 gNet->getPbNet()->getDbNet()->getConstName(),
                 net_slack,
                 gNet->getTotalWeight());
    }
  }

  debugPrint(log_,
             GPL,
             "timing",
             1,
             "Timing-driven: weighted {} nets.",
             weighted_net_count);
  return true;
}

// Fairly obvious what this should do. LLMs were fairly heavily utilized for
// this, however the code seems sensible.
std::vector<ViolatingPath> TimingPass::getViolatingPaths(int path_end_count)
{
  // Filter parameters for finding path ends.
  // nullptr means no restriction on that filter dimension.
  ExceptionFrom* from = nullptr;      // No from-pin filter
  ExceptionThruSeq* thrus = nullptr;  // No thru-pin filter
  ExceptionTo* to = nullptr;          // No to-pin filter
  bool unconstrained = false;         // Only report unconstrained endpoints

  // Empty scene sequence means search all timing scenes (corners)
  SceneSeq scenes;
  // Use max() to consider both min (setup) and max (hold) delay analysis
  const MinMaxAll* delay_min_max = MinMaxAll::max();

  // How many paths to find per path group (max means unlimited)
  int group_path_count = PathGroup::group_path_count_max;
  // Limit to top_n worst paths per unique endpoint pin
  int endpoint_path_count = path_end_count;
  bool unique_pins = false;        // Don't filter for unique pins
  bool unique_edges = false;       // Don't filter for unique edges
  float slack_min = -1e30f;        // Capture all paths (no lower bound)
  float slack_max = slack_offset;  // TODO: Architectural decision: Figure out
                                   // how to deal with near-violations.
  bool sort_by_slack = true;  // Sort results by slack (most negative first)

  // Empty path_groups means search all path groups (e.g., max, min, etc.)
  StdStringSeq path_groups;
  bool setup = true;      // Include setup timing paths
  bool hold = false;      // Exclude hold timing paths
  bool recovery = false;  // Exclude recovery paths
  bool removal = false;   // Exclude removal paths
  bool clk_gating_setup = false;
  bool clk_gating_hold = false;

  // Query STA for path ends matching our filter criteria
  // This returns paths sorted by slack (most critical first)
  PathEndSeq ends = sta_->findPathEnds(from,
                                       thrus,
                                       to,
                                       unconstrained,
                                       scenes,
                                       delay_min_max,
                                       group_path_count,
                                       endpoint_path_count,
                                       unique_pins,
                                       unique_edges,
                                       slack_min,
                                       slack_max,
                                       sort_by_slack,
                                       path_groups,
                                       setup,
                                       hold,
                                       recovery,
                                       removal,
                                       clk_gating_setup,
                                       clk_gating_hold);

  // Get the database network adapter for converting between OpenSTA and OpenDB
  // objects
  sta::dbNetwork* network = sta_->getDbNetwork();
  std::vector<ViolatingPath> violating_paths;

  // Iterate through each path endpoint found by STA
  for (PathEnd* end : ends) {
    // Get the endpoint pin of this path (the sink/flop input or output port)
    const Pin* pin = end->vertex(sta_)->pin();
    // Slack is negative for violating paths, positive for meeting timing.
    // We only query paths with slack <= slack_offset (typically <= 0).
    Slack slack = end->slack(sta_);

    // Skip paths with infinite slack (shouldn't happen with slack_max=0,
    // but guards against edge cases)
    if (sta::fuzzyInf(slack)) {
      continue;
    }

    // Create a violating path record for this endpoint
    ViolatingPath violating_path;
    violating_path.slack = slack;

    // Walk backwards through the path from endpoint to source
    // Each Path object represents a timing point in the path
    Path* path = end->path();
    std::vector<size_t> gCell_indices;

    while (path != nullptr) {
      // Get the pin at this timing point in the path
      const Pin* path_pin = path->pin(sta_);

      // Convert OpenSTA Pin* to OpenDB objects.
      // The network adapter can extract any of: dbITerm, dbBTerm, or dbModITerm
      odb::dbITerm* iterm = nullptr;
      odb::dbBTerm* bterm = nullptr;
      odb::dbModITerm* moditerm = nullptr;
      network->staToDb(path_pin, iterm, bterm, moditerm);

      // Try to find the corresponding GPin in the NesterovBase
      GPin* gPin = nullptr;
      if (iterm != nullptr) {
        // Internal pin (connected to an instance)
        gPin = nbc_->dbToNb(iterm);
      } else if (bterm != nullptr) {
        // Boundary pin (top-level input/output port)
        gPin = nbc_->dbToNb(bterm);
      }
      // moditerm pins (hierarchical) are not yet supported

      // If we found a GPin, extract its GCell and convert to index
      if (gPin != nullptr) {
        GCell* gCell = gPin->getGCell();
        if (gCell != nullptr) {
          // Get the unique index of this GCell in the placement grid
          size_t gCell_index = nbc_->getGCellIndex(gCell);
          gCell_indices.push_back(gCell_index);
        }
      }

      // Move to the previous timing point in the path (towards the source)
      path = path->prevPath();
    }

    // Store the sequence of GCell indices for this violating path
    violating_path.gCellIndexSequence = std::move(gCell_indices);
    violating_paths.push_back(violating_path);
  }

  return violating_paths;
}

TimingPass::gradientPass(NesterovBaseCommon& nbc,
                         NesterovBaseVars& nbv,
                         const std::vector<FloatPoint>& grad)
{
  if (!_enabled) {
    return;
  }

  std::vector<ViolatingPath> paths = getViolatingPaths(top_n);

  for (const auto& path : paths) {
    const auto& gCell_indices = path.gCellIndexSequence;
    if (gCell_indices.size() < 2) {
      continue;
    }

    GCell& end1 = nbc.getGCell(gCell_indices.front());
    GCell& end2 = nbc.getGCell(gCell_indices.back());

    const float end1_x = end1.cx();
    const float end1_y = end1.cy();
    const float end2_x = end2.cx();
    const float end2_y = end2.cy();

    if (std::abs(path.slack) > kMinSlackThreshold) {
      continue;
    }

    // Weight function: exp(-sharpness * (slack + offset))
    // Negative slack (violation) increases weight; zero slack gives weight =
    // exp(-offset).
    const float slack_weight
        = exp(-1.0f * slack_sharpness * (slack + slack_offset));

    for (size_t i = 0; i < gCell_indices.size(); ++i) {
      const size_t cell_idx = gCell_indices[i];
      GCell& cell = nbc.getGCell(cell_idx);

      const FloatPoint cell_pos{static_cast<float>(cell.cx()),
                                static_cast<float>(cell.cy())};
      const FloatPoint end1_pos{end1_x, end1_y};
      const FloatPoint end2_pos{end2_x, end2_y};

      FloatPoint force(0.0f, 0.0f);

      // Endpoint attraction force calc
      const bool is_endpoint = (i == 0 || i == gCell_indices.size() - 1);
      if (end_to_end_weight > 0.0f && is_endpoint) {
        const FloatPoint to_end1{end1_x - cell_pos.x, end1_y - cell_pos.y};
        const FloatPoint to_end2{end2_x - cell_pos.x, end2_y - cell_pos.y};
        const float scaled_force = end_to_end_weight * slack_weight;
        force = (to_end1 + to_end2) * scaled_force;
      }

      // Projection force calc
      if (proj_weight > 0.0f && gCell_indices.size() > 2 && !is_endpoint) {
        const FloatPoint proj_from_end1
            = proj_vector(cell_pos, end1_pos, end2_pos);
        const FloatPoint from_cell_to_proj
            = proj_from_end1 + (end1_pos - cell_pos);
        const float dist_sq = from_cell_to_proj.x * from_cell_to_proj.x
                              + from_cell_to_proj.y * from_cell_to_proj.y;
        const float proj_scaled_force = proj_weight * slack_weight * dist_sq;
        force = force + (from_cell_to_proj * proj_scaled_force);
      }

      grad[cell_idx] = grad[cell_idx] + force;
    }
  }
}

}  // namespace gpl
