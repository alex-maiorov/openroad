// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2020-2025, The OpenROAD Authors

#include "graphicsImpl.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "AbstractGraphics.h"
#include "gui/gui.h"
#include "nesterovBase.h"
#include "nesterovPlace.h"
#include "odb/db.h"
#include "placerBase.h"
#include "point.h"
#include "utl/Logger.h"

namespace gpl {

gui::Chart* GraphicsImpl::main_chart_ = nullptr;
gui::Chart* GraphicsImpl::density_chart_ = nullptr;
gui::Chart* GraphicsImpl::stepLength_chart_ = nullptr;
gui::Chart* GraphicsImpl::routing_chart_ = nullptr;

GraphicsImpl::GraphicsImpl(utl::Logger* logger)
    : HeatMapDataSource(logger, "gpl", "gpl"), logger_(logger), mode_(Mbff)
{
  gui::Gui::get()->registerRenderer(this);
}

GraphicsImpl::~GraphicsImpl() = default;

std::unique_ptr<AbstractGraphics> GraphicsImpl::MakeNew(
    utl::Logger* logger) const
{
  return std::make_unique<GraphicsImpl>(logger);
}

void GraphicsImpl::debugForMbff()
{
  setDebugOn(true);
  mode_ = Mbff;
}

void GraphicsImpl::debugForInitialPlace(
    std::shared_ptr<PlacerBaseCommon> pbc,
    std::vector<std::shared_ptr<PlacerBase>>& pbVec)
{
  setDebugOn(true);
  pbc_ = std::move(pbc);
  pbVec_ = pbVec;
  mode_ = Initial;
}

void GraphicsImpl::debugForNesterovPlace(
    NesterovPlace* np,
    std::shared_ptr<PlacerBaseCommon> pbc,
    std::shared_ptr<NesterovBaseCommon> nbc,
    std::shared_ptr<RouteBase> rb,
    std::vector<std::shared_ptr<PlacerBase>>& pbVec,
    std::vector<std::shared_ptr<NesterovBase>>& nbVec,
    bool draw_bins,
    odb::dbInst* debug_inst)
{
  pbc_ = std::move(pbc);
  nbc_ = std::move(nbc);
  rb_ = std::move(rb);
  pbVec_ = pbVec;
  nbVec_ = nbVec;
  np_ = np;
  draw_bins_ = draw_bins;
  mode_ = Nesterov;

  if (!gui::Gui::enabled()) {
    return;
  }

  if (debug_on_) {
    initCharts();
    addDisplayControl(kDrawInstances, true);
    addDisplayControl(kDrawTimingPaths, true);
    addDisplayControl(kDrawTimingGradientArrows, true);
    gui::Gui::get()->registerRenderer(this);

    if (debug_inst) {
      for (size_t idx = 0; idx < nbc_->getGCells().size(); ++idx) {
        auto cell = nbc_->getGCellByIndex(idx);
        if (cell->contains(debug_inst)) {
          selected_ = idx;
          break;
        }
      }
    }

    for (const auto& nb : nbVec_) {
      for (size_t idx = 0; idx < nb->getGCells().size(); ++idx) {
        GCellHandle cell_handle = nb->getGCells()[idx];
        if (cell_handle->contains(debug_inst)) {
          nb_selected_index_ = &nb - nbVec_.data();
          break;
        }
      }
    }
    initDebugHeatmap();
  }
}

void GraphicsImpl::initDebugHeatmap()
{
  addMultipleChoiceSetting(
      "Type",
      "Type:",
      []() {
        return std::vector<std::string>{
            "Density", "Overflow", "Overflow Normalized"};
      },
      [this]() -> std::string {
        switch (heatmap_type_) {
          case Density:
            return "Density";
          case Overflow:
            return "Overflow";
          case OverflowMinMax:
            return "Overflow Normalized";
        }
        return "Density";
      },
      [this](const std::string& value) {
        if (value == "Density") {
          heatmap_type_ = Density;
        } else if (value == "Overflow") {
          heatmap_type_ = Overflow;
        } else if (value == "Overflow Normalized") {
          heatmap_type_ = OverflowMinMax;
        } else {
          heatmap_type_ = Density;
        }
      });

  setChip(pbc_->db()->getChip());
  registerHeatMap();
}

void GraphicsImpl::initCharts()
{
  if (!gui::Gui::enabled()) {
    return;
  }
  gui::Gui* gui = gui::Gui::get();

  if (main_chart_ == nullptr) {
    main_chart_ = gui->addChart("GPL", "Iteration", {"HPWL (μm)", "Overflow"});
    main_chart_->setXAxisFormat("%d");
    main_chart_->setYAxisFormats({"%.2e", "%.2f"});
    main_chart_->setYAxisMin({std::nullopt, 0});
  }

  if (density_chart_ == nullptr) {
    density_chart_ = gui->addChart(
        "GPL Density Penalty", "Iteration", {"DensityPenalty", "phiCoef"});
    density_chart_->setXAxisFormat("%d");
    density_chart_->setYAxisFormats({"%.2e", "%.2f"});
    if (nbc_) {
      density_chart_->setYAxisMin({0.0, nbc_->getNbVars().minPhiCoef});
    }
  }

  if (stepLength_chart_ == nullptr) {
    stepLength_chart_ = gui->addChart(
        "GPL StepLength",
        "Iteration",
        {"StepLength", "CoordiDistance", "GradDistance", "Std area"});
    stepLength_chart_->setXAxisFormat("%d");
    stepLength_chart_->setYAxisFormats({"%.2e", "%.2f", "%.2f", "%.2f"});
    stepLength_chart_->setYAxisMin({0.0, 0.0, 0.0, 0.0});
  }

  if (routing_chart_ == nullptr && np_->getNpVars().routability_driven_mode) {
    routing_chart_ = gui->addChart(
        "GPL Routing",
        "Iteration",
        {"avg RUDY", "Std area", "% Overflow Tiles", "Total RUDY Overflow"});
    routing_chart_->setXAxisFormat("%d");
    routing_chart_->setYAxisFormats({"%.2f", "%.2f", "%.2f", "%.2f"});
    routing_chart_->setYAxisMin({0.0, 0.0, 0.0, 0.0});
  }
}

void GraphicsImpl::drawBounds(gui::Painter& painter)
{
  // draw core bounds
  auto& die = pbc_->getDie();
  painter.setPen(gui::Painter::kYellow, /* cosmetic */ true);
  painter.drawLine(die.coreLx(), die.coreLy(), die.coreUx(), die.coreLy());
  painter.drawLine(die.coreUx(), die.coreLy(), die.coreUx(), die.coreUy());
  painter.drawLine(die.coreUx(), die.coreUy(), die.coreLx(), die.coreUy());
  painter.drawLine(die.coreLx(), die.coreUy(), die.coreLx(), die.coreLy());
}

void GraphicsImpl::drawInitial(gui::Painter& painter)
{
  drawBounds(painter);

  painter.setPen(gui::Painter::kWhite, /* cosmetic */ true);
  for (auto& inst : pbc_->placeInsts()) {
    int lx = inst->lx();
    int ly = inst->ly();
    int ux = inst->ux();
    int uy = inst->uy();

    gui::Painter::Color color = gui::Painter::kDarkGreen;
    color.a = 180;
    painter.setBrush(color);
    painter.drawRect({lx, ly, ux, uy});
  }
}

void GraphicsImpl::drawField(gui::Painter& painter)
{
  for (size_t nb_idx = 0; nb_idx < nbVec_.size(); ++nb_idx) {
    const auto& nb = nbVec_[nb_idx];
    const auto& bins = nb->getBins();
    if (bins.empty()) {
      continue;
    }
    const auto& bin = *bins.begin();
    const auto size = std::max(bin.dx(), bin.dy());
    if (size * painter.getPixelsPerDBU() < 10) {  // too small
      return;
    }
    float efMax = 0;
    int max_len = std::numeric_limits<int>::max();
    for (auto& bin : bins) {
      efMax = std::max(efMax,
                       std::hypot(bin.electroFieldX(), bin.electroFieldY()));
      max_len = std::min({max_len, bin.dx(), bin.dy()});
    }

    for (auto& bin : bins) {
      float fx = bin.electroFieldX();
      float fy = bin.electroFieldY();
      float f = std::hypot(fx, fy);
      float ratio = f / efMax;
      float dx = fx / f * max_len * ratio;
      float dy = fy / f * max_len * ratio;

      int cx = bin.cx();
      int cy = bin.cy();

      gui::Painter::Color color
          = region_colors_[nb_idx % region_colors_.size()];
      painter.setPen(color, true);
      painter.drawLine(cx, cy, cx + dx, cy + dy);

      // Draw a circle at the outer end of the line
      int circle_x = static_cast<int>(cx + dx);
      int circle_y = static_cast<int>(cy + dy);
      float bin_area = bin.dx() * bin.dy();
      int circle_radius = static_cast<int>(0.05 * std::sqrt(bin_area / M_PI));
      painter.setPen(color, true);
      painter.drawCircle(circle_x, circle_y, circle_radius);
    }
  }
}

void GraphicsImpl::drawCells(const std::vector<GCellHandle>& cells,
                             gui::Painter& painter,
                             size_t nb_index)
{
  for (const auto& handle : cells) {
    const GCell* gCell = handle;
    drawSingleGCell(gCell, painter, nb_index);
  }
}

void GraphicsImpl::drawCells(const std::vector<GCell*>& cells,
                             gui::Painter& painter)
{
  for (const auto& gCell : cells) {
    drawSingleGCell(gCell, painter);
  }
}

void GraphicsImpl::drawSingleGCell(const GCell* gCell,
                                   gui::Painter& painter,
                                   size_t nb_index)
{
  const int gcx = gCell->dCx();
  const int gcy = gCell->dCy();

  int xl = gcx - gCell->dx() / 2;
  int yl = gcy - gCell->dy() / 2;
  int xh = gcx + gCell->dx() / 2;
  int yh = gcy + gCell->dy() / 2;

  gui::Painter::Color color;
  // Highlight modified instances (overrides base color, unless selected)
  switch (gCell->changeType()) {
    case GCell::GCellChange::kRoutability:
      color = gui::Painter::kWhite;
      color.a = 75;
      break;
    case GCell::GCellChange::kNewInstance:
      color = gui::Painter::kDarkRed;
      break;
    case GCell::GCellChange::kDownsize:
      color = gui::Painter::kDarkBlue;
      break;
    case GCell::GCellChange::kUpsize:
      color = gui::Painter::kOrange;
      break;
    case GCell::GCellChange::kResizeNoChange:
      color = gui::Painter::kDarkYellow;
      break;
    default:
      if (gCell->isInstance()) {
        color = gCell->isLocked()
                    ? gui::Painter::kTurquoise
                    : instances_colors_[nb_index % instances_colors_.size()];
      } else if (gCell->isFiller()) {
        // Use different colors for each NesterovBase
        color = region_colors_[nb_index % region_colors_.size()];
      }
      color.a = 180;
      break;
  }

  // Highlight selection (highest priority)
  if (selected_ != kInvalidIndex && gCell == nbc_->getGCellByIndex(selected_)) {
    color = gui::Painter::kYellow;
    color.a = 180;
  }

  gui::Painter::Color outline = gui::Painter::kBlack;
  outline.a = 150;
  painter.setPen(outline, /*cosmetic=*/false, /*width=*/1);
  painter.setBrush(color);
  painter.drawRect({xl, yl, xh, yh});
}

void GraphicsImpl::drawTimingPaths(gui::Painter& painter)
{
  if (!np_ || !np_->getNpVars().timingDrivenMode) {
    return;
  }

  // Draw centerlines between path endpoints for all violating paths.
  // Single color, no slack-based coloring.
  painter.setPen(gui::Painter::kMagenta, true);

  for (const auto& nb : nbVec_) {
    for (const auto& path : nb->getViolatingPathsStored()) {
      if (path.gCellIndexSequence.size() < 2) {
        continue;
      }

      const size_t first_idx = path.gCellIndexSequence.front();
      const size_t last_idx = path.gCellIndexSequence.back();

      if (first_idx >= nbc_->getGCells().size()
          || last_idx >= nbc_->getGCells().size()) {
        continue;
      }

      GCell* end1 = nbc_->getGCellByIndex(first_idx);
      GCell* end2 = nbc_->getGCellByIndex(last_idx);

      // Centerline
      painter.drawLine(end1->dCx(), end1->dCy(), end2->dCx(), end2->dCy());

      // Endpoint markers
      const int r1 = std::max(3, std::min(end1->dx(), end1->dy()) / 12);
      const int r2 = std::max(3, std::min(end2->dx(), end2->dy()) / 12);
      painter.drawCircle(end1->dCx(), end1->dCy(), r1);
      painter.drawCircle(end2->dCx(), end2->dCy(), r2);
    }
  }
}

void GraphicsImpl::drawTimingGradientArrows(gui::Painter& painter)
{
  if (!np_ || !np_->getNpVars().timingDrivenMode) {
    return;
  }

  // Collect all unique GCell indices that participate in violating paths
  std::unordered_set<size_t> path_cell_indices;
  for (const auto& nb : nbVec_) {
    for (const auto& path : nb->getViolatingPathsStored()) {
      for (size_t idx : path.gCellIndexSequence) {
        if (idx < nbc_->getGCells().size()) {
          path_cell_indices.insert(idx);
        }
      }
    }
  }

  if (path_cell_indices.empty()) {
    return;
  }

  // Compute timing gradient for each path cell, aggregated across all NBs
  struct CellGrad
  {
    FloatPoint grad;
    GCell* cell;
  };
  std::vector<CellGrad> cell_grads;
  cell_grads.reserve(path_cell_indices.size());

  for (size_t idx : path_cell_indices) {
    GCell* cell = nbc_->getGCellByIndex(idx);
    FloatPoint grad(0, 0);
    for (const auto& nb : nbVec_) {
      const FloatPoint nb_grad = nb->getTimingGradient(cell);
      grad.x += nb_grad.x;
      grad.y += nb_grad.y;
    }
    cell_grads.push_back({grad, cell});
  }

  // Find max magnitude for uniform scaling
  float max_mag = 0;
  for (const auto& cg : cell_grads) {
    const float mag = std::hypot(cg.grad.x, cg.grad.y);
    if (mag > max_mag) {
      max_mag = mag;
    }
  }

  if (max_mag <= std::numeric_limits<float>::epsilon()) {
    return;
  }

  // Draw arrows
  painter.setPen(gui::Painter::kMagenta, true);
  for (const auto& cg : cell_grads) {
    const float mag = std::hypot(cg.grad.x, cg.grad.y);
    if (mag <= std::numeric_limits<float>::epsilon()) {
      continue;
    }

    const int cx = cg.cell->dCx();
    const int cy = cg.cell->dCy();
    const int max_len = std::max(1, std::min(cg.cell->dx(), cg.cell->dy()));
    const float target_len = 0.45f * static_cast<float>(max_len);

    const float dx = cg.grad.x / max_mag * target_len;
    const float dy = cg.grad.y / max_mag * target_len;

    painter.drawLine(
        cx, cy, cx + static_cast<int>(dx), cy + static_cast<int>(dy));
  }
}

void GraphicsImpl::drawNesterov(gui::Painter& painter)
{
  drawBounds(painter);
  if (draw_bins_) {
    // Draw the bins
    painter.setPen(gui::Painter::kTransparent);

    for (const auto& nb : nbVec_) {
      for (auto& bin : nb->getBins()) {
        int density = bin.getDensity() * 50 + 20;
        gui::Painter::Color color;
        if (density > 255) {
          color = {255, 165, 0, 180};  // orange = out of the range
        } else {
          density = 255 - std::max(density, 20);
          color = {density, density, density, 180};
        }

        painter.setBrush(color);
        painter.drawRect({bin.lx(), bin.ly(), bin.ux(), bin.uy()});
      }
    }
  }

  // Draw the placeable objects
  if (checkDisplayControl(kDrawInstances)) {
    painter.setPen(gui::Painter::kWhite);
    drawCells(nbc_->getGCells(), painter);
    for (size_t nb_idx = 0; nb_idx < nbVec_.size(); ++nb_idx) {
      const auto& nb = nbVec_[nb_idx];
      drawCells(nb->getGCells(), painter, nb_idx);
    }
  }

  // Create lighter versions of the region_colors_ with alpha 50
  std::vector<gui::Painter::Color> light_colors;
  light_colors.reserve(region_colors_.size());
  for (const auto& color : region_colors_) {
    light_colors.emplace_back(color.r, color.g, color.b, 50);
  }

  for (size_t pb_idx = 0; pb_idx < pbVec_.size(); ++pb_idx) {
    const auto& pb = pbVec_[pb_idx];
    gui::Painter::Color color = light_colors[pb_idx % light_colors.size()];
    painter.setBrush(color);

    for (auto& pb_inst : pb->nonPlaceInsts()) {
      painter.drawRect(
          {pb_inst->lx(), pb_inst->ly(), pb_inst->ux(), pb_inst->uy()});
    }
  }

  // Draw timing path centerlines between violating path endpoints
  if (checkDisplayControl(kDrawTimingPaths)) {
    drawTimingPaths(painter);
  }

  // Draw lines to neighbors
  if (selected_ != kInvalidIndex && nbc_->getGCellByIndex(selected_)) {
    painter.setPen(gui::Painter::kYellow, true);
    for (GPin* pin : nbc_->getGCellByIndex(selected_)->gPins()) {
      GNet* net = pin->getGNet();
      if (!net) {
        continue;
      }
      for (GPin* other_pin : net->getGPins()) {
        GCell* neighbor = other_pin->getGCell();
        if (neighbor == nbc_->getGCellByIndex(selected_)) {
          continue;
        }
        painter.drawLine(
            pin->cx(), pin->cy(), other_pin->cx(), other_pin->cy());
      }
    }

    // Draw gradient direction lines in the GUI from the GCell center.
    // We scale vectors to fit nicely within the cell (similar to drawField()).
    const GCell* gcell = nbc_->getGCellByIndex(selected_);
    auto wlCoeffX = np_->getWireLengthCoefX();
    auto wlCoeffY = np_->getWireLengthCoefY();
    size_t nb_index = 0;
    if (nb_selected_index_ != kInvalidIndex) {
      nb_index = nb_selected_index_;
    } else {
      logger_->warn(
          utl::GPL, 317, "Selected instance not found in any NesterovBase");
    }
    FloatPoint densityGrad = nbVec_[nb_index]->getDensityGradient(gcell);
    FloatPoint wlGrad
        = nbc_->getWireLengthGradientWA(gcell, wlCoeffX, wlCoeffY);
    const int cx = gcell->dCx();
    const int cy = gcell->dCy();

    // Calculate the maximum length for the lines based on the GCell size
    const int max_len = std::max(1, std::min(gcell->dx(), gcell->dy()));
    const float target_len = 0.45f * static_cast<float>(max_len);

    // Determine the maximum magnitude for proper scaling
    const float wl_magnitude = std::hypot(wlGrad.x, wlGrad.y);
    FloatPoint timingGrad(0, 0);
    if (np_->getNpVars().timingDrivenMode) {
      timingGrad = nbVec_[nb_index]->getTimingGradient(gcell);
    }
    const float densityPenalty = nbVec_[nb_index]->getDensityPenalty();
    const float density_magnitude = std::hypot(densityPenalty * densityGrad.x,
                                               densityPenalty * densityGrad.y);
    const float timing_magnitude = std::hypot(timingGrad.x, timingGrad.y);
    const float overall_x = wlGrad.x + (densityPenalty * densityGrad.x);
    const float overall_y = wlGrad.y + (densityPenalty * densityGrad.y);
    const float overall_magnitude = std::hypot(overall_x, overall_y);
    const float max_magnitude = std::max(
        {wl_magnitude, density_magnitude, timing_magnitude, overall_magnitude});

    auto scaleVector = [&](float vx, float vy) -> std::pair<float, float> {
      const float magnitude = std::hypot(vx, vy);
      if (magnitude <= std::numeric_limits<float>::epsilon()) {
        return {0.0f, 0.0f};
      }
      return {vx / max_magnitude * target_len, vy / max_magnitude * target_len};
    };

    // Draw WL gradient line
    {
      auto [dx, dy] = scaleVector(wlGrad.x, wlGrad.y);
      painter.setPen(gui::Painter::kRed, true);  // Use red for WL gradient
      painter.drawLine(
          cx, cy, cx + static_cast<int>(dx), cy + static_cast<int>(dy));
    }

    // Draw Density gradient line
    {
      const float scaled_dx = densityPenalty * densityGrad.x;
      const float scaled_dy = densityPenalty * densityGrad.y;
      auto [dx, dy] = scaleVector(scaled_dx, scaled_dy);
      painter.setPen(gui::Painter::kBlue,
                     true);  // Use blue for Density gradient
      painter.drawLine(
          cx, cy, cx + static_cast<int>(dx), cy + static_cast<int>(dy));
    }

    // Draw Overall gradient line
    {
      auto [dx, dy] = scaleVector(overall_x, overall_y);
      painter.setPen(gui::Painter::kBlack,
                     true);  // Use black for Overall gradient
      painter.drawLine(
          cx, cy, cx + static_cast<int>(dx), cy + static_cast<int>(dy));
    }

    // Draw Timing gradient line (if timing driven mode active)
    if (np_->getNpVars().timingDrivenMode) {
      auto [dx, dy] = scaleVector(timingGrad.x, timingGrad.y);
      painter.setPen(gui::Painter::kMagenta,
                     true);  // Use magenta for Timing gradient
      painter.drawLine(
          cx, cy, cx + static_cast<int>(dx), cy + static_cast<int>(dy));
    }
  }

  // Draw per-cell timing gradient arrows
  if (checkDisplayControl(kDrawTimingGradientArrows)) {
    drawTimingGradientArrows(painter);
  }

  // Draw field lines
  if (draw_bins_) {
    drawField(painter);
  }
}

void GraphicsImpl::drawMBFF(gui::Painter& painter)
{
  painter.setPen(gui::Painter::kYellow, /* cosmetic */ true);
  for (const auto& [start, end] : mbff_edges_) {
    painter.drawLine(start, end);
  }

  for (odb::dbInst* inst : mbff_cluster_) {
    odb::Rect bbox = inst->getBBox()->getBox();
    painter.drawRect(bbox);
  }
}

void GraphicsImpl::drawObjects(gui::Painter& painter)
{
  if (!enabled()) {
    return;
  }

  switch (mode_) {
    case Mbff:
      drawMBFF(painter);
      break;
    case Nesterov:
      drawNesterov(painter);
      break;
    case Initial:
      drawInitial(painter);
      break;
  }
}

void GraphicsImpl::reportSelected()
{
  if (selected_ == kInvalidIndex) {
    return;
  }
  const GCell* gcell = nbc_->getGCellByIndex(selected_);
  logger_->report("Inst: {}", gcell->getName());

  if (np_) {
    auto wlCoeffX = np_->getWireLengthCoefX();
    auto wlCoeffY = np_->getWireLengthCoefY();

    logger_->report("  Wire Length Gradient");
    for (auto& gPin : gcell->gPins()) {
      FloatPoint wlGradPin
          = nbc_->getWireLengthGradientPinWA(gPin, wlCoeffX, wlCoeffY);
      const float weight = gPin->getGNet()->getTotalWeight();
      logger_->report("          ({:+.2e}, {:+.2e}) (weight = {}) pin {}",
                      wlGradPin.x,
                      wlGradPin.y,
                      weight,
                      gPin->getPbPin()->getName());
    }

    FloatPoint wlGrad
        = nbc_->getWireLengthGradientWA(gcell, wlCoeffX, wlCoeffY);
    logger_->report("  sum wl  ({: .2e}, {: .2e})", wlGrad.x, wlGrad.y);

    size_t nb_index = 0;
    if (nb_selected_index_ != kInvalidIndex) {
      nb_index = nb_selected_index_;
    } else {
      logger_->warn(
          utl::GPL, 318, "Selected instance not found in any NesterovBase");
    }
    FloatPoint densityGrad = nbVec_[nb_index]->getDensityGradient(gcell);
    float densityPenalty = nbVec_[nb_index]->getDensityPenalty();
    logger_->report("  density ({: .2e}, {: .2e}) (penalty: {})",
                    densityPenalty * densityGrad.x,
                    densityPenalty * densityGrad.y,
                    densityPenalty);

    FloatPoint timingGrad(0, 0);
    if (np_->getNpVars().timingDrivenMode) {
      timingGrad = nbVec_[nb_index]->getTimingGradient(gcell);
      logger_->report(
          "  timing ({: .2e}, {: .2e})", timingGrad.x, timingGrad.y);

      // Report violating path info for this cell
      int paths_through = 0;
      float worst_slack = 0.0f;
      for (const auto& path : nbVec_[nb_index]->getViolatingPathsStored()) {
        const auto& indices = path.gCellIndexSequence;
        if (std::find(indices.begin(), indices.end(), selected_)
            != indices.end()) {
          paths_through++;
          if (path.slack < worst_slack) {
            worst_slack = path.slack;
          }
        }
      }
      if (paths_through > 0) {
        logger_->report("  on {} violating path(s), worst slack: {:.3e}",
                        paths_through,
                        worst_slack);
      }
    }

    logger_->report("  overall ({: .2e}, {: .2e})",
                    wlGrad.x + densityPenalty * densityGrad.x + timingGrad.x,
                    wlGrad.y + densityPenalty * densityGrad.y + timingGrad.y);
  }
}

void GraphicsImpl::addIter(const int iter, const double overflow)
{
  if (!gui::Gui::enabled()) {
    return;
  }
  odb::dbBlock* block = pbc_->db()->getChip()->getBlock();
  main_chart_->addPoint(iter, {block->dbuToMicrons(nbc_->getHpwl()), overflow});

  std::vector<double> values;
  if (!nbVec_.empty() && nbVec_[0]) {
    values.push_back((static_cast<double>(nbVec_[0]->getDensityPenalty())));
    values.push_back(static_cast<double>(nbVec_[0]->getStoredPhiCoef()));
  } else {
    values.push_back(0.0);
    values.push_back(0.0);
  }
  density_chart_->addPoint(iter, values);

  values.clear();
  if (!nbVec_.empty() && nbVec_[0]) {
    values.push_back(static_cast<double>(nbVec_[0]->getStoredStepLength()));
    values.push_back(static_cast<double>(nbVec_[0]->getStoredCoordiDistance()));
    values.push_back(static_cast<double>(nbVec_[0]->getStoredGradDistance()));
    values.push_back(
        block->dbuAreaToMicrons(nbVec_[0]->getNesterovInstsArea()));
  } else {
    values.push_back(0.0);
    values.push_back(0.0);
    values.push_back(0.0);
    values.push_back(0.0);
  }
  stepLength_chart_->addPoint(iter, values);

  if (routing_chart_) {
    values.clear();
    if (!nbVec_.empty() && nbVec_[0] && rb_) {
      values.push_back(static_cast<double>(rb_->getRudyAverage()));
      values.push_back(
          block->dbuAreaToMicrons(nbVec_[0]->getNesterovInstsArea()));
      const double total_tiles = static_cast<double>(rb_->getTotalTilesCount());
      values.push_back(total_tiles > 0.0 ? (static_cast<double>(
                                                rb_->getOverflowedTilesCount())
                                            / total_tiles * 100.0)
                                         : 0.0);
      values.push_back((rb_->getTotalRudyOverflow()));
    } else {
      values.push_back(0.0);
      values.push_back(0.0);
      values.push_back(0.0);
      values.push_back(0.0);
    }
    routing_chart_->addPoint(iter, values);
  }
}

void GraphicsImpl::addTimingDrivenIter(const int iter)
{
  main_chart_->addVerticalMarker(iter, gui::Painter::kTurquoise);
  if (routing_chart_) {
    routing_chart_->addVerticalMarker(iter, gui::Painter::kTurquoise);
  }
}

void GraphicsImpl::addRoutabilitySnapshot(int iter)
{
  main_chart_->addVerticalMarker(iter, gui::Painter::kYellow);
  if (routing_chart_) {
    routing_chart_->addVerticalMarker(iter, gui::Painter::kYellow);
  }
}

void GraphicsImpl::addRoutabilityIter(const int iter, const bool revert)
{
  gui::Painter::Color color
      = revert ? gui::Painter::kRed : gui::Painter::kGreen;
  main_chart_->addVerticalMarker(iter, color);
  if (routing_chart_ && rb_) {
    routing_chart_->addVerticalMarker(
        iter, rb_->isMinRc() ? gui::Painter::kMagenta : gui::Painter::kBlack);
  }
}

void GraphicsImpl::cellPlotImpl(bool pause)
{
  gui::Gui::get()->redraw();
  if (pause) {
    reportSelected();
    gui::Gui::get()->pause();
  }
}

void GraphicsImpl::mbffMapping(const LineSegs& segs)
{
  mbff_edges_ = segs;
  gui::Gui::get()->redraw();
  gui::Gui::get()->pause();
  mbff_edges_.clear();
}

void GraphicsImpl::mbffFlopClusters(const std::vector<odb::dbInst*>& ffs)
{
  mbff_cluster_ = ffs;
  gui::Gui::get()->redraw();
  gui::Gui::get()->pause();
  mbff_cluster_.clear();
}

gui::SelectionSet GraphicsImpl::select(odb::dbTechLayer* layer,
                                       const odb::Rect& region)
{
  selected_ = kInvalidIndex;

  if (layer || !nbc_) {
    return gui::SelectionSet();
  }

  for (size_t idx = 0; idx < nbc_->getGCells().size(); ++idx) {
    auto cell = nbc_->getGCellByIndex(idx);
    const int gcx = cell->dCx();
    const int gcy = cell->dCy();

    int xl = gcx - cell->dx() / 2;
    int yl = gcy - cell->dy() / 2;
    int xh = gcx + cell->dx() / 2;
    int yh = gcy + cell->dy() / 2;

    if (region.xMax() < xl || region.yMax() < yl || region.xMin() > xh
        || region.yMin() > yh) {
      continue;
    }

    selected_ = idx;
    odb::dbInst* db_inst
        = cell->isInstance() ? cell->insts().front()->dbInst() : nullptr;
    if (db_inst != nullptr) {
      for (size_t nb_idx = 0; nb_idx < nbVec_.size(); ++nb_idx) {
        for (size_t gc_idx = 0; gc_idx < nbVec_[nb_idx]->getGCells().size();
             ++gc_idx) {
          GCellHandle cell_handle = nbVec_[nb_idx]->getGCells()[gc_idx];
          if (cell_handle->contains(db_inst)) {
            nb_selected_index_ = nb_idx;
            break;
          }
        }
      }
    }
    gui::Gui::get()->redraw();
    if (cell->isInstance()) {
      reportSelected();
      gui::SelectionSet selected;
      for (Instance* inst : cell->insts()) {
        selected.insert(gui::Gui::get()->makeSelected(inst->dbInst()));
      }
      return selected;
    }
  }
  return gui::SelectionSet();
}

void GraphicsImpl::status(const std::string_view message)
{
  gui::Gui::get()->status(std::string(message));
}

double GraphicsImpl::getGridXSize() const
{
  const BinGrid& grid = nbVec_[0]->getBinGrid();
  return grid.getBinSizeX() / (double) getBlock()->getDbUnitsPerMicron();
}

double GraphicsImpl::getGridYSize() const
{
  const BinGrid& grid = nbVec_[0]->getBinGrid();
  return grid.getBinSizeY() / (double) getBlock()->getDbUnitsPerMicron();
}

odb::Rect GraphicsImpl::getBounds() const
{
  return getBlock()->getCoreArea();
}

bool GraphicsImpl::populateMap()
{
  BinGrid& grid = nbVec_[0]->getBinGrid();
  odb::dbBlock* block = pbc_->db()->getChip()->getBlock();

  double min_value = std::numeric_limits<double>::max();
  double max_value = std::numeric_limits<double>::lowest();

  if (heatmap_type_ == OverflowMinMax) {
    for (const Bin& bin : grid.getBins()) {
      int64_t binArea = bin.getBinArea();
      const float scaledBinArea
          = static_cast<float>(binArea * bin.getTargetDensity());

      double value
          = std::max(0.0f,
                     static_cast<float>(bin.getInstPlacedAreaUnscaled())
                         + static_cast<float>(bin.getNonPlaceAreaUnscaled())
                         - scaledBinArea);
      value = block->dbuAreaToMicrons(value);

      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
    }
  }

  for (const Bin& bin : grid.getBins()) {
    odb::Rect box(bin.lx(), bin.ly(), bin.ux(), bin.uy());
    double value = 0.0;

    if (heatmap_type_ == Density) {
      value = bin.getDensity() * 100.0;
    } else if (heatmap_type_ == Overflow || heatmap_type_ == OverflowMinMax) {
      int64_t binArea = bin.getBinArea();
      const float scaledBinArea
          = static_cast<float>(binArea * bin.getTargetDensity());

      double raw_value
          = std::max(0.0f,
                     static_cast<float>(bin.getInstPlacedAreaUnscaled())
                         + static_cast<float>(bin.getNonPlaceAreaUnscaled())
                         - scaledBinArea);
      raw_value = block->dbuAreaToMicrons(raw_value);

      if (heatmap_type_ == OverflowMinMax && max_value > min_value) {
        value = (raw_value - min_value) / (max_value - min_value) * 100.0;
      } else {
        value = raw_value;
      }
    }

    addToMap(box, value);
  }

  return true;
}

void GraphicsImpl::populateXYGrid()
{
  BinGrid& grid = nbVec_[0]->getBinGrid();
  std::vector<Bin>& bin = grid.getBins();
  int x_grid = grid.getBinCntX();
  int y_grid = grid.getBinCntY();

  std::vector<int> x_grid_set, y_grid_set;
  x_grid_set.reserve(x_grid + 1);
  y_grid_set.reserve(y_grid + 1);

  x_grid_set.push_back(bin[0].lx());
  y_grid_set.push_back(bin[0].ly());

  for (int x = 0; x < x_grid && x < static_cast<int>(bin.size()); x++) {
    x_grid_set.push_back(bin[x].ux());
  }

  for (int y = 0; y < y_grid; y++) {
    size_t index = static_cast<size_t>(y) * static_cast<size_t>(x_grid);
    if (index < bin.size()) {
      y_grid_set.push_back(bin[index].uy());
    }
  }
  setXYMapGrid(x_grid_set, y_grid_set);
}

void GraphicsImpl::combineMapData(bool base_has_value,
                                  double& base,
                                  const double new_data,
                                  const double data_area,
                                  const double intersection_area,
                                  const double rect_area)
{
  base += new_data * intersection_area / rect_area;
}

bool GraphicsImpl::enabled()
{
  return debug_on_ && gui::Gui::enabled();
}

void GraphicsImpl::addFrameLabelImpl(const odb::Rect& bbox,
                                     std::string_view label,
                                     std::string_view label_name,
                                     int image_width_px)
{
  gui::Gui* gui = gui::Gui::get();

  int label_x = bbox.xMin() + 300;
  int label_y = bbox.yMin() + 300;

  gui::Painter::Color color = gui::Painter::kYellow;
  gui::Painter::Anchor anchor = gui::Painter::kBottomLeft;

  int font_size = std::clamp(image_width_px / 50, 15, 24);

  gui->addLabel(label_x,
                label_y,
                std::string(label),
                color,
                font_size,
                anchor,
                std::string(label_name));
}

void GraphicsImpl::saveLabeledImageImpl(std::string_view path,
                                        std::string_view label,
                                        std::string_view heatmap_control,
                                        int image_width_px)
{
  gui::Gui* gui = gui::Gui::get();

  odb::Rect bbox = pbc_->db()->getChip()->getBlock()->getBBox()->getBox();

  if (!heatmap_control.empty()) {
    gui->setDisplayControlsVisible(std::string(heatmap_control), true);
  }

  static int label_id = 0;
  std::string label_name = fmt::format("auto_label_{}", label_id++);

  addFrameLabel(bbox, label, label_name, image_width_px);
  gui->saveImage(std::string(path));
  gui->deleteLabel(label_name);

  if (!heatmap_control.empty()) {
    gui->setDisplayControlsVisible(std::string(heatmap_control), false);
  }

  gui->clearSelections();
}

int GraphicsImpl::gifStart(std::string_view path)
{
  return gui::Gui::get()->gifStart(std::string(path));
}

void GraphicsImpl::gifAddFrameImpl(int key,
                                   const odb::Rect& region,
                                   int width_px,
                                   double dbu_per_pixel,
                                   std::optional<int> delay)
{
  gui::Gui::get()->gifAddFrame(key, region, width_px, dbu_per_pixel, delay);
}

void GraphicsImpl::deleteLabel(std::string_view label_name)
{
  gui::Gui::get()->deleteLabel(std::string(label_name));
}

void GraphicsImpl::gifEnd(int key)
{
  gui::Gui::get()->gifEnd(key);
}

void GraphicsImpl::setDisplayControl(std::string_view name, bool value)
{
  gui::Gui::get()->setDisplayControlsVisible(std::string(name), value);
}

}  // namespace gpl
