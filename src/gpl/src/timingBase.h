// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2018-2025, The OpenROAD Authors

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "db_sta/dbSta.hh"
#include "nesterovPass.h"
#include "sta/Sta.hh"

namespace grt {
class GlobalRouter;
}

namespace rsz {
class Resizer;
}

namespace sta {
class Sta;
}

namespace utl {
class Logger;
}

namespace gpl {

class NesterovBaseCommon;
class NesterovPassBase;
class GNet;

class TimingBase
{
 public:
  TimingBase();
  TimingBase(std::shared_ptr<NesterovBaseCommon> nbc,
             grt::GlobalRouter* grt,
             rsz::Resizer* rs,
             utl::Logger* log);

  // check whether overflow reached the timingOverflow
  bool isTimingNetWeightOverflow(float overflow);
  void addTimingNetWeightOverflow(int overflow);
  void setTimingNetWeightOverflows(const std::vector<int>& overflows);
  void deleteTimingNetWeightOverflow(int overflow);
  void clearTimingNetWeightOverflow();
  size_t getTimingNetWeightOverflowSize() const;

  void setTimingNetWeightMax(float max);

  grt::GlobalRouter* getGlobalRouter() const { return grt_; }
  rsz::Resizer* getResizer() const { return rs_; }

  // updateNetWeight.
  // True: successfully reweighted gnets
  // False: no slacks found
  bool executeTimingDriven(bool run_journal_restore);

 private:
  grt::GlobalRouter* grt_ = nullptr;
  rsz::Resizer* rs_ = nullptr;
  utl::Logger* log_ = nullptr;
  std::shared_ptr<NesterovBaseCommon> nbc_;

  std::vector<int> timingNetWeightOverflow_;
  std::vector<int> timingOverflowChk_;
  float net_weight_max_ = 5;
  void initTimingOverflowChk();
};

struct ViolatingPath
{
  std::vector<size_t> gCellIndexSequence;
  float slack;
};

class TimingPass : public NesterovPassBase
{
 public:
  TimingPass(sta::dbSta* sta,
             utl::Logger* log,
             size_t top_n = 10,
             float proj_weight = 1.0F,
             float end_to_end_weight = 1.0F,
             float slack_sharpness = 1.0F,
             float slack_offset = 0.0F);

  void runSTA()
  {
    sta_->updateTiming(false);
    sta_->ensureLibLinked();
  }
  void gradientPass(NesterovBaseCommon& nbc,
                    NesterovBaseVars& nbv,
                    std::vector<FloatPoint>& grad) override;

 private:
  std::vector<ViolatingPath> getViolatingPaths(int path_end_count, NesterovBaseCommon& nbc);

  bool _enabled = false;
  grt::GlobalRouter* grt_ = nullptr;
  rsz::Resizer* rs_ = nullptr;
  utl::Logger* log_ = nullptr;
  sta::dbSta* sta_ = nullptr;

  size_t top_n = 10;
  float proj_weight = 1.0F;
  float end_to_end_weight = 1.0F;
  float slack_sharpness = 1.0F;
  float slack_offset = 0.0F;

  static constexpr float kMinSlackThreshold = 1e-3f;
};

}  // namespace gpl
