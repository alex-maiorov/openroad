// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2018-2025, The OpenROAD Authors

#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include "nesterovPass.h"


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

struct ViolatingPath{
  std::vector<size_t> gCellIndexSequence;
  float negativeSlack;
};

class TimingPass : public NesterovPassBase
{
public:
  // TimingBase();
  TimingPass(std::shared_ptr<NesterovBaseCommon> nbc,
             grt::GlobalRouter* grt,
             rsz::Resizer* rs,
             sta::Sta* sta,
             utl::Logger* log);

  void runSTA(){
    sta_->updateTiming(false);

    // FIXME: Not sure what it does exactly, but allegedly required. Figure out what it is doing.
    sta_->ensureLibLinked();
  }
  void gradientPass(NesterovBaseCommon& nbc,
                    NesterovBaseVars& nbv,
                    const std::vector<FloatPoint>& grad) override;

private:

  std::vector<ViolatingPath>

  bool _enabled = false;
  grt::GlobalRouter* grt_ = nullptr;
  rsz::Resizer* rs_ = nullptr;
  utl::Logger* log_ = nullptr;
  sta::Sta* sta_ = nullptr;
  std::shared_ptr<NesterovBaseCommon> nbc_;

  size_t top_n=10; // how many violating paths per endpoint to attract
  float violation_weight;
  float

  std::vector<int> timingNetWeightOverflow_;
  std::vector<int> timingOverflowChk_;
  float net_weight_max_ = 5;
};

}  // namespace gpl
