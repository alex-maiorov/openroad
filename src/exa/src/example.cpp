// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025-2025, The OpenROAD Authors

#include "exa/example.h"

#include <memory>
#include <utility>

#include "observer.h"
#include "odb/db.h"
#include "utl/Logger.h"

#include <random>
#include <vector>
#include <thread>
#include <chrono>

namespace exa {

// These need to be defined in the cpp due to the smart pointer in
// example.h to Observer which is only declared there.
Example::Example(odb::dbDatabase* db, utl::Logger* logger)
    : db_(db), logger_(logger)
{
}

Example::~Example() = default;

// Checks that a block exists and errors if not.  error() will throw an
// exception with the message.
odb::dbBlock* Example::getBlock()
{
  odb::dbChip* chip = db_->getChip();
  if (!chip) {
    logger_->error(utl::EXA, 2, "No chip exists.");
  }

  odb::dbBlock* block = chip->getBlock();
  if (!block) {
    logger_->error(utl::EXA, 3, "No block exists.");
  }

  return block;
}

// The example operation.
void Example::makeInstance(const char* name)
{
  logger_->info(utl::EXA, 1, "Making an example instance named {}", name);

  odb::dbBlock* block = getBlock();

  // Find an arbitrary master to instantiate
  odb::dbMaster* master = nullptr;
  for (odb::dbLib* lib : db_->getLibs()) {
    if (!lib->getMasters().empty()) {
      master = *lib->getMasters().begin();
      break;
    }
  }

  if (!master) {
    logger_->error(utl::EXA, 4, "No master found.");
  }

  // Make an instance and mark the instance as placed.  The default
  // is at (0, 0) with R0 orientation.
  odb::dbInst* inst = odb::dbInst::create(block, master, name);
  inst->setPlacementStatus(odb::dbPlacementStatus::PLACED);

  // Notify the observer that something interesting has happened.
  if (debug_observer_) {
    debug_observer_->makeInstance(inst);
  }
}

void Example::exerciseDbLog()
{
  logger_->info(utl::EXA, 100, "Starting exerciseDbLog");

  // 1. Metadata
  logger_->logToDbMetadata(utl::EXA, "test_metadata_key", "test_metadata_value");

  // 2. Single-threaded, non-bulk
  std::mt19937 gen(12345);
  std::uniform_int_distribution<int> dist_int(0, 100);
  std::uniform_real_distribution<double> dist_real(0.0, 1.0);

  for (int i = 0; i < 50; ++i) {
    logger_->logToDb<"id,val">(utl::EXA, 200, "exa_st_nonbulk", i, dist_real(gen));
  }

  // 3. Single-threaded, bulk
  std::vector<int> bulk_ids;
  std::vector<double> bulk_vals;
  std::vector<int> bulk_types;
  for (int i = 0; i < 100; ++i) {
    bulk_ids.push_back(i);
    bulk_vals.push_back(dist_real(gen));
    bulk_types.push_back(dist_int(gen));
  }
  
  logger_->logToDbBulk<"id,val,type">(utl::EXA, 201, "exa_st_bulk", bulk_ids.size(),
                                      bulk_ids.begin(), bulk_vals.begin(), bulk_types.begin());

  // 4. Multi-threaded, non-bulk
  auto mt_nonbulk_worker = [this](int thread_id) {
    std::mt19937 local_gen(thread_id);
    std::uniform_real_distribution<double> local_dist(0.0, 1.0);
    for (int i = 0; i < 50; ++i) {
      logger_->logToDb<"thread_id,iter,val">(utl::EXA, 202, "exa_mt_nonbulk", thread_id, i, local_dist(local_gen));
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back(mt_nonbulk_worker, i);
  }
  for (auto& t : threads) {
    t.join();
  }

  // 5. Multi-threaded, bulk
  auto mt_bulk_worker = [this](int thread_id) {
    std::mt19937 local_gen(thread_id);
    std::uniform_real_distribution<double> local_dist(0.0, 1.0);
    std::vector<int> t_ids;
    std::vector<int> t_iters;
    std::vector<double> t_vals;
    for (int i = 0; i < 100; ++i) {
      t_ids.push_back(thread_id);
      t_iters.push_back(i);
      t_vals.push_back(local_dist(local_gen));
    }
    logger_->logToDbBulk<"thread_id,iter,val">(utl::EXA, 203, "exa_mt_bulk", t_ids.size(),
                                               t_ids.begin(), t_iters.begin(), t_vals.begin());
  };

  threads.clear();
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back(mt_bulk_worker, i);
  }
  for (auto& t : threads) {
    t.join();
  }

  logger_->info(utl::EXA, 101, "Finished exerciseDbLog");
}

void Example::setDebug(std::unique_ptr<Observer>& observer)
{
  debug_observer_ = std::move(observer);
}

}  // namespace exa
