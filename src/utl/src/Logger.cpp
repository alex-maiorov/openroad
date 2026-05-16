// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2020-2025, The OpenROAD Authors

#include "utl/Logger.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <fstream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stack>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <sqlite3.h>

#include "CommandLineProgress.h"
#include "utl/Metrics.h"
#if SPDLOG_VERSION < 10601
#include "spdlog/details/pattern_formatter.h"
#else
#include "spdlog/pattern_formatter.h"
#endif
#include "spdlog/common.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/ostream_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "utl/Progress.h"
#include "utl/prometheus/metrics_server.h"
#include "utl/prometheus/registry.h"

namespace utl {

Logger::Logger(const char* log_filename, const char* metrics_filename)
{
  progress_ = std::make_unique<CommandLineProgress>(this);

  sinks_.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
  if (log_filename) {
    sinks_.push_back(
        std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_filename));
  }

  logger_ = std::make_shared<spdlog::logger>(
      "logger", sinks_.begin(), sinks_.end());
  setFormatter();
  logger_->set_level(spdlog::level::level_enum::debug);

  if (metrics_filename) {
    addMetricsSink(metrics_filename);
  }

  metrics_policies_ = MetricsPolicy::makeDefaultPolicies();

  for (auto& counters : message_counters_) {
    for (auto& counter : counters) {
      counter = 0;
    }
  }

  for (auto& levels : message_levels_) {
    for (auto& level : levels) {
      level.store(spdlog::level::off, std::memory_order_relaxed);
    }
  }

  prometheus_registry_ = std::make_shared<PrometheusRegistry>();
}

Logger::~Logger()
{
  stopLogDb();
  finalizeMetrics();
}

void Logger::addMetricsSink(const char* metrics_filename)
{
  metrics_sinks_.emplace_back(metrics_filename);
}

void Logger::removeMetricsSink(const char* metrics_filename)
{
  auto metrics_file = std::ranges::find(metrics_sinks_, metrics_filename);
  if (metrics_file == metrics_sinks_.end()) {
    this->error(UTL, 11, "{} is not a metrics file", metrics_filename);
  }
  flushMetrics();

  metrics_sinks_.erase(metrics_file);
}

void Logger::startLogDb(const char* filename)
{
  if (db_) {
    return;
  }

  int rc = sqlite3_open_v2(filename, &db_,
                           SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
                           nullptr);
  if (rc != SQLITE_OK) {
    this->error(UTL, 109, "Failed to open SQLite database {}: {}", filename, sqlite3_errmsg(db_));
    sqlite3_close(db_);
    db_ = nullptr;
    return;
  }

  // Set fast mode
  sqlite3_exec(db_, "PRAGMA journal_mode = WAL;", nullptr, nullptr, nullptr);
  sqlite3_exec(db_, "PRAGMA synchronous = OFF;", nullptr, nullptr, nullptr);

  // Create system tables
  sqlite3_exec(db_,
      "CREATE TABLE IF NOT EXISTS tool_names ("
      "tool_id INTEGER PRIMARY KEY, name TEXT)",
      nullptr, nullptr, nullptr);
  sqlite3_exec(db_,
      "CREATE TABLE IF NOT EXISTS table_list ("
      "tool_id INTEGER, message_id INTEGER,"
      " column_types TEXT, column_names TEXT,"
      " PRIMARY KEY(tool_id, message_id))",
      nullptr, nullptr, nullptr);
  sqlite3_exec(db_,
      "CREATE TABLE IF NOT EXISTS metadata ("
      "tool_id INTEGER, key TEXT, value TEXT)",
      nullptr, nullptr, nullptr);

  // Populate tool_names table
  for (int i = 0; i < ToolId::SIZE; ++i) {
    std::string insert = fmt::format(
        "INSERT OR REPLACE INTO tool_names VALUES ({}, '{}')",
        i, tool_names_[i]);
    sqlite3_exec(db_, insert.c_str(), nullptr, nullptr, nullptr);
  }

  log_db_running_ = true;
  log_db_thread_ = std::thread(&Logger::logDbLoop, this);
}

void Logger::stopLogDb()
{
  log_db_running_ = false;
  if (log_db_thread_.joinable()) {
    log_db_thread_.join();
  }

  if (db_) {
    // Drain any remaining metadata rows.
    {
      std::queue<MetadataRow> local_meta;
      {
        std::lock_guard<std::mutex> lock(metadata_queue_mutex_);
        local_meta.swap(metadata_queue_);
      }

      if (!local_meta.empty()) {
        sqlite3_exec(db_, "BEGIN", nullptr, nullptr, nullptr);
        bool meta_ok = true;
        while (!local_meta.empty()) {
          auto [mtool, mkey, mval] = std::move(local_meta.front());
          local_meta.pop();

          char* sql = sqlite3_mprintf(
              "INSERT INTO metadata VALUES (%d, %q, %q)",
              static_cast<int>(mtool), mkey.c_str(), mval.c_str());
          int rc = sqlite3_exec(db_, sql, nullptr, nullptr, nullptr);
          sqlite3_free(sql);

          if (rc != SQLITE_OK) {
            meta_ok = false;
            break;
          }
        }
        sqlite3_exec(db_, meta_ok ? "COMMIT" : "ROLLBACK",
                     nullptr, nullptr, nullptr);
      }
    }

    // Catch any remaining data in queues that may have been registered
    // after the backend thread exited (or data pushed during the final
    // moments).  This also drains data for schemas whose
    // NewSchemaCommand was never processed by the thread.
    auto reg = schema_registry_.get_map();
    for (auto& [key, q] : *reg) {
      q->drain_to_db(db_, SIZE_MAX);
    }

    sqlite3_close(db_);
    db_ = nullptr;
  }
}

// ---------------------------------------------------------------------------
// Free helper utilities (moved from Logger.h)
// ---------------------------------------------------------------------------

namespace {

const char* sqlite_type_name(utl::SQLiteType t) {
  switch (t) {
    case utl::SQLiteType::INTEGER:
      return "INTEGER";
    case utl::SQLiteType::REAL:
      return "REAL";
    case utl::SQLiteType::TEXT:
      return "TEXT";
    case utl::SQLiteType::BLOB:
      return "BLOB";
  }
  return "TEXT";
}

// Split a comma-separated header string into individual column names,
// trimming leading/trailing whitespace from each.
std::vector<std::string> split_header(std::string_view header) {
  std::vector<std::string> fields;
  const char* cur = header.data();
  const char* end = cur + header.size();
  while (cur < end) {
    // Skip leading spaces
    while (cur < end && *cur == ' ') ++cur;
    const char* start = cur;
    // Find next comma or end
    while (cur < end && *cur != ',') ++cur;
    // Trim trailing spaces
    const char* stop = cur;
    while (stop > start && *(stop - 1) == ' ') --stop;
    if (stop > start || (!fields.empty() && stop == start)) {
      fields.emplace_back(start, stop - start);
    }
    if (cur < end) ++cur;  // skip comma
  }
  return fields;
}

std::vector<ColumnDefinition> build_columns_from_runtime(
    std::string_view header,
    const std::vector<SQLiteType>& types) {
  auto names = split_header(header);
  // Safety: if counts don't match, pad with "unknown"
  std::vector<ColumnDefinition> cols;
  cols.reserve(types.size());
  for (size_t i = 0; i < types.size(); ++i) {
    std::string name = (i < names.size()) ? names[i] : "unknown";
    cols.push_back({std::move(name), types[i]});
  }
  return cols;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// logToDb helpers  (called by the thin template in Logger.h)
// ---------------------------------------------------------------------------

std::optional<std::shared_ptr<AbstractQueue>> Logger::logToDbFindQueue(SchemaKey key)
{
  auto map = schema_registry_.get_map();
  auto it = map->find(key);
  if (it != map->end()) {
    return it->second;
  }
  return std::nullopt;
}

SchemaInfo Logger::logToDbBuildSchemaInfo(
    sqlite3* db,
    SchemaKey key,
    std::string_view header,
    const std::vector<SQLiteType>& types)
{
  SchemaInfo info;
  info.table_name
      = std::string(tool_names_[key.tool]) + "_" + std::to_string(key.id);
  info.columns = build_columns_from_runtime(header, types);

  // Build and execute CREATE TABLE IF NOT EXISTS in one shot.
  std::string create_sql
      = "CREATE TABLE IF NOT EXISTS " + info.table_name + " (";
  for (size_t i = 0; i < info.columns.size(); ++i) {
    if (i > 0)
      create_sql += ", ";
    create_sql
        += info.columns[i].name + " " + sqlite_type_name(info.columns[i].type);
  }
  create_sql += ");";

  char* err_msg = nullptr;
  int rc = sqlite3_exec(db, create_sql.c_str(), nullptr, nullptr, &err_msg);
  if (rc != SQLITE_OK) {
    this->error(UTL, 110, "SQLite error creating table '{}': {}",
                info.table_name,
                err_msg ? err_msg : "unknown");
    sqlite3_free(err_msg);
  }

  // Insert into table_list to track this registered schema.
  std::string col_types, col_names;
  for (size_t i = 0; i < info.columns.size(); ++i) {
    if (i > 0) {
      col_types += ",";
      col_names += ",";
    }
    col_types += sqlite_type_name(info.columns[i].type);
    col_names += info.columns[i].name;
  }
  char* tbl_sql = sqlite3_mprintf(
      "INSERT OR REPLACE INTO table_list VALUES (%d, %d, %q, %q)",
      static_cast<int>(key.tool),
      key.id,
      col_types.c_str(),
      col_names.c_str());
  sqlite3_exec(db, tbl_sql, nullptr, nullptr, nullptr);
  sqlite3_free(tbl_sql);

  return info;
}

void Logger::logToDbRegisterQueue(SchemaKey key,
                                   std::shared_ptr<AbstractQueue> queue)
{
  auto registered = schema_registry_.register_schema(key, std::move(queue));

  if (log_db_running_.load(std::memory_order_acquire)) {
    NewSchemaCommand cmd{key, std::move(registered)};
    std::lock_guard<std::mutex> lock(new_schema_queue_mutex_);
    new_schema_queue_.push_back(std::move(cmd));
  }
}

// ---------------------------------------------------------------------------
// logDbLoop  (backend thread)
// ---------------------------------------------------------------------------

void Logger::logDbLoop()
{
  // Local registry for the backend thread: SchemaKey -> AbstractQueue
  std::unordered_map<SchemaKey, std::shared_ptr<AbstractQueue>, SchemaKeyHasher> local_registry;

  while (log_db_running_.load(std::memory_order_acquire)) {
    // --- Phase 1: Drain all pending NewSchemaCommand entries ---
    std::deque<NewSchemaCommand> pending_commands;
    {
      std::lock_guard<std::mutex> lock(new_schema_queue_mutex_);
      pending_commands.swap(new_schema_queue_);
    }

    bool did_work = !pending_commands.empty();

    for (auto& cmd : pending_commands) {
      // If already registered locally (shouldn't happen, but guard), skip.
      if (local_registry.find(cmd.key) != local_registry.end()) {
        continue;
      }

      // Store the AbstractQueue in the local backend registry
      local_registry[cmd.key] = std::move(cmd.queue);
    }

    // --- Phase 2: Drain metadata queue (low-traffic text pairs) ---
    {
      std::queue<MetadataRow> local_meta;
      {
        std::lock_guard<std::mutex> lock(metadata_queue_mutex_);
        local_meta.swap(metadata_queue_);
      }

      if (!local_meta.empty()) {
        sqlite3_exec(db_, "BEGIN", nullptr, nullptr, nullptr);
        bool meta_ok = true;
        while (!local_meta.empty()) {
          auto [mtool, mkey, mval] = std::move(local_meta.front());
          local_meta.pop();

          char* sql = sqlite3_mprintf(
              "INSERT INTO metadata VALUES (%d, %q, %q)",
              static_cast<int>(mtool), mkey.c_str(), mval.c_str());
          int rc = sqlite3_exec(db_, sql, nullptr, nullptr, nullptr);
          sqlite3_free(sql);

          if (rc != SQLITE_OK) {
            meta_ok = false;
            break;
          }
        }
        sqlite3_exec(db_, meta_ok ? "COMMIT" : "ROLLBACK",
                     nullptr, nullptr, nullptr);
        did_work = true;
      }
    }

    // --- Phase 3: Schedule and drain queues ---
    //
    // (Phase 2 above drained the metadata queue.)
    //
    // Scheduling policy (checked in priority order):
    //
    // 1. Global memory pressure
    //    If total buffered bytes across all queues ≥ 80% of the user's global
    //    limit, fully drain the largest queue.
    //
    // 2. Per-channel memory pressure
    //    If any individual queue ≥ 80% of its per-channel limit, drain enough
    //    rows from it to fall back below the 80% threshold.
    //
    // 3. Round-robin
    //    Otherwise, drain a fixed batch (100 rows) from every queue.

    // Total buffered memory across all queues.
    size_t total_mem = 0;
    for (auto& entry : local_registry) {
      total_mem += entry.second->approx_size()
                   * entry.second->row_size_bytes();
    }

    bool drained_some = false;

    // --- Global pressure: fully drain the largest queue ---
    bool global_pressure = false;
    if (db_log_global_max_mem_ > 0) {
      const size_t global_limit
          = static_cast<size_t>(db_log_global_max_mem_
                                * k_queue_mem_high_water_mark);
      global_pressure = (total_mem >= global_limit);
    }

    if (global_pressure) {
      AbstractQueue* largest_q = nullptr;
      size_t largest_bytes = 0;
      for (auto& entry : local_registry) {
        const size_t bytes = entry.second->approx_size()
                             * entry.second->row_size_bytes();
        if (bytes > largest_bytes) {
          largest_bytes = bytes;
          largest_q = entry.second.get();
        }
      }
      if (largest_q) {
        drained_some |= (largest_q->drain_to_db(db_, SIZE_MAX) > 0);
      }
      did_work |= drained_some;
      // Skip round-robin / per-channel when we hit global pressure.
    } else {
      // --- Per-channel pressure: drain enough to get below 80% ---
      for (auto& entry : local_registry) {
        if (db_log_per_channel_max_mem_ == 0) {
          continue;
        }

        auto& q = entry.second;
        const size_t channel_bytes
            = q->approx_size() * q->row_size_bytes();
        const size_t channel_limit
            = static_cast<size_t>(db_log_per_channel_max_mem_
                                  * k_queue_mem_high_water_mark);

        if (channel_bytes >= channel_limit) {
          // Drain enough bytes to fall strictly below the threshold.
          const size_t bytes_to_clear
              = channel_bytes - channel_limit + 1;
          const size_t row_size = q->row_size_bytes();
          // Ceiling division — drain at least one row.
          const size_t rows_to_clear
              = (bytes_to_clear + row_size - 1) / row_size;

          drained_some |= (q->drain_to_db(db_, rows_to_clear) > 0);
        }
      }

      if (!drained_some) {
        // --- Round-robin: fully clear every queue ---
        for (auto& entry : local_registry) {
          drained_some
              |= (entry.second->drain_to_db(db_, SIZE_MAX) > 0);
        }
      }
      did_work |= drained_some;
    }

    // If no work was done, sleep briefly to avoid busy-waiting
    if (!did_work) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  // --- Shutdown drain: flush all remaining data ---
  // Process any pending schema commands that arrived during the final iteration.
  {
    std::lock_guard<std::mutex> lock(new_schema_queue_mutex_);
    for (auto& cmd : new_schema_queue_) {
      if (local_registry.find(cmd.key) == local_registry.end()) {
        local_registry[cmd.key] = std::move(cmd.queue);
      }
    }
    new_schema_queue_.clear();
  }

  // Drain any remaining metadata rows.
  {
    std::queue<MetadataRow> local_meta;
    {
      std::lock_guard<std::mutex> lock(metadata_queue_mutex_);
      local_meta.swap(metadata_queue_);
    }

    if (!local_meta.empty()) {
      sqlite3_exec(db_, "BEGIN", nullptr, nullptr, nullptr);
      bool meta_ok = true;
      while (!local_meta.empty()) {
        auto [mtool, mkey, mval] = std::move(local_meta.front());
        local_meta.pop();

        char* sql = sqlite3_mprintf(
            "INSERT INTO metadata VALUES (%d, %q, %q)",
            static_cast<int>(mtool), mkey.c_str(), mval.c_str());
        int rc = sqlite3_exec(db_, sql, nullptr, nullptr, nullptr);
        sqlite3_free(sql);

        if (rc != SQLITE_OK) {
          meta_ok = false;
          break;
        }
      }
      sqlite3_exec(db_, meta_ok ? "COMMIT" : "ROLLBACK",
                   nullptr, nullptr, nullptr);
    }
  }

  // Drain every queue in the local registry to SQLite before the
  // backend thread exits.
  for (auto& [key, q] : local_registry) {
    q->drain_to_db(db_, SIZE_MAX);
  }
}

ToolId Logger::findToolId(const char* tool_name)
{
  int tool_id = 0;
  for (const char* tool : tool_names_) {
    if (strcmp(tool_name, tool) == 0) {
      return static_cast<ToolId>(tool_id);
    }
    tool_id++;
  }
  return UKN;
}

void Logger::setDebugLevel(ToolId tool, const char* group, int level)
{
  if (level == 0) {
    auto& groups = debug_group_level_[tool];
    auto it = groups.find(group);
    if (it != groups.end()) {
      groups.erase(it);
      debug_on_
          = std::ranges::any_of(debug_group_level_,

                                [](auto& group) { return !group.empty(); });
    }
  } else {
    debug_on_ = true;
    debug_group_level_.at(tool)[group] = level;
  }
}

void Logger::addSink(spdlog::sink_ptr sink)
{
  sinks_.push_back(sink);
  logger_->sinks().emplace_back(std::move(sink));
  setFormatter();  // updates the new sink
}

void Logger::removeSink(const spdlog::sink_ptr& sink)
{
  // remove from local list of sinks_
  auto sinks_find = std::ranges::find(sinks_, sink);
  if (sinks_find != sinks_.end()) {
    sinks_.erase(sinks_find);
  }
  // remove from spdlog list of sinks
  auto& logger_sinks = logger_->sinks();
  auto logger_find = std::ranges::find(logger_sinks, sink);
  if (logger_find != logger_sinks.end()) {
    logger_sinks.erase(logger_find);
  }
}

void Logger::setMetricsStage(std::string_view format)
{
  if (metrics_stages_.empty()) {
    metrics_stages_.emplace(format);
  } else {
    metrics_stages_.top() = format;
  }
}

void Logger::clearMetricsStage()
{
  std::stack<std::string> new_stack;
  metrics_stages_.swap(new_stack);
}

void Logger::pushMetricsStage(std::string_view format)
{
  metrics_stages_.emplace(format);
}

std::string Logger::popMetricsStage()
{
  if (!metrics_stages_.empty()) {
    std::string stage = metrics_stages_.top();
    metrics_stages_.pop();
    return stage;
  }
  return "";
}

void Logger::flushMetrics()
{
  const std::string json = MetricsEntry::assembleJSON(metrics_entries_);

  for (const std::string& sink_path : metrics_sinks_) {
    std::ofstream sink_file(sink_path);
    if (sink_file) {
      sink_file << json;
    } else {
      this->warn(UTL, 10, "Unable to open {} to write metrics", sink_path);
    }
  }
}

void Logger::addWarningMetrics()
{
  // Add metrics for non-zero warnings
  int warning_type_cnt = 0;
  for (int i = 0; i < ToolId::SIZE; ++i) {
    for (int j = 0; j <= max_message_id; ++j) {
      if (message_counters_[i][j] > 0
          && message_levels_[i][j] == spdlog::level::warn) {
        warning_type_cnt++;
        log_metric(
            // NOLINTNEXTLINE(misc-include-cleaner)
            fmt::format("flow__warnings__count:{}-{:04}", tool_names_[i], j),
            std::to_string(message_counters_[i][j]));
      }
    }
  }

  // Add a metric to report the number of unique warning types
  log_metric("flow__warnings__type_count", std::to_string(warning_type_cnt));
}

void Logger::finalizeMetrics()
{
  log_metric("flow__warnings__count", std::to_string(warning_count_));
  log_metric("flow__errors__count", std::to_string(error_count_));

  addWarningMetrics();

  for (MetricsPolicy policy : metrics_policies_) {
    policy.applyPolicy(metrics_entries_);
  }

  flushMetrics();
}

void Logger::suppressMessage(ToolId tool, int id)
{
  message_counters_[tool][id] = max_message_print + 1;
}

void Logger::unsuppressMessage(ToolId tool, int id)
{
  message_counters_[tool][id] = 0;
}

void Logger::redirectFileBegin(const std::string& filename)
{
  assertNoRedirect();

  file_redirect_ = std::make_unique<std::ofstream>(filename);
  setRedirectSink(*file_redirect_);
}

void Logger::redirectFileAppendBegin(const std::string& filename)
{
  assertNoRedirect();

  file_redirect_
      = std::make_unique<std::ofstream>(filename, std::ofstream::app);
  setRedirectSink(*file_redirect_);
}

void Logger::redirectFileEnd()
{
  if (file_redirect_ == nullptr) {
    return;
  }

  restoreFromRedirect();

  file_redirect_->close();
  file_redirect_ = nullptr;
}

void Logger::teeFileBegin(const std::string& filename)
{
  assertNoRedirect();

  file_redirect_ = std::make_unique<std::ofstream>(filename);
  setRedirectSink(*file_redirect_, true);
}

void Logger::teeFileAppendBegin(const std::string& filename)
{
  assertNoRedirect();

  file_redirect_
      = std::make_unique<std::ofstream>(filename, std::ofstream::app);
  setRedirectSink(*file_redirect_, true);
}

void Logger::teeFileEnd()
{
  redirectFileEnd();
}

void Logger::redirectStringBegin()
{
  assertNoRedirect();

  string_redirect_ = std::make_unique<std::ostringstream>();
  setRedirectSink(*string_redirect_);
}

std::string Logger::redirectStringEnd()
{
  if (string_redirect_ == nullptr) {
    return "";
  }

  restoreFromRedirect();

  std::string string = string_redirect_->str();
  string_redirect_ = nullptr;

  return string;
}

void Logger::teeStringBegin()
{
  assertNoRedirect();

  string_redirect_ = std::make_unique<std::ostringstream>();
  setRedirectSink(*string_redirect_, true);
}

std::string Logger::teeStringEnd()
{
  return redirectStringEnd();
}

Logger* Logger::defaultLogger()
{
  static Logger default_logger;
  return &default_logger;
}

void Logger::assertNoRedirect()
{
  if (string_redirect_ != nullptr || file_redirect_ != nullptr) {
    this->error(
        UTL, 102, "Unable to start new log redirect while another is active.");
  }
}

void Logger::setRedirectSink(std::ostream& sink_stream, bool keep_sinks)
{
  if (!keep_sinks) {
    logger_->sinks().clear();
  }

  logger_->sinks().push_back(
      std::make_shared<spdlog::sinks::ostream_sink_mt>(sink_stream, true));
  setFormatter();
}

void Logger::restoreFromRedirect()
{
  logger_->sinks().clear();
  logger_->sinks().insert(
      logger_->sinks().begin(), sinks_.begin(), sinks_.end());
}

void Logger::startPrometheusEndpoint(uint16_t port)
{
  if (prometheus_metrics_) {
    return;
  }

  prometheus_metrics_ = std::make_unique<PrometheusMetricsServer>(
      prometheus_registry_, this, port);
}

std::shared_ptr<PrometheusRegistry> Logger::getRegistry()
{
  return prometheus_registry_;
}

bool Logger::isPrometheusServerReadyToServe()
{
  if (!prometheus_metrics_) {
    return false;
  }

  return prometheus_metrics_->is_ready() && prometheus_metrics_->port() != 0;
}

bool Logger::hasPrometheusServerStartupFailed()
{
  if (!prometheus_metrics_) {
    return false;
  }

  return prometheus_metrics_->has_startup_failed();
}

uint16_t Logger::getPrometheusPort()
{
  if (!prometheus_metrics_) {
    return 0;
  }

  return prometheus_metrics_->port();
}

void Logger::setFormatter()
{
  // create formatter without a newline
  std::unique_ptr<spdlog::formatter> formatter
      = std::make_unique<spdlog::pattern_formatter>(
          pattern_, spdlog::pattern_time_type::local, "");
  logger_->set_formatter(std::move(formatter));
}

std::unique_ptr<Progress> Logger::swapProgress(Progress* progress)
{
  std::unique_ptr<Progress> current_progress = std::move(progress_);
  progress_.reset(progress);

  return current_progress;
}

void Logger::setDbLogGlobalMaxMem(size_t bytes)
{
  db_log_global_max_mem_ = bytes;
}

size_t Logger::getDbLogGlobalMaxMem() const
{
  return db_log_global_max_mem_;
}

void Logger::setDbLogPerChannelMaxMem(size_t bytes)
{
  db_log_per_channel_max_mem_ = bytes;
}

size_t Logger::getDbLogPerChannelMaxMem() const
{
  return db_log_per_channel_max_mem_;
}

void Logger::logMetadata(ToolId tool, std::string key, std::string value)
{
  if (!db_) {
    return;
  }
  std::lock_guard<std::mutex> lock(metadata_queue_mutex_);
  metadata_queue_.emplace(tool, std::move(key), std::move(value));
}

}  // namespace utl
