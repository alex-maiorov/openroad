# High-Performance Typed Relational Logging Plan (SQLite)

## 1. Overview
This document outlines the architecture for a high-performance, structured "Trace/Decision Logger" integrated into the `utl::Logger` system. This system, called **Typed Relational Logging (TRL)**, moves away from a single wide-table approach to a "Table-per-Schema" model. This allows each data producer to have its own well-structured, named tables while maintaining near-zero impact on the performance of the producer threads.

## 2. Design Goals
1.  **Maximum Writing Speed**: The "Producer Tax" (overhead on the tool threads) must be near zero.
2.  **Well-Structured Data**: Data must be stored in specialized relational tables with meaningful column names, allowing for complex mathematical and logical queries.
3.  **Low Contention**: Avoid global locks that would bottleneck multi-threaded tools.
4.  **Integration**: Must respect existing `ToolId` and `debugCheck()` verbosity levels.

## 3. Architecture: The "Direct-Bind" Pattern

The system follows a **Producer-Consumer** pattern using **Binary Serialization** and **Direct-Bind SQLite** storage.

### A. The Producer (Tool Threads)
To minimize latency, producers perform a "Binary Dump" of raw numeric data.

*   **Verbosity Check**: Every call first executes `debugCheck(tool, group, level)`. If the level is too low, the function returns immediately.
*   **Binary Packing**: If enabled, the producer writes a tiny, fixed-size header followed by raw numeric bytes (integers or doubles) directly into a thread-local buffer or a lock-free queue.
*   **Zero-Allocation**: Use of stack-allocated headers and move semantics to avoid heap overhead.

**Proposed Binary Header (`TraceHeader`):**
| Field | Size | Description |
| :--- | :--- | :--- |
| `schema_id` | 2 bytes | Maps to a registered relational schema |
| `timestamp` | 8 bytes | Nanosecond precision |
| `tool_id` | 1 byte | Existing `utl::ToolId` |
| `payload_count` | 1 byte | Number of numeric values following the header |

### B. The Transport (Concurrency Layer)
We will utilize **`boost::lockfree::queue`** (Multi-Producer Single-Consumer) to move pointers to binary "Slabs" from the tool threads to the background logger thread.
*   **Batching**: Instead of pushing individual records, producers will push large contiguous memory blocks (Slabs) once they are full.

### C. The Consumer (Background Writer Thread)
A single, dedicated thread is responsible for all heavy lifting.

1.  **Drain**: Pulls all available Slabs from the MPSC queue.
2.  **Statement Management**: Maintains a cache of prepared `sqlite3_stmt*` objects, one for each registered `schema_id`.
3.  **Direct-to-Bind (The Optimized Path)**: For each record in a slab, the consumer:
    *   Identifies the `schema_id`.
    *   Retrieves the cached prepared statement.
    *   Iterates through the binary payload, using the `TableDefinition` to perform direct type-casting (e.g., `static_cast<double*>`) and `sqlite3_bind_*` calls. This avoids the "weird blob conversion" and mapping to generic `val1, val2...` columns.
4.  **Batch Transaction**: Performs a single, massive SQLite transaction for the entire batch to maximize disk throughput.

### D. The Storage Layer (SQLite)
SQLite will be configured for "Extreme Write Mode" and will host multiple specialized tables.

*   **Configuration (PRAGMAs)**:
    *   `journal_mode = WAL`
    *   `synchronous = OFF`
    *   `locking_mode = EXCLUSIVE`
*   **Schema (Relational Design)**:
    *   Instead of a single `trace_data` table, each registered schema defines its own table: `CREATE TABLE <table_name> (<col1> <type>, <col2> <type>, ...)`.

## 4. Required Component Implementation

### `utl::SchemaRegistry`
*   `registerSchema(string tableName, vector<pair<string, ColumnType>> columns)`: Defines a new table and its structure.
*   `getSchema(schema_id)`: Returns the `TableDefinition` (table name, column names, and types).

### `utl::Logger` (Frontend API)
*   `registerSchema(...)`: Public API for components to define their specialized tables.
*   `trace(tool, group, message, schema_id, ...args)`: The primary high-speed API. Uses variadic templates to accept raw numbers and pack them into the binary buffer.

### `utl::Logger` (Backend Implementation)
*   `backgroundWriterLoop()`: The core loop of the worker thread.
*   `prepareStatements()`: Logic to initialize and cache `sqlite3_stmt*` for all registered schemas.
*   `commitBatchToSqlite(vector<ParsedRecord> records)`: Logic to execute the bulk `INSERT` via prepared statements.

## 5. Complexity Analysis
*   **Producer Complexity**: $O(1)$ (Memory copy + Atomic increment).
*   **Consumer Complexity**: $O(N)$ (Iterating payload and binding values to statements).
*   **Space Complexity**: $O(S)$ where $S$ is the size of the buffered slabs.
