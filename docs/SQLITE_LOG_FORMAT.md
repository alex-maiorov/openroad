# SQLite Database Log Format

This document describes the SQLite database schema written by OpenROAD's
`Logger::startLogDb()`/`logToDb()` infrastructure.  It is intended for
engineers who need to consume or inspect these databases from external
tools (Python scripts, analytics pipelines, etc.).

> **Note:** The database is created with `PRAGMA journal_mode = WAL` and
> `PRAGMA synchronous = OFF` — optimised for **write throughput** at the
> cost of durability.  A crash may lose the last few batches of rows.
> The WAL journal is harmless for readers; SQLite readers
> automatically see a consistent snapshot.

---

## 1. System Tables

Three system tables are created at startup and are present in every
database.  They describe the schema of all data tables and provide a
tool-name lookup.

### 1.1 `tool_names`

```sql
CREATE TABLE IF NOT EXISTS tool_names (
    tool_id INTEGER PRIMARY KEY,
    name    TEXT
);
```

Pre-populated with one row per tool (37 tools at time of writing):

| tool_id | name |
|---------|------|
| 0       | ANT  |
| 1       | CGT  |
| 2       | CHK  |
| 3       | CTS  |
| 4       | CUT  |
| 5       | DFT  |
| 6       | DPL  |
| 7       | DRT  |
| 8       | DST  |
| 9       | EST  |
| **10**  | **EXA** |
| 11      | FIN  |
| 12      | FLW  |
| 13      | GPL  |
| 14      | GRT  |
| 15      | GUI  |
| 16      | IFP  |
| 17      | MPL  |
| 18      | ODB  |
| 19      | ORD  |
| 20      | PAD  |
| 21      | PAR  |
| 22      | PDN  |
| 23      | PPL  |
| 24      | PSM  |
| 25      | RAM  |
| 26      | RCX  |
| 27      | RMP  |
| 28      | RSZ  |
| 29      | STA  |
| 30      | STT  |
| 31      | TAP  |
| 32      | TST  |
| 33      | UKN  |
| 34      | UPF  |
| 35      | UTL  |
| 36      | WEB  |

A consumer can `JOIN` against this table to convert a `tool_id` foreign
key in other tables into a human-readable three-letter code.

### 1.2 `table_list`

```sql
CREATE TABLE IF NOT EXISTS table_list (
    tool_id      INTEGER,
    message_id   INTEGER,
    column_types TEXT,
    column_names TEXT,
    PRIMARY KEY (tool_id, message_id)
);
```

**Purpose:** A schema registry recording the structure of every data
table.  A reader can discover all data tables and their column types
without parsing individual `CREATE TABLE` statements.

| Column        | Type    | Description |
|---------------|---------|-------------|
| `tool_id`     | INTEGER | Foreign key into `tool_names`. |
| `message_id`  | INTEGER | The `id` parameter of the `logToDb`/`logToDbBulk` call. |
| `column_types`| TEXT    | Comma-separated SQLite type names, e.g. `"INTEGER,REAL"`. |
| `column_names`| TEXT    | Comma-separated column names, matched positionally to `column_types`. |

**Example rows** (from the EXA module's `exerciseDbLog()`):

| tool_id | message_id | column_types          | column_names |
|---------|------------|-----------------------|--------------|
| 10      | 6          | INTEGER               | value        |
| 10      | 7          | REAL                  | ratio        |
| 10      | 8          | INTEGER,REAL          | id,weight    |
| 10      | 9          | INTEGER,INTEGER,INTEGER | x,y,z      |
| 10      | 10         | INTEGER,REAL          | id,val       |
| 10      | 11         | INTEGER               | code         |
| 10      | 12         | INTEGER,INTEGER,INTEGER | x,y,z      |
| 10      | 15         | INTEGER               | toggle       |

### 1.3 `metadata`

```sql
CREATE TABLE IF NOT EXISTS metadata (
    tool_id INTEGER,
    key     TEXT,
    value   TEXT
);
```

**Purpose:** Free-form key-value store for text metadata.  Unlike the
typed data tables (section 2), this table stores TEXT in both value
columns.  It has **no primary key** — duplicate `(tool_id, key)` rows
are allowed.

| Column    | Type    | Description |
|-----------|---------|-------------|
| `tool_id` | INTEGER | Foreign key into `tool_names`. |
| `key`     | TEXT    | Metadata key name. |
| `value`   | TEXT    | Metadata value. |

**Example rows:**

```text
tool_id | key          | value
--------|--------------|----------------------
10      | experiment   | exercise_db_log
10      | author       | openroad
10      | description  | trivial fake usage of the db log infrastructure
```

Metadata rows are submitted via `logger->logMetadata(tool, key, value)`
from C++ code and are drained to the database by the backend thread in
batches (one transaction per drain cycle).

---

## 2. Data Tables

### 2.1 Naming Convention

Every `logToDb` or `logToDbBulk` call site with a unique `(tool, id)`
pair creates a **single data table**.  The table name is:

```
{TOOL_NAME}_{MESSAGE_ID}
```

where `TOOL_NAME` is the three-letter uppercase code from `tool_names`
and `MESSAGE_ID` is the integer `id` parameter.

**Examples:**

| `logToDb` call (conceptual) | Tool | ID | Table name |
|---|---|---|---|
| `logToDb<"value">(EXA, 6, 42)` | EXA | 6 | `EXA_6` |
| `logToDb<"ratio">(EXA, 7, 3.14)` | EXA | 7 | `EXA_7` |
| `logToDbBulk<"id,val">(EXA, 10, ...)` | EXA | 10 | `EXA_10` |
| `logToDb<"code">(CGT, 1, 0xabcd)` | CGT | 1 | `CGT_1` |

The `id` field is clamped to the range `[0, 9999]`.

### 2.2 Column Types

SQLite supports exactly two column types:

| C++ type family           | SQLite type name |
|---------------------------|------------------|
| All integral types        | `INTEGER`        |
| All floating-point types  | `REAL`           |

The mapping is defined at compile time via `TypeToSQLite<T>`:

```cpp
template <typename T>
struct TypeToSQLite<T, std::enable_if_t<std::is_integral_v<T>>> {
    static constexpr SQLiteType value = SQLiteType::INTEGER;
};

template <typename T>
struct TypeToSQLite<T, std::enable_if_t<std::is_floating_point_v<T>>> {
    static constexpr SQLiteType value = SQLiteType::REAL;
};
```

**There is no TEXT, BLOB, or NULL support** in the data tables.
Non-arithmetic types are rejected at compile time — `logToDb` accepts
only integral/floating-point arguments, and `logToDbBulk` additionally
requires all iterator value types to be arithmetic (`static_assert`).

### 2.3 Column Names

Column names are specified via the `Header` template parameter as a
comma-separated string literal:

```cpp
logger_->logToDb<"id,weight">(utl::EXA, 8, 1, 1.0);
//                                      ↑  ↑
//                              column "id"   column "weight"
```

- Allowed characters: `a-z`, `A-Z`, `0-9`, `_` (underscore), `,` (comma
  separator), ` ` (space, trimmed from field edges).
- Any other character (including `'`, `"`, `;`, `-`, `.`, `(`, `)`) is
  **rejected at compile time** — this is a deliberate SQL injection
  defence.
- Spaces adjacent to commas are stripped: `"id , weight"` produces
  the same columns as `"id,weight"`.
- The column name **count** must match the argument/iterator **count**
  (enforced by `static_assert` at each call site).
- A single-column header is just a single name: `"value"`, `"ratio"`,
  `"code"`, `"toggle"`.

### 2.4 Example Data Tables

Given the `exerciseDbLog()` calls in the EXA module, here is what the
data tables contain:

**`EXA_6`** — single INTEGER column, single row:
```sql
sqlite> SELECT * FROM EXA_6;
value
-----
42
```

**`EXA_7`** — single REAL column, single row:
```sql
sqlite> SELECT * FROM EXA_7;
ratio
-----
3.14
```

**`EXA_8`** — two columns (INTEGER, REAL), single row:
```sql
sqlite> SELECT * FROM EXA_8;
id          weight
----------  ------
1           1.0
```

**`EXA_10`** — two columns, bulk-loaded with 5 rows:
```sql
sqlite> SELECT * FROM EXA_10;
id          val
----------  ----------
101         1.1
102         2.2
103         3.3
104         4.4
105         5.5
```

**`EXA_12`** — three INTEGER columns, bulk-loaded with 4 rows:
```sql
sqlite> SELECT * FROM EXA_12;
x           y           z
----------  ----------  ----------
0           10          100
1           20          200
2           30          300
3           40          400
```

---

## 3. `logToDb` vs `logToDbBulk`

| Aspect | `logToDb` | `logToDbBulk` |
|--------|-----------|---------------|
| Signature | `logToDb<Header>(tool, id, args...)` | `logToDbBulk<Header>(tool, id, count, iters...)` |
| Input | Individual values (variadic) | One iterator per column + `count` |
| Rows per call | 1 | `count` |
| Value types | Integral or floating-point | Must be **arithmetic** (stricter check) |
| Use case | Occasional single-row logging | Batch logging from vectors/arrays |

Both push data into the same underlying queue infrastructure; the
SQLite tables created are identical in structure.

---

## 4. Lifecycle

### 4.1 Startup

```cpp
logger_->startLogDb("/path/to/db.log");
```

1. A **backend thread** is spawned.
2. The backend thread opens (or creates) the SQLite file with
   `SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE`.
3. Pragmas are applied: `journal_mode = WAL`, `synchronous = OFF`.
4. The three system tables (`tool_names`, `table_list`, `metadata`)
   are created if they do not exist.
5. The `tool_names` table is populated with all known tools via
   `INSERT OR REPLACE`.
6. `db_ready_` is set to `true` and `startLogDb()` returns.
7. **From this point**, any `logToDb`/`logToDbBulk`/`logMetadata` call
   will enqueue data.  Before this point, such calls silently return.

### 4.2 Runtime

The backend thread loops:
1. **Phase 1** — Drain pending `CreateTableCommand` entries (DDL).
   These have highest priority because callers are blocked on their
   `std::future`.
2. **Phase 2** — Drain pending `NewSchemaCommand` entries (queue
   registration).
3. **Phase 3** — Drain the metadata queue.
4. **Phase 4** — Drain data queues, applying pressure-based scheduling:
   - If global memory usage exceeds 80% of `setDbLogGlobalMaxMem`,
     the single largest queue is fully drained.
   - Otherwise, per-channel drains triggered at 80% of
     `setDbLogPerChannelMaxMem`.
   - Finally, a round-robin pass fully drains every queue.
5. If nothing was done, sleep 10 ms.

### 4.3 Shutdown

```cpp
logger_->stopLogDb();
```

1. A shutdown flag is set.
2. The caller joins the backend thread.
3. The backend repeatedly drains **all** queues (CreateTable, NewSchema,
   metadata, data) until every queue is empty.
4. All `TypedQueue` objects are destroyed, finalising their prepared
   statements.
5. `sqlite3_close()` is called.

### 4.4 Enable / Disable

Individual `(tool, id)` pairs can be toggled:

```cpp
logger_->setDbLogEnabled(utl::EXA, 15, false);   // disable
logger_->setDbLogEnabled(utl::EXA, 15, true);    // re-enable
```

- When disabled, `logToDb`/`logToDbBulk` silently return for that pair.
- The schema is removed from the registry so that re-enabling triggers
  a fresh table lookup (re-discovering the schema).

---

## 5. Performance Characteristics

### 5.1 Batching

All inserts are batched in explicit transactions:

```
BEGIN
  bind_field() + sqlite3_step() × N
COMMIT
```

If any step fails, the transaction is rolled back (`ROLLBACK`) rather
than committing a partial batch.

### 5.2 Prepared statements

Each `TypedQueue` builds a prepared insert statement on first drain:

```sql
INSERT INTO {table_name} (col1, col2, ...) VALUES (?, ?, ...);
```

This prepared statement is reused across drain cycles and finalised
when the queue is destroyed.

### 5.3 Memory backpressure

Two knobs control in-memory buffering:

- **`setDbLogGlobalMaxMem(bytes)`** — Total memory across all queues.
  At 80% utilisation the largest queue is fully drained.
- **`setDbLogPerChannelMaxMem(bytes)`** — Memory per (tool, id) pair.
  At 80% utilisation that channel is drained enough to drop below 80%.

Both default to `0` (unlimited).

### 5.4 WAL mode

`PRAGMA journal_mode = WAL` allows concurrent readers to see a
consistent snapshot without blocking the writer.  The WAL file
(`-wal`) is automatically checkpointed by SQLite.

---

## 6. Parsing Strategy

### 6.1 Discover available tables

Start by reading `table_list` joined with `tool_names`:

```sql
SELECT tn.name, tl.message_id, tl.column_types, tl.column_names
FROM table_list tl
JOIN tool_names tn ON tl.tool_id = tn.tool_id
ORDER BY tn.name, tl.message_id;
```

This tells you every data table's name (`{name}_{message_id}`), its
column names, and their SQLite types.

### 6.2 Query a specific table

```sql
SELECT * FROM EXA_10;
```

The column names are always the ones recorded in `table_list`.

### 6.3 Handle missing tables

A `(tool, id)` pair only creates a table when `logToDb`/`logToDbBulk`
is first called for that pair.  If a tool registers a schema but never
produces data, or if the data was rolled back, the table simply will
not exist.  Always check via `table_list` rather than assuming a
table name exists.

### 6.4 Thread safety for readers

Because the database uses WAL mode, readers can open the database file
while the OpenROAD process is writing to it.  The reader will see a
consistent snapshot as of the moment it opened the connection.  To see
the latest writes, close and re-open the connection.

### 6.5 Python quick-start

```python
import sqlite3

conn = sqlite3.connect("path/to/db.log")
conn.row_factory = sqlite3.Row

# Discover tables
rows = conn.execute("""
    SELECT tn.name, tl.message_id, tl.column_types, tl.column_names
    FROM table_list tl
    JOIN tool_names tn ON tl.tool_id = tn.tool_id
    ORDER BY tn.name, tl.message_id
""").fetchall()

for r in rows:
    table = f"{r['name']}_{r['message_id']}"
    print(f"{table}: columns={r['column_names']} types={r['column_types']}")
    data = conn.execute(f"SELECT * FROM [{table}]").fetchall()
    for row in data:
        print(f"  {dict(row)}")
```

---

## 7. Constraints and Invariants

| Property | Constraint |
|----------|-----------|
| Table name format | `{TOOL_NAME}_{id}`, e.g. `EXA_6` |
| `id` range | `[0, 9999]` |
| Column types | Only `INTEGER` and `REAL` |
| Header characters | Only `a-z`, `A-Z`, `0-9`, `_`, `,`, ` ` (space) |
| Argument count | Must equal `Header.count_fields()` (compile-time) |
| ID uniqueness | IDs are shared between ordinary messages and DB_LOG entries — duplicates across `info`/`warn`/`error`/`debug`/`logToDb` are errors |
| `table_list` | One row per `(tool, id)` pair, inserted via `INSERT OR REPLACE` |
| `metadata` | No primary key; duplicates allowed |
| Lifetime | Calls made before `startLogDb()` or after `stopLogDb()` silently return |
| Disabled pairs | Silently skip logging when `setDbLogEnabled(tool, id, false)` is in effect |

---

## 8. `find_messages.py` Integration

The `etc/find_messages.py` script scans C++ source files for
`logToDb<...>` and `logToDbBulk<...>` calls and registers them in
the same `(tool, id)` namespace as ordinary message IDs.  This means:

- A DB_LOG entry with `(EXA, 6)` competes with an `info(EXA, 6, ...)`
  in the same file — if both exist, `find_messages.py` reports a
  duplicate-ID error.
- The output format for DB_LOG entries is:
  ```
  EXA 0006   src/exa/src/example.cpp:89    (database log)           DB_LOG <github_url>
  ```
- DB_LOG IDs are reported alongside all other message IDs, making it
  easy to verify there are no collisions before committing.

---

## 9. Error Handling

The following error IDs are used for SQLite-related failures:

| ID  | Tool | Severity | Message |
|-----|------|----------|---------|
| 109 | UTL  | error    | Failed to open SQLite database |
| 110 | UTL  | error    | SQLite error creating table |
| 111 | UTL  | error    | SQLite error inserting metadata |
| 112 | UTL  | error    | SQLite error creating prepared insert statement |
| 113 | UTL  | error    | SQLite error during batch insert |
| 115 | UTL  | warn     | Failed to insert into `table_list` |
| 116 | UTL  | warn     | Failed to insert metadata row |

If the database cannot be opened at startup (`UTL-109`), the error is
reported and `startLogDb()` re-throws.  All other errors are logged
but do not halt execution — data may be silently lost.
