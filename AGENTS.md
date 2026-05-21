# OpenROAD - AI agent context

This file provides project-specific guidance for AI agent sessions working on the OpenROAD codebase.

## Quick Reference

Detailed guides are in `docs/agents/` subdirectory:

- **Build Pitfalls**: See `docs/agents/build.md`
- **Testing Guide**: See `docs/agents/testing.md`
- **Git & CI**: See `docs/agents/ci.md`
- **Coding Patterns**: See `docs/agents/coding.md`

## Critical Rules

1. **Ask before modifying `src/sta/` files** -- OpenSTA is managed upstream (`Sdc.cc`, `Power.cc`, `Sdc.tcl`, etc.). Prefer fixes in OpenROAD code (e.g., src/dbSta/, src/rsz/) when possible. 
2. Run `clang-format -i <files>` for C++ files before commit. **NEVER** for `src/sta/*` and `*.i` files.
3. **Always use `git commit -s`** for DCO compliance.
4. When amending submodule commits, parent repo submodule reference must also be updated via `git submodule update --init --recursive`. It is needed after any merge/pull.
5. **Trace bugs upstream** -- when a bug appears in output (e.g., Verilog), find the data creation point (e.g., `buffer_ports`, `remove_buffers`), not the serialization point (e.g., `VerilogWriter`).
6. **Temporary scripts**: All temporary development scripts MUST go into the `tmp/` folder in the project.

## ⚠️ Container-Only Build & Test Rule

**All compilation and testing MUST be done inside Docker containers.** Never compile or test natively on the host.

The approved workflow uses `cached.Dockerfile` (note: NOT `Dockerfile`):

```
docker build -f cached.Dockerfile --tag <container-tag> .
```

Where `<container-tag>` is a tag for the resulting image (e.g. `openroad:exa-db-log`).

Build logs are saved to `tmp/<log-file>.log` (e.g. `tmp/build10.log`). Use sequential numbering.

### Why containers?
- The native host lacks required dependencies (GTest, spdlog dev, Python3 dev, etc.).
- `cached.Dockerfile` uses `--mount=type=cache` to persist the CMake build directory
  between Docker rebuilds, making incremental builds fast.
- The dev stage installs all deps; the builder stage compiles; the final stage
  contains only the runtime binary.

### Workflow
1. Make code changes
2. `docker build -f cached.Dockerfile --tag <container-tag> . 2>&1 | tee tmp/<log-file>.log`
3. On first build the full dep install + compile takes ~30-60 min; subsequent builds
   reuse cached layers and the persisted build directory (only changed files recompile).
4. Run compiled code inside a container derived from the image (see testing.md).

### Timeouts
**Use very large (or no) timeouts for build commands.** Timeouts are only for interactive
testing scenarios. A full build with `cached.Dockerfile` can take 30+ minutes on the
first run and several minutes even for incremental rebuilds. Set `timeout=0` (infinite)
or at least `timeout=1800000` (30 min) when running `docker build`.

## AI Agent Skills

Skills are located in `.agents/skills/` (with `.claude/skills` symlink for Claude Code).

| Skill | Purpose | Invocation |
|-------|---------|------------|
| `triage-issue` | Reproduce bug and minimize test case with whittle.py | `/triage-issue <issue#>` |
| `fix-bug` | Trace root cause, implement fix, create tests, prepare commit | `/fix-bug <issue#-or-error-code>` |
| `add-test` | Add integration/unit tests with dual CMake+Bazel registration | `/add-test <module> [description]` |
| `review-pr` | Draft local PR review notes (correctness > QoR > testing); human posts | `/review-pr <pr#-or-url>` |
