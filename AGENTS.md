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

**All compilation and testing MUST be done inside containers.** Never compile or test natively on the host.

### ✅ Preferred: Apptainer Incremental Build (`etc/BuildApptainer.sh`)

**Use this flow by default** for all code changes and testing. It is faster for
iterative development because the build directory persists on the host filesystem
and supports true incremental compilation (only changed files recompile).

```
# First time (or after dependency changes): build the pre-requisite image
./etc/BuildApptainer.sh --force

# Subsequent incremental builds (reuses existing pre image):
./etc/BuildApptainer.sh

# Build only (skip pre-image rebuild):
./etc/BuildApptainer.sh -build

# Rebuild pre-image only:
./etc/BuildApptainer.sh -pre

# Package final SIF only:
./etc/BuildApptainer.sh -final
```

**How it works:**
1. **Phase 1 (`-pre`)**: Builds `openroad_pre.sif` — an Apptainer image containing
   all build dependencies (Boost, CUDD, Eigen, spdlog, GTest, OR-Tools, Abseil, etc.).
   The dependency versions and their CMake flags are managed centrally by
   `etc/DependencyInstaller.sh` and saved to `/opt/openroad_deps_prefixes.txt`
   inside the image.
2. **Phase 2 (`-build`)**: Mounts the host source tree into the pre image and runs
   CMake + make. The `build/` directory lives on the host, so subsequent
   compilations are incremental.
3. **Phase 3 (`-final`)**: Packages `build/bin/openroad` into a minimal
   `openroad.sif` using `openroad.def` (inherits from `openroad_pre.sif`).

**Build logs**: Save to `tmp/<log-file>.log` (e.g. `tmp/apptainer_build1.log`).
Use sequential numbering.

### Timeouts
**Use very large (or no) timeouts for build commands.** Timeouts are only for interactive
testing scenarios. A full build can take 30+ minutes on the first run and several
minutes even for incremental rebuilds. Set `timeout=0` (infinite) or at least
`timeout=1800000` (30 min) when running build commands.

## AI Agent Skills

Skills are located in `.agents/skills/` (with `.claude/skills` symlink for Claude Code).

| Skill | Purpose | Invocation |
|-------|---------|------------|
| `triage-issue` | Reproduce bug and minimize test case with whittle.py | `/triage-issue <issue#>` |
| `fix-bug` | Trace root cause, implement fix, create tests, prepare commit | `/fix-bug <issue#-or-error-code>` |
| `add-test` | Add integration/unit tests with dual CMake+Bazel registration | `/add-test <module> [description]` |
| `review-pr` | Draft local PR review notes (correctness > QoR > testing); human posts | `/review-pr <pr#-or-url>` |
