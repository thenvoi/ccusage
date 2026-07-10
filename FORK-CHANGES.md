# thenvoi/ccusage — fork changes (branch `lib-facade`)

This is **thenvoi's embedding fork** of [ccusage](https://github.com/ccusage/ccusage)
(MIT, © 2025 ryoppippi). It exists for exactly one consumer: **jam** (`jamd`)
statically links `rust/crates/ccusage` as a library to compute usage & cost
reports in-process (see jam's doc `28-ccusage-usage-cost-integration.md`).
Upstream ships the crate **binary-only** — no `[lib]` target, every type
`pub(crate)`, not published to crates.io — so embedding requires a fork.

**Ground rules for this branch:**
- Keep the delta minimal and mechanical so rebasing onto upstream releases stays
  cheap. New surface goes into `src/api.rs`; upstream files get the smallest
  possible touches.
- The shipped **binary behavior is unchanged** (default features, identical
  output) with one deliberate exception noted below (the embedding API's
  inclusive `until` bound — the CLI keeps stock behavior).
- jam pins this fork **by commit rev**, never by branch. Changing anything here
  means: commit → push → bump the `rev` in jam's `Cargo.toml`.

---

## Changes, newest first (what · why)

### 5. `license = "MIT"` declared on all four crates (`1fa488b`)
**What:** Added the `license` field to `ccusage`, `ccusage-cli`,
`ccusage-terminal`, `ccusage-test-support` manifests.
**Why:** Upstream keeps a repo-root `LICENSE` only; the crate manifests said
nothing. Downstream license tooling (cargo-about / cargo-deny) cannot
synthesize an expression for a git dependency without it, which broke jam's
`THIRD-PARTY-LICENSES.txt` generation. The field states what the repo LICENSE
already grants. *Good upstream PR candidate.*

### 4. Inclusive `until` bound for sessions in the embedding API (`3436c23`)
**What:** `api::claude_sessions` compares the **date prefix** of a session's
last activity against the `until` bound, instead of the whole RFC3339 stamp.
Regression test included.
**Why (a real bug, found by cross-verifying against ccusage 20.0.6):** the
upstream 20.0.11 `session` filter compares `"2026-06-10T12:00:00.000Z"` (dashes
stripped) lexically against `"20260610"` — the longer string sorts *greater*,
so a session last active **on** the `until` day silently vanishes from bounded
reports. 20.0.6 did not have this behavior. The API documents its bounds as
inclusive, so the API fixes it; `commands::run_session` (the CLI) deliberately
keeps stock behavior to stay diff-identical with the upstream binary.
*Upstream PR candidate (the fix belongs in `run_session` there).*

### 3. Self-contained `WeekStart` enum in the API (`7287fb5`)
**What:** `api::claude_weekly` takes an API-owned `WeekStart` enum (default
Sunday, matching the CLI) instead of the internal `cli::WeekDay`.
**Why:** `WeekDay` re-exports through a `pub(crate)` module, so embedders could
not name it — the function was public but uncallable from outside.

### 2. Explicit data-directory override (`4f9e877`)
**What:** `api::UsageOptions.claude_dirs: Option<Vec<PathBuf>>`, threaded as an
optional override through the claude loaders (`load_entries_in` /
`load_daily_summaries_in`; the existing functions delegate with `None`).
Validates like `CLAUDE_CONFIG_DIR`: entries without `projects/` are skipped, an
override with no valid entries is an error.
**Why:** the only override before was the `CLAUDE_CONFIG_DIR` env var —
process-global, racy under parallel tests, and `std::env::set_var` is *unsafe*
in edition-2024 consumers (jam forbids `unsafe_code`). Embedders and tests need
a scoped, data-flow override.

### 1. Library target + embedding API + optional SQLite (`b731155 → 4efc417`)
**What:**
- `src/main.rs` → `src/lib.rs` (the crate root: modules, `CliError`, `run()`),
  with a 7-line `main.rs` shim. `Result`/`CliError`/`run` became `pub`. The
  musl `#[global_allocator]` moved to the **bin** shim so library consumers
  don't inherit mimalloc.
- New `src/api.rs`: a stable, self-contained programmatic surface
  (`UsageOptions` → `PeriodUsage` / `SessionUsage` / `BlockUsage` for
  daily/weekly/monthly/sessions/blocks). It defines its **own public types**
  and converts internally — zero visibility churn in upstream code, and
  embedders never depend on internals. Terminal progress output is suppressed.
- New **on-by-default `sqlite-adapters` feature**: the OpenCode/Kilo/Hermes/
  Goose database readers (and their sqlite-typed parser helpers + tests) are
  gated; with the feature off, the bundled `sqlite` dependency leaves the
  graph entirely and those adapters report no usage.
**Why (lib):** jam needs typed, in-process report calls — no subprocess, no
JSON re-parse, no sidecar binary to distribute.
**Why (sqlite gate):** jam already statically links SQLCipher via `rusqlite`
(`libsqlite3-sys`, cargo `links = "sqlite3"`). Two crates claiming the same
`links` key cannot coexist in one build graph, so for jam this gate is a hard
requirement, not hygiene. Claude/Codex reports don't need those adapters.

---

## Maintenance

- **Rebase cadence:** onto upstream tags quarterly, or immediately when a
  Claude log-format change breaks parsing. The delta is ~6 files of mechanical
  changes plus the self-contained `api.rs`; conflicts should be rare.
- **Upstreaming:** items marked *upstream PR candidate* above shrink this fork
  when accepted. The lib target itself is also worth proposing.
- **Pricing snapshot:** the build script embeds LiteLLM pricing; jam pins it
  via `CCUSAGE_PRICING_JSON_PATH` to a vendored snapshot (see jam's
  `just update-pricing`). Nothing about that lives in this fork — upstream's
  env-var escape hatch is used as-is.

## Automation: weekly facade rebase (`.github/workflows/jam-rebase-facade.yml`)

Weekly (and via workflow_dispatch), the fork rebases its facade commits onto the
newest upstream `v*` tag as a pre-staged branch `lib-facade-v<X.Y.Z>`,
compile-checks the `ccusage` crate, and pushes it. Facade commits are derived
(`rev-list HEAD --not --remotes=upstream`), never stored as files. A red run =
upstream drifted under the facade; resolve by hand and push the pre-staged
branch. Upgrading tjam = bump the `ccusage` rev to the pre-staged head,
`cargo update -p ccusage`, `just update-pricing`, re-exempt new transitives in
cargo-vet. (Same design as thenvoi/tauri's `jam-rebase-patch.yml`.)
