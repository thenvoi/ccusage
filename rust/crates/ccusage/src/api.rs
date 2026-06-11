//! Embedding API (thenvoi fork addition).
//!
//! A stable, self-contained programmatic surface over the report pipelines the
//! CLI commands use (`daily` / `weekly` / `monthly` / `session` / `blocks`),
//! returning plain owned structs instead of printing. The types here are
//! deliberately decoupled from the crate's internal types so embedders never
//! depend on internals.
//!
//! Data directories resolve exactly like the CLI: `CLAUDE_CONFIG_DIR`
//! (comma-separated) when set, else `$XDG_CONFIG_HOME/claude` and
//! `~/.claude`. Missing directories yield empty reports, not errors; an
//! explicitly-set `CLAUDE_CONFIG_DIR` with no valid entries is an error
//! (mirroring the CLI).

use std::path::PathBuf;

use crate::{
    adapter::claude::{load_daily_summaries_in, load_entries_in},
    calculate_burn_rate,
    cli::{normalize_date_bound, CostMode, SharedArgs, SortOrder, WeekDay},
    filter_and_sort_summaries, filter_blocks_by_date, identify_session_blocks, sort_blocks,
    sort_summaries, summarize_by_key, summarize_summaries_by_bucket, BucketKind, ModelBreakdown,
    Result, SessionAccumulator, SessionBlock, UsageSummary, DEFAULT_SESSION_DURATION_HOURS,
};

/// How costs are derived from usage entries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CostBasis {
    /// Prefer the `costUSD` recorded in the log entry, falling back to a
    /// token-based calculation when absent (the CLI default).
    #[default]
    Auto,
    /// Always calculate from tokens and pricing, ignoring recorded costs.
    Calculate,
    /// Only use recorded `costUSD` values (entries without one cost 0).
    Display,
}

/// Options shared by every report function.
#[derive(Debug, Clone, Default)]
pub struct UsageOptions {
    /// Inclusive lower date bound, `YYYY-MM-DD` or `YYYYMMDD`.
    pub since: Option<String>,
    /// Inclusive upper date bound, `YYYY-MM-DD` or `YYYYMMDD`.
    pub until: Option<String>,
    /// Skip network pricing refresh; use only the embedded pricing snapshot.
    pub offline: bool,
    /// IANA timezone for date grouping (default: system local).
    pub timezone: Option<String>,
    /// Cost derivation mode.
    pub cost_basis: CostBasis,
    /// Explicit Claude config directories (each containing `projects/`),
    /// overriding `CLAUDE_CONFIG_DIR` / home discovery. Like the env var,
    /// entries without a `projects/` subdirectory are skipped and an
    /// override with no valid entries is an error.
    pub claude_dirs: Option<Vec<PathBuf>>,
}

/// Per-model usage within a report row.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelUsage {
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
    pub cost: f64,
    /// True when no pricing was found for this model (its cost may be 0).
    pub missing_pricing: bool,
}

/// One row of a daily / weekly / monthly report.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PeriodUsage {
    /// The grouping key: a date (`2026-06-11`), a week start date, or a
    /// month (`2026-06`).
    pub period: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
    pub total_cost: f64,
    pub models: Vec<ModelUsage>,
}

/// One session's aggregated usage. `session_id` is the Claude Code session
/// UUID (the `.jsonl` file name under `projects/<project>/`); subagent logs
/// roll up into their parent session.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionUsage {
    pub session_id: String,
    pub project_path: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
    pub total_cost: f64,
    pub models: Vec<ModelUsage>,
    /// RFC3339 milliseconds, e.g. `2026-06-11T09:00:00.000Z`.
    pub first_activity: Option<String>,
    pub last_activity: Option<String>,
}

/// Tokens-per-minute and cost-per-hour over a block's active span.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BurnRateInfo {
    pub tokens_per_minute: f64,
    pub cost_per_hour: f64,
}

/// Projected end-of-block totals for an active block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectionInfo {
    pub total_tokens: u64,
    pub total_cost: f64,
    pub remaining_minutes: u64,
}

/// One 5-hour (by default) billing block.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BlockUsage {
    /// Block start, epoch milliseconds (floored to the hour).
    pub start_ms: i64,
    /// Block end (start + session duration), epoch milliseconds.
    pub end_ms: i64,
    /// Timestamp of the last entry in the block, epoch milliseconds.
    pub actual_end_ms: Option<i64>,
    pub is_active: bool,
    /// A synthetic block covering a gap longer than the session duration.
    pub is_gap: bool,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
    pub total_cost: f64,
    pub models: Vec<String>,
    pub burn_rate: Option<BurnRateInfo>,
    pub projection: Option<ProjectionInfo>,
}

/// The default billing-block length, in hours.
pub const DEFAULT_BLOCK_HOURS: f64 = DEFAULT_SESSION_DURATION_HOURS;

fn shared_args(opts: &UsageOptions) -> SharedArgs {
    SharedArgs {
        since: opts.since.as_deref().map(normalize_date_bound),
        until: opts.until.as_deref().map(normalize_date_bound),
        // Suppresses interactive progress output on stderr.
        json: true,
        mode: match opts.cost_basis {
            CostBasis::Auto => CostMode::Auto,
            CostBasis::Calculate => CostMode::Calculate,
            CostBasis::Display => CostMode::Display,
        },
        order: SortOrder::Asc,
        offline: opts.offline,
        timezone: opts.timezone.clone(),
        ..SharedArgs::default()
    }
}

fn resolve_dirs(opts: &UsageOptions) -> Result<Option<Vec<PathBuf>>> {
    let Some(dirs) = &opts.claude_dirs else {
        return Ok(None);
    };
    let valid: Vec<PathBuf> = dirs
        .iter()
        .filter(|dir| dir.join("projects").is_dir())
        .cloned()
        .collect();
    if valid.is_empty() {
        return Err(crate::cli_error(
            "no valid Claude data directories in claude_dirs (each must contain projects/)",
        ));
    }
    Ok(Some(valid))
}

fn model_usage(b: &ModelBreakdown) -> ModelUsage {
    ModelUsage {
        model: b.model_name.clone(),
        input_tokens: b.input_tokens,
        output_tokens: b.output_tokens,
        cache_creation_tokens: b.cache_creation_tokens,
        cache_read_tokens: b.cache_read_tokens,
        cost: b.cost,
        missing_pricing: b.missing_pricing,
    }
}

fn period_usage(row: &UsageSummary, period: String) -> PeriodUsage {
    PeriodUsage {
        period,
        input_tokens: row.input_tokens,
        output_tokens: row.output_tokens,
        cache_creation_tokens: row.cache_creation_tokens,
        cache_read_tokens: row.cache_read_tokens,
        total_cost: row.total_cost,
        models: row.model_breakdowns.iter().map(model_usage).collect(),
    }
}

/// Daily Claude Code usage, one row per date (in the configured timezone).
pub fn claude_daily(opts: &UsageOptions) -> Result<Vec<PeriodUsage>> {
    let shared = shared_args(opts);
    let dirs = resolve_dirs(opts)?;
    let mut rows = load_daily_summaries_in(&shared, None, false, dirs.as_deref())?;
    filter_and_sort_summaries(&mut rows, &shared, |row| {
        row.date.as_deref().unwrap_or_default()
    });
    Ok(rows
        .iter()
        .map(|row| period_usage(row, row.date.clone().unwrap_or_default()))
        .collect())
}

/// Weekly Claude Code usage; weeks start on `week_starts_on` (the CLI
/// default is Sunday) and are keyed by the week's start date.
pub fn claude_weekly(opts: &UsageOptions, week_starts_on: WeekDay) -> Result<Vec<PeriodUsage>> {
    let shared = shared_args(opts);
    let dirs = resolve_dirs(opts)?;
    let entries = load_entries_in(&shared, None, dirs.as_deref())?;
    let mut daily = summarize_by_key(
        &entries,
        |entry| entry.date.clone(),
        |key| (key.to_string(), None),
    )?;
    filter_and_sort_summaries(&mut daily, &shared, |row| {
        row.date.as_deref().unwrap_or_default()
    });
    let mut weekly = summarize_summaries_by_bucket(&daily, BucketKind::Weekly, week_starts_on);
    sort_summaries(&mut weekly, &shared.order, |row| {
        row.week.as_deref().unwrap_or_default()
    });
    Ok(weekly
        .iter()
        .map(|row| period_usage(row, row.week.clone().unwrap_or_default()))
        .collect())
}

/// Monthly Claude Code usage, keyed `YYYY-MM`.
pub fn claude_monthly(opts: &UsageOptions) -> Result<Vec<PeriodUsage>> {
    let shared = shared_args(opts);
    let dirs = resolve_dirs(opts)?;
    let entries = load_entries_in(&shared, None, dirs.as_deref())?;
    let mut daily = summarize_by_key(
        &entries,
        |entry| entry.date.clone(),
        |key| (key.to_string(), None),
    )?;
    filter_and_sort_summaries(&mut daily, &shared, |row| {
        row.date.as_deref().unwrap_or_default()
    });
    let mut monthly = summarize_summaries_by_bucket(&daily, BucketKind::Monthly, WeekDay::Sunday);
    sort_summaries(&mut monthly, &shared.order, |row| {
        row.month.as_deref().unwrap_or_default()
    });
    Ok(monthly
        .iter()
        .map(|row| period_usage(row, row.month.clone().unwrap_or_default()))
        .collect())
}

/// Per-session Claude Code usage, sorted by cost (highest first), mirroring
/// the `session` CLI report: grouped by `(project_path, session_id)`,
/// zero-token sessions dropped, date bounds applied to the last activity.
pub fn claude_sessions(opts: &UsageOptions) -> Result<Vec<SessionUsage>> {
    use crate::fast::FxHashMap;
    use std::sync::Arc;

    let shared = shared_args(opts);
    let dirs = resolve_dirs(opts)?;
    let entries = load_entries_in(&shared, None, dirs.as_deref())?;
    let mut grouped = Vec::<SessionAccumulator>::new();
    let mut group_indexes = FxHashMap::<(Arc<str>, Arc<str>), usize>::default();
    for entry in &entries {
        let key = (
            Arc::clone(&entry.project_path),
            Arc::clone(&entry.session_id),
        );
        let index = *group_indexes.entry(key).or_insert_with(|| {
            let index = grouped.len();
            grouped.push(SessionAccumulator::default());
            index
        });
        grouped[index].add_entry(entry);
    }

    let mut rows = Vec::with_capacity(grouped.len());
    for group in grouped {
        rows.push(group.into_summary()?);
    }
    if shared.since.is_some() || shared.until.is_some() {
        rows.retain(|row| {
            let date = row
                .last_activity
                .as_deref()
                .unwrap_or_default()
                .replace('-', "");
            shared.since.as_ref().is_none_or(|since| &date >= since)
                && shared.until.as_ref().is_none_or(|until| &date <= until)
        });
    }
    rows.retain(|row| {
        row.input_tokens + row.output_tokens + row.cache_creation_tokens + row.cache_read_tokens > 0
    });
    rows.sort_by(|a, b| b.total_cost.total_cmp(&a.total_cost));

    Ok(rows
        .into_iter()
        .map(|row| SessionUsage {
            session_id: row.session_id.clone().unwrap_or_default(),
            project_path: row.project_path.clone().unwrap_or_default(),
            input_tokens: row.input_tokens,
            output_tokens: row.output_tokens,
            cache_creation_tokens: row.cache_creation_tokens,
            cache_read_tokens: row.cache_read_tokens,
            total_cost: row.total_cost,
            models: row.model_breakdowns.iter().map(model_usage).collect(),
            first_activity: row.first_activity,
            last_activity: row.last_activity,
        })
        .collect())
}

/// Billing blocks (`session_hours`-long windows, gap blocks included),
/// sorted by start time. With `active_only`, only the currently-active
/// block (if any) is returned.
pub fn claude_blocks(
    opts: &UsageOptions,
    session_hours: f64,
    active_only: bool,
) -> Result<Vec<BlockUsage>> {
    if session_hours <= 0.0 {
        return Err(crate::cli_error("session_hours must be positive"));
    }
    let shared = shared_args(opts);
    let dirs = resolve_dirs(opts)?;
    let entries = load_entries_in(&shared, None, dirs.as_deref())?;
    let mut blocks = identify_session_blocks(entries, session_hours);
    filter_blocks_by_date(&mut blocks, &shared);
    sort_blocks(&mut blocks, &shared.order);
    if active_only {
        blocks.retain(|block| block.is_active);
    }
    Ok(blocks.iter().map(block_usage).collect())
}

fn block_usage(block: &SessionBlock) -> BlockUsage {
    let burn = calculate_burn_rate(block);
    BlockUsage {
        start_ms: block.start_time.as_millis(),
        end_ms: block.end_time.as_millis(),
        actual_end_ms: block.actual_end_time.map(|t| t.as_millis()),
        is_active: block.is_active,
        is_gap: block.is_gap,
        input_tokens: block.token_counts.input_tokens,
        output_tokens: block.token_counts.output_tokens,
        cache_creation_tokens: block.token_counts.cache_creation_tokens,
        cache_read_tokens: block.token_counts.cache_read_tokens,
        total_cost: block.cost_usd,
        models: block.models.clone(),
        burn_rate: burn.map(|b| BurnRateInfo {
            tokens_per_minute: b.tokens_per_minute,
            cost_per_hour: b.cost_per_hour,
        }),
        projection: crate::blocks::project_block_usage(block).map(|p| ProjectionInfo {
            total_tokens: p.total_tokens,
            total_cost: p.total_cost,
            remaining_minutes: p.remaining_minutes,
        }),
    }
}

#[cfg(test)]
mod tests {
    use ccusage_test_support::fs_fixture;

    use super::*;

    fn in_dir(root: &std::path::Path) -> UsageOptions {
        UsageOptions {
            claude_dirs: Some(vec![root.to_path_buf()]),
            ..options()
        }
    }

    fn entry(ts: &str, msg: &str, req: &str, model: &str, input: u64, cost: f64) -> String {
        format!(
            r#"{{"timestamp":"{ts}","message":{{"id":"{msg}","model":"{model}","usage":{{"input_tokens":{input},"output_tokens":10,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}},"requestId":"{req}","costUSD":{cost}}}"#
        )
    }

    fn options() -> UsageOptions {
        UsageOptions {
            offline: true,
            timezone: Some("UTC".to_string()),
            ..UsageOptions::default()
        }
    }

    #[test]
    fn daily_groups_by_date_and_sums_costs() {
        let fixture = fs_fixture!({
            "projects/proj-a/11111111-1111-4111-8111-111111111111.jsonl": [
                entry("2026-01-10T10:00:00.000Z", "m1", "r1", "claude-opus-4-6", 100, 0.5),
                entry("2026-01-10T11:00:00.000Z", "m2", "r2", "claude-opus-4-6", 200, 0.25),
                entry("2026-01-11T09:00:00.000Z", "m3", "r3", "claude-opus-4-6", 50, 0.1),
            ].join("\n"),
        });

        let rows = claude_daily(&in_dir(fixture.root())).unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].period, "2026-01-10");
        assert!((rows[0].total_cost - 0.75).abs() < f64::EPSILON);
        assert_eq!(rows[0].input_tokens, 300);
        assert_eq!(rows[1].period, "2026-01-11");
        assert_eq!(rows[0].models.len(), 1);
        assert_eq!(rows[0].models[0].model, "claude-opus-4-6");
    }

    #[test]
    fn sessions_key_by_jsonl_file_name_and_roll_up_subagents() {
        let session = "22222222-2222-4222-8222-222222222222";
        let fixture = fs_fixture!({
            "projects/proj-a/22222222-2222-4222-8222-222222222222.jsonl":
                entry("2026-01-10T10:00:00.000Z", "m1", "r1", "claude-opus-4-6", 100, 0.5),
            "projects/proj-a/22222222-2222-4222-8222-222222222222/subagents/sub-1.jsonl":
                entry("2026-01-10T10:05:00.000Z", "m2", "r2", "claude-opus-4-6", 40, 0.2),
        });

        let rows = claude_sessions(&in_dir(fixture.root())).unwrap();

        assert_eq!(rows.len(), 1, "subagent log must merge into its parent");
        assert_eq!(rows[0].session_id, session);
        assert_eq!(rows[0].input_tokens, 140);
        assert!((rows[0].total_cost - 0.7).abs() < f64::EPSILON);
        assert_eq!(
            rows[0].last_activity.as_deref(),
            Some("2026-01-10T10:05:00.000Z")
        );
    }

    #[test]
    fn blocks_split_on_gaps_and_mark_nothing_active_for_old_data() {
        let fixture = fs_fixture!({
            "projects/proj-a/33333333-3333-4333-8333-333333333333.jsonl": [
                entry("2026-01-10T10:00:00.000Z", "m1", "r1", "claude-opus-4-6", 100, 0.5),
                // 12h later: a new block (plus a gap block in between).
                entry("2026-01-10T22:00:00.000Z", "m2", "r2", "claude-opus-4-6", 200, 0.25),
            ].join("\n"),
        });

        let blocks = claude_blocks(&in_dir(fixture.root()), 5.0, false).unwrap();

        let real: Vec<_> = blocks.iter().filter(|b| !b.is_gap).collect();
        let gaps: Vec<_> = blocks.iter().filter(|b| b.is_gap).collect();
        assert_eq!(real.len(), 2);
        assert_eq!(gaps.len(), 1);
        assert!(blocks.iter().all(|b| !b.is_active));

        let active = claude_blocks(&in_dir(fixture.root()), 5.0, true).unwrap();
        assert!(active.is_empty());
    }

    #[test]
    fn missing_data_dirs_yield_empty_reports_without_env() {
        // A valid data dir (has projects/) with no logs yields empty reports.
        let fixture = fs_fixture!({
            "projects/.keep": "",
        });

        let daily = claude_daily(&in_dir(fixture.root())).unwrap();
        let sessions = claude_sessions(&in_dir(fixture.root())).unwrap();
        let blocks = claude_blocks(&in_dir(fixture.root()), 5.0, false).unwrap();

        assert!(daily.is_empty());
        assert!(sessions.is_empty());
        assert!(blocks.is_empty());
    }

    #[test]
    fn invalid_explicit_config_dir_is_an_error() {
        let fixture = fs_fixture!({
            "not-projects/.keep": "",
        });

        let result = claude_daily(&in_dir(fixture.root()));

        assert!(result.is_err(), "explicit bad CLAUDE_CONFIG_DIR must error");
    }

    #[test]
    fn since_until_bound_daily_rows_inclusively() {
        let fixture = fs_fixture!({
            "projects/proj-a/44444444-4444-4444-8444-444444444444.jsonl": [
                entry("2026-01-09T10:00:00.000Z", "m1", "r1", "claude-opus-4-6", 1, 0.1),
                entry("2026-01-10T10:00:00.000Z", "m2", "r2", "claude-opus-4-6", 2, 0.2),
                entry("2026-01-11T10:00:00.000Z", "m3", "r3", "claude-opus-4-6", 3, 0.3),
            ].join("\n"),
        });
        let opts = UsageOptions {
            since: Some("2026-01-10".to_string()),
            until: Some("2026-01-10".to_string()),
            ..in_dir(fixture.root())
        };

        let rows = claude_daily(&opts).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].period, "2026-01-10");
    }
}
