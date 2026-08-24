//! Embedding API (thenvoi fork addition).
//!
//! A stable, self-contained programmatic surface over the report pipelines the
//! CLI commands use (`daily` / `weekly` / `monthly` / `session` / `blocks`),
//! returning plain owned structs instead of printing. The types here are
//! deliberately decoupled from the crate's internal types so embedders never
//! depend on internals.
//!
//! Claude-specific functions resolve data directories exactly like the CLI:
//! `CLAUDE_CONFIG_DIR` (comma-separated) when set, else `$XDG_CONFIG_HOME/claude`
//! and `~/.claude`. Missing directories yield empty reports, not errors; an
//! explicitly-set `CLAUDE_CONFIG_DIR` with no valid entries is an error
//! (mirroring the CLI). All-provider functions scan every supported adapter;
//! `UsageOptions::claude_dirs` only overrides Claude Code discovery.

use std::{collections::BTreeMap, fs, io::Read, path::PathBuf};

use sha2::{Digest, Sha256};

use crate::{
    BucketKind, DEFAULT_SESSION_DURATION_HOURS, ModelBreakdown, Result, SessionAccumulator,
    SessionBlock, UsageSummary,
    adapter::{
        all::{loader::load_rows_in, types::AllRow},
        claude::{load_daily_summaries_in, load_entries_from_captured_files, load_entries_in},
        codex::load_codex_events_from_captured_manifest,
    },
    calculate_burn_rate,
    cli::{AgentReportKind, CostMode, SharedArgs, SortOrder, WeekDay, normalize_date_bound},
    filter_and_sort_summaries, filter_blocks_by_date, identify_session_blocks, sort_blocks,
    sort_summaries, summarize_by_key, summarize_summaries_by_bucket,
};

/// Provider supported by the body-free detailed event prototype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DetailedUsageProvider {
    Claude,
    Codex,
}

/// Provider-native identity parts. An absent identity is reported as `None`;
/// embedders must never synthesize one from timestamp or token values.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProviderEventId {
    pub primary: String,
    pub secondary: Option<String>,
}

/// Whether token values were emitted as a delta or derived from a cumulative
/// counter. A decrease remains explicitly ambiguous when provider data cannot
/// distinguish reset from correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DetailedCounterMode {
    Delta,
    CumulativeDelta,
    CumulativeDecreaseAmbiguous,
}

/// Permission state for parsing detailed provider logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetailedCapturePermission {
    Denied,
    Granted,
}

/// Explicit, bounded inputs for detailed capture. Calling this API represents
/// detailed-capture consent; it never performs default home-directory discovery.
#[derive(Debug, Clone)]
pub struct DetailedUsageOptions {
    /// Must be `Granted`. Denied calls return before directory enumeration.
    pub capture_permission: DetailedCapturePermission,
    /// Explicit Claude config roots, each containing `projects/`.
    pub claude_dirs: Vec<PathBuf>,
    /// Explicit Codex `sessions/` or `archived_sessions/` directories.
    pub codex_session_dirs: Vec<PathBuf>,
    /// Maximum combined size of JSONL sources. Checked before any log parsing.
    pub max_source_bytes: u64,
    /// Maximum number of JSONL files captured in one manifest.
    pub max_source_files: usize,
    /// Maximum recursive directory depth below each explicit provider root.
    pub max_directory_depth: usize,
    /// Maximum explicit roots plus directory entries inspected, including
    /// directories and files that are not JSONL sources.
    pub max_discovery_entries: usize,
    /// Maximum normalized events returned from one scan.
    pub max_events: usize,
}

/// One body-free normalized token event.
#[derive(Debug, Clone, PartialEq)]
pub struct DetailedUsageEvent {
    pub provider: DetailedUsageProvider,
    pub session_id: String,
    pub provider_event_id: Option<ProviderEventId>,
    pub timestamp: String,
    pub model: Option<String>,
    /// Non-cached input tokens. Cached input is reported separately.
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
    /// Included in `output_tokens` for Codex; never add it again to totals.
    pub reasoning_output_tokens: u64,
    pub total_cost: Option<f64>,
    pub missing_pricing: bool,
    pub counter_mode: DetailedCounterMode,
    /// Proven source epoch. Remains zero when reset provenance is unavailable.
    pub counter_epoch: u64,
}

/// Failed proof obligations that prohibit durable detailed-event capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetailedCaptureBlocker {
    AuthoritativeCorrectionOrderUnavailable,
    ProviderEventIdentityUnavailable,
    CumulativeResetVsCorrectionAmbiguous,
    ExactCostReconciliationUnavailable,
    BaselineCarryInReconciliationUnavailable,
}

/// Provider-specific graduation result. Events from a blocked prototype may
/// be inspected in memory for reconciliation tests but must not be persisted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetailedProviderFeasibility {
    pub durable_capture_allowed: bool,
    pub blockers: Vec<DetailedCaptureBlocker>,
}

/// Provider-scoped evidence needed by a caller's feasibility gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetailedSourceSnapshot {
    pub provider: DetailedUsageProvider,
    /// SHA-256 revision of canonical body-free usage and correction fields.
    pub source_revision: String,
    pub source_bytes: u64,
    pub event_count: usize,
    pub observed_lower_bound: Option<String>,
    pub observed_upper_bound: Option<String>,
    pub identity_complete: bool,
    pub feasibility: DetailedProviderFeasibility,
}

/// Result of one consented, bounded scan.
#[derive(Debug, Clone, PartialEq)]
pub struct DetailedUsageScan {
    pub events: Vec<DetailedUsageEvent>,
    pub sources: Vec<DetailedSourceSnapshot>,
    pub source_bytes: u64,
}

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
    /// Optional all-provider adapter filter for embedders and tests. `None`
    /// means scan every supported adapter.
    pub providers: Option<Vec<String>>,
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

/// Per-model usage within an all-provider report row.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentModelUsage {
    pub provider: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
    pub cost: f64,
    /// True when no pricing was found for this model (its cost may be 0).
    pub missing_pricing: bool,
}

/// One provider-specific row of an all-provider daily report.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentPeriodUsage {
    /// Provider/adapter key, e.g. `claude`, `codex`, or `opencode`.
    pub provider: String,
    /// The grouping key: a date (`2026-06-11`).
    pub period: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
    pub total_cost: f64,
    pub models: Vec<AgentModelUsage>,
}

/// One provider-specific session row from the all-provider scan.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentSessionUsage {
    /// Provider/adapter key, e.g. `claude`, `codex`, or `opencode`.
    pub provider: String,
    pub session_id: String,
    pub project_path: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
    pub total_cost: f64,
    pub models: Vec<AgentModelUsage>,
    /// RFC3339 milliseconds when the adapter exposes it.
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

fn agent_model_usage(provider: &str, b: &ModelBreakdown) -> AgentModelUsage {
    AgentModelUsage {
        provider: provider.to_string(),
        model: b.model_name.clone(),
        input_tokens: b.input_tokens,
        output_tokens: b.output_tokens,
        cache_creation_tokens: b.cache_creation_tokens,
        cache_read_tokens: b.cache_read_tokens,
        cost: b.cost,
        missing_pricing: b.missing_pricing,
    }
}

fn agent_period_usage(row: &AllRow) -> AgentPeriodUsage {
    AgentPeriodUsage {
        provider: row.agent.to_string(),
        period: row.period.clone(),
        input_tokens: row.input_tokens,
        output_tokens: row.output_tokens,
        cache_creation_tokens: row.cache_creation_tokens,
        cache_read_tokens: row.cache_read_tokens,
        total_cost: row.total_cost,
        models: row
            .model_breakdowns
            .iter()
            .map(|b| agent_model_usage(row.agent, b))
            .collect(),
    }
}

fn agent_session_usage(row: &AllRow) -> AgentSessionUsage {
    AgentSessionUsage {
        provider: row.agent.to_string(),
        session_id: row.period.clone(),
        project_path: metadata_string(row, "projectPath").unwrap_or_default(),
        input_tokens: row.input_tokens,
        output_tokens: row.output_tokens,
        cache_creation_tokens: row.cache_creation_tokens,
        cache_read_tokens: row.cache_read_tokens,
        total_cost: row.total_cost,
        models: row
            .model_breakdowns
            .iter()
            .map(|b| agent_model_usage(row.agent, b))
            .collect(),
        first_activity: metadata_string(row, "firstActivity"),
        last_activity: metadata_string(row, "lastActivity"),
    }
}

fn metadata_string(row: &AllRow, key: &str) -> Option<String> {
    row.metadata.as_ref()?.get(key)?.as_str().map(str::to_owned)
}

fn flatten_agent_rows(rows: Vec<AllRow>) -> Vec<AllRow> {
    rows.into_iter()
        .flat_map(|mut row| row.agent_breakdowns.take().unwrap_or_else(|| vec![row]))
        .filter(|row| row.agent != "all")
        .collect()
}

fn provider_allowed(opts: &UsageOptions, provider: &str) -> bool {
    opts.providers
        .as_ref()
        .is_none_or(|providers| providers.iter().any(|p| p == provider))
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

/// Daily usage from every supported local coding-agent adapter, returned as
/// provider-specific rows. Unlike the CLI's `agent daily` table, rows are not
/// collapsed into a single `all` provider because embedders usually need to
/// persist and aggregate by provider.
pub fn all_daily(opts: &UsageOptions) -> Result<Vec<AgentPeriodUsage>> {
    let shared = shared_args(opts);
    let dirs = resolve_dirs(opts)?;
    let rows = load_rows_in(
        AgentReportKind::Daily,
        &shared,
        dirs.as_deref(),
        opts.providers.as_deref(),
    )?;
    Ok(flatten_agent_rows(rows.rows)
        .iter()
        .filter(|row| provider_allowed(opts, row.agent))
        .map(agent_period_usage)
        .collect())
}

/// First day of the week for weekly grouping.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WeekStart {
    /// The CLI default.
    #[default]
    Sunday,
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
}

impl WeekStart {
    fn week_day(self) -> WeekDay {
        match self {
            WeekStart::Sunday => WeekDay::Sunday,
            WeekStart::Monday => WeekDay::Monday,
            WeekStart::Tuesday => WeekDay::Tuesday,
            WeekStart::Wednesday => WeekDay::Wednesday,
            WeekStart::Thursday => WeekDay::Thursday,
            WeekStart::Friday => WeekDay::Friday,
            WeekStart::Saturday => WeekDay::Saturday,
        }
    }
}

/// Weekly Claude Code usage; weeks start on `week_starts_on` (the CLI
/// default is Sunday) and are keyed by the week's start date.
pub fn claude_weekly(opts: &UsageOptions, week_starts_on: WeekStart) -> Result<Vec<PeriodUsage>> {
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
    let mut weekly =
        summarize_summaries_by_bucket(&daily, BucketKind::Weekly, week_starts_on.week_day());
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
            // Compare the DATE PREFIX only. The upstream CLI compares the whole
            // RFC3339 stamp against the compact date bound lexically, which
            // silently excludes sessions last active ON the `until` day
            // ("…0610T12:00…" > "20260610"); bounds here are documented
            // inclusive, so a session on the boundary day must stay.
            let date = row
                .last_activity
                .as_deref()
                .unwrap_or_default()
                .get(..10)
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

/// Per-session usage from every supported local coding-agent adapter. Session
/// identifiers are provider-local; consumers should key by `(provider,
/// session_id)`, not session ID alone.
pub fn all_sessions(opts: &UsageOptions) -> Result<Vec<AgentSessionUsage>> {
    let shared = shared_args(opts);
    let dirs = resolve_dirs(opts)?;
    let rows = load_rows_in(
        AgentReportKind::Session,
        &shared,
        dirs.as_deref(),
        opts.providers.as_deref(),
    )?;
    Ok(rows
        .rows
        .iter()
        .filter(|row| provider_allowed(opts, row.agent))
        .map(agent_session_usage)
        .collect())
}

#[derive(Debug)]
struct ManifestFile {
    root: PathBuf,
    path: PathBuf,
    content: Vec<u8>,
}

#[derive(Debug)]
struct SourceManifest {
    files: Vec<ManifestFile>,
    source_bytes: u64,
    discovery_entries: usize,
}

fn charge_discovery_entry(observed: &mut usize, limit: usize) -> Result<()> {
    *observed = observed
        .checked_add(1)
        .ok_or_else(|| crate::cli_error("detailed discovery entry count overflow"))?;
    if *observed > limit {
        return Err(crate::cli_error(format!(
            "detailed discovery entry limit exceeded: more than {limit}"
        )));
    }
    Ok(())
}

fn source_manifest(
    roots: impl IntoIterator<Item = (PathBuf, PathBuf)>,
    max_source_bytes: u64,
    max_source_files: usize,
    max_directory_depth: usize,
    max_discovery_entries: usize,
) -> Result<SourceManifest> {
    let mut discovered = BTreeMap::<PathBuf, PathBuf>::new();
    let mut discovery_entries = 0_usize;
    for (provider_root, scan_root) in roots {
        charge_discovery_entry(&mut discovery_entries, max_discovery_entries)?;
        let mut pending = vec![(scan_root, 0_usize)];
        while let Some((dir, depth)) = pending.pop() {
            let entries = fs::read_dir(&dir).map_err(|error| {
                crate::cli_error(format!(
                    "cannot enumerate detailed source {}: {error}",
                    dir.display()
                ))
            })?;
            for entry in entries {
                charge_discovery_entry(&mut discovery_entries, max_discovery_entries)?;
                let entry = entry?;
                let file_type = entry.file_type()?;
                let path = entry.path();
                if file_type.is_dir() {
                    if depth >= max_directory_depth {
                        return Err(crate::cli_error(format!(
                            "detailed source directory depth limit exceeded at {}",
                            path.display()
                        )));
                    }
                    pending.push((path, depth + 1));
                } else if file_type.is_file()
                    && path
                        .extension()
                        .is_some_and(|extension| extension == "jsonl")
                {
                    discovered
                        .entry(path)
                        .or_insert_with(|| provider_root.clone());
                    if discovered.len() > max_source_files {
                        return Err(crate::cli_error(format!(
                            "detailed source file limit exceeded: more than {max_source_files}"
                        )));
                    }
                }
            }
        }
    }

    let mut files = Vec::with_capacity(discovered.len());
    let mut source_bytes = 0_u64;
    for (path, root) in discovered {
        let len = fs::metadata(&path)?.len();
        source_bytes = source_bytes
            .checked_add(len)
            .ok_or_else(|| crate::cli_error("detailed source byte count overflow"))?;
        if source_bytes > max_source_bytes {
            return Err(crate::cli_error(format!(
                "detailed source byte limit exceeded: {source_bytes} > {max_source_bytes}"
            )));
        }
        let capacity = usize::try_from(len)
            .map_err(|_| crate::cli_error("detailed source is too large for this platform"))?;
        let mut content = Vec::with_capacity(capacity);
        let mut input = fs::File::open(&path)?.take(len.saturating_add(1));
        input.read_to_end(&mut content)?;
        if content.len() as u64 != len {
            return Err(crate::cli_error(format!(
                "detailed source changed while capturing manifest: {}",
                path.display()
            )));
        }
        files.push(ManifestFile {
            root,
            path,
            content,
        });
    }
    Ok(SourceManifest {
        files,
        source_bytes,
        discovery_entries,
    })
}

fn revision_field(hasher: &mut Sha256, value: impl AsRef<[u8]>) {
    let value = value.as_ref();
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn finish_revision(hasher: Sha256) -> String {
    let digest = hasher.finalize();
    let mut revision = String::with_capacity("sha256:".len() + digest.len() * 2);
    revision.push_str("sha256:");
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(revision, "{byte:02x}");
    }
    revision
}

fn hash_detailed_event(hasher: &mut Sha256, event: &DetailedUsageEvent) {
    revision_field(hasher, format!("{:?}", event.provider));
    revision_field(hasher, &event.session_id);
    if let Some(id) = &event.provider_event_id {
        revision_field(hasher, &id.primary);
        revision_field(hasher, id.secondary.as_deref().unwrap_or(""));
    } else {
        revision_field(hasher, "<missing-id>");
    }
    revision_field(hasher, &event.timestamp);
    revision_field(hasher, event.model.as_deref().unwrap_or(""));
    for value in [
        event.input_tokens,
        event.output_tokens,
        event.cache_creation_tokens,
        event.cache_read_tokens,
        event.reasoning_output_tokens,
        event.total_cost.map(f64::to_bits).unwrap_or_default(),
        event.counter_epoch,
    ] {
        hasher.update(value.to_le_bytes());
    }
    hasher.update([event.missing_pricing as u8]);
    revision_field(hasher, format!("{:?}", event.counter_mode));
}

fn detailed_revision(events: &[DetailedUsageEvent]) -> String {
    let mut hasher = Sha256::new();
    for event in events {
        hash_detailed_event(&mut hasher, event);
    }
    finish_revision(hasher)
}

fn claude_revision(manifest: &SourceManifest, events: &[DetailedUsageEvent]) -> String {
    let mut hasher = Sha256::new();
    for file in &manifest.files {
        let relative_path = file.path.strip_prefix(&file.root).unwrap_or(&file.path);
        revision_field(&mut hasher, relative_path.to_string_lossy().as_bytes());
        for line in crate::fast::byte_lines(&file.content) {
            let Ok(entry) = serde_json::from_slice::<crate::UsageEntry>(line) else {
                continue;
            };
            revision_field(&mut hasher, &entry.timestamp);
            revision_field(&mut hasher, entry.session_id.as_deref().unwrap_or(""));
            revision_field(&mut hasher, entry.message.id.as_deref().unwrap_or(""));
            revision_field(&mut hasher, entry.request_id.as_deref().unwrap_or(""));
            revision_field(&mut hasher, entry.message.model.as_deref().unwrap_or(""));
            let usage = entry.message.usage;
            for value in [
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_token_count(),
                usage.cache_read_input_tokens,
                entry.cost_usd.map(f64::to_bits).unwrap_or_default(),
            ] {
                hasher.update(value.to_le_bytes());
            }
        }
    }
    for event in events {
        hash_detailed_event(&mut hasher, event);
    }
    finish_revision(hasher)
}

fn detailed_snapshot(
    provider: DetailedUsageProvider,
    events: &[DetailedUsageEvent],
    manifest: &SourceManifest,
    source_revision: String,
) -> DetailedSourceSnapshot {
    let blockers = match provider {
        DetailedUsageProvider::Claude => vec![
            DetailedCaptureBlocker::AuthoritativeCorrectionOrderUnavailable,
            DetailedCaptureBlocker::BaselineCarryInReconciliationUnavailable,
        ],
        DetailedUsageProvider::Codex => vec![
            DetailedCaptureBlocker::ProviderEventIdentityUnavailable,
            DetailedCaptureBlocker::CumulativeResetVsCorrectionAmbiguous,
            DetailedCaptureBlocker::ExactCostReconciliationUnavailable,
            DetailedCaptureBlocker::BaselineCarryInReconciliationUnavailable,
        ],
    };
    DetailedSourceSnapshot {
        provider,
        source_revision,
        source_bytes: manifest.source_bytes,
        event_count: events.len(),
        observed_lower_bound: events.iter().map(|event| &event.timestamp).min().cloned(),
        observed_upper_bound: events.iter().map(|event| &event.timestamp).max().cloned(),
        identity_complete: events.iter().all(|event| event.provider_event_id.is_some()),
        feasibility: DetailedProviderFeasibility {
            durable_capture_allowed: blockers.is_empty(),
            blockers,
        },
    }
}

fn sort_detailed_events(events: &mut [DetailedUsageEvent]) {
    events.sort_by(|left, right| {
        (
            left.provider,
            &left.timestamp,
            &left.session_id,
            &left.provider_event_id,
            &left.model,
        )
            .cmp(&(
                right.provider,
                &right.timestamp,
                &right.session_id,
                &right.provider_event_id,
                &right.model,
            ))
    });
}

/// Read normalized, body-free Claude and Codex token events from explicit
/// consented directories. The call is bounded by source bytes and event count,
/// and performs no implicit provider discovery.
pub fn detailed_usage_events(opts: &DetailedUsageOptions) -> Result<DetailedUsageScan> {
    if opts.capture_permission != DetailedCapturePermission::Granted {
        return Err(crate::cli_error(
            "detailed capture permission has not been granted",
        ));
    }
    let claude_manifest = source_manifest(
        opts.claude_dirs
            .iter()
            .cloned()
            .map(|root| (root.clone(), root.join("projects"))),
        opts.max_source_bytes,
        opts.max_source_files,
        opts.max_directory_depth,
        opts.max_discovery_entries,
    )?;
    let codex_manifest = source_manifest(
        opts.codex_session_dirs
            .iter()
            .cloned()
            .map(|root| (root.clone(), root)),
        opts.max_source_bytes
            .saturating_sub(claude_manifest.source_bytes),
        opts.max_source_files
            .saturating_sub(claude_manifest.files.len()),
        opts.max_directory_depth,
        opts.max_discovery_entries
            .saturating_sub(claude_manifest.discovery_entries),
    )?;
    let total_bytes = claude_manifest
        .source_bytes
        .checked_add(codex_manifest.source_bytes)
        .ok_or_else(|| crate::cli_error("detailed source byte count overflow"))?;
    if total_bytes > opts.max_source_bytes {
        return Err(crate::cli_error(format!(
            "detailed source byte limit exceeded: {total_bytes} > {}",
            opts.max_source_bytes
        )));
    }

    let shared = SharedArgs {
        json: true,
        offline: true,
        ..SharedArgs::default()
    };
    let mut claude_events = if opts.claude_dirs.is_empty() {
        Vec::new()
    } else {
        load_entries_from_captured_files(
            &shared,
            claude_manifest
                .files
                .iter()
                .map(|file| (file.path.as_path(), file.content.as_slice())),
            opts.max_events,
        )?
        .into_iter()
        .map(|entry| {
            let usage = entry.data.message.usage;
            DetailedUsageEvent {
                provider: DetailedUsageProvider::Claude,
                session_id: entry.session_id.to_string(),
                provider_event_id: entry.data.message.id.map(|primary| ProviderEventId {
                    primary,
                    secondary: entry.data.request_id,
                }),
                timestamp: entry.data.timestamp,
                model: entry.model,
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                cache_creation_tokens: usage.cache_creation_token_count(),
                cache_read_tokens: usage.cache_read_input_tokens,
                reasoning_output_tokens: 0,
                total_cost: Some(entry.cost),
                missing_pricing: entry.missing_pricing_model.is_some(),
                counter_mode: DetailedCounterMode::Delta,
                counter_epoch: 0,
            }
        })
        .collect::<Vec<_>>()
    };

    let mut codex_events = load_codex_events_from_captured_manifest(
        codex_manifest.files.iter().map(|file| {
            (
                file.root.as_path(),
                file.path.as_path(),
                file.content.as_slice(),
            )
        }),
        opts.max_events.saturating_sub(claude_events.len()),
    )?
    .into_iter()
    .map(|event| DetailedUsageEvent {
        provider: DetailedUsageProvider::Codex,
        session_id: event.session_id,
        // Codex token_count records do not carry a provider event
        // ID. Deliberately report the gap instead of synthesizing
        // identity from mutable token or timestamp fields.
        provider_event_id: None,
        timestamp: event.timestamp,
        model: event.model,
        input_tokens: event.input_tokens.saturating_sub(event.cached_input_tokens),
        output_tokens: event.output_tokens,
        cache_creation_tokens: 0,
        cache_read_tokens: event.cached_input_tokens,
        reasoning_output_tokens: event.reasoning_output_tokens,
        total_cost: None,
        missing_pricing: true,
        counter_mode: match event.counter_mode {
            crate::CodexCounterMode::Delta => DetailedCounterMode::Delta,
            crate::CodexCounterMode::CumulativeDelta => DetailedCounterMode::CumulativeDelta,
            crate::CodexCounterMode::CumulativeDecreaseAmbiguous => {
                DetailedCounterMode::CumulativeDecreaseAmbiguous
            }
        },
        counter_epoch: event.counter_epoch,
    })
    .collect::<Vec<_>>();
    sort_detailed_events(&mut claude_events);
    sort_detailed_events(&mut codex_events);

    let event_count = claude_events
        .len()
        .checked_add(codex_events.len())
        .ok_or_else(|| crate::cli_error("detailed event count overflow"))?;
    if event_count > opts.max_events {
        return Err(crate::cli_error(format!(
            "detailed event limit exceeded: {event_count} > {}",
            opts.max_events
        )));
    }

    let sources = vec![
        detailed_snapshot(
            DetailedUsageProvider::Claude,
            &claude_events,
            &claude_manifest,
            claude_revision(&claude_manifest, &claude_events),
        ),
        detailed_snapshot(
            DetailedUsageProvider::Codex,
            &codex_events,
            &codex_manifest,
            detailed_revision(&codex_events),
        ),
    ];
    claude_events.extend(codex_events);
    sort_detailed_events(&mut claude_events);
    Ok(DetailedUsageScan {
        events: claude_events,
        sources,
        source_bytes: total_bytes,
    })
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

    fn codex_cumulative(ts: &str, input: u64, cached: u64, output: u64) -> String {
        serde_json::json!({
            "timestamp": ts,
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": input,
                        "cached_input_tokens": cached,
                        "output_tokens": output,
                        "total_tokens": input + output,
                    },
                    "model": "gpt-5",
                },
            },
        })
        .to_string()
    }

    #[test]
    fn detailed_claude_events_are_body_free_and_keep_provider_identity() {
        let fixture = fs_fixture!({
            "projects/proj-a/session.jsonl": format!(
                r#"{{"timestamp":"2026-01-10T10:00:00.000Z","message":{{"id":"m1","model":"claude-opus-4-6","content":"DO_NOT_RETAIN_SENTINEL","usage":{{"input_tokens":100,"output_tokens":10,"cache_creation_input_tokens":2,"cache_read_input_tokens":3}}}},"requestId":"r1","costUSD":0.5}}"#
            ),
        });
        let opts = DetailedUsageOptions {
            capture_permission: DetailedCapturePermission::Granted,
            claude_dirs: vec![fixture.root().to_path_buf()],
            codex_session_dirs: Vec::new(),
            max_source_bytes: 1024 * 1024,
            max_source_files: 10,
            max_directory_depth: 8,
            max_discovery_entries: 100,
            max_events: 10,
        };

        let scan = detailed_usage_events(&opts).unwrap();

        assert_eq!(scan.events.len(), 1);
        let event = &scan.events[0];
        assert_eq!(event.provider, DetailedUsageProvider::Claude);
        assert_eq!(event.session_id, "session");
        assert_eq!(
            event.provider_event_id,
            Some(ProviderEventId {
                primary: "m1".to_string(),
                secondary: Some("r1".to_string()),
            })
        );
        assert_eq!(event.counter_mode, DetailedCounterMode::Delta);
        assert_eq!(event.counter_epoch, 0);
        assert_eq!(event.input_tokens, 100);
        assert_eq!(event.output_tokens, 10);
        assert_eq!(event.cache_creation_tokens, 2);
        assert_eq!(event.cache_read_tokens, 3);
        assert!((event.total_cost.unwrap() - 0.5).abs() < f64::EPSILON);
        assert!(!format!("{scan:?}").contains("DO_NOT_RETAIN_SENTINEL"));
        assert!(!scan.sources[0].feasibility.durable_capture_allowed);
        assert!(
            scan.sources[0]
                .feasibility
                .blockers
                .contains(&DetailedCaptureBlocker::AuthoritativeCorrectionOrderUnavailable)
        );
    }

    #[test]
    fn detailed_codex_events_expose_cumulative_deltas_ambiguous_decreases_and_missing_identity() {
        let fixture = fs_fixture!({
            "session.jsonl": [
                codex_cumulative("2026-01-10T10:00:00.000Z", 100, 20, 10),
                codex_cumulative("2026-01-10T10:01:00.000Z", 160, 30, 25),
                codex_cumulative("2026-01-10T10:02:00.000Z", 40, 5, 30),
            ].join("\n"),
        });
        let opts = DetailedUsageOptions {
            capture_permission: DetailedCapturePermission::Granted,
            claude_dirs: Vec::new(),
            codex_session_dirs: vec![fixture.root().to_path_buf()],
            max_source_bytes: 1024 * 1024,
            max_source_files: 10,
            max_directory_depth: 8,
            max_discovery_entries: 100,
            max_events: 10,
        };

        let scan = detailed_usage_events(&opts).unwrap();

        assert_eq!(scan.events.len(), 3);
        assert!(
            scan.events
                .iter()
                .all(|event| event.provider_event_id.is_none())
        );
        assert_eq!(
            scan.events[0].counter_mode,
            DetailedCounterMode::CumulativeDelta
        );
        assert_eq!(scan.events[0].counter_epoch, 0);
        assert_eq!(scan.events[1].input_tokens, 50);
        assert_eq!(scan.events[1].cache_read_tokens, 10);
        assert_eq!(
            scan.events[1].counter_mode,
            DetailedCounterMode::CumulativeDelta
        );
        assert_eq!(scan.events[1].counter_epoch, 0);
        assert_eq!(scan.events[2].input_tokens, 0);
        assert_eq!(scan.events[2].cache_read_tokens, 0);
        assert_eq!(scan.events[2].output_tokens, 5);
        assert_eq!(
            scan.events[2].counter_mode,
            DetailedCounterMode::CumulativeDecreaseAmbiguous
        );
        assert_eq!(scan.events[2].counter_epoch, 0);
        assert!(!scan.sources[1].feasibility.durable_capture_allowed);
        assert!(
            scan.sources[1]
                .feasibility
                .blockers
                .contains(&DetailedCaptureBlocker::ProviderEventIdentityUnavailable)
        );
        assert!(
            scan.sources[1]
                .feasibility
                .blockers
                .contains(&DetailedCaptureBlocker::CumulativeResetVsCorrectionAmbiguous)
        );
    }

    #[test]
    fn detailed_scan_rejects_sources_over_the_byte_limit_before_parsing() {
        let fixture = fs_fixture!({
            "projects/proj-a/session.jsonl": entry(
                "2026-01-10T10:00:00.000Z",
                "m1",
                "r1",
                "claude-opus-4-6",
                100,
                0.5,
            ),
        });
        let opts = DetailedUsageOptions {
            capture_permission: DetailedCapturePermission::Granted,
            claude_dirs: vec![fixture.root().to_path_buf()],
            codex_session_dirs: Vec::new(),
            max_source_bytes: 1,
            max_source_files: 10,
            max_directory_depth: 8,
            max_discovery_entries: 100,
            max_events: 10,
        };

        let error = detailed_usage_events(&opts).unwrap_err();

        assert!(error.to_string().contains("source byte limit"));
    }

    #[test]
    fn detailed_scan_requires_capture_permission_before_discovery() {
        let opts = DetailedUsageOptions {
            capture_permission: DetailedCapturePermission::Denied,
            claude_dirs: vec![PathBuf::from("/definitely/not/a/provider/root")],
            codex_session_dirs: vec![PathBuf::from("/also/not/a/provider/root")],
            max_source_bytes: u64::MAX,
            max_source_files: usize::MAX,
            max_directory_depth: usize::MAX,
            max_discovery_entries: usize::MAX,
            max_events: usize::MAX,
        };

        let error = detailed_usage_events(&opts).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("permission has not been granted")
        );
    }

    #[test]
    fn detailed_claude_correction_replaces_event_and_changes_revision() {
        let original = fs_fixture!({
            "projects/proj-a/session.jsonl": entry(
                "2026-01-10T10:00:00.000Z",
                "m1",
                "r1",
                "claude-opus-4-6",
                100,
                0.5,
            ),
        });
        let corrected = fs_fixture!({
            "projects/proj-a/session.jsonl": [
                entry(
                    "2026-01-10T10:00:00.000Z",
                    "m1",
                    "r1",
                    "claude-opus-4-6",
                    100,
                    0.5,
                ),
                entry(
                    "2026-01-10T10:00:00.000Z",
                    "m1",
                    "r1",
                    "claude-opus-4-6",
                    200,
                    0.75,
                ),
            ].join("\n"),
        });
        let scan = |root: &std::path::Path| {
            detailed_usage_events(&DetailedUsageOptions {
                capture_permission: DetailedCapturePermission::Granted,
                claude_dirs: vec![root.to_path_buf()],
                codex_session_dirs: Vec::new(),
                max_source_bytes: 1024 * 1024,
                max_source_files: 10,
                max_directory_depth: 8,
                max_discovery_entries: 100,
                max_events: 10,
            })
            .unwrap()
        };

        let before = scan(original.root());
        let after = scan(corrected.root());

        assert_eq!(after.events.len(), 1);
        assert_eq!(after.events[0].input_tokens, 200);
        assert!((after.events[0].total_cost.unwrap() - 0.75).abs() < f64::EPSILON);
        assert_ne!(
            before.sources[0].source_revision,
            after.sources[0].source_revision
        );
    }

    #[test]
    fn detailed_claude_token_categories_reconcile_with_aggregate_facade() {
        let fixture = fs_fixture!({
            "projects/proj-a/session.jsonl": [
                entry("2026-01-10T10:00:00.000Z", "m1", "r1", "claude-opus-4-6", 100, 0.5),
                entry("2026-01-10T10:01:00.000Z", "m2", "r2", "claude-opus-4-6", 200, 0.25),
            ].join("\n"),
        });
        let scan = detailed_usage_events(&DetailedUsageOptions {
            capture_permission: DetailedCapturePermission::Granted,
            claude_dirs: vec![fixture.root().to_path_buf()],
            codex_session_dirs: Vec::new(),
            max_source_bytes: 1024 * 1024,
            max_source_files: 10,
            max_directory_depth: 8,
            max_discovery_entries: 100,
            max_events: 10,
        })
        .unwrap();
        let aggregate = claude_daily(&in_dir(fixture.root())).unwrap();

        assert_eq!(
            scan.events
                .iter()
                .map(|event| event.input_tokens)
                .sum::<u64>(),
            aggregate[0].input_tokens
        );
        assert_eq!(
            scan.events
                .iter()
                .map(|event| event.output_tokens)
                .sum::<u64>(),
            aggregate[0].output_tokens
        );
        assert_eq!(
            scan.events
                .iter()
                .map(|event| event.cache_creation_tokens)
                .sum::<u64>(),
            aggregate[0].cache_creation_tokens
        );
        assert_eq!(
            scan.events
                .iter()
                .map(|event| event.cache_read_tokens)
                .sum::<u64>(),
            aggregate[0].cache_read_tokens
        );
        assert!(
            (scan
                .events
                .iter()
                .filter_map(|event| event.total_cost)
                .sum::<f64>()
                - aggregate[0].total_cost)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn detailed_manifest_revision_observes_downward_correction_even_when_adapter_blocks_it() {
        let original = fs_fixture!({
            "projects/proj-a/session.jsonl": entry(
                "2026-01-10T10:00:00.000Z",
                "m1",
                "r1",
                "claude-opus-4-6",
                200,
                0.75,
            ),
        });
        let corrected = fs_fixture!({
            "projects/proj-a/session.jsonl": [
                entry("2026-01-10T10:00:00.000Z", "m1", "r1", "claude-opus-4-6", 200, 0.75),
                entry("2026-01-10T10:00:00.000Z", "m1", "r1", "claude-opus-4-6", 100, 0.25),
            ].join("\n"),
        });
        let scan = |root: &std::path::Path| {
            detailed_usage_events(&DetailedUsageOptions {
                capture_permission: DetailedCapturePermission::Granted,
                claude_dirs: vec![root.to_path_buf()],
                codex_session_dirs: Vec::new(),
                max_source_bytes: 1024 * 1024,
                max_source_files: 10,
                max_directory_depth: 8,
                max_discovery_entries: 100,
                max_events: 10,
            })
            .unwrap()
        };

        let before = scan(original.root());
        let after = scan(corrected.root());

        assert_ne!(
            before.sources[0].source_revision,
            after.sources[0].source_revision
        );
        assert_eq!(after.events[0].input_tokens, 200);
        assert!(
            after.sources[0]
                .feasibility
                .blockers
                .contains(&DetailedCaptureBlocker::AuthoritativeCorrectionOrderUnavailable)
        );
    }

    #[test]
    fn detailed_revision_does_not_fingerprint_message_bodies() {
        let usage_line = |body: &str| {
            format!(
                r#"{{"timestamp":"2026-01-10T10:00:00.000Z","message":{{"id":"m1","model":"claude-opus-4-6","content":"{body}","usage":{{"input_tokens":100,"output_tokens":10,"cache_creation_input_tokens":2,"cache_read_input_tokens":3}}}},"requestId":"r1","costUSD":0.5}}"#
            )
        };
        let first = fs_fixture!({
            "projects/proj-a/session.jsonl": usage_line("FIRST_PRIVATE_BODY"),
        });
        let second = fs_fixture!({
            "projects/proj-a/session.jsonl": usage_line("OTHER_PRIVATE_BODY"),
        });
        let scan = |root: &std::path::Path| {
            detailed_usage_events(&DetailedUsageOptions {
                capture_permission: DetailedCapturePermission::Granted,
                claude_dirs: vec![root.to_path_buf()],
                codex_session_dirs: Vec::new(),
                max_source_bytes: 1024 * 1024,
                max_source_files: 10,
                max_directory_depth: 8,
                max_discovery_entries: 100,
                max_events: 10,
            })
            .unwrap()
        };

        let first = scan(first.root());
        let second = scan(second.root());

        assert_eq!(
            first.sources[0].source_revision,
            second.sources[0].source_revision
        );
        assert!(!format!("{first:?}").contains("FIRST_PRIVATE_BODY"));
        assert!(!format!("{second:?}").contains("OTHER_PRIVATE_BODY"));
    }

    #[test]
    fn detailed_manifest_enforces_file_depth_and_event_limits() {
        let files = fs_fixture!({
            "projects/proj-a/a.jsonl": entry("2026-01-10T10:00:00.000Z", "m1", "r1", "claude-opus-4-6", 1, 0.1),
            "projects/proj-a/b.jsonl": entry("2026-01-10T10:01:00.000Z", "m2", "r2", "claude-opus-4-6", 1, 0.1),
        });
        let options = |max_source_files, max_directory_depth, max_events| DetailedUsageOptions {
            capture_permission: DetailedCapturePermission::Granted,
            claude_dirs: vec![files.root().to_path_buf()],
            codex_session_dirs: Vec::new(),
            max_source_bytes: 1024 * 1024,
            max_source_files,
            max_directory_depth,
            max_discovery_entries: 100,
            max_events,
        };

        assert!(
            detailed_usage_events(&options(1, 8, 10))
                .unwrap_err()
                .to_string()
                .contains("file limit")
        );
        assert!(
            detailed_usage_events(&options(10, 0, 10))
                .unwrap_err()
                .to_string()
                .contains("depth limit")
        );
        assert!(
            detailed_usage_events(&options(10, 8, 1))
                .unwrap_err()
                .to_string()
                .contains("event limit")
        );
    }

    #[test]
    fn detailed_manifest_bounds_wide_irrelevant_discovery() {
        let fixture = fs_fixture!({
            "projects/empty-a/.keep": "",
            "projects/empty-b/.keep": "",
            "projects/empty-c/.keep": "",
            "projects/readme.txt": "not usage",
        });
        let opts = DetailedUsageOptions {
            capture_permission: DetailedCapturePermission::Granted,
            claude_dirs: vec![fixture.root().to_path_buf()],
            codex_session_dirs: Vec::new(),
            max_source_bytes: 1024 * 1024,
            max_source_files: 10,
            max_directory_depth: 8,
            max_discovery_entries: 3,
            max_events: 10,
        };

        let error = detailed_usage_events(&opts).unwrap_err();

        assert!(error.to_string().contains("discovery entry limit"));
    }

    #[test]
    fn detailed_codex_scan_does_not_tuple_dedupe_distinct_sessions() {
        let line = codex_cumulative("2026-01-10T10:00:00.000Z", 100, 20, 10);
        let fixture = fs_fixture!({
            "session-a.jsonl": line.clone(),
            "session-b.jsonl": line,
        });

        let scan = detailed_usage_events(&DetailedUsageOptions {
            capture_permission: DetailedCapturePermission::Granted,
            claude_dirs: Vec::new(),
            codex_session_dirs: vec![fixture.root().to_path_buf()],
            max_source_bytes: 1024 * 1024,
            max_source_files: 10,
            max_directory_depth: 8,
            max_discovery_entries: 100,
            max_events: 10,
        })
        .unwrap();

        assert_eq!(scan.events.len(), 2);
        assert_eq!(scan.events[0].session_id, "session-a");
        assert_eq!(scan.events[1].session_id, "session-b");
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
    fn all_daily_returns_provider_rows_not_collapsed_all_rows() {
        let fixture = fs_fixture!({
            "projects/proj-a/12121212-1212-4212-8212-121212121212.jsonl": [
                entry("2026-01-10T10:00:00.000Z", "m1", "r1", "claude-opus-4-6", 100, 0.5),
                entry("2026-01-10T11:00:00.000Z", "m2", "r2", "claude-opus-4-6", 200, 0.25),
            ].join("\n"),
        });

        let rows = all_daily(&in_dir(fixture.root())).unwrap();

        assert!(
            rows.iter().all(|row| row.provider != "all"),
            "embedder API must expose provider-specific rows"
        );
        let claude: Vec<_> = rows.iter().filter(|row| row.provider == "claude").collect();
        assert_eq!(claude.len(), 1);
        assert_eq!(claude[0].period, "2026-01-10");
        assert_eq!(claude[0].input_tokens, 300);
        assert!((claude[0].total_cost - 0.75).abs() < f64::EPSILON);
        assert_eq!(claude[0].models[0].provider, "claude");
    }

    #[test]
    fn all_daily_can_filter_to_named_providers() {
        let fixture = fs_fixture!({
            "projects/proj-a/13131313-1313-4313-8313-131313131313.jsonl":
                entry("2026-01-10T10:00:00.000Z", "m1", "r1", "claude-opus-4-6", 100, 0.5),
        });
        let opts = UsageOptions {
            providers: Some(vec!["claude".to_string()]),
            ..in_dir(fixture.root())
        };

        let rows = all_daily(&opts).unwrap();

        assert!(!rows.is_empty());
        assert!(rows.iter().all(|row| row.provider == "claude"));
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
    fn all_sessions_return_provider_local_session_rows() {
        let session = "23232323-2323-4232-8232-232323232323";
        let fixture = fs_fixture!({
            "projects/proj-a/23232323-2323-4232-8232-232323232323.jsonl":
                entry("2026-01-10T10:00:00.000Z", "m1", "r1", "claude-opus-4-6", 100, 0.5),
        });

        let rows = all_sessions(&in_dir(fixture.root())).unwrap();
        let row = rows
            .iter()
            .find(|row| row.provider == "claude" && row.session_id == session)
            .expect("expected Claude session row from explicit fixture");

        assert_eq!(row.input_tokens, 100);
        assert!((row.total_cost - 0.5).abs() < f64::EPSILON);
        assert_eq!(row.models[0].provider, "claude");
        assert_eq!(
            row.last_activity.as_deref(),
            Some("2026-01-10T10:00:00.000Z")
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
    fn sessions_until_bound_includes_the_boundary_day() {
        let fixture = fs_fixture!({
            "projects/proj-a/55555555-5555-4555-8555-555555555555.jsonl":
                entry("2026-01-10T22:30:00.000Z", "m1", "r1", "claude-opus-4-6", 100, 0.5),
            "projects/proj-a/66666666-6666-4666-8666-666666666666.jsonl":
                entry("2026-01-11T08:00:00.000Z", "m2", "r2", "claude-opus-4-6", 10, 0.1),
        });
        let opts = UsageOptions {
            since: Some("2026-01-01".to_string()),
            until: Some("2026-01-10".to_string()),
            ..in_dir(fixture.root())
        };

        let rows = claude_sessions(&opts).unwrap();

        assert_eq!(
            rows.len(),
            1,
            "a session last active ON the until day is inside the inclusive bound"
        );
        assert_eq!(rows[0].session_id, "55555555-5555-4555-8555-555555555555");
    }

    #[test]
    fn all_sessions_until_bound_includes_the_boundary_day() {
        let fixture = fs_fixture!({
            "projects/proj-a/57575757-5757-4575-8575-575757575757.jsonl":
                entry("2026-01-10T22:30:00.000Z", "m1", "r1", "claude-opus-4-6", 100, 0.5),
            "projects/proj-a/68686868-6868-4686-8686-686868686868.jsonl":
                entry("2026-01-11T08:00:00.000Z", "m2", "r2", "claude-opus-4-6", 10, 0.1),
        });
        let opts = UsageOptions {
            since: Some("2026-01-01".to_string()),
            until: Some("2026-01-10".to_string()),
            ..in_dir(fixture.root())
        };

        let rows = all_sessions(&opts).unwrap();
        let claude_ids: Vec<_> = rows
            .iter()
            .filter(|row| row.provider == "claude")
            .map(|row| row.session_id.as_str())
            .collect();

        assert_eq!(claude_ids, vec!["57575757-5757-4575-8575-575757575757"]);
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
