//! JSONL append-only writer + rolling rotation.
//!
//! RAM contract: a single event lands in two allocations (the JSON line
//! that goes to disk + the `serde_json::Value` clone that goes to the
//! broadcast hook). Rolling rotation streams through `BufReader::lines`
//! into a temp file rather than slurping the whole file into a `String`.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::broadcast::current_broadcast_hook;
use crate::config::{LogConfig, ResolvedPolicy, StoragePolicy};
use crate::event::LogEvent;
use crate::migrate;
use crate::observer_bridge;
use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde_json::Value;

struct WriterState {
    policy: ResolvedPolicy,
    write_lock: Mutex<()>,
    last_hash: Mutex<String>,
    audit_key: Vec<u8>,
}

static WRITER: OnceLock<parking_lot::RwLock<Option<Arc<WriterState>>>> = OnceLock::new();

fn slot() -> &'static parking_lot::RwLock<Option<Arc<WriterState>>> {
    WRITER.get_or_init(|| parking_lot::RwLock::new(None))
}

fn current_state() -> Option<Arc<WriterState>> {
    slot().read().clone()
}

fn read_last_hash(path: &Path) -> String {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    if !path.exists() {
        return String::new();
    }

    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return String::new(),
    };

    let reader = BufReader::new(file);
    let mut last_line = None;
    for l in reader.lines().map_while(Result::ok) {
        let trimmed = l.trim();
        if !trimmed.is_empty() {
            last_line = Some(trimmed.to_string());
        }
    }

    if let Some(line) = last_line
        && let Ok(event) = serde_json::from_str::<LogEvent>(&line)
        && let Some(hash) = event.hash
    {
        return hash;
    }

    String::new()
}

fn compute_next_hash(previous_hash: &str, event: &mut LogEvent) -> String {
    use sha2::{Digest, Sha256};

    event.hash = None;
    event.signature = None;
    let serialized = serde_json::to_string(event).unwrap_or_default();

    let mut hasher = Sha256::new();
    hasher.update(previous_hash.as_bytes());
    hasher.update(serialized.as_bytes());
    let result = hasher.finalize();

    result.iter().map(|b| format!("{b:02x}")).collect()
}

fn load_or_create_audit_key(workspace_dir: &Path) -> Result<Vec<u8>> {
    let key_path = workspace_dir.join(".audit_key");
    if key_path.exists() {
        let hex_key = fs::read_to_string(&key_path)
            .with_context(|| format!("Failed to read audit key from {}", key_path.display()))?;
        let bytes = hex_decode(hex_key.trim()).context("Audit key file is corrupt")?;
        if bytes.len() == 32 {
            return Ok(bytes);
        }
    }

    // Generate 32 bytes using Uuid v4
    let mut key = Vec::with_capacity(32);
    key.extend_from_slice(uuid::Uuid::new_v4().as_bytes());
    key.extend_from_slice(uuid::Uuid::new_v4().as_bytes());

    if let Some(parent) = key_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&key_path, hex_encode(&key))
        .with_context(|| format!("Failed to write audit key to {}", key_path.display()))?;

    // Set restrictive permissions
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&key_path, fs::Permissions::from_mode(0o600));
    }

    Ok(key)
}

pub(crate) fn hex_encode(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 2);
    for b in data {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

pub(crate) fn hex_decode(s: &str) -> Result<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        anyhow::bail!("Odd-length hex string");
    }
    let mut res = Vec::with_capacity(s.len() / 2);
    for i in (0..s.len()).step_by(2) {
        let b = u8::from_str_radix(&s[i..i + 2], 16)
            .map_err(|e| anyhow::Error::msg(format!("Invalid hex digit: {e}")))?;
        res.push(b);
    }
    Ok(res)
}

pub(crate) fn compute_signature(key: &[u8], hash: &str) -> String {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    type HmacSha256 = Hmac<Sha256>;

    let mut mac =
        HmacSha256::new_from_slice(key).unwrap_or_else(|_| HmacSha256::new(&Default::default()));
    mac.update(hash.as_bytes());
    let result = mac.finalize().into_bytes();
    hex_encode(&result)
}

/// Initialize (or disable) the persistence writer from config. Idempotent.
/// When enabled, runs a streaming in-place migration of any schema-1 rows
/// in the existing file before resuming appends.
pub fn init_from_config(config: &LogConfig, workspace_dir: &Path) {
    let policy = ResolvedPolicy::from_config(config, workspace_dir);

    if policy.storage.is_enabled()
        && policy.path.exists()
        && let Err(err) = migrate::migrate_legacy_jsonl_in_place(&policy.path)
    {
        tracing::warn!(
            target: "zeroclaw_log",
            error = ?err,
            path = %policy.path.display(),
            "log: legacy JSONL migration failed; daemon continuing with mixed-shape file"
        );
    }

    let last_hash = if policy.storage.is_enabled() && policy.path.exists() {
        read_last_hash(&policy.path)
    } else {
        String::new()
    };

    let audit_key = if policy.storage.is_enabled() {
        load_or_create_audit_key(workspace_dir).unwrap_or_default()
    } else {
        Vec::new()
    };

    let state = Arc::new(WriterState {
        policy,
        write_lock: Mutex::new(()),
        last_hash: Mutex::new(last_hash),
        audit_key,
    });
    *slot().write() = Some(state);
}

/// Public accessor for the canonical log file path. Used by the gateway's
/// `/api/logs` endpoint to know which file to stream.
pub fn runtime_trace_path() -> Option<PathBuf> {
    current_state().map(|s| s.policy.path.clone())
}

/// Emit one event. Always fans out to the broadcast hook + tracing event.
/// If persistence is enabled, also appends a JSON line to disk.
///
/// This is the function the `record!` macro expands into. Direct callers
/// (the schema migration tool, tests) can invoke it too, but production
/// code should go through the macro so the `tracing::event!` carries the
/// correct `file:line` source info.
pub fn record_event(mut event: LogEvent) {
    let Some(state) = current_state() else {
        let value = match serde_json::to_value(&event) {
            Ok(v) => v,
            Err(err) => {
                tracing::warn!(
                    target: "zeroclaw_log_internal",
                    error = ?err,
                    "log: event serialization failed"
                );
                return;
            }
        };

        observer_bridge::forward(&event);

        if let Some(hook) = current_broadcast_hook() {
            let _ = hook.send(value);
        }
        return;
    };

    let _guard = state.write_lock.lock();

    if state.policy.storage.is_enabled() {
        let mut last_hash_guard = state.last_hash.lock();
        let next_hash = compute_next_hash(&last_hash_guard, &mut event);
        event.hash = Some(next_hash.clone());
        if !state.audit_key.is_empty() {
            let signature = compute_signature(&state.audit_key, &next_hash);
            event.signature = Some(signature);
        }
        *last_hash_guard = next_hash;
    }

    let value = match serde_json::to_value(&event) {
        Ok(v) => v,
        Err(err) => {
            tracing::warn!(
                target: "zeroclaw_log_internal",
                error = ?err,
                "log: event serialization failed"
            );
            return;
        }
    };

    observer_bridge::forward(&event);

    if let Some(hook) = current_broadcast_hook() {
        let _ = hook.send(value.clone());
    }

    if !state.policy.storage.is_enabled() {
        return;
    }

    if let Err(err) = append_line_locked(&state, &value) {
        tracing::warn!(
            target: "zeroclaw_log_internal",
            error = ?err,
            path = %state.policy.path.display(),
            "log: append failed",
        );
    }
}

fn append_line_locked(state: &Arc<WriterState>, value: &Value) -> Result<()> {
    if let Some(parent) = state.policy.path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating log directory {}", parent.display()))?;
    }

    let mut options = OpenOptions::new();
    options.create(true).append(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    let file = options
        .open(&state.policy.path)
        .with_context(|| format!("opening log file {}", state.policy.path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value).context("serializing log line")?;
    writer.write_all(b"\n").context("writing newline")?;
    writer.flush().context("flushing log line")?;
    let file = writer
        .into_inner()
        .context("taking log file out of buf writer")?;
    file.sync_data().context("fsync log line")?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&state.policy.path, fs::Permissions::from_mode(0o600));
    }

    if state.policy.storage == StoragePolicy::Rolling {
        trim_to_last_entries(state)?;
    }

    Ok(())
}

/// Rolling trim. Streams the file line-by-line into a temp file, keeping
/// the last `max_entries` lines, then atomically renames. Never loads the
/// whole file into memory.
fn trim_to_last_entries(state: &Arc<WriterState>) -> Result<()> {
    // Count lines first (cheap pass).
    let total = count_nonempty_lines(&state.policy.path)?;
    if total <= state.policy.max_entries {
        return Ok(());
    }
    let skip = total - state.policy.max_entries;

    let tmp = state.policy.path.with_extension(format!(
        "tmp.{}.{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default(),
    ));

    {
        let mut opts = OpenOptions::new();
        opts.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let out_file = opts
            .open(&tmp)
            .with_context(|| format!("creating trim temp file {}", tmp.display()))?;
        let mut out = BufWriter::new(out_file);

        let in_file = fs::File::open(&state.policy.path)
            .with_context(|| format!("opening log for trim: {}", state.policy.path.display()))?;
        let reader = BufReader::new(in_file);

        let mut index: usize = 0;
        for line in reader.lines() {
            let line = line.context("reading log line during trim")?;
            if line.trim().is_empty() {
                continue;
            }
            if index >= skip {
                out.write_all(line.as_bytes())
                    .context("writing trim line")?;
                out.write_all(b"\n").context("writing trim newline")?;
            }
            index += 1;
        }
        out.flush().context("flushing trim file")?;
        out.into_inner()
            .context("taking trim file out of buf writer")?
            .sync_data()
            .context("fsync trim file")?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600));
    }
    fs::rename(&tmp, &state.policy.path).with_context(|| {
        format!(
            "renaming trim temp {} → {}",
            tmp.display(),
            state.policy.path.display()
        )
    })?;

    Ok(())
}

fn count_nonempty_lines(path: &Path) -> Result<usize> {
    let file = fs::File::open(path)
        .with_context(|| format!("opening log to count lines: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut n = 0usize;
    for line in reader.lines() {
        let line = line.context("reading log line for count")?;
        if !line.trim().is_empty() {
            n += 1;
        }
    }
    Ok(n)
}

/// Shared test-time mutex for tests that mutate the global writer state.
/// Re-exported `pub(crate)` so `macro::tests` etc. can serialize against
/// the same lock as `writer::tests`.
#[cfg(test)]
pub(crate) static WRITER_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EventCategory, Severity};

    fn install_writer(dir: &Path, max_entries: usize) {
        let cfg = LogConfig {
            log_persistence: "rolling".into(),
            log_persistence_max_entries: max_entries,
            ..LogConfig::default()
        };
        init_from_config(&cfg, dir);
    }

    #[test]
    fn append_and_rolling_keeps_only_max_entries() {
        let _guard = WRITER_TEST_LOCK.lock();
        let tmp = tempfile::tempdir().unwrap();
        install_writer(tmp.path(), 3);

        for i in 0..10 {
            let mut ev = LogEvent::new(Severity::Info, "test", EventCategory::Agent);
            ev.message = Some(format!("event-{i}"));
            record_event(ev);
        }

        let path = runtime_trace_path().unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(lines.len(), 3);
        // Last three should be 7, 8, 9 (oldest to newest order preserved).
        for (idx, &line) in lines.iter().enumerate() {
            let v: Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["message"].as_str().unwrap(), format!("event-{}", idx + 7));
        }
    }

    #[test]
    fn disabled_storage_does_not_write_file() {
        let _guard = WRITER_TEST_LOCK.lock();
        let tmp = tempfile::tempdir().unwrap();
        let cfg = LogConfig {
            log_persistence: "none".into(),
            ..LogConfig::default()
        };
        init_from_config(&cfg, tmp.path());

        let event = LogEvent::new(Severity::Info, "test", EventCategory::Agent);
        record_event(event);

        let path = runtime_trace_path().unwrap();
        assert!(
            !path.exists(),
            "no file should exist when storage is disabled"
        );
    }

    #[test]
    fn hash_chain_verification_and_integrity() {
        let _guard = WRITER_TEST_LOCK.lock();
        let tmp = tempfile::tempdir().unwrap();
        install_writer(tmp.path(), 10);

        for i in 0..5 {
            let mut ev = LogEvent::new(Severity::Info, "test", EventCategory::Agent);
            ev.message = Some(format!("event-{i}"));
            record_event(ev);
        }

        let path = runtime_trace_path().unwrap();
        *slot().write() = None;

        // The log integrity check should succeed initially
        let verified = crate::reader::verify_log_integrity(&path).unwrap();

        assert!(verified, "initial log integrity check failed");

        // Now let's tamper with the file by modifying a log message in the middle
        let contents = fs::read_to_string(&path).unwrap();
        let mut lines: Vec<String> = contents.lines().map(|s| s.to_string()).collect();
        assert_eq!(lines.len(), 5);

        // Parse line 2, modify message, and put it back
        let mut val: Value = serde_json::from_str(&lines[2]).unwrap();
        val["message"] = Value::String("tampered message".to_string());
        lines[2] = serde_json::to_string(&val).unwrap();

        let tampered_contents = lines.join("\n") + "\n";
        fs::write(&path, tampered_contents).unwrap();

        // The log integrity check should now fail!
        let verified_after_tamper = crate::reader::verify_log_integrity(&path).unwrap();
        assert!(
            !verified_after_tamper,
            "tampered log check unexpectedly succeeded"
        );
    }

    #[test]
    fn hash_chain_preserves_across_reinitialization() {
        let _guard = WRITER_TEST_LOCK.lock();
        let tmp = tempfile::tempdir().unwrap();
        install_writer(tmp.path(), 10);

        // Write 3 events
        for i in 0..3 {
            let mut ev = LogEvent::new(Severity::Info, "test", EventCategory::Agent);
            ev.message = Some(format!("event-{i}"));
            record_event(ev);
        }

        let path = runtime_trace_path().unwrap();
        let verified = crate::reader::verify_log_integrity(&path).unwrap();
        assert!(verified);

        // Reinitialize the writer pointing to the same directory (simulating restart)
        install_writer(tmp.path(), 10);

        // Write 2 more events
        for i in 3..5 {
            let mut ev = LogEvent::new(Severity::Info, "test", EventCategory::Agent);
            ev.message = Some(format!("event-{i}"));
            record_event(ev);
        }

        // Verify the entire file is still a single continuous valid hash chain!
        let verified_full = crate::reader::verify_log_integrity(&path).unwrap();
        assert!(verified_full, "hash chain broken across reinitialization");
    }

    #[test]
    fn signature_verification_and_tampering() {
        let _guard = WRITER_TEST_LOCK.lock();
        let tmp = tempfile::tempdir().unwrap();
        install_writer(tmp.path(), 10);

        // Write 3 events
        for i in 0..3 {
            let mut ev = LogEvent::new(Severity::Info, "test", EventCategory::Agent);
            ev.message = Some(format!("event-{i}"));
            record_event(ev);
        }

        let path = runtime_trace_path().unwrap();
        *slot().write() = None;

        let verified = crate::reader::verify_log_integrity(&path).unwrap();

        assert!(verified, "Initial verification failed");

        // 1. Check that audit key was created with correct permissions
        let key_path = tmp.path().join(".audit_key");
        assert!(key_path.exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(&key_path).unwrap();
            let mode = metadata.permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "Audit key file must have 0600 permissions");
        }

        // 2. Tamper: Remove signature from an event
        let contents = fs::read_to_string(&path).unwrap();
        let mut lines: Vec<String> = contents.lines().map(|s| s.to_string()).collect();
        let mut val: Value = serde_json::from_str(&lines[1]).unwrap();
        val.as_object_mut().unwrap().remove("signature");
        lines[1] = serde_json::to_string(&val).unwrap();
        fs::write(&path, lines.join("\n") + "\n").unwrap();

        let verified_no_sig = crate::reader::verify_log_integrity(&path).unwrap();
        assert!(
            !verified_no_sig,
            "Verification should fail if signature is missing"
        );

        // Restore original contents
        fs::write(&path, &contents).unwrap();

        // 3. Tamper: Modify signature value to an invalid one
        let mut lines: Vec<String> = contents.lines().map(|s| s.to_string()).collect();
        let mut val: Value = serde_json::from_str(&lines[1]).unwrap();
        val["signature"] = Value::String("a".repeat(64));
        lines[1] = serde_json::to_string(&val).unwrap();
        fs::write(&path, lines.join("\n") + "\n").unwrap();

        let verified_bad_sig = crate::reader::verify_log_integrity(&path).unwrap();
        assert!(
            !verified_bad_sig,
            "Verification should fail with invalid signature"
        );

        // Restore original contents
        fs::write(&path, &contents).unwrap();

        // 4. Tamper: Modify audit key content
        let original_key = fs::read_to_string(&key_path).unwrap();
        let tampered_key = "b".repeat(64);
        fs::write(&key_path, tampered_key).unwrap();

        let verified_bad_key = crate::reader::verify_log_integrity(&path).unwrap();
        assert!(
            !verified_bad_key,
            "Verification should fail when audit key is modified"
        );

        // Restore key
        fs::write(&key_path, original_key).unwrap();
        assert!(crate::reader::verify_log_integrity(&path).unwrap());

        // 5. Audit key missing (should verify only hashes)
        fs::remove_file(&key_path).unwrap();
        let verified_no_key = crate::reader::verify_log_integrity(&path).unwrap();
        assert!(
            verified_no_key,
            "Should fall back to hash-only check and succeed if key is missing"
        );
    }
}
