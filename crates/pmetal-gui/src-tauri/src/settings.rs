//! The app's settings file.
//!
//! A small JSON file in the user's configuration directory. It holds
//! preferences only: the Hugging Face token is in the keychain
//! ([`crate::secrets`]), and a token an earlier version wrote here is moved
//! there on first load. The file is replaced atomically and readable by its
//! owner only.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::secrets::HfToken;

/// The colour scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    System,
    Light,
    #[default]
    Dark,
}

/// What a person has chosen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Settings {
    pub default_model: Option<String>,
    pub theme: Theme,
    /// Directories scanned for models besides the Hub cache.
    pub custom_model_dirs: Vec<String>,
}

/// Where the settings file is: `<config dir>/pmetal/config.json`.
pub fn default_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("pmetal")
        .join("config.json")
}

/// What loading the file found out besides the settings.
#[derive(Debug, Default, PartialEq)]
pub struct LoadReport {
    /// A token from an earlier version was moved to the keychain.
    pub token_migrated: bool,
    /// Why the token could not be moved. The file is left as it was, so the
    /// next start tries again.
    pub token_migration_error: Option<String>,
    /// The file could not be read as settings and was set aside under this
    /// name; the defaults are in use.
    pub set_aside: Option<PathBuf>,
}

/// Read the settings at `path`, moving a plain-text token to `token`.
///
/// A missing file is the defaults. A file that is not valid settings is
/// renamed out of the way (never silently overwritten) and the defaults used.
pub fn load(path: &Path, token: &HfToken) -> (Settings, LoadReport) {
    let mut report = LoadReport::default();
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return (Settings::default(), report);
        }
        Err(e) => {
            tracing::warn!("cannot read {}: {e}", path.display());
            report.set_aside = set_aside(path);
            return (Settings::default(), report);
        }
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        tracing::warn!("{} is not valid JSON", path.display());
        report.set_aside = set_aside(path);
        return (Settings::default(), report);
    };
    let settings: Settings = serde_json::from_value(value.clone()).unwrap_or_else(|e| {
        tracing::warn!(
            "{} has settings this version does not understand: {e}",
            path.display()
        );
        report.set_aside = set_aside(path);
        Settings::default()
    });

    let legacy = value
        .get("hf_token")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|t| !t.is_empty());
    if let Some(legacy) = legacy {
        match token.set(legacy) {
            Ok(()) => match save(path, &settings) {
                Ok(()) => {
                    report.token_migrated = true;
                    tracing::info!(
                        "moved the Hugging Face token from the settings file to the keychain"
                    );
                }
                Err(e) => {
                    // The token is safe in the keychain; the old copy is still
                    // in the file. Say so, and try the rewrite again next time.
                    report.token_migration_error = Some(format!(
                        "The token is in the keychain, but {} could not be rewritten without it: {e}",
                        path.display()
                    ));
                }
            },
            Err(e) => report.token_migration_error = Some(e.to_string()),
        }
    } else if value.get("hf_token").is_some() {
        // An empty or null token: drop the key.
        let _ = save(path, &settings);
    }
    (settings, report)
}

fn set_aside(path: &Path) -> Option<PathBuf> {
    let target = path.with_extension("json.unreadable");
    std::fs::rename(path, &target).ok().map(|()| target)
}

/// Write `settings` to `path`: a temporary file next to it, flushed, then
/// renamed over it, so a crash leaves the old file or the new one. Readable by
/// the owner only.
pub fn save(path: &Path, settings: &Settings) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir)?;
    let temporary = path.with_extension(format!("json.tmp{}", std::process::id()));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(serde_json::to_string_pretty(settings)?.as_bytes())?;
        file.sync_all()?;
        std::fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets::in_memory;
    use std::os::unix::fs::PermissionsExt;

    fn path(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("pmetal").join("config.json")
    }

    #[test]
    fn a_missing_file_is_the_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let (settings, report) = load(&path(&dir), &in_memory());
        assert_eq!(settings, Settings::default());
        assert_eq!(report, LoadReport::default());
    }

    #[test]
    fn settings_round_trip_and_are_private_to_the_owner() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        let settings = Settings {
            default_model: Some("Qwen/Qwen3-0.6B".into()),
            theme: Theme::Light,
            custom_model_dirs: vec!["/models".into()],
        };
        save(&p, &settings).unwrap();
        assert_eq!(
            std::fs::metadata(&p).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let (loaded, _) = load(&p, &in_memory());
        assert_eq!(loaded, settings);
        // No temporary file is left behind.
        assert_eq!(std::fs::read_dir(p.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn a_plain_text_token_moves_to_the_keychain_and_leaves_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(
            &p,
            r#"{"cache_dir":"/x","hf_token":"hf_legacyToken","default_model":"m","theme":"light","custom_model_dirs":["/a"]}"#,
        )
        .unwrap();
        let token = in_memory();

        let (settings, report) = load(&p, &token);

        assert!(report.token_migrated, "{report:?}");
        assert_eq!(
            token.get().unwrap().unwrap().expose_secret(),
            "hf_legacyToken"
        );
        let after = std::fs::read_to_string(&p).unwrap();
        assert!(!after.contains("hf_legacyToken"), "{after}");
        assert!(!after.contains("hf_token"), "{after}");
        // Everything else survives.
        assert_eq!(settings.default_model.as_deref(), Some("m"));
        assert_eq!(settings.theme, Theme::Light);
        assert_eq!(settings.custom_model_dirs, ["/a"]);
        let (again, report) = load(&p, &token);
        assert_eq!(again, settings);
        assert!(!report.token_migrated);
    }

    #[test]
    fn a_token_the_keychain_refuses_stays_in_the_file_to_be_tried_again() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, r#"{"hf_token":"hf_legacyToken"}"#).unwrap();
        let (_, report) = load(&p, &crate::secrets::unavailable());
        assert!(!report.token_migrated);
        assert!(report.token_migration_error.is_some());
        assert!(
            std::fs::read_to_string(&p)
                .unwrap()
                .contains("hf_legacyToken")
        );
    }

    #[test]
    fn a_file_that_is_not_settings_is_set_aside_not_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, "{ this is not json").unwrap();
        let (settings, report) = load(&p, &in_memory());
        assert_eq!(settings, Settings::default());
        let aside = report.set_aside.expect("the file is set aside");
        assert_eq!(
            std::fs::read_to_string(aside).unwrap(),
            "{ this is not json"
        );
        assert!(!p.exists());
    }

    #[test]
    fn settings_from_an_older_version_still_load() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(&dir);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        // No custom_model_dirs yet, an unknown theme name.
        std::fs::write(&p, r#"{"cache_dir":"/old","theme":"dark"}"#).unwrap();
        let (settings, report) = load(&p, &in_memory());
        assert_eq!(settings.theme, Theme::Dark);
        assert!(settings.custom_model_dirs.is_empty());
        assert!(report.set_aside.is_none());
    }
}
