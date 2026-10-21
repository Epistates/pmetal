use std::path::Path;

use super::*;
use crate::secrets::in_memory;

struct Fixture {
    dir: tempfile::TempDir,
    state: AppState,
}

fn sink() -> crate::hub::DownloadSink {
    Arc::new(|_| {})
}

fn fixture_with(binary: std::result::Result<PathBuf, String>) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let paths = AppPaths {
        settings: dir.path().join("config/config.json"),
        output_root: dir.path().join("out"),
        hub_cache: dir.path().join("hub"),
    };
    let state = AppState::new(paths, in_memory(), binary, Downloads::new(2, sink()));
    Fixture { dir, state }
}

fn fixture() -> Fixture {
    fixture_with(Ok(PathBuf::from("/bin/sleep")))
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

#[tokio::test]
async fn the_token_goes_to_the_keychain_and_never_to_the_file_or_the_webview() {
    let f = fixture();
    let view = f
        .state
        .set_token("  hf_secretValue123  ".into())
        .await
        .unwrap();
    assert!(view.hf_token_set);

    // What the webview gets has no token in it, in any field.
    let json = serde_json::to_string(&f.state.config().await).unwrap();
    assert!(!json.contains("hf_secretValue123"), "{json}");
    assert!(json.contains("\"hf_token_set\":true"));

    // Saving other settings writes a file without it.
    f.state
        .update_settings(SettingsPatch {
            theme: Some(Theme::Light),
            ..Default::default()
        })
        .await
        .unwrap();
    let file = std::fs::read_to_string(&f.state.paths.settings).unwrap();
    assert!(!file.contains("hf_secretValue123"), "{file}");
    assert!(!file.contains("hf_token"), "{file}");

    // The backend reads it for the Hub and for jobs.
    assert_eq!(
        f.state.token_for_hub().await.unwrap().expose_secret(),
        "hf_secretValue123"
    );
    let cleared = f.state.clear_token().await.unwrap();
    assert!(!cleared.hf_token_set);
    assert!(f.state.token_for_hub().await.is_none());
}

#[tokio::test]
async fn a_refused_token_is_not_saved() {
    let f = fixture();
    assert!(f.state.set_token("has a space".into()).await.is_err());
    assert!(!f.state.config().await.hf_token_set);
}

#[tokio::test]
async fn an_old_settings_file_with_a_token_is_migrated_at_startup() {
    let dir = tempfile::tempdir().unwrap();
    let settings = dir.path().join("config.json");
    std::fs::write(
        &settings,
        r#"{"cache_dir":"/old","hf_token":"hf_oldPlaintext","theme":"light"}"#,
    )
    .unwrap();
    let token = in_memory();
    let paths = AppPaths {
        settings: settings.clone(),
        output_root: dir.path().join("out"),
        hub_cache: dir.path().join("hub"),
    };
    let state = AppState::new(
        paths,
        token.clone(),
        Ok("/bin/sleep".into()),
        Downloads::new(1, sink()),
    );

    assert_eq!(
        token.get().unwrap().unwrap().expose_secret(),
        "hf_oldPlaintext"
    );
    assert!(
        !std::fs::read_to_string(&settings)
            .unwrap()
            .contains("hf_oldPlaintext")
    );
    let view = state.config().await;
    assert!(view.hf_token_set);
    assert_eq!(view.theme, Theme::Light);
    assert!(view.notes.is_empty());
}

#[tokio::test]
async fn a_token_that_could_not_be_moved_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let settings = dir.path().join("config.json");
    std::fs::write(&settings, r#"{"hf_token":"hf_oldPlaintext"}"#).unwrap();
    let paths = AppPaths {
        settings,
        output_root: dir.path().join("out"),
        hub_cache: dir.path().join("hub"),
    };
    let state = AppState::new(
        paths,
        crate::secrets::unavailable(),
        Ok("/bin/sleep".into()),
        Downloads::new(1, sink()),
    );
    let view = state.config().await;
    assert!(!view.hf_token_set);
    assert!(
        view.notes.iter().any(|n| n.contains("keychain")),
        "{:?}",
        view.notes
    );
}

#[tokio::test]
async fn changing_one_setting_leaves_the_others() {
    let f = fixture();
    write(&f.dir.path().join("models/m/config.json"), "{}");
    f.state
        .add_model_directory(f.dir.path().join("models").to_str().unwrap())
        .await
        .unwrap();
    f.state
        .update_settings(SettingsPatch {
            default_model: Some("Qwen/Qwen3-0.6B".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    let view = f
        .state
        .update_settings(SettingsPatch {
            theme: Some(Theme::System),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(view.default_model.as_deref(), Some("Qwen/Qwen3-0.6B"));
    assert_eq!(view.custom_model_dirs.len(), 1);
    assert_eq!(view.theme, Theme::System);

    // An empty name clears the default.
    let view = f
        .state
        .update_settings(SettingsPatch {
            default_model: Some("  ".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(view.default_model, None);
}

#[tokio::test]
async fn a_folder_added_for_models_is_checked_saved_and_scanned() {
    let f = fixture();
    write(&f.dir.path().join("models/m/config.json"), "{}");
    let models = f
        .state
        .add_model_directory(f.dir.path().join("models").to_str().unwrap())
        .await
        .unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "m");

    // It survives a restart.
    let paths = f.state.paths.clone();
    let again = AppState::new(
        paths,
        in_memory(),
        Ok("/bin/sleep".into()),
        Downloads::new(1, sink()),
    );
    assert_eq!(again.config().await.custom_model_dirs.len(), 1);

    // Refused: not there, not a folder, not a full path, a system folder.
    let file = f.dir.path().join("a-file");
    std::fs::write(&file, "x").unwrap();
    for bad in [
        "/no/such/folder",
        file.to_str().unwrap(),
        "relative",
        "/",
        "/System",
    ] {
        assert!(
            f.state.add_model_directory(bad).await.is_err(),
            "{bad} was accepted"
        );
    }

    let models = f
        .state
        .remove_model_directory(&f.state.config().await.custom_model_dirs[0])
        .await
        .unwrap();
    assert!(models.is_empty());
    assert!(f.state.config().await.custom_model_dirs.is_empty());
}

#[tokio::test]
async fn a_trained_model_can_be_deleted_but_not_the_folder_it_is_in() {
    let f = fixture();
    write(&f.dir.path().join("out/run-1/adapter_config.json"), "{}");
    write(
        &f.dir.path().join("out/run-1/adapter_model.safetensors"),
        "w",
    );
    let models = f.state.refresh_models().await;
    assert_eq!(models.len(), 1);

    f.state.delete_model("run-1").await.unwrap();

    assert!(!f.dir.path().join("out/run-1").exists());
    assert!(f.dir.path().join("out").exists());
    assert!(f.state.models().await.is_empty());
}

#[tokio::test]
async fn a_model_that_is_a_scanned_folder_itself_is_not_deleted() {
    let f = fixture();
    // The person added a folder that is itself a model.
    write(&f.dir.path().join("mine/config.json"), "{}");
    f.state
        .add_model_directory(f.dir.path().join("mine").to_str().unwrap())
        .await
        .unwrap();
    let err = f.state.delete_model("mine").await.unwrap_err();
    assert!(
        err.message().contains("not inside a folder PMetal manages"),
        "{err}"
    );
    assert!(f.dir.path().join("mine/config.json").exists());
}

#[tokio::test]
async fn a_model_that_is_not_listed_is_not_deleted() {
    let f = fixture();
    let victim = f.dir.path().join("precious");
    write(&victim.join("config.json"), "{}");
    let err = f
        .state
        .delete_model(victim.to_str().unwrap())
        .await
        .unwrap_err();
    assert!(err.message().contains("not in the model list"));
    assert!(victim.exists());
}

#[tokio::test]
async fn a_hub_model_is_removed_from_the_cache_folder_it_was_found_in() {
    let f = fixture();
    let repo = f.dir.path().join("hub/models--acme--m");
    write(&repo.join("blobs/b"), "weights");
    write(&repo.join("snapshots/abc/config.json"), "{}");
    write(&repo.join("snapshots/abc/model.safetensors"), "w");
    write(&repo.join("refs/main"), "abc");
    write(
        &f.dir.path().join("hub/models--acme--other/blobs/b"),
        "weights",
    );
    f.state.refresh_models().await;
    assert!(f.state.find_model("acme/m").await.is_some());

    f.state.delete_model("acme/m").await.unwrap();

    assert!(!repo.exists());
    assert!(f.dir.path().join("hub/models--acme--other").exists());
}

#[tokio::test]
async fn a_model_a_running_job_uses_is_not_deleted_from_under_it() {
    let f = fixture_with(Ok({
        use std::os::unix::fs::PermissionsExt;
        let script = f_script();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script
    }));
    write(&f.dir.path().join("out/run-1/adapter_config.json"), "{}");
    f.state.refresh_models().await;
    let model = f.state.find_model("run-1").await.unwrap();

    let mut values = std::collections::BTreeMap::new();
    values.insert("model".to_string(), model.reference.clone());
    values.insert("port".to_string(), "9511".to_string());
    f.state.jobs.start("serve", &values, None).unwrap();

    let err = f.state.delete_model("run-1").await.unwrap_err();
    assert!(err.message().contains("is using it"), "{err}");
    assert!(f.dir.path().join("out/run-1").exists());
    f.state.jobs.shutdown().await;
}

/// A `pmetal` that runs until stopped, in a directory that outlives the test.
fn f_script() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pmetal-gui-state-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pmetal");
    std::fs::write(&path, "#!/bin/sh\nsleep 300\n").unwrap();
    path
}

#[tokio::test]
async fn a_missing_command_line_tool_is_part_of_the_config_the_webview_reads() {
    let f = fixture_with(Err("The pmetal command-line tool was not found.".into()));
    let view = f.state.config().await;
    assert!(view.pmetal_binary.is_none());
    assert!(view.pmetal_binary_error.unwrap().contains("not found"));
}

#[tokio::test]
async fn the_config_names_the_hub_cache_it_reads() {
    let f = fixture();
    let view = f.state.config().await;
    assert_eq!(
        view.hf_cache_dir,
        f.dir.path().join("hub").display().to_string()
    );
    assert_eq!(
        view.output_dir,
        f.dir.path().join("out").display().to_string()
    );
}
