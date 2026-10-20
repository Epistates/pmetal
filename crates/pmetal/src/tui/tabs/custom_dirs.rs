//! The extra directories a person asked the Models and Datasets tabs to look in.

use std::path::{Path, PathBuf};

/// A remembered list of directories, kept in a JSON file.
#[derive(Debug, Clone)]
pub struct CustomDirs {
    file: PathBuf,
    dirs: Vec<PathBuf>,
}

impl CustomDirs {
    /// The list in `file`; empty if there is none or it cannot be read.
    pub fn load(file: PathBuf) -> Self {
        let dirs = std::fs::read_to_string(&file)
            .ok()
            .and_then(|text| serde_json::from_str::<Vec<PathBuf>>(&text).ok())
            .unwrap_or_default();
        Self { file, dirs }
    }

    /// The list for models, in the pmetal cache directory.
    pub fn models() -> Self {
        Self::load(pmetal_hub::pmetal_cache_dir().join("custom_model_dirs.json"))
    }

    /// The list for datasets, in the pmetal cache directory.
    pub fn datasets() -> Self {
        Self::load(pmetal_hub::pmetal_cache_dir().join("custom_dataset_dirs.json"))
    }

    pub fn dirs(&self) -> &[PathBuf] {
        &self.dirs
    }

    pub fn contains(&self, dir: &Path) -> bool {
        self.dirs.iter().any(|d| d == dir)
    }

    /// Add `dir` and remember it. Returns whether it was new. A list that
    /// cannot be saved still holds the directory for this session.
    pub fn add(&mut self, dir: PathBuf) -> bool {
        if self.contains(&dir) {
            return false;
        }
        self.dirs.push(dir);
        self.save();
        true
    }

    fn save(&self) {
        if let Some(parent) = self.file.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(&self.dirs) {
            let _ = std::fs::write(&self.file, json);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directories_are_remembered_between_sessions_without_repeats() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("nested").join("dirs.json");
        let mut list = CustomDirs::load(file.clone());
        assert!(list.dirs().is_empty());
        assert!(list.add(PathBuf::from("/a")));
        assert!(list.add(PathBuf::from("/b")));
        assert!(!list.add(PathBuf::from("/a")));
        let again = CustomDirs::load(file);
        assert_eq!(again.dirs(), [PathBuf::from("/a"), PathBuf::from("/b")]);
    }

    #[test]
    fn a_damaged_file_is_an_empty_list_not_a_crash() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("dirs.json");
        std::fs::write(&file, "{not json").unwrap();
        assert!(CustomDirs::load(file).dirs().is_empty());
    }
}
