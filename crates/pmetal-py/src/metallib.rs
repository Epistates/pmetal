//! Pointing MLX at the `mlx.metallib` built with it.
//!
//! MLX loads its Metal kernels from an `mlx.metallib`, and only the one built
//! with the MLX this module links holds the kernels it asks for. In a build
//! tree the runtime finds that file itself. A wheel outlives its build tree, so
//! it carries the file in the package, next to this module, and the package's
//! `__init__` hands the directory to [`configure_metallib`] before anything
//! touches the GPU.

use std::path::{Path, PathBuf};

use pmetal_runtime::metallib;
use pyo3::prelude::*;

/// The candidate locations for the metallib in the package directory.
fn candidates(package_dir: &Path) -> Vec<PathBuf> {
    vec![package_dir.join("mlx.metallib")]
}

/// Point MLX at the metallib for this module's MLX and return its path, or
/// None if none was found (MLX then falls back to its own search, and a GPU
/// operation fails with MLX's message).
///
/// In order: `PMETAL_METALLIB_PATH`; the user cache entry for this MLX; the
/// file the build left, while it exists; `mlx.metallib` in `package_dir`.
/// Every file is checked first, and one that is damaged or from another MLX
/// is skipped with a warning.
#[pyfunction]
pub fn configure_metallib(py: Python<'_>, package_dir: PathBuf) -> PyResult<Option<String>> {
    let found = metallib::locate(
        metallib::BUILT_SHA256,
        &candidates(&package_dir),
        |path, e| {
            let _ = PyModule::import(py, "warnings").and_then(|w| {
                w.call_method1(
                    "warn",
                    (format!(
                        "skipping unusable mlx.metallib {}: {e}",
                        path.display()
                    ),),
                )
            });
        },
    );
    Ok(found.map(|located| {
        let path = located.path.to_string_lossy().into_owned();
        pmetal_runtime::inline_array::set_metallib_path(&path);
        path
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The package directory is searched for `mlx.metallib` by that name.
    #[test]
    fn the_package_directory_is_a_candidate() {
        let dir = Path::new("/some/pkg");
        assert_eq!(candidates(dir), vec![dir.join("mlx.metallib")]);
    }
}
