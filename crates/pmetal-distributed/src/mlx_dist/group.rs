//! A communication group of MLX processes.
//!
//! The group is initialized once at process startup via
//! [`DistributedGroup::init`]; sub-groups come from
//! [`DistributedGroup::split`]. It wraps [`pmetal_bridge::distributed::Group`],
//! which calls MLX's C++ distributed API directly.

use pmetal_bridge::distributed::{self, Group};

/// A communication group for distributed operations.
pub struct DistributedGroup {
    pub(crate) inner: Group,
}

impl DistributedGroup {
    /// Check if MLX was built with any distributed backend (ring, JACCL,
    /// MPI). Whether this process was launched as part of a group is up to
    /// the environment (`mlx.launch` or the `MLX_*` variables).
    pub fn is_available() -> bool {
        distributed::is_available()
    }

    /// Initialize the distributed group.
    ///
    /// When `strict` is `true`, returns `None` (and logs why) if no backend
    /// could be set up. When `false`, returns `None` if MLX has no backend;
    /// a process launched without a distributed configuration gets a group
    /// of one, whose collectives return their input.
    ///
    /// This should be called once at process startup.
    pub fn init(strict: bool) -> Option<Self> {
        if !strict && !Self::is_available() {
            return None;
        }
        match Group::init(strict) {
            Ok(inner) => Some(Self { inner }),
            Err(e) => {
                tracing::warn!("MLX distributed init failed: {e}");
                None
            }
        }
    }

    /// Get this process's rank within the group (0-indexed).
    pub fn rank(&self) -> i32 {
        self.inner.rank()
    }

    /// Get the total number of processes in the group.
    pub fn size(&self) -> i32 {
        self.inner.size()
    }

    /// Split the group into sub-groups.
    ///
    /// Processes with the same `color` end up in the same sub-group.
    /// `key` controls the rank ordering within the new group (use -1
    /// to preserve the original ordering).
    ///
    /// Returns `None` (and logs why) if MLX can't split the group.
    pub fn split(&self, color: i32, key: i32) -> Option<Self> {
        match self.inner.split(color, key) {
            Ok(inner) => Some(Self { inner }),
            Err(e) => {
                tracing::warn!("MLX distributed group split failed: {e}");
                None
            }
        }
    }
}

impl std::fmt::Debug for DistributedGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DistributedGroup(rank={}, size={})",
            self.rank(),
            self.size()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_non_strict_returns_none_when_unavailable() {
        if !DistributedGroup::is_available() {
            assert!(DistributedGroup::init(false).is_none());
        }
    }

    /// Not launched as part of a group, this process is a group of one.
    #[test]
    fn a_lone_process_is_rank_zero_of_one() {
        if let Some(group) = DistributedGroup::init(false) {
            assert_eq!((group.rank(), group.size()), (0, 1));
        }
    }
}
