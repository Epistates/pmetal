//! Pieces every kind of tool definition shares.

use turbomcp::ToolAnnotations;

/// The behavior hints of a tool, which a client uses to decide whether to ask
/// before calling it. Every tool states all of them: a client treats a missing
/// hint as the most cautious reading (`destructive` and `openWorld` default to
/// true, `readOnly` and `idempotent` to false), which would put a confirmation
/// in front of every call.
#[derive(Debug, Clone, Copy)]
pub struct Hints {
    /// Changes nothing.
    pub read_only: bool,
    /// May overwrite or delete what exists. Only meaningful when not read-only.
    pub destructive: bool,
    /// Calling it again with the same arguments changes nothing more.
    pub idempotent: bool,
    /// Reaches beyond this machine (the Hub, say).
    pub open_world: bool,
}

impl Hints {
    /// Reads, changes nothing.
    pub const READ: Self = Self {
        read_only: true,
        destructive: false,
        idempotent: true,
        open_world: false,
    };
    /// Creates something new (a job, a file) without overwriting what exists.
    pub const CREATE: Self = Self {
        read_only: false,
        destructive: false,
        idempotent: false,
        open_world: false,
    };
    /// Writes results that replace earlier ones of the same name.
    pub const OVERWRITE: Self = Self {
        read_only: false,
        destructive: true,
        idempotent: true,
        open_world: false,
    };

    #[must_use]
    pub const fn open_world(mut self) -> Self {
        self.open_world = true;
        self
    }

    #[must_use]
    pub const fn idempotent(mut self, value: bool) -> Self {
        self.idempotent = value;
        self
    }

    #[must_use]
    pub const fn destructive(mut self) -> Self {
        self.read_only = false;
        self.destructive = true;
        self
    }

    pub fn annotations(self) -> ToolAnnotations {
        let annotations = ToolAnnotations::default()
            .with_read_only(self.read_only)
            .with_idempotent(self.idempotent)
            .with_open_world(self.open_world);
        if self.read_only {
            annotations
        } else {
            annotations.with_destructive(self.destructive)
        }
    }
}
