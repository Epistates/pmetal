//! Safe Rust wrappers for the MLX distributed C API.
//!
//! Provides access to MLX's native collective communication backends:
//! - **JACCL**: RDMA over Thunderbolt 5 (~3µs latency, 45 GB/s)
//! - **Ring**: TCP-based ring communication
//!
//! These wrap [`pmetal_bridge::distributed`], which calls MLX's C++
//! distributed API (`mlx/distributed/`) in the libmlx the bridge links.
//!
//! # Environment Variables
//!
//! MLX distributed backends are configured via environment variables
//! (set before process launch via `mlx.launch` or manually):
//!
//! - `MLX_RANK` — this process's rank
//! - `MLX_WORLD_SIZE` — total number of processes
//! - `MLX_IBV_DEVICES` — JACCL device file (Thunderbolt device names)
//! - `MLX_JACCL_COORDINATOR` — coordinator IP:port for JACCL handshake
//!
//! # Example
//!
//! ```ignore
//! use pmetal_distributed::mlx_dist::{DistributedGroup, ops};
//!
//! if DistributedGroup::is_available() {
//!     let group = DistributedGroup::init(false).expect("distributed init");
//!     println!("rank {}/{}", group.rank(), group.size());
//!
//!     let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
//!     let sum = ops::all_sum(&x, Some(&group)).unwrap();
//! }
//! ```

pub mod backend;
pub mod group;
pub mod ops;

pub use backend::MlxDistributedBackend;
pub use group::DistributedGroup;
