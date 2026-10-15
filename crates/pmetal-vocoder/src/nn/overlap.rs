//! Overlap-add of strided patches.
//!
//! Both a transposed convolution (each input sample scatters a kernel-long
//! patch) and an inverse STFT (each frame scatters a window-long patch) are the
//! same operation: patch `t` is added into the output at offset `t * hop`.

use pmetal_runtime::compat::{Array, ops};

/// Overlap-add `patches` (`[batch, count, length, channels]`) with stride
/// `hop`, giving `[batch, (count - 1) * hop + length, channels]`.
///
/// The patch axis is padded to a whole number of hops and split into
/// `ceil(length / hop)` blocks; block `c` of patch `t` lands in output block
/// `t + c`, so the sum is `ceil(length / hop)` shifted adds with no scatter.
pub fn overlap_add(patches: &Array, hop: i32) -> Array {
    let (batch, count, length, channels) = (
        patches.dim(0),
        patches.dim(1),
        patches.dim(2),
        patches.dim(3),
    );
    let blocks = (length + hop - 1) / hop;
    let padded = if blocks * hop != length {
        ops::pad(
            patches,
            &[(0, 0), (0, 0), (0, blocks * hop - length), (0, 0)],
            None,
            None,
        )
    } else {
        patches.clone()
    };
    let padded = padded.reshape(&[batch, count, blocks, hop, channels]);
    let mut acc: Option<Array> = None;
    for c in 0..blocks {
        let block = ops::slice_axis(&padded, 2, c, c + 1).reshape(&[batch, count, hop, channels]);
        let placed = ops::pad(
            &block,
            &[(0, 0), (c, blocks - 1 - c), (0, 0), (0, 0)],
            None,
            None,
        );
        acc = Some(match acc {
            Some(a) => a.add(&placed),
            None => placed,
        });
    }
    let full = acc.expect("a patch has at least one block").reshape(&[
        batch,
        (count + blocks - 1) * hop,
        channels,
    ]);
    ops::slice_axis(&full, 1, 0, (count - 1) * hop + length)
}
