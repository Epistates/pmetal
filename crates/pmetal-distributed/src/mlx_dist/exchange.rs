//! Neighbour exchange: send to the next rank while receiving from the
//! previous one, without deadlock.
//!
//! MLX's point-to-point `send` does not return until the neighbour has read
//! the data. If every rank sent first, every rank would block in `send` with
//! nobody reading, as soon as the message is larger than the socket buffers.
//! Even ranks therefore send before they receive and odd ranks receive before
//! they send. Rank 1 always reads first, which breaks the cycle for any ring
//! size, odd or even.
//!
//! The transfers are started asynchronously: [`start`] returns at once so the
//! caller can compute while the data moves, and [`Exchange::finish`] waits.
//! The ring backend only connects neighbours, so only the next and previous
//! rank can be reached; [`crate::expert_parallel`] builds its all-to-all on
//! top of this.

use super::group::DistributedGroup;
use super::ops;
use pmetal_runtime::compat::{Array, Dtype, Exception};

fn runtime_error(e: pmetal_runtime::RuntimeError) -> Exception {
    Exception::custom(e.to_string())
}

/// A neighbour exchange in flight.
#[must_use = "an exchange must be finished to know it succeeded"]
pub struct Exchange {
    sent: Option<Array>,
    received: Option<Array>,
}

/// Start sending `outgoing` (if any) to the next rank and receiving an array
/// of `incoming` shape and dtype (if any) from the previous rank.
///
/// Every rank of the group must call this in the same round, with `outgoing`
/// matching the next rank's `incoming` and vice versa.
pub fn start(
    group: &DistributedGroup,
    outgoing: Option<&Array>,
    incoming: Option<(&[i32], Dtype)>,
) -> Result<Exchange, Exception> {
    let (rank, size) = (group.rank(), group.size());
    let next = (rank + 1) % size;
    let prev = (rank + size - 1) % size;
    let sent = outgoing
        .map(|x| ops::send(x, next, Some(group)))
        .transpose()?;
    let received = incoming
        .map(|(shape, dtype)| ops::recv(shape, dtype, prev, Some(group)))
        .transpose()?;
    let launch = |a: &Option<Array>| -> Result<(), Exception> {
        a.as_ref()
            .map_or(Ok(()), |a| a.try_async_eval().map_err(runtime_error))
    };
    if rank % 2 == 0 {
        launch(&sent)?;
        launch(&received)?;
    } else {
        launch(&received)?;
        launch(&sent)?;
    }
    Ok(Exchange { sent, received })
}

impl Exchange {
    /// Wait for the transfers and return what was received.
    pub fn finish(self) -> Result<Option<Array>, Exception> {
        for a in [self.sent.as_ref(), self.received.as_ref()]
            .into_iter()
            .flatten()
        {
            a.try_eval().map_err(runtime_error)?;
        }
        Ok(self.received)
    }
}

/// [`start`] and [`Exchange::finish`] in one call.
pub fn exchange(
    group: &DistributedGroup,
    outgoing: Option<&Array>,
    incoming: Option<(&[i32], Dtype)>,
) -> Result<Option<Array>, Exception> {
    start(group, outgoing, incoming)?.finish()
}
