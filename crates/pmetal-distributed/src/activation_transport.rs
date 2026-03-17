//! Activation transport for pipeline-parallel inference.
//!
//! Transfers intermediate hidden states between pipeline shards using
//! length-prefixed messages over the existing TCP transport layer.

use crate::error::{DistributedError, DistributedResult};
use crate::transport::{TransportReceiver, TransportSender};

/// Wire format for activation tensors sent between pipeline stages.
///
/// Layout:
/// - 8 bytes: nonce (u64 hash of request ID for routing)
/// - 4 bytes: layer_id (u32, the layer index that produced this activation)
/// - 4 bytes: ndim (u32, number of shape dimensions)
/// - ndim * 4 bytes: shape dimensions (u32 each)
/// - 1 byte: dtype tag (0=f16, 1=f32, 2=bf16)
/// - N bytes: tensor data
#[derive(Debug, Clone)]
pub struct ActivationMessage {
    /// Request nonce for routing multi-request pipelines.
    pub nonce: u64,
    /// The layer index that produced this activation.
    pub layer_id: u32,
    /// Tensor shape.
    pub shape: Vec<u32>,
    /// Data type tag.
    pub dtype: DtypeTag,
    /// Raw tensor data bytes.
    pub data: Vec<u8>,
}

/// Data type tag for wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DtypeTag {
    Float16 = 0,
    Float32 = 1,
    BFloat16 = 2,
}

impl DtypeTag {
    /// Bytes per element for this dtype.
    pub fn element_size(self) -> usize {
        match self {
            Self::Float16 | Self::BFloat16 => 2,
            Self::Float32 => 4,
        }
    }

    pub fn from_u8(v: u8) -> DistributedResult<Self> {
        match v {
            0 => Ok(Self::Float16),
            1 => Ok(Self::Float32),
            2 => Ok(Self::BFloat16),
            _ => Err(DistributedError::Protocol(format!(
                "unknown dtype tag: {v}"
            ))),
        }
    }
}

impl ActivationMessage {
    /// Total number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Serialize to wire format (without length prefix — transport layer adds its own).
    pub fn serialize(&self) -> Vec<u8> {
        let header_size = 8 + 4 + 4 + self.shape.len() * 4 + 1;
        let total = header_size + self.data.len();
        let mut buf = Vec::with_capacity(total);

        // Nonce
        buf.extend_from_slice(&self.nonce.to_le_bytes());
        // Layer ID
        buf.extend_from_slice(&self.layer_id.to_le_bytes());
        // Ndim
        let ndim = self.shape.len() as u32;
        buf.extend_from_slice(&ndim.to_le_bytes());
        // Shape
        for &d in &self.shape {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        // Dtype tag
        buf.push(self.dtype as u8);
        // Data
        buf.extend_from_slice(&self.data);

        buf
    }

    /// Deserialize from wire format (excluding the 4-byte length prefix).
    pub fn deserialize(buf: &[u8]) -> DistributedResult<Self> {
        if buf.len() < 17 {
            return Err(DistributedError::Protocol(
                "activation message too short".into(),
            ));
        }

        let mut offset = 0;

        let nonce = u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let layer_id = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
        offset += 4;

        let ndim = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if buf.len() < offset + ndim * 4 + 1 {
            return Err(DistributedError::Protocol(
                "activation message truncated at shape".into(),
            ));
        }

        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(u32::from_le_bytes(
                buf[offset..offset + 4].try_into().unwrap(),
            ));
            offset += 4;
        }

        let dtype = DtypeTag::from_u8(buf[offset])?;
        offset += 1;

        let data = buf[offset..].to_vec();

        Ok(Self {
            nonce,
            layer_id,
            shape,
            dtype,
            data,
        })
    }
}

/// Send an activation message over the transport.
pub async fn send_activation(
    sender: &mut TransportSender,
    msg: &ActivationMessage,
) -> DistributedResult<()> {
    let bytes = msg.serialize();
    sender
        .send(&bytes)
        .await
        .map_err(|e| DistributedError::Protocol(format!("send activation failed: {e}")))?;
    Ok(())
}

/// Receive an activation message from the transport.
///
/// Uses `recv_vec` which reads the length-prefixed framing and allocates
/// the buffer dynamically (activation messages have variable size).
pub async fn recv_activation(
    receiver: &mut TransportReceiver,
) -> DistributedResult<ActivationMessage> {
    let body = receiver
        .recv_vec()
        .await
        .map_err(|e| DistributedError::Protocol(format!("recv activation failed: {e}")))?;

    ActivationMessage::deserialize(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_serialization() {
        let msg = ActivationMessage {
            nonce: 0xDEAD_BEEF_CAFE_BABE,
            layer_id: 15,
            shape: vec![1, 512, 4096],
            dtype: DtypeTag::Float16,
            data: vec![0u8; 512 * 4096 * 2],
        };

        let bytes = msg.serialize();
        // No length prefix — transport layer handles framing
        let deserialized = ActivationMessage::deserialize(&bytes).unwrap();

        assert_eq!(deserialized.nonce, msg.nonce);
        assert_eq!(deserialized.layer_id, msg.layer_id);
        assert_eq!(deserialized.shape, msg.shape);
        assert_eq!(deserialized.dtype, msg.dtype);
        assert_eq!(deserialized.data.len(), msg.data.len());
    }
}
