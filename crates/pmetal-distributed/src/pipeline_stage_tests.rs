use super::*;
use crate::transport::in_memory_channel;

fn stage_pair(
    wire_dtype: DtypeTag,
    codec: ActivationCodec,
) -> (PipelineStageRuntime, PipelineStageRuntime) {
    let (tx, rx) = in_memory_channel(4);
    let config = |rank, is_first, is_last| PipelineStageConfig {
        rank,
        world_size: 2,
        layer_range: 0..1,
        is_first,
        is_last,
        wire_dtype,
        codec,
    };
    (
        PipelineStageRuntime::new(config(0, true, false), Some(tx), None, None, None),
        PipelineStageRuntime::new(config(1, false, true), None, Some(rx), None, None),
    )
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A stage that computes in f32 and sends fp16 must tag the message fp16:
/// the receiver validates the payload against the tag, and used to reject
/// every such message because the tag still said f32.
#[tokio::test]
async fn f32_activations_sent_as_fp16_arrive_and_widen_back() {
    let (mut first, mut second) = stage_pair(DtypeTag::Float32, ActivationCodec::Float16);
    let values = [0.5f32, -1.25, 3.0, 1024.0];
    first
        .send_to_next(&f32_bytes(&values), &[1, 4], 7, 0)
        .await
        .expect("send");
    let msg = second.recv_from_prev().await.expect("recv");
    assert_eq!(
        (msg.nonce, msg.dtype, msg.data.len()),
        (7, DtypeTag::Float16, 8)
    );
    assert_eq!(
        msg.into_dtype(DtypeTag::Float32).unwrap(),
        f32_bytes(&values)
    );
}

#[tokio::test]
async fn uncompressed_activations_keep_their_dtype() {
    let (mut first, mut second) = stage_pair(DtypeTag::Float32, ActivationCodec::None);
    let values = [0.1f32, 0.2];
    first
        .send_to_next(&f32_bytes(&values), &[2], 1, 0)
        .await
        .expect("send");
    let msg = second.recv_from_prev().await.expect("recv");
    assert_eq!(
        (msg.dtype, msg.data),
        (DtypeTag::Float32, f32_bytes(&values))
    );
}
