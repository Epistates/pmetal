//! UltraFusion-aware local execution planning.
//!
//! This module models a dual-die UltraFusion Mac as a local multi-stage
//! pipeline target. It does not claim hard die affinity for GPU dispatches;
//! instead it provides:
//!
//! - per-die resource planning
//! - layer partitioning across local stages
//! - same-process in-memory transport for stage-to-stage activation flow
//!
//! This gives PMetal a real local execution primitive for Ultra hardware
//! without routing same-machine activations through TCP.

use crate::activation_codec::ActivationCodec;
use crate::activation_transport::DtypeTag;
use crate::error::{DistributedError, DistributedResult};
use crate::layer_assignment::{assign_layers_bandwidth_aware, assign_layers_proportional};
use crate::pipeline::{PipelineStageConfig, PipelineStageRuntime};
use crate::transport::in_memory_channel;
use std::ops::Range;

/// Heuristic UltraFusion interconnect bandwidth: ~32 TB/s.
pub const DEFAULT_ULTRAFUSION_INTERCONNECT_BYTES_PER_SEC: u64 = 32_000_000_000_000;

/// Default channel capacity for local stage-to-stage activation flow.
pub const DEFAULT_LOCAL_CHANNEL_CAPACITY: usize = 8;

/// Per-die hardware slice used for local execution planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UltraFusionDieProfile {
    /// Logical die identifier within the local machine.
    pub die_id: usize,
    /// Available unified memory budget for this die's working set.
    pub available_ram_bytes: u64,
    /// GPU cores attributed to this die.
    pub gpu_cores: u32,
    /// ANE cores attributed to this die.
    pub ane_cores: u32,
}

/// Local UltraFusion execution configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UltraFusionExecutionConfig {
    /// Local die slices. Ultra hardware typically exposes two.
    pub dies: Vec<UltraFusionDieProfile>,
    /// Estimated die-to-die link bandwidth.
    pub interconnect_bandwidth_bytes_per_sec: u64,
    /// Activation wire dtype for local transport.
    pub wire_dtype: DtypeTag,
    /// Activation codec for local transport.
    pub codec: ActivationCodec,
    /// Bounded queue depth for each in-memory stage link.
    pub channel_capacity: usize,
}

/// Planned local execution stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UltraFusionStagePlan {
    /// Local stage rank.
    pub rank: usize,
    /// Logical die id this stage is assigned to.
    pub die_id: usize,
    /// Decoder layers assigned to this stage.
    pub layer_range: Range<usize>,
    /// Whether this stage owns embedding / prompt ingress.
    pub is_first: bool,
    /// Whether this stage owns final norm / lm_head / result egress.
    pub is_last: bool,
    /// Available RAM slice on this die.
    pub available_ram_bytes: u64,
    /// GPU cores on this die.
    pub gpu_cores: u32,
    /// ANE cores on this die.
    pub ane_cores: u32,
}

/// Complete local UltraFusion plan for one model shape.
#[derive(Debug, Clone, PartialEq)]
pub struct UltraFusionPlan {
    /// Total decoder layer count this plan covers.
    pub num_layers: usize,
    /// Planned stages, one per die slice.
    pub stages: Vec<UltraFusionStagePlan>,
    /// Estimated inter-stage link bandwidth.
    pub interconnect_bandwidth_bytes_per_sec: u64,
    /// Heuristic per-token latency estimate in milliseconds.
    pub estimated_latency_ms: f64,
    /// Queue depth used by the local in-memory runtime.
    pub channel_capacity: usize,
}

impl UltraFusionExecutionConfig {
    /// Construct a uniform per-die configuration from machine totals.
    pub fn from_uniform_hardware(
        total_available_ram_bytes: u64,
        total_gpu_cores: u32,
        total_ane_cores: u32,
        die_count: usize,
    ) -> Self {
        let die_count = die_count.max(1);
        let base_ram = total_available_ram_bytes / die_count as u64;
        let ram_remainder = total_available_ram_bytes % die_count as u64;
        let base_gpu = total_gpu_cores / die_count as u32;
        let gpu_remainder = total_gpu_cores % die_count as u32;
        let base_ane = total_ane_cores / die_count as u32;
        let ane_remainder = total_ane_cores % die_count as u32;

        let dies = (0..die_count)
            .map(|die_id| UltraFusionDieProfile {
                die_id,
                available_ram_bytes: base_ram + u64::from(die_id == die_count - 1) * ram_remainder,
                gpu_cores: base_gpu + u32::from(die_id == die_count - 1) * gpu_remainder,
                ane_cores: base_ane + u32::from(die_id == die_count - 1) * ane_remainder,
            })
            .collect();

        Self {
            dies,
            interconnect_bandwidth_bytes_per_sec: DEFAULT_ULTRAFUSION_INTERCONNECT_BYTES_PER_SEC,
            wire_dtype: DtypeTag::Float16,
            codec: ActivationCodec::Float16,
            channel_capacity: DEFAULT_LOCAL_CHANNEL_CAPACITY,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> DistributedResult<()> {
        if self.dies.len() < 2 {
            return Err(DistributedError::Config(
                "UltraFusion local execution requires at least 2 die slices".to_string(),
            ));
        }
        if self.interconnect_bandwidth_bytes_per_sec == 0 {
            return Err(DistributedError::Config(
                "UltraFusion interconnect bandwidth must be > 0".to_string(),
            ));
        }
        if self.channel_capacity == 0 {
            return Err(DistributedError::Config(
                "UltraFusion local channel capacity must be > 0".to_string(),
            ));
        }
        if self.dies.iter().any(|die| die.available_ram_bytes == 0) {
            return Err(DistributedError::Config(
                "all UltraFusion die slices must have available RAM".to_string(),
            ));
        }
        Ok(())
    }

    /// Plan local pipeline-parallel inference across the configured die slices.
    pub fn plan_inference(&self, num_layers: usize) -> DistributedResult<UltraFusionPlan> {
        self.validate()?;

        if num_layers < self.dies.len() {
            return Err(DistributedError::Config(format!(
                "num_layers {} must be >= local stage count {}",
                num_layers,
                self.dies.len()
            )));
        }

        let available_ram: Vec<u64> = self
            .dies
            .iter()
            .map(|die| die.available_ram_bytes)
            .collect();
        let bandwidths = vec![self.interconnect_bandwidth_bytes_per_sec; self.dies.len()];
        let ranges = if bandwidths.windows(2).all(|pair| pair[0] == pair[1]) {
            assign_layers_proportional(num_layers, &available_ram)
        } else {
            assign_layers_bandwidth_aware(num_layers, &available_ram, &bandwidths)
        };

        let stages = ranges
            .into_iter()
            .zip(self.dies.iter())
            .enumerate()
            .map(|(rank, (layer_range, die))| UltraFusionStagePlan {
                rank,
                die_id: die.die_id,
                is_first: rank == 0,
                is_last: rank == self.dies.len() - 1,
                layer_range,
                available_ram_bytes: die.available_ram_bytes,
                gpu_cores: die.gpu_cores,
                ane_cores: die.ane_cores,
            })
            .collect::<Vec<_>>();

        let estimated_latency_ms = estimate_ultrafusion_latency_ms(
            &stages,
            self.interconnect_bandwidth_bytes_per_sec,
            self.wire_dtype,
        );

        Ok(UltraFusionPlan {
            num_layers,
            stages,
            interconnect_bandwidth_bytes_per_sec: self.interconnect_bandwidth_bytes_per_sec,
            estimated_latency_ms,
            channel_capacity: self.channel_capacity,
        })
    }

    /// Build local same-process pipeline runtimes from a planned stage layout.
    pub fn build_stage_runtimes(
        &self,
        plan: &UltraFusionPlan,
    ) -> DistributedResult<Vec<PipelineStageRuntime>> {
        self.validate()?;

        if plan.stages.len() != self.dies.len() {
            return Err(DistributedError::Config(format!(
                "plan stage count {} does not match die count {}",
                plan.stages.len(),
                self.dies.len()
            )));
        }

        let stage_count = plan.stages.len();
        let mut forward_senders = Vec::with_capacity(stage_count.saturating_sub(1));
        let mut forward_receivers = Vec::with_capacity(stage_count.saturating_sub(1));
        for _ in 0..stage_count.saturating_sub(1) {
            let (sender, receiver) = in_memory_channel(self.channel_capacity);
            forward_senders.push(Some(sender));
            forward_receivers.push(Some(receiver));
        }

        let (result_sender, result_receiver) = in_memory_channel(self.channel_capacity);
        let mut runtimes = Vec::with_capacity(stage_count);
        let mut result_sender = Some(result_sender);
        let mut result_receiver = Some(result_receiver);

        for (rank, stage) in plan.stages.iter().enumerate() {
            let config = PipelineStageConfig {
                rank,
                world_size: stage_count,
                layer_range: stage.layer_range.clone(),
                is_first: stage.is_first,
                is_last: stage.is_last,
                wire_dtype: self.wire_dtype,
                codec: self.codec,
            };

            let next_sender = if rank < stage_count - 1 {
                forward_senders[rank].take()
            } else {
                None
            };
            let prev_receiver = if rank > 0 {
                forward_receivers[rank - 1].take()
            } else {
                None
            };
            let runtime = PipelineStageRuntime::new(
                config,
                next_sender,
                prev_receiver,
                if stage.is_last {
                    result_sender.take()
                } else {
                    None
                },
                if stage.is_first {
                    result_receiver.take()
                } else {
                    None
                },
            );
            runtimes.push(runtime);
        }

        Ok(runtimes)
    }
}

fn estimate_ultrafusion_latency_ms(
    stages: &[UltraFusionStagePlan],
    interconnect_bandwidth_bytes_per_sec: u64,
    wire_dtype: DtypeTag,
) -> f64 {
    if stages.is_empty() {
        return 0.0;
    }

    // Heuristic compute cost: ~0.5 ms / layer / token for a mid-sized decoder.
    let max_compute_ms = stages
        .iter()
        .map(|stage| stage.layer_range.len() as f64 * 0.5)
        .fold(0.0, f64::max);

    // Heuristic activation payload: one token at hidden size 4096.
    let activation_bytes = 4096usize * wire_dtype.element_size();
    let transfer_ms = if interconnect_bandwidth_bytes_per_sec == 0 {
        0.0
    } else {
        ((stages.len().saturating_sub(1) * activation_bytes) as f64
            / interconnect_bandwidth_bytes_per_sec as f64)
            * 1000.0
    };

    max_compute_ms + transfer_ms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_hardware_splits_resources_evenly() {
        let config =
            UltraFusionExecutionConfig::from_uniform_hardware(128 * 1024 * 1024 * 1024, 80, 32, 2);

        assert_eq!(config.dies.len(), 2);
        assert_eq!(config.dies[0].available_ram_bytes, 64 * 1024 * 1024 * 1024);
        assert_eq!(config.dies[1].available_ram_bytes, 64 * 1024 * 1024 * 1024);
        assert_eq!(config.dies[0].gpu_cores, 40);
        assert_eq!(config.dies[1].gpu_cores, 40);
        assert_eq!(config.dies[0].ane_cores, 16);
        assert_eq!(config.dies[1].ane_cores, 16);
    }

    #[test]
    fn validation_rejects_single_die_configs() {
        let config =
            UltraFusionExecutionConfig::from_uniform_hardware(64 * 1024 * 1024 * 1024, 40, 16, 1);
        let err = config
            .validate()
            .expect_err("single-die config should fail");
        assert!(err.to_string().contains("at least 2 die slices"));
    }

    #[test]
    fn plan_inference_generates_contiguous_stage_ranges() {
        let config =
            UltraFusionExecutionConfig::from_uniform_hardware(128 * 1024 * 1024 * 1024, 80, 32, 2);
        let plan = config.plan_inference(80).expect("plan");

        assert_eq!(plan.stages.len(), 2);
        assert_eq!(plan.stages[0].layer_range.start, 0);
        assert_eq!(
            plan.stages[0].layer_range.end,
            plan.stages[1].layer_range.start
        );
        assert_eq!(plan.stages[1].layer_range.end, 80);
        assert!(plan.estimated_latency_ms > 0.0);
    }

    #[test]
    fn plan_inference_biases_toward_larger_die_ram() {
        let config = UltraFusionExecutionConfig {
            dies: vec![
                UltraFusionDieProfile {
                    die_id: 0,
                    available_ram_bytes: 96 * 1024 * 1024 * 1024,
                    gpu_cores: 48,
                    ane_cores: 16,
                },
                UltraFusionDieProfile {
                    die_id: 1,
                    available_ram_bytes: 32 * 1024 * 1024 * 1024,
                    gpu_cores: 32,
                    ane_cores: 16,
                },
            ],
            interconnect_bandwidth_bytes_per_sec: DEFAULT_ULTRAFUSION_INTERCONNECT_BYTES_PER_SEC,
            wire_dtype: DtypeTag::Float16,
            codec: ActivationCodec::Float16,
            channel_capacity: DEFAULT_LOCAL_CHANNEL_CAPACITY,
        };

        let plan = config.plan_inference(64).expect("plan");
        assert!(plan.stages[0].layer_range.len() > plan.stages[1].layer_range.len());
    }

    #[tokio::test]
    async fn build_stage_runtimes_wires_local_pipeline_channels() {
        let config =
            UltraFusionExecutionConfig::from_uniform_hardware(128 * 1024 * 1024 * 1024, 80, 32, 2);
        let plan = config.plan_inference(8).expect("plan");
        let mut runtimes = config.build_stage_runtimes(&plan).expect("runtimes");
        assert_eq!(runtimes.len(), 2);

        let mut first = runtimes.remove(0);
        let mut last = runtimes.remove(0);

        first
            .send_to_next(&[1, 2, 3, 4], &[1, 2], 7, 3)
            .await
            .expect("send to next");
        let msg = last.recv_from_prev().await.expect("recv from prev");
        assert_eq!(msg.nonce, 7);
        assert_eq!(msg.layer_id, 3);
        assert_eq!(msg.shape, vec![1, 2]);
        assert_eq!(msg.data, vec![1, 2, 3, 4]);

        last.send_result(&42u32.to_le_bytes(), &[1], 7)
            .await
            .expect("send result");
        let result = first.recv_result().await.expect("recv result");
        assert_eq!(result.nonce, 7);
        assert_eq!(result.shape, vec![1]);
        assert_eq!(result.data, 42u32.to_le_bytes());
    }
}
