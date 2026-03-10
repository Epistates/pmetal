//! Device information tab — GPU, ANE, and memory details.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Widget, Wrap};

use crate::tui::theme::{gauge_color, THEME};
use crate::tui::widgets::KeyValueList;
use crate::tui::widgets::key_value::KvPair;

/// Cached device information (queried once at startup).
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub gpu_family: String,
    pub device_tier: String,
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_memory: u64,
    pub has_unified_memory: bool,
    pub recommended_working_set: u64,
    pub max_buffer_length: u64,
    pub has_dynamic_caching: bool,
    pub has_ray_tracing: bool,
    pub has_mesh_shaders: bool,
    pub supports_ane: bool,
    pub batch_size_multiplier: usize,
    pub tile_size: (u32, u32, u32),
}

impl DeviceInfo {
    /// Query device info from the Metal context.
    pub fn query() -> Option<Self> {
        let ctx = pmetal_metal::context::MetalContext::global().ok()?;
        let props = ctx.properties();
        Some(Self {
            name: props.name.clone(),
            gpu_family: format!("{:?}", props.gpu_family),
            device_tier: format!("{:?}", props.device_tier),
            max_threads_per_threadgroup: props.max_threads_per_threadgroup,
            max_threadgroup_memory: props.max_threadgroup_memory_length,
            has_unified_memory: props.has_unified_memory,
            recommended_working_set: props.recommended_working_set_size,
            max_buffer_length: props.max_buffer_length,
            has_dynamic_caching: props.has_dynamic_caching,
            has_ray_tracing: props.has_hardware_ray_tracing,
            has_mesh_shaders: props.has_mesh_shaders,
            supports_ane: ctx.supports_neural_engine(),
            batch_size_multiplier: props.batch_size_multiplier(),
            tile_size: props.recommended_tile_size(),
        })
    }
}

/// Memory statistics snapshot.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub total_gb: f64,
    pub used_gb: f64,
    pub peak_gb: f64,
    pub available_gb: f64,
}

impl MemorySnapshot {
    pub fn query() -> Self {
        let stats = pmetal_mlx::memory::get_memory_stats();
        Self {
            total_gb: stats.total_gb(),
            used_gb: stats.used_gb(),
            peak_gb: stats.peak_gb(),
            available_gb: stats.available_gb(),
        }
    }

    pub fn used_ratio(&self) -> f64 {
        if self.total_gb > 0.0 {
            self.used_gb / self.total_gb
        } else {
            0.0
        }
    }

    pub fn peak_ratio(&self) -> f64 {
        if self.total_gb > 0.0 {
            self.peak_gb / self.total_gb
        } else {
            0.0
        }
    }
}

/// Device tab state.
pub struct DeviceTab {
    pub device: Option<DeviceInfo>,
    pub memory: MemorySnapshot,
}

impl DeviceTab {
    pub fn new() -> Self {
        Self {
            device: DeviceInfo::query(),
            memory: MemorySnapshot::query(),
        }
    }

    /// Refresh memory stats (called on tick).
    pub fn refresh_memory(&mut self) {
        self.memory = MemorySnapshot::query();
    }
}

impl Widget for &DeviceTab {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let [top, bottom] =
            Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(area);

        let [gpu_area, features_area] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(top);

        let [memory_area, tuning_area] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(bottom);

        self.render_gpu_info(gpu_area, buf);
        self.render_features(features_area, buf);
        self.render_memory(memory_area, buf);
        self.render_tuning(tuning_area, buf);
    }
}

impl DeviceTab {
    fn render_gpu_info(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" GPU ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let Some(dev) = &self.device else {
            Paragraph::new("No Metal device found")
                .style(THEME.text_error)
                .render(inner, buf);
            return;
        };

        let pairs = [
            KvPair::new("Device", &dev.name),
            KvPair::new("GPU Family", &dev.gpu_family),
            KvPair::new("Tier", &dev.device_tier),
            KvPair::new("Unified Memory", if dev.has_unified_memory { "Yes" } else { "No" }),
            KvPair::new("Max Threads/TG", dev.max_threads_per_threadgroup.to_string()),
            KvPair::new(
                "TG Memory",
                format!("{} KB", dev.max_threadgroup_memory / 1024),
            ),
            KvPair::new(
                "Working Set",
                format!("{:.1} GB", dev.recommended_working_set as f64 / (1024.0 * 1024.0 * 1024.0)),
            ),
            KvPair::new(
                "Max Buffer",
                format!("{:.1} GB", dev.max_buffer_length as f64 / (1024.0 * 1024.0 * 1024.0)),
            ),
        ];

        KeyValueList::new(&pairs).render(inner, buf);
    }

    fn render_features(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Features ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let Some(dev) = &self.device else {
            return;
        };

        let check_style = |b: bool| {
            if b {
                THEME.text_success
            } else {
                THEME.text_muted
            }
        };

        let features = [
            ("Dynamic Caching", dev.has_dynamic_caching),
            ("Ray Tracing", dev.has_ray_tracing),
            ("Mesh Shaders", dev.has_mesh_shaders),
            ("Neural Engine", dev.supports_ane),
        ];

        for (i, (name, enabled)) in features.iter().enumerate() {
            let y = inner.y + i as u16;
            if y >= inner.y + inner.height {
                break;
            }
            let row = Rect::new(inner.x, y, inner.width, 1);
            Line::from(vec![
                Span::styled(format!("{:>18}: ", name), THEME.kv_key),
                Span::styled(
                    if *enabled { " YES " } else { " NO  " },
                    check_style(*enabled),
                ),
            ])
            .render(row, buf);
        }

        // Kernel tuning info
        let y_offset = features.len() as u16 + 1;
        if inner.y + y_offset < inner.y + inner.height {
            let row = Rect::new(inner.x, inner.y + y_offset, inner.width, 1);
            Line::from(vec![
                Span::styled("Batch Multiplier: ", THEME.kv_key),
                Span::styled(
                    format!("{}x", dev.batch_size_multiplier),
                    THEME.kv_value,
                ),
            ])
            .render(row, buf);
        }

        let y_offset = y_offset + 1;
        if inner.y + y_offset < inner.y + inner.height {
            let row = Rect::new(inner.x, inner.y + y_offset, inner.width, 1);
            let (m, n, k) = dev.tile_size;
            Line::from(vec![
                Span::styled("   Tile Size MNK: ", THEME.kv_key),
                Span::styled(format!("{m}x{n}x{k}"), THEME.kv_value),
            ])
            .render(row, buf);
        }
    }

    fn render_memory(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Memory ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let mem = &self.memory;

        // Memory gauge
        let [gauge_area, stats_area] =
            Layout::vertical([Constraint::Length(3), Constraint::Fill(1)]).areas(inner);

        // Used memory gauge
        let ratio = mem.used_ratio();
        let gauge_label = format!(
            "{:.1} / {:.1} GB ({:.0}%)",
            mem.used_gb,
            mem.total_gb,
            ratio * 100.0
        );
        Gauge::default()
            .block(Block::default().title("Used").title_style(THEME.gauge_label))
            .gauge_style(gauge_color(ratio))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(gauge_label)
            .render(gauge_area, buf);

        // Stats
        let pairs = [
            KvPair::new("Total", format!("{:.2}", mem.total_gb)).with_unit("GB"),
            KvPair::new("Used", format!("{:.2}", mem.used_gb)).with_unit("GB"),
            KvPair::new("Available", format!("{:.2}", mem.available_gb)).with_unit("GB"),
            KvPair::new("Peak", format!("{:.2}", mem.peak_gb)).with_unit("GB"),
        ];
        let stats_inner = Rect::new(stats_area.x, stats_area.y + 1, stats_area.width, stats_area.height.saturating_sub(1));
        KeyValueList::new(&pairs).render(stats_inner, buf);
    }

    fn render_tuning(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Kernel Tuning ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let Some(dev) = &self.device else {
            return;
        };

        let tier = &dev.device_tier;

        let lines = vec![
            Line::from(vec![
                Span::styled("FlashAttention (d=128): ", THEME.kv_key),
                Span::styled(
                    match tier.as_str() {
                        "Max" | "Ultra" => "64x64",
                        _ => "32x32",
                    },
                    THEME.kv_value,
                ),
            ]),
            Line::from(vec![
                Span::styled("Fused NormLora TG:      ", THEME.kv_key),
                Span::styled(
                    match tier.as_str() {
                        "Max" | "Ultra" => "256",
                        _ => "128",
                    },
                    THEME.kv_value,
                ),
            ]),
            Line::from(vec![
                Span::styled("Fused SwiGLU TG:        ", THEME.kv_key),
                Span::styled(
                    match tier.as_str() {
                        "Max" | "Ultra" => "512",
                        _ => "256",
                    },
                    THEME.kv_value,
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    "Tuning is tier-based. See docs/hardware-support.md",
                    THEME.text_muted,
                ),
            ]),
        ];

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}
