//! System tab: the GPU, the Neural Engine and memory.
//!
//! Everything here is read from the machine: the Metal device for the chip,
//! its cores and the working set the GPU may address, the kernel's virtual
//! memory statistics for how much memory is in use. Nothing is derived from
//! the amount of RAM.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Widget, Wrap};

use crate::tui::theme::{THEME, gauge_color};
use crate::tui::widgets::KeyValueList;
use crate::tui::widgets::key_value::KvPair;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Cached device information (queried once at startup).
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub gpu_family: String,
    pub device_tier: String,
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_memory: u64,
    pub has_unified_memory: bool,
    /// What the GPU may address, as the device reports it.
    pub recommended_working_set: u64,
    pub max_buffer_length: u64,
    pub has_dynamic_caching: bool,
    pub has_ray_tracing: bool,
    pub has_mesh_shaders: bool,
    pub has_nax: bool,
    pub supports_ane: bool,
    pub batch_size_multiplier: usize,
    pub architecture_gen: u32,
    pub memory_bandwidth_gbps: f64,
    pub gpu_core_count: u32,
    pub ane_core_count: u32,
    pub is_ultra_fusion: bool,
    pub die_count: u32,
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
            has_nax: props.has_nax,
            supports_ane: ctx.supports_neural_engine(),
            batch_size_multiplier: props.batch_size_multiplier(),
            architecture_gen: props.architecture_gen,
            memory_bandwidth_gbps: props.memory_bandwidth_gbps,
            gpu_core_count: props.gpu_core_count,
            ane_core_count: props.ane_core_count,
            is_ultra_fusion: props.is_ultra_fusion,
            die_count: props.die_count,
        })
    }
}

/// System memory at one moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemorySnapshot {
    /// Installed memory, as the Metal device reports it (the GPU shares it).
    pub total_bytes: u64,
    /// Memory in use; `None` until the kernel has been asked.
    pub used_bytes: Option<u64>,
}

impl MemorySnapshot {
    /// Installed memory only; the usage arrives when it has been read.
    pub fn installed() -> Self {
        Self {
            total_bytes: pmetal_runtime::memory::get_system_memory().unwrap_or(0),
            used_bytes: None,
        }
    }

    pub fn used_ratio(&self) -> f64 {
        match (self.used_bytes, self.total_bytes) {
            (Some(used), total) if total > 0 => (used as f64 / total as f64).clamp(0.0, 1.0),
            _ => 0.0,
        }
    }

    fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / GIB
    }

    fn used_gb(&self) -> Option<f64> {
        self.used_bytes.map(|b| b as f64 / GIB)
    }
}

/// Memory in use, from the text of `vm_stat`.
///
/// Available memory is what the system can hand out without paging: free,
/// inactive and speculative pages (the file cache among them) plus purgeable
/// ones. In use is what is left of the installed memory.
pub fn memory_in_use(vm_stat: &str, installed_bytes: u64) -> Option<u64> {
    let page_size: u64 = vm_stat
        .lines()
        .next()?
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let pages = |label: &str| -> Option<u64> {
        vm_stat.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name.trim().trim_matches('"') == label)
                .then(|| value.trim().trim_end_matches('.').parse().ok())
                .flatten()
        })
    };
    let available_pages = pages("Pages free")?
        + pages("Pages inactive").unwrap_or(0)
        + pages("Pages speculative").unwrap_or(0)
        + pages("Pages purgeable").unwrap_or(0);
    Some(installed_bytes.saturating_sub(available_pages * page_size))
}

/// Ask the kernel how much memory is in use. Blocking (runs `vm_stat`).
pub fn read_memory_in_use(installed_bytes: u64) -> Option<u64> {
    let output = std::process::Command::new("vm_stat")
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    memory_in_use(&String::from_utf8_lossy(&output.stdout), installed_bytes)
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
            memory: MemorySnapshot::installed(),
        }
    }

    /// A tab for a device that is not queried (tests).
    pub fn with(device: Option<DeviceInfo>, total_bytes: u64) -> Self {
        Self {
            device,
            memory: MemorySnapshot {
                total_bytes,
                used_bytes: None,
            },
        }
    }

    /// The kernel reported how much memory is in use.
    pub fn set_used(&mut self, used_bytes: Option<u64>) {
        if used_bytes.is_some() {
            self.memory.used_bytes = used_bytes;
        }
    }
}

impl Default for DeviceTab {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for &DeviceTab {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let [top, bottom] =
            Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(area);

        let [gpu_area, features_area] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(top);

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

        let cores = |count: u32, unknown: &str| {
            if count == 0 {
                unknown.to_string()
            } else if dev.is_ultra_fusion && dev.die_count > 0 {
                format!("{count} ({}x{} dies)", count / dev.die_count, dev.die_count)
            } else {
                count.to_string()
            }
        };
        let tier_str = if dev.is_ultra_fusion {
            format!("{} (UltraFusion)", dev.device_tier)
        } else {
            dev.device_tier.clone()
        };

        let pairs = [
            KvPair::new("Device", &dev.name),
            KvPair::new(
                "GPU Family",
                format!("{} (gen {})", dev.gpu_family, dev.architecture_gen),
            ),
            KvPair::new("Tier", &tier_str),
            KvPair::new("GPU Cores", cores(dev.gpu_core_count, "Unknown")),
            KvPair::new("ANE Cores", cores(dev.ane_core_count, "N/A")),
            KvPair::new(
                "Bandwidth",
                format!("{:.0} GB/s", dev.memory_bandwidth_gbps),
            ),
            KvPair::new(
                "Working Set",
                format!("{:.1} GB", dev.recommended_working_set as f64 / GIB),
            ),
            KvPair::new(
                "Max Buffer",
                format!("{:.1} GB", dev.max_buffer_length as f64 / GIB),
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

        let features = [
            ("Dynamic Caching", dev.has_dynamic_caching),
            ("Ray Tracing", dev.has_ray_tracing),
            ("Mesh Shaders", dev.has_mesh_shaders),
            ("NAX (GPU Neural)", dev.has_nax),
            ("Neural Engine", dev.supports_ane),
            ("UltraFusion", dev.is_ultra_fusion),
        ];

        for (i, (name, enabled)) in features.iter().enumerate() {
            let y = inner.y + i as u16;
            if y >= inner.bottom() {
                break;
            }
            // Spelled out as well as colored.
            Line::from(vec![
                Span::styled(format!("{name:>18}: "), THEME.kv_key),
                Span::styled(
                    if *enabled { "yes" } else { "no" },
                    if *enabled {
                        THEME.text_success
                    } else {
                        THEME.text_muted
                    },
                ),
            ])
            .render(Rect::new(inner.x, y, inner.width, 1), buf);
        }

        let y = inner.y + features.len() as u16 + 1;
        if y < inner.bottom() {
            Line::from(vec![
                Span::styled("Batch Multiplier: ", THEME.kv_key),
                Span::styled(format!("{}x", dev.batch_size_multiplier), THEME.kv_value),
            ])
            .render(Rect::new(inner.x, y, inner.width, 1), buf);
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
        let [gauge_area, stats_area] =
            Layout::vertical([Constraint::Length(3), Constraint::Fill(1)]).areas(inner);

        let ratio = mem.used_ratio();
        let label = match mem.used_gb() {
            Some(used) => format!(
                "{used:.1} / {:.1} GB ({:.0}%)",
                mem.total_gb(),
                ratio * 100.0
            ),
            None => format!("{:.1} GB installed", mem.total_gb()),
        };
        Gauge::default()
            .block(
                Block::default()
                    .title("In use")
                    .title_style(THEME.gauge_label),
            )
            .gauge_style(gauge_color(ratio))
            .ratio(ratio)
            .label(label)
            .render(gauge_area, buf);

        let mut pairs =
            vec![KvPair::new("Installed", format!("{:.1}", mem.total_gb())).with_unit("GB")];
        if let Some(used) = mem.used_gb() {
            pairs.push(KvPair::new("In use", format!("{used:.1}")).with_unit("GB"));
            pairs.push(
                KvPair::new(
                    "Available",
                    format!("{:.1}", (mem.total_gb() - used).max(0.0)),
                )
                .with_unit("GB"),
            );
        }
        if let Some(dev) = &self.device {
            let share = if mem.total_bytes > 0 {
                format!(
                    "{:.1} ({:.0}% of installed)",
                    dev.recommended_working_set as f64 / GIB,
                    dev.recommended_working_set as f64 / mem.total_bytes as f64 * 100.0
                )
            } else {
                format!("{:.1}", dev.recommended_working_set as f64 / GIB)
            };
            pairs.push(KvPair::new("GPU may use", share).with_unit("GB"));
        }
        let stats = Rect::new(
            stats_area.x,
            stats_area.y + 1,
            stats_area.width,
            stats_area.height.saturating_sub(1),
        );
        KeyValueList::new(&pairs).render(stats, buf);
    }

    fn render_tuning(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Kernel Tuning ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        if self.device.is_none() {
            return;
        }

        let lines = vec![
            Line::from(vec![
                Span::styled("Attention, GEMM, RoPE, MLP, LoRA: ", THEME.kv_key),
                Span::styled("tensor runtime", THEME.kv_value),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Tuning is tier-based. See docs/hardware-support.md",
                THEME.text_muted,
            )),
        ];

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VM_STAT: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages free:                               20000.\n\
Pages active:                            500000.\n\
Pages inactive:                          100000.\n\
Pages speculative:                         5000.\n\
Pages throttled:                              0.\n\
Pages wired down:                        300000.\n\
Pages purgeable:                          10000.\n\
\"Translation faults\":                 123456789.\n";

    fn device() -> DeviceInfo {
        DeviceInfo {
            name: "Apple M4 Max".into(),
            gpu_family: "Apple9".into(),
            device_tier: "Max".into(),
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory: 32768,
            has_unified_memory: true,
            recommended_working_set: 100 << 30,
            max_buffer_length: 64 << 30,
            has_dynamic_caching: true,
            has_ray_tracing: true,
            has_mesh_shaders: true,
            has_nax: false,
            supports_ane: true,
            batch_size_multiplier: 4,
            architecture_gen: 16,
            memory_bandwidth_gbps: 546.0,
            gpu_core_count: 40,
            ane_core_count: 16,
            is_ultra_fusion: false,
            die_count: 1,
        }
    }

    fn draw(tab: &DeviceTab, w: u16, h: u16) -> String {
        let area = Rect::new(0, 0, w, h);
        let mut buf = Buffer::empty(area);
        tab.render(area, &mut buf);
        (0..h)
            .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn memory_in_use_is_what_is_left_after_what_the_system_could_hand_out() {
        let installed = 128u64 << 30;
        let used = memory_in_use(VM_STAT, installed).unwrap();
        let available = (20_000 + 100_000 + 5_000 + 10_000) * 16384u64;
        assert_eq!(used, installed - available);
        assert_eq!(memory_in_use("", installed), None);
        assert_eq!(memory_in_use("garbage\nPages free: 1.", installed), None);
    }

    #[test]
    fn a_machine_reporting_more_free_than_installed_is_not_negative() {
        let used = memory_in_use(VM_STAT, 1 << 20).unwrap();
        assert_eq!(used, 0);
    }

    #[test]
    fn the_tab_shows_the_devices_own_report() {
        let mut tab = DeviceTab::with(Some(device()), 128 << 30);
        let screen = draw(&tab, 120, 36);
        assert!(screen.contains("Apple M4 Max"), "{screen}");
        assert!(screen.contains("546 GB/s"), "{screen}");
        assert!(screen.contains("100.0 GB"), "the working set: {screen}");
        assert!(screen.contains("GPU may use"), "{screen}");
        assert!(screen.contains("128.0 GB installed"), "{screen}");
        assert!(!screen.contains("Used"), "usage is not known yet: {screen}");

        tab.set_used(Some(32 << 30));
        let screen = draw(&tab, 120, 36);
        assert!(screen.contains("32.0 / 128.0 GB (25%)"), "{screen}");
        assert!(screen.contains("96.0"), "available: {screen}");
        // A failed reading keeps the last good one.
        tab.set_used(None);
        assert_eq!(tab.memory.used_bytes, Some(32 << 30));
    }

    #[test]
    fn a_machine_without_a_gpu_says_so() {
        let tab = DeviceTab::with(None, 16 << 30);
        assert!(draw(&tab, 100, 30).contains("No Metal device found"));
    }

    #[test]
    fn the_tab_draws_at_any_size() {
        let tab = DeviceTab::with(Some(device()), 128 << 30);
        for &(w, h) in &[(0, 0), (1, 1), (10, 4), (40, 10), (100, 30), (200, 60)] {
            draw(&tab, w, h);
        }
    }
}
