//! Centralized theme for the PMetal TUI.
//!
//! All colors and styles are defined here to ensure visual consistency.
//! Uses Tailwind-inspired palettes via ratatui's built-in color support.

use ratatui::style::{Color, Modifier, Style};

/// PMetal brand palette — Apple-inspired with warm neutrals.
#[allow(dead_code)]
pub mod palette {
    use ratatui::style::Color;

    // Primary — electric blue
    pub const PRIMARY: Color = Color::Rgb(59, 130, 246); // blue-500
    pub const PRIMARY_DIM: Color = Color::Rgb(37, 99, 235); // blue-600
    pub const PRIMARY_BRIGHT: Color = Color::Rgb(96, 165, 250); // blue-400

    // Accent — amber/gold
    pub const ACCENT: Color = Color::Rgb(251, 191, 36); // amber-400
    pub const ACCENT_DIM: Color = Color::Rgb(217, 119, 6); // amber-600

    // Success
    pub const SUCCESS: Color = Color::Rgb(34, 197, 94); // green-500
    pub const SUCCESS_DIM: Color = Color::Rgb(22, 163, 74); // green-600
    pub const SUCCESS_BRIGHT: Color = Color::Rgb(74, 222, 128); // green-400

    // Warning
    pub const WARNING: Color = Color::Rgb(234, 179, 8); // yellow-500
    pub const WARNING_DIM: Color = Color::Rgb(202, 138, 4); // yellow-600

    // Error / destructive
    pub const ERROR: Color = Color::Rgb(239, 68, 68); // red-500
    pub const ERROR_DIM: Color = Color::Rgb(220, 38, 38); // red-600

    // Surfaces
    pub const SURFACE_0: Color = Color::Rgb(15, 23, 42); // slate-900
    pub const SURFACE_1: Color = Color::Rgb(30, 41, 59); // slate-800
    pub const SURFACE_2: Color = Color::Rgb(51, 65, 85); // slate-700
    pub const SURFACE_3: Color = Color::Rgb(71, 85, 105); // slate-600

    // Text
    pub const TEXT: Color = Color::Rgb(226, 232, 240); // slate-200
    pub const TEXT_DIM: Color = Color::Rgb(148, 163, 184); // slate-400
    pub const TEXT_MUTED: Color = Color::Rgb(100, 116, 139); // slate-500
    pub const TEXT_BRIGHT: Color = Color::Rgb(248, 250, 252); // slate-50

    // Borders
    pub const BORDER: Color = Color::Rgb(51, 65, 85); // slate-700
    pub const BORDER_FOCUS: Color = Color::Rgb(59, 130, 246); // blue-500

    // Chart-specific
    pub const CHART_1: Color = Color::Rgb(34, 197, 94); // green-500
    pub const CHART_2: Color = Color::Rgb(59, 130, 246); // blue-500
    pub const CHART_3: Color = Color::Rgb(168, 85, 247); // purple-500
    pub const CHART_4: Color = Color::Rgb(251, 191, 36); // amber-400
    pub const CHART_5: Color = Color::Rgb(236, 72, 153); // pink-500
    pub const CHART_6: Color = Color::Rgb(20, 184, 166); // teal-500

    // Gauge colors
    pub const GAUGE_LOW: Color = Color::Rgb(34, 197, 94); // green
    pub const GAUGE_MED: Color = Color::Rgb(234, 179, 8); // yellow
    pub const GAUGE_HIGH: Color = Color::Rgb(239, 68, 68); // red
}

/// Application-wide theme with semantic style names.
#[allow(dead_code)]
pub struct Theme {
    // Layout
    pub root: Style,
    pub header: Style,
    pub header_title: Style,
    pub footer: Style,
    pub footer_key: Style,
    pub footer_desc: Style,
    pub content: Style,

    // Tabs
    pub tab_active: Style,
    pub tab_inactive: Style,

    // Blocks / panels
    pub block: Style,
    pub block_title: Style,
    pub block_focused: Style,
    pub block_title_focused: Style,

    // Text
    pub text: Style,
    pub text_dim: Style,
    pub text_muted: Style,
    pub text_bright: Style,
    pub text_success: Style,
    pub text_warning: Style,
    pub text_error: Style,

    // Table
    pub table_header: Style,
    pub table_row: Style,
    pub table_row_alt: Style,
    pub table_selected: Style,

    // Charts
    pub chart_axis: Style,
    pub chart_label: Style,

    // Gauge
    pub gauge_label: Style,

    // Key-value pairs
    pub kv_key: Style,
    pub kv_value: Style,
    pub kv_unit: Style,

    // Status indicators
    pub status_running: Style,
    pub status_success: Style,
    pub status_error: Style,
    pub status_idle: Style,

    // Logo
    pub logo: Style,
}

/// Global theme singleton.
pub const THEME: Theme = Theme {
    root: Style::new().bg(palette::SURFACE_0).fg(palette::TEXT),
    header: Style::new().bg(palette::SURFACE_1).fg(palette::TEXT),
    header_title: Style::new()
        .fg(palette::PRIMARY_BRIGHT)
        .add_modifier(Modifier::BOLD),
    footer: Style::new().bg(palette::SURFACE_1).fg(palette::TEXT_DIM),
    footer_key: Style::new()
        .fg(palette::ACCENT)
        .add_modifier(Modifier::BOLD),
    footer_desc: Style::new().fg(palette::TEXT_DIM),
    content: Style::new().bg(palette::SURFACE_0).fg(palette::TEXT),

    tab_active: Style::new()
        .fg(palette::PRIMARY_BRIGHT)
        .bg(palette::SURFACE_0)
        .add_modifier(Modifier::BOLD),
    tab_inactive: Style::new().fg(palette::TEXT_DIM).bg(palette::SURFACE_1),

    block: Style::new().fg(palette::BORDER),
    block_title: Style::new().fg(palette::TEXT).add_modifier(Modifier::BOLD),
    block_focused: Style::new().fg(palette::BORDER_FOCUS),
    block_title_focused: Style::new()
        .fg(palette::PRIMARY_BRIGHT)
        .add_modifier(Modifier::BOLD),

    text: Style::new().fg(palette::TEXT),
    text_dim: Style::new().fg(palette::TEXT_DIM),
    text_muted: Style::new().fg(palette::TEXT_MUTED),
    text_bright: Style::new()
        .fg(palette::TEXT_BRIGHT)
        .add_modifier(Modifier::BOLD),
    text_success: Style::new().fg(palette::SUCCESS),
    text_warning: Style::new().fg(palette::WARNING),
    text_error: Style::new().fg(palette::ERROR),

    table_header: Style::new()
        .fg(palette::TEXT_BRIGHT)
        .bg(palette::SURFACE_2)
        .add_modifier(Modifier::BOLD),
    table_row: Style::new().fg(palette::TEXT).bg(palette::SURFACE_0),
    table_row_alt: Style::new().fg(palette::TEXT).bg(palette::SURFACE_1),
    table_selected: Style::new()
        .fg(palette::TEXT_BRIGHT)
        .bg(palette::PRIMARY_DIM)
        .add_modifier(Modifier::BOLD),

    chart_axis: Style::new().fg(palette::TEXT_DIM),
    chart_label: Style::new().fg(palette::TEXT_MUTED),

    gauge_label: Style::new().fg(palette::TEXT).add_modifier(Modifier::BOLD),

    kv_key: Style::new().fg(palette::TEXT_DIM),
    kv_value: Style::new()
        .fg(palette::TEXT_BRIGHT)
        .add_modifier(Modifier::BOLD),
    kv_unit: Style::new().fg(palette::TEXT_MUTED),

    status_running: Style::new()
        .fg(palette::PRIMARY_BRIGHT)
        .add_modifier(Modifier::BOLD),
    status_success: Style::new()
        .fg(palette::SUCCESS)
        .add_modifier(Modifier::BOLD),
    status_error: Style::new().fg(palette::ERROR).add_modifier(Modifier::BOLD),
    status_idle: Style::new().fg(palette::TEXT_MUTED),

    logo: Style::new()
        .fg(palette::PRIMARY_BRIGHT)
        .add_modifier(Modifier::BOLD),
};

/// Get a color interpolated between two colors based on ratio (0.0-1.0).
pub fn lerp_color(a: Color, b: Color, t: f64) -> Color {
    let t = t.clamp(0.0, 1.0) as f32;
    match (a, b) {
        (Color::Rgb(r1, g1, b1), Color::Rgb(r2, g2, b2)) => {
            let r = (r1 as f32 + (r2 as f32 - r1 as f32) * t) as u8;
            let g = (g1 as f32 + (g2 as f32 - g1 as f32) * t) as u8;
            let b = (b1 as f32 + (b2 as f32 - b1 as f32) * t) as u8;
            Color::Rgb(r, g, b)
        }
        _ => {
            if t < 0.5 {
                a
            } else {
                b
            }
        }
    }
}

/// Get gauge color based on utilization ratio (0.0-1.0).
/// Green → Yellow → Red as utilization increases.
pub fn gauge_color(ratio: f64) -> Color {
    if ratio < 0.6 {
        lerp_color(palette::GAUGE_LOW, palette::GAUGE_MED, ratio / 0.6)
    } else {
        lerp_color(palette::GAUGE_MED, palette::GAUGE_HIGH, (ratio - 0.6) / 0.4)
    }
}
