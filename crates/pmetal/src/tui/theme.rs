//! Centralized theme for the PMetal TUI.
//!
//! Every color and style is defined here so the interface looks like one thing.
//! The palette is written in 24-bit color and adapted to what the terminal can
//! show: 256 colors, the 16 ANSI colors, or none at all (`NO_COLOR`, a dumb
//! terminal), so it never prints garbage escape codes on a terminal that does
//! not understand them. Every text color keeps at least a 4.5:1 contrast ratio
//! against the surface it is drawn on (WCAG AA), and every border 3:1; a test
//! holds the palette to it. State is never carried by color alone: statuses
//! have glyphs and words as well.

use std::sync::LazyLock;

use ratatui::style::{Color, Modifier, Style};

/// How many colors the terminal can show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorDepth {
    /// No color: `NO_COLOR` is set or the terminal is `dumb`.
    Mono,
    /// The 16 ANSI colors.
    Ansi16,
    /// The 256-color palette.
    Ansi256,
    /// 24-bit color.
    TrueColor,
}

impl ColorDepth {
    /// What the environment says the terminal supports.
    pub fn detect() -> Self {
        Self::from_env(|name| std::env::var(name).ok())
    }

    fn from_env(var: impl Fn(&str) -> Option<String>) -> Self {
        // https://no-color.org: any non-empty value turns color off.
        if var("NO_COLOR").is_some_and(|v| !v.is_empty()) {
            return Self::Mono;
        }
        let term = var("TERM").unwrap_or_default();
        if term == "dumb" {
            return Self::Mono;
        }
        let colorterm = var("COLORTERM").unwrap_or_default().to_ascii_lowercase();
        if colorterm == "truecolor" || colorterm == "24bit" {
            return Self::TrueColor;
        }
        if term.contains("256color") || term.contains("direct") {
            return Self::Ansi256;
        }
        // Terminal.app answers TERM=xterm-256color; a bare xterm or linux
        // console has the 16.
        Self::Ansi16
    }
}

/// The palette: Apple-inspired, on slate surfaces.
pub mod palette {
    use ratatui::style::Color;

    // Primary: electric blue
    pub const PRIMARY_DIM: Color = Color::Rgb(37, 99, 235); // blue-600
    pub const PRIMARY_BRIGHT: Color = Color::Rgb(96, 165, 250); // blue-400

    // Accent: amber
    pub const ACCENT: Color = Color::Rgb(251, 191, 36); // amber-400

    pub const SUCCESS: Color = Color::Rgb(74, 222, 128); // green-400
    pub const WARNING: Color = Color::Rgb(250, 204, 21); // yellow-400
    pub const ERROR: Color = Color::Rgb(248, 113, 113); // red-400

    // Surfaces
    pub const SURFACE_0: Color = Color::Rgb(15, 23, 42); // slate-900
    pub const SURFACE_1: Color = Color::Rgb(30, 41, 59); // slate-800
    pub const SURFACE_2: Color = Color::Rgb(51, 65, 85); // slate-700

    // Text
    pub const TEXT: Color = Color::Rgb(226, 232, 240); // slate-200
    pub const TEXT_DIM: Color = Color::Rgb(160, 174, 192); // between slate-400 and 300
    pub const TEXT_MUTED: Color = Color::Rgb(138, 153, 176); // lighter than slate-500
    pub const TEXT_BRIGHT: Color = Color::Rgb(248, 250, 252); // slate-50

    // Borders
    pub const BORDER: Color = Color::Rgb(100, 116, 139); // slate-500
    pub const BORDER_FOCUS: Color = Color::Rgb(96, 165, 250); // blue-400

    // Charts
    pub const CHART_1: Color = Color::Rgb(74, 222, 128); // green-400
    pub const CHART_2: Color = Color::Rgb(96, 165, 250); // blue-400
    pub const CHART_3: Color = Color::Rgb(192, 132, 252); // purple-400
    pub const CHART_4: Color = Color::Rgb(251, 191, 36); // amber-400
    pub const CHART_5: Color = Color::Rgb(244, 114, 182); // pink-400

    // Gauges
    pub const GAUGE_LOW: Color = SUCCESS;
    pub const GAUGE_MED: Color = WARNING;
    pub const GAUGE_HIGH: Color = ERROR;

    /// The chart colors in order, for series that need `n` distinguishable ones.
    pub const CHARTS: [Color; 5] = [CHART_1, CHART_2, CHART_3, CHART_4, CHART_5];
}

/// Application-wide theme with semantic style names.
pub struct Theme {
    pub depth: ColorDepth,

    // Layout
    pub root: Style,
    pub header: Style,
    pub header_title: Style,
    pub footer: Style,
    pub footer_key: Style,
    pub footer_desc: Style,

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

/// The theme for this terminal.
pub static THEME: LazyLock<Theme> = LazyLock::new(|| Theme::for_depth(ColorDepth::detect()));

/// One of the chart colors, adapted to the terminal.
pub fn chart(index: usize) -> Color {
    adapt(palette::CHARTS[index % palette::CHARTS.len()], THEME.depth)
}

impl Theme {
    /// The theme for a terminal that shows `depth` colors.
    pub fn for_depth(depth: ColorDepth) -> Self {
        let c = |color: Color| adapt(color, depth);
        let fg = |color: Color| Style::new().fg(c(color));
        let fgbg = |f: Color, b: Color| Style::new().fg(c(f)).bg(c(b));
        let bold = Modifier::BOLD;
        let mono = depth == ColorDepth::Mono;
        // With no color, emphasis has to come from the terminal's own
        // attributes, or "selected" and "error" would look like everything else.
        let selected = if mono {
            Style::new().add_modifier(Modifier::REVERSED | bold)
        } else {
            fgbg(palette::TEXT_BRIGHT, palette::PRIMARY_DIM).add_modifier(bold)
        };
        let error = if mono {
            Style::new().add_modifier(bold | Modifier::UNDERLINED)
        } else {
            fg(palette::ERROR)
        };
        Self {
            depth,
            root: if mono {
                Style::new()
            } else {
                fgbg(palette::TEXT, palette::SURFACE_0)
            },
            header: if mono {
                Style::new().add_modifier(Modifier::REVERSED)
            } else {
                fgbg(palette::TEXT, palette::SURFACE_1)
            },
            header_title: fg(palette::PRIMARY_BRIGHT).add_modifier(bold),
            footer: if mono {
                Style::new().add_modifier(Modifier::REVERSED)
            } else {
                fgbg(palette::TEXT_DIM, palette::SURFACE_1)
            },
            footer_key: fg(palette::ACCENT).add_modifier(bold),
            footer_desc: fg(palette::TEXT_DIM),

            tab_active: if mono {
                Style::new().add_modifier(bold | Modifier::UNDERLINED)
            } else {
                fgbg(palette::PRIMARY_BRIGHT, palette::SURFACE_0).add_modifier(bold)
            },
            tab_inactive: if mono {
                Style::new().add_modifier(Modifier::REVERSED)
            } else {
                fgbg(palette::TEXT_DIM, palette::SURFACE_1)
            },

            block: fg(palette::BORDER),
            block_title: fg(palette::TEXT).add_modifier(bold),
            block_focused: fg(palette::BORDER_FOCUS),
            block_title_focused: fg(palette::PRIMARY_BRIGHT).add_modifier(bold),

            text: fg(palette::TEXT),
            text_dim: fg(palette::TEXT_DIM),
            text_muted: fg(palette::TEXT_MUTED),
            text_bright: fg(palette::TEXT_BRIGHT).add_modifier(bold),
            text_success: fg(palette::SUCCESS),
            text_warning: fg(palette::WARNING),
            text_error: error,

            table_header: fgbg(palette::TEXT_BRIGHT, palette::SURFACE_2).add_modifier(bold),
            table_row: fgbg(palette::TEXT, palette::SURFACE_0),
            table_row_alt: fgbg(palette::TEXT, palette::SURFACE_1),
            table_selected: selected,

            chart_axis: fg(palette::TEXT_DIM),

            gauge_label: fg(palette::TEXT).add_modifier(bold),

            kv_key: fg(palette::TEXT_DIM),
            kv_value: fg(palette::TEXT_BRIGHT).add_modifier(bold),
            kv_unit: fg(palette::TEXT_MUTED),

            status_running: fg(palette::PRIMARY_BRIGHT).add_modifier(bold),
            status_success: fg(palette::SUCCESS).add_modifier(bold),
            status_error: error.add_modifier(bold),
            status_idle: fg(palette::TEXT_MUTED),

            logo: fg(palette::PRIMARY_BRIGHT).add_modifier(bold),
        }
    }
}

// ---------------------------------------------------------------------------
// Color adaptation
// ---------------------------------------------------------------------------

/// `color` as a terminal showing `depth` colors can draw it.
pub fn adapt(color: Color, depth: ColorDepth) -> Color {
    let Color::Rgb(r, g, b) = color else {
        return color;
    };
    match depth {
        ColorDepth::TrueColor => color,
        ColorDepth::Ansi256 => Color::Indexed(nearest_256(r, g, b)),
        ColorDepth::Ansi16 => nearest_16(r, g, b),
        ColorDepth::Mono => Color::Reset,
    }
}

/// The 6x6x6 color cube and the 24-step gray ramp, whichever is closer.
fn nearest_256(r: u8, g: u8, b: u8) -> u8 {
    const STEPS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let nearest_step = |v: u8| {
        (0..6)
            .min_by_key(|&i| (i32::from(STEPS[i]) - i32::from(v)).abs())
            .unwrap_or(0)
    };
    let (ri, gi, bi) = (nearest_step(r), nearest_step(g), nearest_step(b));
    let cube = (
        STEPS[ri],
        STEPS[gi],
        STEPS[bi],
        16 + 36 * ri as u8 + 6 * gi as u8 + bi as u8,
    );
    let average = ((u16::from(r) + u16::from(g) + u16::from(b)) / 3) as u8;
    let gray_index = (average.saturating_sub(8) / 10).min(23);
    let gray_level = 8 + 10 * gray_index;
    let gray = (gray_level, gray_level, gray_level, 232 + gray_index);
    let distance = |(cr, cg, cb, _): (u8, u8, u8, u8)| {
        let d = |a: u8, b: u8| (i32::from(a) - i32::from(b)).pow(2);
        d(cr, r) + d(cg, g) + d(cb, b)
    };
    if distance(gray) < distance(cube) {
        gray.3
    } else {
        cube.3
    }
}

/// The nearest of the 16 ANSI colors by hue and brightness.
fn nearest_16(r: u8, g: u8, b: u8) -> Color {
    // xterm's default values, which is what most terminals approximate.
    const ANSI: [(Color, (u8, u8, u8)); 16] = [
        (Color::Black, (0, 0, 0)),
        (Color::Red, (205, 0, 0)),
        (Color::Green, (0, 205, 0)),
        (Color::Yellow, (205, 205, 0)),
        (Color::Blue, (0, 0, 238)),
        (Color::Magenta, (205, 0, 205)),
        (Color::Cyan, (0, 205, 205)),
        (Color::Gray, (229, 229, 229)),
        (Color::DarkGray, (127, 127, 127)),
        (Color::LightRed, (255, 0, 0)),
        (Color::LightGreen, (0, 255, 0)),
        (Color::LightYellow, (255, 255, 0)),
        (Color::LightBlue, (92, 92, 255)),
        (Color::LightMagenta, (255, 0, 255)),
        (Color::LightCyan, (0, 255, 255)),
        (Color::White, (255, 255, 255)),
    ];
    ANSI.iter()
        .min_by_key(|(_, (cr, cg, cb))| {
            let d = |a: u8, b: u8| (i32::from(a) - i32::from(b)).pow(2);
            d(*cr, r) + d(*cg, g) + d(*cb, b)
        })
        .map_or(Color::Reset, |(color, _)| *color)
}

// ---------------------------------------------------------------------------
// Gauges
// ---------------------------------------------------------------------------

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

/// Gauge color for a utilization ratio (0.0-1.0): green, then yellow, then red
/// as it climbs. A terminal with no color gets the default foreground; the
/// gauge's label says the number.
pub fn gauge_color(ratio: f64) -> Color {
    let color = if ratio < 0.6 {
        lerp_color(palette::GAUGE_LOW, palette::GAUGE_MED, ratio / 0.6)
    } else {
        lerp_color(palette::GAUGE_MED, palette::GAUGE_HIGH, (ratio - 0.6) / 0.4)
    };
    adapt(color, THEME.depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WCAG relative luminance of an sRGB color.
    fn luminance(color: Color) -> f64 {
        let Color::Rgb(r, g, b) = color else {
            panic!("only palette colors are checked");
        };
        let channel = |v: u8| {
            let v = f64::from(v) / 255.0;
            if v <= 0.03928 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    }

    fn contrast(a: Color, b: Color) -> f64 {
        let (la, lb) = (luminance(a), luminance(b));
        let (hi, lo) = if la > lb { (la, lb) } else { (lb, la) };
        (hi + 0.05) / (lo + 0.05)
    }

    #[test]
    fn text_colors_meet_wcag_aa_on_every_surface_they_are_drawn_on() {
        use palette::*;
        let text = [
            ("TEXT", TEXT),
            ("TEXT_DIM", TEXT_DIM),
            ("TEXT_MUTED", TEXT_MUTED),
            ("TEXT_BRIGHT", TEXT_BRIGHT),
            ("PRIMARY_BRIGHT", PRIMARY_BRIGHT),
            ("ACCENT", ACCENT),
            ("SUCCESS", SUCCESS),
            ("WARNING", WARNING),
            ("ERROR", ERROR),
        ];
        for (name, fg) in text {
            for (surface_name, bg) in [("SURFACE_0", SURFACE_0), ("SURFACE_1", SURFACE_1)] {
                let ratio = contrast(fg, bg);
                assert!(
                    ratio >= 4.5,
                    "{name} on {surface_name} is {ratio:.2}:1, below 4.5:1"
                );
            }
        }
        // Table headers and the selected row put text on their own backgrounds.
        assert!(contrast(TEXT_BRIGHT, SURFACE_2) >= 4.5);
        assert!(contrast(TEXT_BRIGHT, PRIMARY_DIM) >= 4.5);
        // Footer text sits on SURFACE_1.
        assert!(contrast(TEXT_DIM, SURFACE_1) >= 4.5);
    }

    #[test]
    fn borders_and_chart_series_meet_the_non_text_contrast_of_three_to_one() {
        use palette::*;
        for (name, color) in [("BORDER", BORDER), ("BORDER_FOCUS", BORDER_FOCUS)] {
            for bg in [SURFACE_0, SURFACE_1] {
                let ratio = contrast(color, bg);
                assert!(ratio >= 3.0, "{name} is {ratio:.2}:1");
            }
        }
        for (i, series) in CHARTS.into_iter().enumerate() {
            let ratio = contrast(series, SURFACE_0);
            assert!(ratio >= 3.0, "chart color {i} is {ratio:.2}:1");
        }
    }

    #[test]
    fn the_color_depth_follows_the_environment() {
        let depth = |pairs: &[(&str, &str)]| {
            ColorDepth::from_env(|name| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| v.to_string())
            })
        };
        assert_eq!(depth(&[("COLORTERM", "truecolor")]), ColorDepth::TrueColor);
        assert_eq!(depth(&[("COLORTERM", "24bit")]), ColorDepth::TrueColor);
        assert_eq!(depth(&[("TERM", "xterm-256color")]), ColorDepth::Ansi256);
        assert_eq!(depth(&[("TERM", "xterm")]), ColorDepth::Ansi16);
        assert_eq!(depth(&[]), ColorDepth::Ansi16);
        assert_eq!(depth(&[("TERM", "dumb")]), ColorDepth::Mono);
        // NO_COLOR wins over everything, and an empty one is ignored.
        assert_eq!(
            depth(&[("NO_COLOR", "1"), ("COLORTERM", "truecolor")]),
            ColorDepth::Mono
        );
        assert_eq!(
            depth(&[("NO_COLOR", ""), ("COLORTERM", "truecolor")]),
            ColorDepth::TrueColor
        );
    }

    #[test]
    fn a_palette_color_becomes_one_the_terminal_can_show() {
        let blue = palette::PRIMARY_BRIGHT;
        assert_eq!(adapt(blue, ColorDepth::TrueColor), blue);
        assert!(matches!(
            adapt(blue, ColorDepth::Ansi256),
            Color::Indexed(_)
        ));
        assert!(!matches!(adapt(blue, ColorDepth::Ansi16), Color::Rgb(..)));
        assert_eq!(adapt(blue, ColorDepth::Mono), Color::Reset);
        // Named colors pass through.
        assert_eq!(adapt(Color::Red, ColorDepth::Ansi16), Color::Red);
        // A pure gray lands on the gray ramp, not a tinted cube corner.
        assert_eq!(
            adapt(Color::Rgb(128, 128, 128), ColorDepth::Ansi256),
            Color::Indexed(244)
        );
        assert_eq!(
            adapt(Color::Rgb(255, 0, 0), ColorDepth::Ansi256),
            Color::Indexed(196)
        );
    }

    #[test]
    fn without_color_selection_and_errors_still_stand_out() {
        let mono = Theme::for_depth(ColorDepth::Mono);
        assert!(
            mono.table_selected
                .add_modifier
                .contains(Modifier::REVERSED)
        );
        assert!(mono.text_error.add_modifier.contains(Modifier::UNDERLINED));
        assert_eq!(mono.text.fg, Some(Color::Reset));
        // Every style the mono theme builds from the palette resets color
        // rather than naming one.
        for style in [mono.text, mono.text_dim, mono.block, mono.kv_key] {
            assert!(matches!(style.fg, None | Some(Color::Reset)), "{style:?}");
        }
    }
}
