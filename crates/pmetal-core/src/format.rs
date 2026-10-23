//! Text for people: sizes.

/// A file or download size in decimal units, as the Hub and Finder show them:
/// `512 B`, `48 KB`, `250 MB`, `1.5 GB`, `2.00 TB`.
pub fn human_size(bytes: u64) -> String {
    scaled(bytes, 1000.0)
}

/// A memory size in binary units, labelled the way Apple labels a Mac's memory
/// (a machine with 16 GiB of memory is a "16 GB" Mac).
pub fn human_memory(bytes: u64) -> String {
    scaled(bytes, 1024.0)
}

fn scaled(bytes: u64, base: f64) -> String {
    let value = bytes as f64;
    let (kb, mb, gb, tb) = (base, base * base, base.powi(3), base.powi(4));
    if value >= tb {
        format!("{:.2} TB", value / tb)
    } else if value >= gb {
        format!("{:.1} GB", value / gb)
    } else if value >= mb {
        format!("{:.0} MB", value / mb)
    } else if value >= kb {
        format!("{:.0} KB", value / kb)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_sizes_are_decimal() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(999), "999 B");
        assert_eq!(human_size(48_000), "48 KB");
        assert_eq!(human_size(250_000_000), "250 MB");
        assert_eq!(human_size(1_500_000_000), "1.5 GB");
        assert_eq!(human_size(2_000_000_000_000), "2.00 TB");
    }

    #[test]
    fn memory_is_binary() {
        assert_eq!(human_memory(512), "512 B");
        assert_eq!(human_memory(2048), "2 KB");
        assert_eq!(human_memory(16 << 30), "16.0 GB");
        assert_eq!(human_memory(3 << 20), "3 MB");
    }
}
