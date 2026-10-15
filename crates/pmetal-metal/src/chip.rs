#![allow(unsafe_code)]

//! What the device says about itself, and the published specifications of the
//! chips that report it.
//!
//! Metal answers some questions directly (GPU family, architecture name,
//! working-set size) and none about others (GPU core count, memory
//! bandwidth). This module asks the device for everything it can answer,
//! reads the GPU core count from the IO registry, and only then consults a
//! table of published peak bandwidths keyed by the generation, the tier and
//! the core count the device reported. Nothing here is derived from the
//! installed RAM size.

use std::ffi::{CStr, c_char, c_void};

use crate::context::DeviceTier;

/// The generation number in a Metal architecture name: `applegpu_g13g` is 13
/// (M1), `applegpu_g16s` is 16 (M4), `applegpu_g17s` is 17 (M5). The runtime
/// reads the same two digits, so the numbers agree across the workspace.
/// Returns 0 for a name that doesn't fit the pattern.
pub(crate) fn architecture_gen(architecture: &str) -> u32 {
    let Some(rest) = architecture.strip_prefix("applegpu_g") else {
        return 0;
    };
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().unwrap_or(0)
}

/// The variant letter that ends an architecture name (`s` in `applegpu_g16s`).
pub(crate) fn architecture_variant(architecture: &str) -> Option<char> {
    architecture
        .chars()
        .last()
        .filter(char::is_ascii_alphabetic)
}

/// Whether the GPU has neural accelerators the Metal 4 tensor operations can
/// use: generation 17 and later (18 for the phone-class `p` variants), on
/// macOS 26.2 or later. This is the gate the runtime applies to its own
/// NAX kernels, so both stacks agree on which device runs which path.
pub(crate) fn has_neural_accelerators(architecture: &str, os_at_least_26_2: bool) -> bool {
    let needed = if architecture_variant(architecture) == Some('p') {
        18
    } else {
        17
    };
    os_at_least_26_2 && architecture_gen(architecture) >= needed
}

/// The tier of a chip from its marketing name ("Apple M4 Max"). Metal exposes
/// no tier, so the name is the only source.
pub(crate) fn tier_from_name(name: &str) -> DeviceTier {
    if name.contains("Ultra") {
        DeviceTier::Ultra
    } else if name.contains("Max") {
        DeviceTier::Max
    } else if name.contains("Pro") {
        DeviceTier::Pro
    } else {
        DeviceTier::Base
    }
}

/// GPU core count of the full (unbinned) chip. Only a fallback for when the
/// IO registry won't say; binned parts have fewer.
pub(crate) fn full_bin_gpu_cores(arch_gen: u32, tier: DeviceTier) -> u32 {
    use DeviceTier::{Base, Max, Pro, Ultra};
    match (arch_gen, tier) {
        (16.., Ultra) => 80,
        (16.., Max) => 40,
        (16.., Pro) => 20,
        (16.., Base) => 10,
        (15, Ultra) => 80,
        (15, Max) => 40,
        (15, Pro) => 18,
        (15, Base) => 10,
        (14, Ultra) => 76,
        (14, Max) => 38,
        (14, Pro) => 19,
        (14, Base) => 10,
        (13, Ultra) => 64,
        (13, Max) => 32,
        (13, Pro) => 16,
        (13, Base) => 8,
        _ => 0,
    }
}

/// Neural Engine core count: 16 on every chip, 32 on the dual-die Ultra.
pub(crate) fn ane_cores(tier: DeviceTier) -> u32 {
    match tier {
        DeviceTier::Ultra => 32,
        _ => 16,
    }
}

/// Published peak unified-memory bandwidth in GB/s for a chip, from Apple's
/// specifications. `gpu_cores` selects between binned and full parts where
/// the memory bus differs (Max chips from the M3 generation on). A chip the
/// table doesn't know gets `None`.
pub(crate) fn spec_bandwidth_gbps(arch_gen: u32, tier: DeviceTier, gpu_cores: u32) -> Option<f64> {
    use DeviceTier::{Base, Max, Pro, Ultra};
    // `gpu_cores == 0` means unknown: assume the full part.
    let binned = |full: u32| gpu_cores != 0 && gpu_cores < full;
    let gbps = match (arch_gen, tier) {
        // M1
        (13, Base) => 68.25,
        (13, Pro) => 200.0,
        (13, Max) => 400.0,
        (13, Ultra) => 800.0,
        // M2
        (14, Base) => 102.4,
        (14, Pro) => 200.0,
        (14, Max) => 400.0,
        (14, Ultra) => 800.0,
        // M3: the 30-core Max has a narrower bus; the Pro's is 192 bits.
        (15, Base) => 102.4,
        (15, Pro) => 150.0,
        (15, Max) => {
            if binned(40) {
                300.0
            } else {
                400.0
            }
        }
        (15, Ultra) => 819.2,
        // M4: the 32-core Max has a narrower bus.
        (16, Base) => 120.0,
        (16, Pro) => 273.0,
        (16, Max) => {
            if binned(40) {
                409.6
            } else {
                546.0
            }
        }
        // M5: the 32-core Max has a narrower bus.
        (17, Base) => 153.6,
        (17, Pro) => 307.0,
        (17, Max) => {
            if binned(40) {
                460.0
            } else {
                614.0
            }
        }
        // An Ultra is two Max dies.
        (16 | 17, Ultra) => 2.0 * spec_bandwidth_gbps(arch_gen, Max, gpu_cores.div_ceil(2))?,
        _ => return None,
    };
    Some(gbps)
}

/// A bandwidth to report for a chip no table knows, by tier alone.
pub(crate) fn tier_bandwidth_gbps(tier: DeviceTier) -> f64 {
    match tier {
        DeviceTier::Base => 100.0,
        DeviceTier::Pro => 200.0,
        DeviceTier::Max => 400.0,
        DeviceTier::Ultra => 800.0,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IO registry
// ─────────────────────────────────────────────────────────────────────────────

type CFTypeRef = *const c_void;

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(name: *const c_char) -> *mut c_void;
    fn IOServiceGetMatchingService(main_port: u32, matching: *mut c_void) -> u32;
    fn IORegistryEntryCreateCFProperty(
        entry: u32,
        key: CFTypeRef,
        allocator: CFTypeRef,
        options: u32,
    ) -> CFTypeRef;
    fn IOObjectRelease(object: u32) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(
        allocator: CFTypeRef,
        c_str: *const c_char,
        encoding: u32,
    ) -> CFTypeRef;
    fn CFNumberGetValue(number: CFTypeRef, the_type: isize, value_ptr: *mut c_void) -> u8;
    fn CFRelease(cf: CFTypeRef);
}

const CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
const CF_NUMBER_SINT32_TYPE: isize = 3;

/// The number of GPU cores the machine has, from the `gpu-core-count`
/// property of the GPU's IO registry entry. Metal itself reports no core
/// count, and chips are binned (an M4 Max has 32 or 40 cores), so the chip
/// name alone can't say. `None` if the registry has no such property.
pub(crate) fn registry_gpu_core_count() -> Option<u32> {
    const SERVICE: &CStr = c"AGXAccelerator";
    const KEY: &CStr = c"gpu-core-count";
    // SAFETY: each call below follows its documented contract. `matching` is
    // consumed by `IOServiceGetMatchingService` (it releases the dictionary
    // whether or not it finds a service). The property and the key are
    // released once, and `entry` is released once if it is non-zero.
    unsafe {
        let matching = IOServiceMatching(SERVICE.as_ptr());
        if matching.is_null() {
            return None;
        }
        // 0 is `kIOMainPortDefault`.
        let entry = IOServiceGetMatchingService(0, matching);
        if entry == 0 {
            return None;
        }
        let key =
            CFStringCreateWithCString(std::ptr::null(), KEY.as_ptr(), CF_STRING_ENCODING_UTF8);
        let mut count = None;
        if !key.is_null() {
            let property = IORegistryEntryCreateCFProperty(entry, key, std::ptr::null(), 0);
            if !property.is_null() {
                let mut value: i32 = 0;
                if CFNumberGetValue(
                    property,
                    CF_NUMBER_SINT32_TYPE,
                    (&mut value as *mut i32).cast(),
                ) != 0
                    && value > 0
                {
                    count = Some(value as u32);
                }
                CFRelease(property);
            }
            CFRelease(key);
        }
        IOObjectRelease(entry);
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use DeviceTier::{Base, Max, Pro, Ultra};

    #[test]
    fn architecture_names_parse_to_their_generation() {
        assert_eq!(architecture_gen("applegpu_g13g"), 13);
        assert_eq!(architecture_gen("applegpu_g14s"), 14);
        assert_eq!(architecture_gen("applegpu_g15d"), 15);
        assert_eq!(architecture_gen("applegpu_g16s"), 16);
        assert_eq!(architecture_gen("applegpu_g17p"), 17);
        assert_eq!(architecture_gen("amd_vega"), 0);
        assert_eq!(architecture_gen(""), 0);
        assert_eq!(architecture_gen("applegpu_g"), 0);
    }

    #[test]
    fn neural_accelerators_follow_the_runtime_gate() {
        assert!(has_neural_accelerators("applegpu_g17s", true));
        assert!(has_neural_accelerators("applegpu_g17c", true));
        assert!(!has_neural_accelerators("applegpu_g17s", false));
        // The phone-class variant needs the next generation.
        assert!(!has_neural_accelerators("applegpu_g17p", true));
        assert!(has_neural_accelerators("applegpu_g18p", true));
        assert!(!has_neural_accelerators("applegpu_g16s", true));
        assert!(!has_neural_accelerators("", true));
    }

    #[test]
    fn tiers_come_from_the_marketing_name() {
        assert_eq!(tier_from_name("Apple M3 Ultra"), Ultra);
        assert_eq!(tier_from_name("Apple M5 Max"), Max);
        assert_eq!(tier_from_name("Apple M4 Pro"), Pro);
        assert_eq!(tier_from_name("Apple M2"), Base);
        assert_eq!(tier_from_name("Apple Paravirtual device"), Base);
    }

    /// Apple's published peak memory bandwidth, chip by chip (GB/s).
    #[test]
    fn spec_bandwidth_matches_the_published_figure_for_every_released_chip() {
        let published: &[(&str, u32, DeviceTier, u32, f64)] = &[
            ("M1", 13, Base, 8, 68.25),
            ("M1 Pro", 13, Pro, 16, 200.0),
            ("M1 Max 24-core", 13, Max, 24, 400.0),
            ("M1 Max", 13, Max, 32, 400.0),
            ("M1 Ultra", 13, Ultra, 64, 800.0),
            ("M2", 14, Base, 10, 102.4),
            ("M2 Pro", 14, Pro, 19, 200.0),
            ("M2 Max", 14, Max, 38, 400.0),
            ("M2 Ultra", 14, Ultra, 76, 800.0),
            ("M3", 15, Base, 10, 102.4),
            ("M3 Pro", 15, Pro, 18, 150.0),
            ("M3 Max 30-core", 15, Max, 30, 300.0),
            ("M3 Max", 15, Max, 40, 400.0),
            ("M3 Ultra", 15, Ultra, 80, 819.2),
            ("M4", 16, Base, 10, 120.0),
            ("M4 Pro", 16, Pro, 20, 273.0),
            ("M4 Max 32-core", 16, Max, 32, 409.6),
            ("M4 Max", 16, Max, 40, 546.0),
            ("M5", 17, Base, 10, 153.6),
            ("M5 Pro", 17, Pro, 20, 307.0),
            ("M5 Max 32-core", 17, Max, 32, 460.0),
            ("M5 Max", 17, Max, 40, 614.0),
        ];
        for &(chip, arch_gen, tier, cores, gbps) in published {
            assert_eq!(
                spec_bandwidth_gbps(arch_gen, tier, cores),
                Some(gbps),
                "{chip}"
            );
        }
    }

    #[test]
    fn an_unknown_core_count_means_the_full_part() {
        assert_eq!(spec_bandwidth_gbps(16, Max, 0), Some(546.0));
        assert_eq!(spec_bandwidth_gbps(17, Max, 0), Some(614.0));
    }

    #[test]
    fn an_unreleased_ultra_is_two_max_dies() {
        assert_eq!(spec_bandwidth_gbps(17, Ultra, 80), Some(1228.0));
        assert_eq!(spec_bandwidth_gbps(16, Ultra, 64), Some(2.0 * 409.6));
    }

    #[test]
    fn an_unknown_generation_has_no_spec() {
        assert_eq!(spec_bandwidth_gbps(0, Max, 40), None);
        assert_eq!(spec_bandwidth_gbps(99, Base, 10), None);
    }

    #[test]
    fn full_bin_core_counts() {
        assert_eq!(full_bin_gpu_cores(16, Max), 40);
        assert_eq!(full_bin_gpu_cores(15, Pro), 18);
        assert_eq!(full_bin_gpu_cores(14, Ultra), 76);
        assert_eq!(full_bin_gpu_cores(13, Base), 8);
        assert_eq!(full_bin_gpu_cores(0, Base), 0);
    }

    /// The registry reports a plausible core count for the GPU in this machine.
    #[test]
    fn the_io_registry_reports_this_machines_gpu_cores() {
        let cores = registry_gpu_core_count().expect("an AGXAccelerator with a core count");
        assert!((7..=80).contains(&cores), "{cores} GPU cores");
    }
}
