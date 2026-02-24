#![no_main]

use libfuzzer_sys::fuzz_target;
use pmetal_gguf::reader::GgufContent;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // Attempt to parse the GGUF header/metadata and tensor info.
    // This exercises alignment validation, metadata parsing, tensor info reading,
    // and all bounds-checking paths including alignment=0 rejection and
    // integer overflow detection.
    let mut cursor = Cursor::new(data);
    if let Ok(content) = GgufContent::read(&mut cursor) {
        // If we successfully parsed the metadata, attempt to read any tensors
        // that report lengths within a reasonable limit to avoid OOM
        for name in content.tensor_names() {
            if let Some(info) = content.get_tensor_info(name) {
                // Use usize::MAX (not u64::MAX) for portability
                let byte_size = info.byte_size_checked().unwrap_or(usize::MAX);
                if byte_size < 1024 * 1024 {
                    // Only read up to 1MB tensors during fuzzing
                    let mut read_cursor = Cursor::new(data);
                    let _ = content.read_tensor_data(&mut read_cursor, name);
                }
            }
        }
    }
});
