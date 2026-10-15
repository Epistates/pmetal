//! Builds PyTorch checkpoints for tests: the ZIP container, and the pickle
//! `torch.save` writes for a state dict. Public only with the `testing`
//! feature, which this crate's own integration tests turn on.

/// Build a stored ZIP archive of `entries`, optionally with ZIP64 records
/// (extended sizes and offsets in the central directory, a ZIP64 end record
/// and locator), the way PyTorch writes archives past 4 GiB.
pub fn stored_zip(entries: &[(&str, &[u8])], zip64: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let offset = out.len() as u64;
        out.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
        out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        out.extend_from_slice(&0u32.to_le_bytes()); // crc (unchecked)
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);

        central.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]);
        central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        central.extend_from_slice(&0u32.to_le_bytes());
        let extra: Vec<u8> = if zip64 {
            let mut e = vec![0x01, 0x00, 24, 0];
            e.extend_from_slice(&(data.len() as u64).to_le_bytes());
            e.extend_from_slice(&(data.len() as u64).to_le_bytes());
            e.extend_from_slice(&offset.to_le_bytes());
            e
        } else {
            Vec::new()
        };
        let (size32, off32) = if zip64 {
            (0xFFFF_FFFFu32, 0xFFFF_FFFFu32)
        } else {
            (data.len() as u32, offset as u32)
        };
        central.extend_from_slice(&size32.to_le_bytes());
        central.extend_from_slice(&size32.to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&(extra.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0; 10]);
        central.extend_from_slice(&off32.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
        central.extend_from_slice(&extra);
    }
    let cd_offset = out.len() as u64;
    out.extend_from_slice(&central);
    let count = entries.len() as u16;
    if zip64 {
        let record = out.len() as u64;
        out.extend_from_slice(&[0x50, 0x4b, 0x06, 0x06]);
        out.extend_from_slice(&44u64.to_le_bytes());
        out.extend_from_slice(&[45, 0, 45, 0]);
        out.extend_from_slice(&[0; 8]);
        out.extend_from_slice(&u64::from(count).to_le_bytes());
        out.extend_from_slice(&u64::from(count).to_le_bytes());
        out.extend_from_slice(&(central.len() as u64).to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&[0x50, 0x4b, 0x06, 0x07]);
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&record.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0]);
        out.extend_from_slice(&0xFFFFu16.to_le_bytes());
        out.extend_from_slice(&0xFFFFu16.to_le_bytes());
        out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    } else {
        out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0]);
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&(cd_offset as u32).to_le_bytes());
    }
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

/// A tensor to put in a fixture checkpoint.
#[derive(Debug, Clone)]
pub struct FixtureTensor {
    /// State-dict name.
    pub name: String,
    /// The `torch.<Class>` of its storage: `FloatStorage`, `BFloat16Storage`, ...
    pub storage_class: &'static str,
    /// Shape.
    pub shape: Vec<usize>,
    /// Row-major little-endian element bytes.
    pub data: Vec<u8>,
}

impl FixtureTensor {
    /// A float32 tensor.
    pub fn f32(name: &str, shape: &[usize], values: &[f32]) -> Self {
        Self {
            name: name.to_owned(),
            storage_class: "FloatStorage",
            shape: shape.to_vec(),
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    /// A bfloat16 tensor from raw bit patterns.
    pub fn bf16_bits(name: &str, shape: &[usize], bits: &[u16]) -> Self {
        Self {
            name: name.to_owned(),
            storage_class: "BFloat16Storage",
            shape: shape.to_vec(),
            data: bits.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }
}

fn push_unicode(p: &mut Vec<u8>, s: &str) {
    p.push(b'X');
    p.extend_from_slice(&(s.len() as u32).to_le_bytes());
    p.extend_from_slice(s.as_bytes());
}

fn push_int(p: &mut Vec<u8>, v: usize) {
    if v < 256 {
        p.extend_from_slice(&[b'K', v as u8]);
    } else {
        p.push(b'J');
        p.extend_from_slice(&(v as u32).to_le_bytes());
    }
}

fn push_int_tuple(p: &mut Vec<u8>, values: &[usize]) {
    p.push(b'(');
    for v in values {
        push_int(p, *v);
    }
    p.push(b't');
}

fn push_ordered_dict_start(p: &mut Vec<u8>) {
    p.extend_from_slice(b"ccollections\nOrderedDict\n)R");
}

/// The pickle `torch.save` writes for an `OrderedDict` of tensors (protocol
/// 2): each tensor is `torch._utils._rebuild_tensor_v2(storage, offset,
/// size, stride, requires_grad, backward_hooks)` with the storage a
/// persistent id.
fn state_dict_pickle(tensors: &[FixtureTensor], first_storage_key: usize) -> Vec<u8> {
    let mut p = vec![0x80, 2];
    push_ordered_dict_start(&mut p);
    p.push(b'(');
    for (i, t) in tensors.iter().enumerate() {
        push_unicode(&mut p, &t.name);
        p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
        p.push(b'(');
        // Persistent id: ('storage', torch.<Class>, key, 'cpu', numel)
        p.push(b'(');
        push_unicode(&mut p, "storage");
        p.extend_from_slice(format!("ctorch\n{}\n", t.storage_class).as_bytes());
        push_unicode(&mut p, &(first_storage_key + i).to_string());
        push_unicode(&mut p, "cpu");
        let width = match t.storage_class {
            "DoubleStorage" | "LongStorage" => 8,
            "FloatStorage" | "IntStorage" => 4,
            "HalfStorage" | "BFloat16Storage" | "ShortStorage" => 2,
            _ => 1,
        };
        push_int(&mut p, t.data.len() / width);
        p.push(b't');
        p.push(b'Q');
        push_int(&mut p, 0); // storage offset
        push_int_tuple(&mut p, &t.shape);
        // Row-major strides.
        let mut stride = vec![1usize; t.shape.len()];
        for d in (0..t.shape.len().saturating_sub(1)).rev() {
            stride[d] = stride[d + 1] * t.shape[d + 1];
        }
        push_int_tuple(&mut p, &stride);
        p.push(0x89); // requires_grad = False
        push_ordered_dict_start(&mut p); // backward_hooks
        p.push(b't');
        p.push(b'R');
    }
    p.push(b'u');
    p.push(b'.');
    p
}

/// How to lay out a fixture checkpoint.
#[derive(Debug, Clone, Default)]
pub struct CheckpointLayout {
    /// Wrap the state dict in `{wrapper: state_dict}` (a trainer's checkpoint).
    pub wrapper: Option<String>,
    /// Write ZIP64 records.
    pub zip64: bool,
}

/// A PyTorch checkpoint (`pytorch_model.bin` style) holding `tensors`.
pub fn checkpoint(tensors: &[FixtureTensor], layout: &CheckpointLayout) -> Vec<u8> {
    let mut pickle = state_dict_pickle(tensors, 0);
    if let Some(wrapper) = &layout.wrapper {
        // {wrapper: <state dict>}: EMPTY_DICT, key, the pickled dict, SETITEM.
        let inner = state_dict_pickle(tensors, 0);
        let mut p = vec![0x80, 2, b'}'];
        push_unicode(&mut p, wrapper);
        // Drop the inner pickle's PROTO header and STOP.
        p.extend_from_slice(&inner[2..inner.len() - 1]);
        p.push(b's');
        p.push(b'.');
        pickle = p;
    }
    let mut entries: Vec<(String, Vec<u8>)> = vec![
        ("archive/data.pkl".to_owned(), pickle),
        ("archive/byteorder".to_owned(), b"little".to_vec()),
    ];
    for (i, t) in tensors.iter().enumerate() {
        entries.push((format!("archive/data/{i}"), t.data.clone()));
    }
    entries.push(("archive/version".to_owned(), b"3\n".to_vec()));
    let borrowed: Vec<(&str, &[u8])> = entries
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect();
    stored_zip(&borrowed, layout.zip64)
}
