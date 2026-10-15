//! A restricted unpickler for PyTorch state dicts.
//!
//! A `.pt` file's `data.pkl` is a Python pickle, which in general can run
//! arbitrary code on load. This interpreter understands only the opcodes
//! `torch.save` emits for plain containers and tensors, and builds objects
//! only for a short allow-list of globals (`OrderedDict` and the tensor
//! rebuilders). Anything else is an error, never an execution.

use crate::error::{Result, VocoderError};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Element type of a tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDtype {
    F32,
    F64,
    F16,
    Bf16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl StorageDtype {
    fn from_class(name: &str) -> Option<Self> {
        Some(match name {
            "FloatStorage" => Self::F32,
            "DoubleStorage" => Self::F64,
            "HalfStorage" => Self::F16,
            "BFloat16Storage" => Self::Bf16,
            "LongStorage" => Self::I64,
            "IntStorage" => Self::I32,
            "ShortStorage" => Self::I16,
            "CharStorage" => Self::I8,
            "ByteStorage" => Self::U8,
            "BoolStorage" => Self::Bool,
            _ => return None,
        })
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }
}

/// A reference to a storage file inside the archive.
#[derive(Debug, Clone)]
pub struct StorageRef {
    /// The `data/<key>` entry name.
    pub key: String,
    /// Element type.
    pub dtype: StorageDtype,
}

/// A tensor as a view into a storage.
#[derive(Debug, Clone)]
pub struct TensorRef {
    /// The backing storage.
    pub storage: StorageRef,
    /// Offset into the storage, in elements.
    pub offset: usize,
    /// Shape.
    pub shape: Vec<usize>,
    /// Strides, in elements.
    pub stride: Vec<usize>,
}

/// A value on the unpickler's stack.
#[derive(Debug, Clone)]
pub enum Obj {
    None,
    Int(i64),
    Str(Rc<str>),
    Tuple(Rc<Vec<Obj>>),
    List(Rc<RefCell<Vec<Obj>>>),
    Dict(Rc<RefCell<Vec<(Obj, Obj)>>>),
    Global(Rc<str>, Rc<str>),
    Storage(StorageRef),
    Tensor(TensorRef),
    /// An object of a class we do not model (kept only so loading continues).
    Opaque,
    Mark,
}

impl Obj {
    /// The string, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Obj::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The `(key, value)` pairs, if this is a dict.
    pub fn dict_items(&self) -> Option<Vec<(Obj, Obj)>> {
        match self {
            Obj::Dict(d) => Some(d.borrow().clone()),
            _ => None,
        }
    }

    /// The value stored under string key `key`, if this is a dict that has it.
    pub fn get(&self, key: &str) -> Option<Obj> {
        match self {
            Obj::Dict(d) => d
                .borrow()
                .iter()
                .find(|(k, _)| k.as_str() == Some(key))
                .map(|(_, v)| v.clone()),
            _ => None,
        }
    }
}

fn bad(msg: impl Into<String>) -> VocoderError {
    VocoderError::WeightLoad(format!(
        "unsupported PyTorch checkpoint pickle: {}",
        msg.into()
    ))
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.data.len());
        let end = end.ok_or_else(|| bad("truncated pickle"))?;
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    fn line(&mut self) -> Result<&'a str> {
        let rest = &self.data[self.pos..];
        let n = rest
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| bad("unterminated text line"))?;
        let s = std::str::from_utf8(&rest[..n]).map_err(|_| bad("non-UTF-8 text"))?;
        self.pos += n + 1;
        Ok(s)
    }

    fn string(&mut self, n: usize) -> Result<Obj> {
        let b = self.take(n)?;
        let s = std::str::from_utf8(b).map_err(|_| bad("non-UTF-8 string"))?;
        Ok(Obj::Str(Rc::from(s)))
    }
}

/// Unpickle `data` into an object tree (tensors stay references).
pub fn load(data: &[u8]) -> Result<Obj> {
    let mut r = Reader { data, pos: 0 };
    let mut stack: Vec<Obj> = Vec::new();
    let mut memo: HashMap<u32, Obj> = HashMap::new();
    let mut next_memo = 0u32;

    fn pop(stack: &mut Vec<Obj>) -> Result<Obj> {
        stack.pop().ok_or_else(|| bad("stack underflow"))
    }
    fn pop_mark(stack: &mut Vec<Obj>) -> Result<Vec<Obj>> {
        let at = stack
            .iter()
            .rposition(|o| matches!(o, Obj::Mark))
            .ok_or_else(|| bad("missing MARK"))?;
        let items = stack.split_off(at + 1);
        stack.pop();
        Ok(items)
    }
    fn tuple(items: Vec<Obj>) -> Obj {
        Obj::Tuple(Rc::new(items))
    }
    fn dict_set(dict: &Obj, items: Vec<Obj>) -> Result<()> {
        let Obj::Dict(d) = dict else {
            return Err(bad("SETITEM(S) on a non-dict"));
        };
        let mut d = d.borrow_mut();
        for kv in items.as_chunks::<2>().0 {
            d.push((kv[0].clone(), kv[1].clone()));
        }
        Ok(())
    }
    fn list_append(list: &Obj, items: Vec<Obj>) -> Result<()> {
        let Obj::List(l) = list else {
            return Err(bad("APPEND(S) on a non-list"));
        };
        l.borrow_mut().extend(items);
        Ok(())
    }

    loop {
        let op = r.u8()?;
        match op {
            0x80 => {
                r.u8()?; // PROTO
            }
            0x95 => {
                r.u64()?; // FRAME
            }
            b'.' => return pop(&mut stack),
            b'(' => stack.push(Obj::Mark),
            b'N' => stack.push(Obj::None),
            // Booleans and floats only appear in optimizer state, which is not read.
            0x88 | 0x89 => stack.push(Obj::Opaque),
            b'K' => stack.push(Obj::Int(i64::from(r.u8()?))),
            b'M' => stack.push(Obj::Int(i64::from(r.u16()?))),
            b'J' => stack.push(Obj::Int(i64::from(r.u32()? as i32))),
            0x8a => {
                let n = r.u8()? as usize;
                if n > 8 {
                    return Err(bad("integer wider than 64 bits"));
                }
                let b = r.take(n)?;
                let mut v: i64 = 0;
                for (i, &byte) in b.iter().enumerate() {
                    v |= i64::from(byte) << (8 * i);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v |= -1i64 << (8 * n);
                }
                stack.push(Obj::Int(v));
            }
            b'G' => {
                r.take(8)?;
                stack.push(Obj::Opaque);
            }
            b'X' => {
                let n = r.u32()? as usize;
                stack.push(r.string(n)?);
            }
            0x8c => {
                let n = r.u8()? as usize;
                stack.push(r.string(n)?);
            }
            0x8d => {
                let n = r.u64()? as usize;
                stack.push(r.string(n)?);
            }
            b'U' => {
                let n = r.u8()? as usize;
                stack.push(r.string(n)?);
            }
            b'T' => {
                let n = r.u32()? as usize;
                stack.push(r.string(n)?);
            }
            b'B' => {
                let n = r.u32()? as usize;
                r.take(n)?;
                stack.push(Obj::Opaque);
            }
            b'C' => {
                let n = r.u8()? as usize;
                r.take(n)?;
                stack.push(Obj::Opaque);
            }
            b')' => stack.push(tuple(Vec::new())),
            0x85 => {
                let a = pop(&mut stack)?;
                stack.push(tuple(vec![a]));
            }
            0x86 => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(tuple(vec![a, b]));
            }
            0x87 => {
                let c = pop(&mut stack)?;
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(tuple(vec![a, b, c]));
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(tuple(items));
            }
            b']' => stack.push(Obj::List(Rc::new(RefCell::new(Vec::new())))),
            b'l' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Obj::List(Rc::new(RefCell::new(items))));
            }
            b'a' => {
                let item = pop(&mut stack)?;
                let list = stack.last().ok_or_else(|| bad("stack underflow"))?;
                list_append(list, vec![item])?;
            }
            b'e' => {
                let items = pop_mark(&mut stack)?;
                let list = stack.last().ok_or_else(|| bad("stack underflow"))?;
                list_append(list, items)?;
            }
            b'}' => stack.push(Obj::Dict(Rc::new(RefCell::new(Vec::new())))),
            b'd' => {
                let items = pop_mark(&mut stack)?;
                let dict = Obj::Dict(Rc::new(RefCell::new(Vec::new())));
                dict_set(&dict, items)?;
                stack.push(dict);
            }
            b's' => {
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                let dict = stack.last().ok_or_else(|| bad("stack underflow"))?;
                dict_set(dict, vec![k, v])?;
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                let dict = stack.last().ok_or_else(|| bad("stack underflow"))?;
                dict_set(dict, items)?;
            }
            b'q' => {
                let k = u32::from(r.u8()?);
                let top = stack.last().ok_or_else(|| bad("stack underflow"))?.clone();
                memo.insert(k, top);
            }
            b'r' => {
                let k = r.u32()?;
                let top = stack.last().ok_or_else(|| bad("stack underflow"))?.clone();
                memo.insert(k, top);
            }
            0x94 => {
                let top = stack.last().ok_or_else(|| bad("stack underflow"))?.clone();
                memo.insert(next_memo, top);
                next_memo += 1;
            }
            b'h' => {
                let k = u32::from(r.u8()?);
                stack.push(memo.get(&k).cloned().ok_or_else(|| bad("bad memo index"))?);
            }
            b'j' => {
                let k = r.u32()?;
                stack.push(memo.get(&k).cloned().ok_or_else(|| bad("bad memo index"))?);
            }
            b'c' => {
                let module = r.line()?;
                let name = r.line()?;
                stack.push(Obj::Global(Rc::from(module), Rc::from(name)));
            }
            0x93 => {
                let name = pop(&mut stack)?;
                let module = pop(&mut stack)?;
                match (module.as_str(), name.as_str()) {
                    (Some(m), Some(n)) => stack.push(Obj::Global(Rc::from(m), Rc::from(n))),
                    _ => return Err(bad("STACK_GLOBAL with non-string operands")),
                }
            }
            b'Q' => {
                let pid = pop(&mut stack)?;
                stack.push(persistent_load(&pid)?);
            }
            b'R' => {
                let args = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                stack.push(reduce(&callable, &args)?);
            }
            0x81 => {
                pop(&mut stack)?; // args
                pop(&mut stack)?; // class
                stack.push(Obj::Opaque);
            }
            b'b' => {
                // BUILD: the state of OrderedDict is only its `_metadata`.
                pop(&mut stack)?;
            }
            b'0' => {
                pop(&mut stack)?;
            }
            b'2' => {
                let top = stack.last().ok_or_else(|| bad("stack underflow"))?.clone();
                stack.push(top);
            }
            other => {
                return Err(bad(format!("opcode 0x{other:02x} at byte {}", r.pos - 1)));
            }
        }
    }
}

/// `('storage', <class>, key, location, numel)` -> a storage reference.
fn persistent_load(pid: &Obj) -> Result<Obj> {
    let Obj::Tuple(items) = pid else {
        return Err(bad("persistent id is not a tuple"));
    };
    match items.as_slice() {
        [tag, class, key, _location, ..] if tag.as_str() == Some("storage") => {
            let class_name = match class {
                Obj::Global(_, name) => name.to_string(),
                _ => return Err(bad("storage class is not a global")),
            };
            let dtype = StorageDtype::from_class(&class_name)
                .ok_or_else(|| bad(format!("storage type `{class_name}`")))?;
            let key = match key {
                Obj::Str(s) => s.to_string(),
                Obj::Int(i) => i.to_string(),
                _ => return Err(bad("storage key")),
            };
            Ok(Obj::Storage(StorageRef { key, dtype }))
        }
        _ => Err(bad("persistent id is not a storage")),
    }
}

fn usize_tuple(o: &Obj) -> Result<Vec<usize>> {
    let Obj::Tuple(items) = o else {
        return Err(bad("expected a tuple of integers"));
    };
    items
        .iter()
        .map(|i| match i {
            Obj::Int(v) if *v >= 0 => Ok(*v as usize),
            _ => Err(bad("expected a non-negative integer")),
        })
        .collect()
}

/// The allow-listed calls: everything else is refused.
fn reduce(callable: &Obj, args: &Obj) -> Result<Obj> {
    let Obj::Global(module, name) = callable else {
        return Err(bad("call of a non-global"));
    };
    let Obj::Tuple(argv) = args else {
        return Err(bad("call arguments are not a tuple"));
    };
    match (&**module, &**name) {
        ("collections", "OrderedDict") => {
            let dict = Obj::Dict(Rc::new(RefCell::new(Vec::new())));
            if let Some(Obj::List(items)) = argv.first() {
                if let Obj::Dict(d) = &dict {
                    for pair in items.borrow().iter() {
                        if let Obj::Tuple(kv) = pair {
                            if let [k, v] = kv.as_slice() {
                                d.borrow_mut().push((k.clone(), v.clone()));
                            }
                        }
                    }
                }
            }
            Ok(dict)
        }
        ("torch._utils", "_rebuild_tensor_v2" | "_rebuild_tensor") => {
            let [Obj::Storage(storage), Obj::Int(offset), size, stride, ..] = argv.as_slice()
            else {
                return Err(bad("malformed tensor rebuild call"));
            };
            let shape = usize_tuple(size)?;
            let stride = usize_tuple(stride)?;
            if shape.len() != stride.len() {
                return Err(bad("tensor shape and stride differ in rank"));
            }
            Ok(Obj::Tensor(TensorRef {
                storage: storage.clone(),
                offset: usize::try_from(*offset).map_err(|_| bad("negative storage offset"))?,
                shape,
                stride,
            }))
        }
        ("torch._utils", "_rebuild_parameter" | "_rebuild_parameter_with_state") => argv
            .first()
            .cloned()
            .ok_or_else(|| bad("parameter rebuild without data")),
        ("torch", "Size") => argv
            .first()
            .cloned()
            .ok_or_else(|| bad("torch.Size without data")),
        (m, n) => Err(bad(format!("call of `{m}.{n}`"))),
    }
}
