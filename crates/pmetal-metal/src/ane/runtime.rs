#![allow(unsafe_code)]

//! ANE private API FFI via dlopen + objc2.
//!
//! Wraps the private ObjC classes from `AppleNeuralEngine.framework`:
//! - `_ANEInMemoryModelDescriptor`
//! - `_ANEInMemoryModel`
//! - `_ANERequest`
//! - `_ANEIOSurfaceObject`
//! - `_ANEPerformanceStats` (optional hardware timing)
//!
//! The framework is loaded at runtime via `dlopen` (not linked at build time).
//! Classes are resolved via `NSClassFromString`. Returns `AneNotAvailable`
//! gracefully if the framework is missing.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;

use objc2::runtime::{AnyClass, AnyObject, Bool};
use objc2::{
    encode::{Encode, Encoding, RefEncode},
    msg_send,
};
use objc2_foundation::{NSArray, NSData, NSDictionary, NSError, NSFileManager, NSNumber, NSString};
use parking_lot::Mutex;

use crate::error::{MetalError, Result};

// dlopen FFI — avoid libc dependency
const RTLD_NOW: c_int = 0x2;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
}

/// QoS constant used for all ANE operations. Value 21 = userInteractive.
/// The ANE reference confirmed this has no latency impact vs other values.
const ANE_QOS: u32 = 21;

#[repr(C)]
struct __IOSurface {
    _private: [u8; 0],
}

unsafe impl Encode for __IOSurface {
    const ENCODING: Encoding = Encoding::Struct("__IOSurface", &[]);
}

unsafe impl RefEncode for __IOSurface {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

/// Global ANE runtime singleton.
static ANE_RUNTIME: OnceLock<std::result::Result<AneRuntime, MetalError>> = OnceLock::new();

/// Safe wrapper around the ANE private API runtime.
///
/// Holds references to the private ObjC classes needed for
/// compilation and evaluation. Created once via [`AneRuntime::global()`].
pub struct AneRuntime {
    /// `_ANEInMemoryModelDescriptor`
    descriptor_class: &'static AnyClass,
    /// `_ANEInMemoryModel`
    model_class: &'static AnyClass,
    /// `_ANERequest`
    request_class: &'static AnyClass,
    /// `_ANEIOSurfaceObject`
    io_surface_class: &'static AnyClass,
    /// `_ANEPerformanceStats` — hardware execution time counters.
    /// Available on all Apple Silicon; populated after eval when perfStatsMask is set.
    perf_stats_class: Option<&'static AnyClass>,
}

// SAFETY: The ObjC classes are process-global singletons and thread-safe for class method dispatch.
unsafe impl Send for AneRuntime {}
unsafe impl Sync for AneRuntime {}

impl AneRuntime {
    /// Get the global ANE runtime, loading the framework on first call.
    ///
    /// Returns `Ok(&AneRuntime)` on M1+ hardware, `Err(AneNotAvailable)` otherwise.
    pub fn global() -> Result<&'static AneRuntime> {
        ANE_RUNTIME
            .get_or_init(AneRuntime::init)
            .as_ref()
            .map_err(|e| e.clone())
    }

    /// Load the private framework and resolve all four classes.
    fn init() -> std::result::Result<AneRuntime, MetalError> {
        // dlopen the private framework
        let path =
            c"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            return Err(MetalError::AneNotAvailable);
        }

        // Resolve classes via NSClassFromString
        let descriptor_class = resolve_class(c"_ANEInMemoryModelDescriptor")?;
        let model_class = resolve_class(c"_ANEInMemoryModel")?;
        let request_class = resolve_class(c"_ANERequest")?;
        let io_surface_class = resolve_class(c"_ANEIOSurfaceObject")?;

        // Probe for performance stats API
        let perf_stats_class = AnyClass::get(c"_ANEPerformanceStats");
        if perf_stats_class.is_some() {
            tracing::debug!("ANE performance stats API (_ANEPerformanceStats) detected");
        }

        Ok(AneRuntime {
            descriptor_class,
            model_class,
            request_class,
            io_surface_class,
            perf_stats_class,
        })
    }

    /// Compile a MIL program with weights into an ANE model.
    ///
    /// This performs the full pipeline: descriptor → model → compile → load.
    /// The returned `AneModel` implements `Drop` for RAII cleanup.
    pub fn compile(&self, mil_text: &[u8], weight_dict: Option<&WeightDict>) -> Result<AneModel> {
        // The framework hands back autoreleased objects, the descriptor among
        // them, and it holds a copy of every weight. A Rust thread has no
        // pool to drain them, so without this one each kernel's weights
        // stayed in memory for the life of the process.
        objc2::rc::autoreleasepool(|_| self.compile_in_pool(mil_text, weight_dict))
    }

    fn compile_in_pool(
        &self,
        mil_text: &[u8],
        weight_dict: Option<&WeightDict>,
    ) -> Result<AneModel> {
        // Several weight files fail macOS 27's bundle hash check; see pack_weights.
        let packed = weight_dict.filter(|_| macos_27_or_later()).and_then(|wd| {
            let mil = std::str::from_utf8(mil_text).ok()?;
            pack_weights(mil, wd)
        });
        let (mil_text, weight_dict) = match &packed {
            Some((mil, wd)) => (mil.as_bytes(), Some(wd)),
            None => (mil_text, weight_dict),
        };

        // SAFETY: All ObjC message sends use valid class/object pointers obtained
        // from the framework. Memory management follows ObjC retain/release rules.
        unsafe {
            let mil_data = NSData::with_bytes(mil_text);

            // Build weight dictionary (empty dict if no weights — ANE requires non-nil)
            let empty_wd_storage;
            let wdict_obj = match weight_dict {
                Some(wd) => wd.to_ns_dict(),
                None => {
                    empty_wd_storage = WeightDict::new();
                    empty_wd_storage.to_ns_dict()
                }
            };

            // Create descriptor: modelWithMILText:weights:optionsPlist:
            let wdict_ptr: *const AnyObject = wdict_obj.as_ref() as *const _;

            let desc: *mut AnyObject = msg_send![
                self.descriptor_class,
                modelWithMILText: &*mil_data,
                weights: wdict_ptr,
                optionsPlist: std::ptr::null::<AnyObject>()
            ];
            if desc.is_null() {
                return Err(MetalError::AneCompileFailed(
                    "descriptor creation failed".into(),
                ));
            }

            // Create model: inMemoryModelWithDescriptor:
            let model: *mut AnyObject = msg_send![
                self.model_class,
                inMemoryModelWithDescriptor: desc
            ];
            if model.is_null() {
                return Err(MetalError::AneCompileFailed("model creation failed".into()));
            }

            // Get temp directory from hexStringIdentifier
            let hex_id: *mut AnyObject = msg_send![model, hexStringIdentifier];
            let tmp_base = NSString::from_str(&std::env::temp_dir().to_string_lossy());
            let tmp_dir: *mut AnyObject =
                msg_send![&*tmp_base, stringByAppendingPathComponent: hex_id];
            let tmp_dir_str = ns_string_to_rust(tmp_dir as *const AnyObject);
            let _compile_dir = CompileDir(tmp_dir_str.clone());

            // aned keeps compiled programs across processes, keyed by the
            // hash in this directory's name (MIL, weights, options), so a
            // kernel compiled by an earlier run only needs loading.
            // Recompiling it took ~0.3 s for each Qwen3-4B FFN kernel.
            let cached: Bool = msg_send![model, compiledModelExists];
            let mut from_cache = cached.as_bool();
            if !from_cache {
                stage_and_compile(model, &tmp_dir_str, &mil_data, weight_dict)?;
            }
            if let Err(err) = load_model(model) {
                if !from_cache {
                    return Err(err);
                }
                from_cache = false;
                // Purged between the check and the load.
                tracing::debug!(error = %err, "cached ANE program didn't load; recompiling");
                stage_and_compile(model, &tmp_dir_str, &mil_data, weight_dict)?;
                load_model(model)?;
            }

            // Retain the model object
            let _: *mut AnyObject = msg_send![model, retain];

            // Only the inner model: see read_io_layouts.
            let inner_model: *mut AnyObject = msg_send![model, model];
            let (input_layouts, output_layouts) = if inner_model.is_null() {
                (Vec::new(), Vec::new())
            } else {
                read_io_layouts(inner_model)
            };
            tracing::debug!(
                inputs = ?input_layouts,
                outputs = ?output_layouts,
                "ANE kernel I/O layouts"
            );

            // The descriptor holds a copy of every weight, about as much
            // memory again as the loaded program. A loaded model evaluates,
            // unloads and reloads without it.
            let _: () = msg_send![model, setDescriptor: std::ptr::null::<AnyObject>()];

            Ok(AneModel {
                model,
                request_class: self.request_class,
                io_surface_class: self.io_surface_class,
                input_layouts,
                output_layouts,
                staging: Mutex::new(Staging::default()),
                from_cache,
            })
        }
    }

    /// Get a reference to the `_ANEIOSurfaceObject` class for wrapping IOSurfaces.
    pub fn io_surface_class(&self) -> &'static AnyClass {
        self.io_surface_class
    }

    /// Get a reference to the `_ANERequest` class.
    pub fn request_class(&self) -> &'static AnyClass {
        self.request_class
    }

    /// Check if ANE performance stats can be requested on this system.
    pub fn perf_stats_available(&self) -> bool {
        self.perf_stats_class.is_some() && perf_stats_request_supported()
    }
}

/// ANE hardware performance stats from a single evaluation.
#[derive(Debug, Clone, Default)]
pub struct AnePerformanceStats {
    /// Hardware execution time in nanoseconds.
    pub hw_execution_time_ns: u64,
}

/// A compiled ANE model ready for evaluation.
///
/// Implements `Drop` for RAII: unloads from ANE hardware.
pub struct AneModel {
    model: *mut AnyObject,
    request_class: &'static AnyClass,
    io_surface_class: &'static AnyClass,
    input_layouts: Vec<AneTensorLayout>,
    output_layouts: Vec<AneTensorLayout>,
    /// Surfaces laid out the model's way, for callers' packed ones whose
    /// layout is padded. Allocated on first use; held for the evaluation.
    staging: Mutex<Staging>,
    from_cache: bool,
}

#[derive(Default)]
struct Staging {
    inputs: Vec<Option<crate::ane::iosurface::IoSurface>>,
    outputs: Vec<Option<crate::ane::iosurface::IoSurface>>,
}

// SAFETY: ANE model objects are thread-safe for evaluation dispatch.
unsafe impl Send for AneModel {}
unsafe impl Sync for AneModel {}

impl AneModel {
    /// The layout the model expects for each input, in order. Empty when the
    /// model doesn't report it.
    pub fn input_layouts(&self) -> &[AneTensorLayout] {
        &self.input_layouts
    }

    /// The layout the model produces for each output, in order.
    pub fn output_layouts(&self) -> &[AneTensorLayout] {
        &self.output_layouts
    }

    /// Whether the program came from aned's compiled-program cache, which
    /// outlives the process, rather than being compiled for this call.
    pub fn from_cache(&self) -> bool {
        self.from_cache
    }

    /// Build a request and evaluate the model.
    ///
    /// `inputs` and `outputs` are IOSurface references for data transfer.
    pub fn evaluate(&self, inputs: &[*mut c_void], outputs: &[*mut c_void]) -> Result<()> {
        self.evaluate_inner(inputs, outputs, false).map(|_| ())
    }

    /// Evaluate the model and collect hardware performance stats.
    ///
    /// Returns the performance stats including hardware execution time.
    /// Requires the ANE perf stats class to be available (always true on M1+).
    /// Falls back to regular evaluation silently if the class is absent.
    pub fn evaluate_with_stats(
        &self,
        inputs: &[*mut c_void],
        outputs: &[*mut c_void],
    ) -> Result<AnePerformanceStats> {
        self.evaluate_inner(inputs, outputs, true)
    }

    /// Evaluate, restriding any packed surface the model expects padded:
    /// inputs are copied into staging surfaces laid out the model's way, and
    /// outputs are copied back after. Surfaces that already match pass
    /// through untouched, so on layouts without padding this costs nothing.
    fn evaluate_inner(
        &self,
        inputs: &[*mut c_void],
        outputs: &[*mut c_void],
        collect_stats: bool,
    ) -> Result<AnePerformanceStats> {
        let mut staging = self.staging.lock();
        let ins = stage(inputs, &self.input_layouts, &mut staging.inputs)?;
        let outs = stage(outputs, &self.output_layouts, &mut staging.outputs)?;
        for (i, layout, surface) in &ins.restrided {
            // SAFETY: both are live surfaces sized for this layout (see stage).
            unsafe { restride(inputs[*i], *surface, layout, Direction::ToModel) };
        }
        check_surface_sizes("input", &ins.surfaces, &self.input_layouts)?;
        check_surface_sizes("output", &outs.surfaces, &self.output_layouts)?;
        let stats = self.evaluate_raw(&ins.surfaces, &outs.surfaces, collect_stats)?;
        for (i, layout, surface) in &outs.restrided {
            unsafe { restride(*surface, outputs[*i], layout, Direction::FromModel) };
        }
        Ok(stats)
    }

    fn evaluate_raw(
        &self,
        inputs: &[*mut c_void],
        outputs: &[*mut c_void],
        collect_stats: bool,
    ) -> Result<AnePerformanceStats> {
        unsafe {
            let rt = AneRuntime::global()?;
            let perf_stats = if collect_stats {
                let Some(perf_class) = rt
                    .perf_stats_class
                    .filter(|_| perf_stats_request_supported())
                else {
                    self.evaluate_raw(inputs, outputs, false)?;
                    return Ok(AnePerformanceStats::default());
                };

                let zero = NSNumber::new_u64(0);
                let perf_stats: *mut AnyObject = msg_send![
                    perf_class,
                    statsWithHardwareExecutionNS: &*zero
                ];
                if perf_stats.is_null() {
                    self.evaluate_raw(inputs, outputs, false)?;
                    return Ok(AnePerformanceStats::default());
                }
                perf_stats
            } else {
                std::ptr::null_mut()
            };

            let request = self.build_request(inputs, outputs, perf_stats);
            self.evaluate_request(request)?;

            let hw_time = if perf_stats.is_null() {
                0
            } else {
                msg_send![perf_stats, hwExecutionTime]
            };

            Ok(AnePerformanceStats {
                hw_execution_time_ns: hw_time,
            })
        }
    }

    unsafe fn build_request(
        &self,
        inputs: &[*mut c_void],
        outputs: &[*mut c_void],
        perf_stats: *mut AnyObject,
    ) -> *mut AnyObject {
        let mut wrapped_inputs: Vec<*mut AnyObject> = Vec::with_capacity(inputs.len());
        let mut input_indices: Vec<objc2::rc::Retained<NSNumber>> =
            Vec::with_capacity(inputs.len());
        for (i, &surface) in inputs.iter().enumerate() {
            let surface = surface.cast::<__IOSurface>();
            let wrapped: *mut AnyObject =
                msg_send![self.io_surface_class, objectWithIOSurface: surface];
            wrapped_inputs.push(wrapped);
            input_indices.push(NSNumber::new_usize(i));
        }

        let mut wrapped_outputs: Vec<*mut AnyObject> = Vec::with_capacity(outputs.len());
        let mut output_indices: Vec<objc2::rc::Retained<NSNumber>> =
            Vec::with_capacity(outputs.len());
        for (i, &surface) in outputs.iter().enumerate() {
            let surface = surface.cast::<__IOSurface>();
            let wrapped: *mut AnyObject =
                msg_send![self.io_surface_class, objectWithIOSurface: surface];
            wrapped_outputs.push(wrapped);
            output_indices.push(NSNumber::new_usize(i));
        }

        let ns_inputs = unsafe { ns_array_from_raw(&wrapped_inputs) };
        let ns_input_idx = unsafe { ns_array_from_numbers(&input_indices) };
        let ns_outputs = unsafe { ns_array_from_raw(&wrapped_outputs) };
        let ns_output_idx = unsafe { ns_array_from_numbers(&output_indices) };
        let zero = NSNumber::new_usize(0);

        msg_send![
            self.request_class,
            requestWithInputs: &*ns_inputs,
            inputIndices: &*ns_input_idx,
            outputs: &*ns_outputs,
            outputIndices: &*ns_output_idx,
            weightsBuffer: std::ptr::null::<AnyObject>(),
            perfStats: perf_stats,
            procedureIndex: &*zero
        ]
    }

    unsafe fn evaluate_request(&self, request: *mut AnyObject) -> Result<()> {
        let mut error: *mut NSError = std::ptr::null_mut();
        let empty_dict = empty_options_dict();
        let ok: Bool = msg_send![
            self.model,
            evaluateWithQoS: ANE_QOS,
            options: &*empty_dict,
            request: request,
            error: &mut error
        ];

        if !ok.as_bool() {
            return Err(MetalError::AneEvalFailed(unsafe { error_message(error) }));
        }

        Ok(())
    }
}

impl Drop for AneModel {
    /// Unload the model from ANE hardware and release it.
    fn drop(&mut self) {
        unsafe {
            let mut error: *mut NSError = std::ptr::null_mut();
            let _: Bool = msg_send![self.model, unloadWithQoS: ANE_QOS, error: &mut error];
            let _: () = msg_send![self.model, release];
        }
    }
}

/// Weight dictionary for ANE model compilation.
///
/// Maps weight file paths (e.g., `"@model_path/weights/wq.bin"`) to raw blob data.
pub struct WeightDict {
    /// Entries mapping path → blob data.
    pub entries: Vec<(String, Vec<u8>)>,
}

impl WeightDict {
    /// Create a new empty weight dictionary.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a weight entry.
    pub fn add(&mut self, path: &str, data: Vec<u8>) {
        self.entries.push((path.to_string(), data));
    }

    /// Convert to NSDictionary for the ANE API.
    ///
    /// Format: `{ "@model_path/weights/name.bin": { "offset": 0, "data": NSData } }`
    fn to_ns_dict(&self) -> objc2::rc::Retained<NSDictionary<NSString, AnyObject>> {
        // `new` returns an owned object, which `Retained` takes over. Retaining
        // it again, as this used to, leaked the dictionary and with it a copy
        // of every weight the kernel was compiled with.
        let mutable_dict = objc2::runtime::AnyClass::get(c"NSMutableDictionary").unwrap();
        unsafe {
            let dict: objc2::rc::Retained<AnyObject> = msg_send![mutable_dict, new];

            for (path, data) in &self.entries {
                let key = NSString::from_str(path);
                let ns_data = NSData::with_bytes(data);

                // Build inner dict: { "offset": @0, "data": ns_data }
                let inner: objc2::rc::Retained<AnyObject> = msg_send![mutable_dict, new];
                let offset_key = NSString::from_str("offset");
                let data_key = NSString::from_str("data");
                let zero = NSNumber::new_i32(0);

                let _: () = msg_send![&*inner, setObject: &*zero, forKey: &*offset_key];
                let _: () = msg_send![&*inner, setObject: &*ns_data, forKey: &*data_key];
                let _: () = msg_send![&*dict, setObject: &*inner, forKey: &*key];
            }

            objc2::rc::Retained::cast_unchecked(dict)
        }
    }
}

impl Default for WeightDict {
    fn default() -> Self {
        Self::new()
    }
}

/// How the ANE expects one model input or output laid out in memory, as the
/// loaded model reports it (`modelAttributes` → `NetworkStatusList`).
///
/// Surfaces are written channel-major with each channel's row packed, which
/// matches this only when `row_stride` equals the row's natural width. On
/// macOS 27 rows are padded to 64 bytes, so a short row (a 16-element fp16
/// sequence, say) needs a bigger surface than the tensor's element count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AneTensorLayout {
    /// Channel count (C in `[1, C, H, W]`).
    pub channels: usize,
    /// Rows per channel.
    pub height: usize,
    /// Elements per row (the sequence or spatial length).
    pub width: usize,
    /// Depth (1 for the 4-D tensors pmetal uses).
    pub depth: usize,
    /// Batch count.
    pub batches: usize,
    /// Bytes from one row to the next.
    pub row_stride: usize,
    /// Bytes from one channel's plane to the next.
    pub plane_stride: usize,
    /// Bytes from one batch to the next.
    pub batch_stride: usize,
    /// Bytes per element (2 for Float16, 4 for Float32; 0 if unreported).
    pub element_bytes: usize,
}

impl AneTensorLayout {
    /// The size of the same tensor with every row packed, which is how
    /// pmetal's surfaces are written.
    pub fn packed_size(&self) -> usize {
        self.channels * self.height * self.width * self.depth * self.batches * self.element_bytes
    }

    /// Whether rows are padded past their natural width, so a packed surface
    /// has to be restrided before the ANE reads it.
    pub fn is_padded(&self) -> bool {
        self.element_bytes > 0 && self.row_stride > self.width * self.element_bytes
    }

    /// The surface size, in bytes, this tensor needs.
    pub fn byte_size(&self) -> usize {
        (self.batch_stride * self.batches)
            .max(self.plane_stride * self.channels * self.depth * self.batches)
    }
}

/// Read a loaded model's input and output layouts. Both are empty when the
/// model doesn't report them.
///
/// # Safety
/// `model` must be the inner `_ANEModel`. On the `_ANEInMemoryModel` wrapper,
/// `modelAttributes` can raise `-externConstants: unrecognized selector`,
/// which aborts the process.
unsafe fn read_io_layouts(model: *mut AnyObject) -> (Vec<AneTensorLayout>, Vec<AneTensorLayout>) {
    unsafe {
        let responds: Bool = msg_send![model, respondsToSelector: objc2::sel!(modelAttributes)];
        if !responds.as_bool() {
            return (Vec::new(), Vec::new());
        }
        let attrs: *mut AnyObject = msg_send![model, modelAttributes];
        let get = |dict: *mut AnyObject, key: &str| -> *mut AnyObject {
            if dict.is_null() {
                return std::ptr::null_mut();
            }
            let key = NSString::from_str(key);
            msg_send![dict, objectForKey: &*key]
        };
        let count = |array: *mut AnyObject| -> usize {
            if array.is_null() {
                0
            } else {
                msg_send![array, count]
            }
        };
        let status = get(attrs, "NetworkStatusList");
        if count(status) == 0 {
            return (Vec::new(), Vec::new());
        }
        let network: *mut AnyObject = msg_send![status, objectAtIndex: 0usize];
        let layouts = |key: &str| -> Vec<AneTensorLayout> {
            let list = get(network, key);
            (0..count(list))
                .map(|i| {
                    let d: *mut AnyObject = msg_send![list, objectAtIndex: i];
                    let n = |k: &str| -> usize {
                        let v = get(d, k);
                        if v.is_null() {
                            0
                        } else {
                            let x: u64 = msg_send![v, unsignedLongLongValue];
                            x as usize
                        }
                    };
                    let element_bytes = {
                        let t = get(d, "Type");
                        if t.is_null() {
                            0
                        } else {
                            match ns_string_to_rust(t as *const AnyObject).as_str() {
                                "Float16" => 2,
                                "Float32" => 4,
                                _ => 0,
                            }
                        }
                    };
                    AneTensorLayout {
                        element_bytes,
                        channels: n("Channels"),
                        height: n("Height"),
                        width: n("Width"),
                        depth: n("Depth").max(1),
                        batches: n("Batches").max(1),
                        row_stride: n("RowStride"),
                        plane_stride: n("PlaneStride"),
                        batch_stride: n("BatchStride"),
                    }
                })
                .collect()
        };
        (layouts("LiveInputList"), layouts("LiveOutputList"))
    }
}

/// The surfaces to evaluate with, and which of them stand in for a caller's.
struct Staged {
    surfaces: Vec<*mut c_void>,
    /// (index, layout, staging surface) for each restrided tensor.
    restrided: Vec<(usize, AneTensorLayout, *mut c_void)>,
}

/// Swap in a staging surface for each caller surface written packed against
/// a layout whose rows are padded. Everything else passes through, and
/// `check_surface_sizes` still reports a surface that's simply too small.
fn stage(
    surfaces: &[*mut c_void],
    layouts: &[AneTensorLayout],
    pool: &mut Vec<Option<crate::ane::iosurface::IoSurface>>,
) -> Result<Staged> {
    use crate::ane::iosurface::IoSurface;
    let mut staged = Staged {
        surfaces: surfaces.to_vec(),
        restrided: Vec::new(),
    };
    if surfaces.len() != layouts.len() {
        return Ok(staged);
    }
    pool.resize_with(surfaces.len(), || None);
    for (i, (&surface, layout)) in surfaces.iter().zip(layouts).enumerate() {
        let have = unsafe { IoSurface::declared_bytes(surface) };
        let packed_input =
            have < layout.byte_size() && have >= layout.packed_size() && layout.packed_size() > 0;
        if !(layout.is_padded()
            && layout.height >= 1
            && layout.depth == 1
            && layout.batches == 1
            && packed_input)
        {
            continue;
        }
        if pool[i]
            .as_ref()
            .is_none_or(|s| s.size_bytes() < layout.byte_size())
        {
            pool[i] = Some(IoSurface::new(layout.byte_size())?);
        }
        let ptr = pool[i]
            .as_ref()
            .map_or(std::ptr::null_mut(), IoSurface::as_ptr);
        staged.surfaces[i] = ptr;
        staged.restrided.push((i, *layout, ptr));
    }
    Ok(staged)
}

#[derive(Clone, Copy)]
enum Direction {
    /// Packed caller surface → padded staging surface.
    ToModel,
    /// Padded staging surface → packed caller surface.
    FromModel,
}

/// Copy a tensor between its packed form and the model's padded layout, one
/// row at a time.
///
/// # Safety
/// `src` and `dst` must be live surfaces: the packed one at least
/// `layout.packed_size()` bytes, the padded one `layout.byte_size()`.
unsafe fn restride(
    src: *mut c_void,
    dst: *mut c_void,
    layout: &AneTensorLayout,
    direction: Direction,
) {
    let row_bytes = layout.width * layout.element_bytes;
    let rows = layout.channels * layout.height;
    let (height, plane, row) = (layout.height, layout.plane_stride, layout.row_stride);
    let packed = move |r: usize| r * row_bytes;
    let padded = move |r: usize| (r / height) * plane + (r % height) * row;
    unsafe {
        match direction {
            Direction::ToModel => crate::ane::iosurface::IoSurface::copy_rows(
                src, dst, rows, row_bytes, packed, padded,
            ),
            Direction::FromModel => crate::ane::iosurface::IoSurface::copy_rows(
                src, dst, rows, row_bytes, padded, packed,
            ),
        }
    }
}

/// Fail before evaluation, with the sizes, when a surface is smaller than the
/// model's layout for it. The ANE's own error for this (Code=42) names
/// neither the tensor nor the size it wanted.
fn check_surface_sizes(
    kind: &str,
    surfaces: &[*mut c_void],
    layouts: &[AneTensorLayout],
) -> Result<()> {
    if surfaces.len() != layouts.len() {
        return Ok(());
    }
    for (i, (&surface, layout)) in surfaces.iter().zip(layouts).enumerate() {
        let have = unsafe { crate::ane::iosurface::IoSurface::declared_bytes(surface) };
        let need = layout.byte_size();
        if have < need {
            return Err(MetalError::AneEvalFailed(format!(
                "{kind} {i}: surface is {have} bytes but the model expects {need} \
                 ({} channels x {} wide, row stride {} bytes)",
                layout.channels, layout.width, layout.row_stride
            )));
        }
    }
    Ok(())
}

/// Path of the single weight file [`pack_weights`] produces.
const PACKED_WEIGHTS_PATH: &str = "@model_path/weights/weight.bin";

/// Merge a kernel's one-blob-per-file weights into one file and point the
/// MIL's `BLOBFILE` references at each blob's offset in it.
///
/// On macOS 27 the descriptor hashes the weight entries in one order and
/// `ANECompilerService` rehashes the files in the bundle in another, so a
/// kernel with several distinct weight files fails `verifyBundleAtPath` with
/// a hash mismatch (Code=10; #34). With one file there is no order. The
/// layout is the one CoreML's weight.bin uses, which the blobs already follow
/// individually: a 64-byte header (blob count, version 2), then per blob a
/// 64-byte metadata record (0xDEADBEEF, dtype, size, data offset) and its
/// data, each record 64-byte aligned. `BLOBFILE(offset=…)` names the record.
///
/// Returns `None`, leaving the inputs as they are, when there is at most one
/// weight file or a blob or reference isn't in the expected form.
fn pack_weights(mil_text: &str, weights: &WeightDict) -> Option<(String, WeightDict)> {
    const HEADER: usize = 64;
    const RECORD: usize = 64;
    if weights.entries.len() < 2 {
        return None;
    }
    let align = |n: usize| n.div_ceil(64) * 64;
    let u32_at = |b: &[u8], at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());

    let mut entries: Vec<&(String, Vec<u8>)> = weights.entries.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut packed = vec![0u8; HEADER];
    packed[0..4].copy_from_slice(&(entries.len() as u32).to_le_bytes());
    packed[4..8].copy_from_slice(&2u32.to_le_bytes());
    let mut mil = mil_text.to_string();

    for (path, blob) in entries {
        // One blob per file: count 1, its record at 64, data at 128.
        if blob.len() < HEADER + RECORD
            || u32_at(blob, 0) != 1
            || u32_at(blob, 64) != 0xDEAD_BEEF
            || u32_at(blob, 80) != (HEADER + RECORD) as u32
        {
            return None;
        }
        let size = u32_at(blob, 72) as usize;
        if HEADER + RECORD + size > blob.len() {
            return None;
        }
        let reference = format!("BLOBFILE(path=string(\"{path}\"), offset=uint64(64))");
        if !mil.contains(&reference) {
            return None;
        }

        let record_at = packed.len();
        // Within the record: size at +8, data offset at +16 (file offsets 72, 80).
        let mut record = blob[HEADER..HEADER + RECORD].to_vec();
        record[8..16].copy_from_slice(&(size as u64).to_le_bytes());
        record[16..24].copy_from_slice(&((record_at + RECORD) as u64).to_le_bytes());
        packed.extend_from_slice(&record);
        packed.extend_from_slice(&blob[HEADER + RECORD..HEADER + RECORD + size]);
        packed.resize(align(packed.len()), 0);

        mil = mil.replace(
            &reference,
            &format!(
                "BLOBFILE(path=string(\"{PACKED_WEIGHTS_PATH}\"), offset=uint64({record_at}))"
            ),
        );
    }

    let mut out = WeightDict::new();
    out.add(PACKED_WEIGHTS_PATH, packed);
    Some((mil, out))
}

// ============================================================================
// Helper functions
// ============================================================================

/// Resolve a class by name via NSClassFromString.
fn resolve_class(name: &CStr) -> std::result::Result<&'static AnyClass, MetalError> {
    AnyClass::get(name).ok_or(MetalError::AneNotAvailable)
}

/// Get a Rust string from an NSString pointer.
///
/// # Safety
/// `obj` must be a valid NSString pointer or null.
unsafe fn ns_string_to_rust(obj: *const AnyObject) -> String {
    if obj.is_null() {
        return String::new();
    }
    let utf8: *const std::ffi::c_char = msg_send![obj, UTF8String];
    if utf8.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(utf8) }
        .to_string_lossy()
        .into_owned()
}

/// Get the description string from an NSError.
///
/// # Safety
/// `error` must be a valid NSError pointer.
unsafe fn ns_error_description(error: *mut NSError) -> String {
    let desc: *const AnyObject = msg_send![error, description];
    unsafe { ns_string_to_rust(desc) }
}

/// Write a kernel's MIL and weight files into `dir`, where the compiler reads
/// them, and compile it.
///
/// # Safety
/// `model` must be a valid `_ANEInMemoryModel` pointer.
unsafe fn stage_and_compile(
    model: *mut AnyObject,
    dir: &str,
    mil_data: &NSData,
    weight_dict: Option<&WeightDict>,
) -> Result<()> {
    let fm = NSFileManager::defaultManager();
    let weights_dir = NSString::from_str(&format!("{dir}/weights"));
    let _: Bool = msg_send![
        &*fm,
        createDirectoryAtPath: &*weights_dir,
        withIntermediateDirectories: Bool::YES,
        attributes: std::ptr::null::<AnyObject>(),
        error: std::ptr::null_mut::<*mut NSError>()
    ];

    let mil_path = NSString::from_str(&format!("{dir}/model.mil"));
    let _: Bool = msg_send![mil_data, writeToFile: &*mil_path, atomically: Bool::YES];

    for (name, data) in weight_dict.map_or(&[][..], |wd| &wd.entries[..]) {
        let rel = name.replace("@model_path/", "");
        // Reject path traversal attempts in weight key names.
        if rel.contains("..") || rel.starts_with('/') || rel.starts_with('\\') || rel.contains('\0')
        {
            return Err(MetalError::InvalidConfig(format!(
                "Invalid weight key (path traversal attempt): {name:?}"
            )));
        }
        let path = NSString::from_str(&format!("{dir}/{rel}"));
        let ns_data = NSData::with_bytes(data);
        let _: Bool = msg_send![&*ns_data, writeToFile: &*path, atomically: Bool::YES];
    }

    let mut error: *mut NSError = std::ptr::null_mut();
    let empty_dict = empty_options_dict();
    let ok: Bool = msg_send![
        model,
        compileWithQoS: ANE_QOS,
        options: &*empty_dict,
        error: &mut error
    ];
    if !ok.as_bool() {
        return Err(MetalError::AneCompileFailed(unsafe {
            error_message(error)
        }));
    }
    Ok(())
}

/// Load a compiled kernel onto the ANE.
///
/// # Safety
/// `model` must be a valid `_ANEInMemoryModel` pointer.
unsafe fn load_model(model: *mut AnyObject) -> Result<()> {
    let mut error: *mut NSError = std::ptr::null_mut();
    let empty_dict = empty_options_dict();
    let ok: Bool = msg_send![
        model,
        loadWithQoS: ANE_QOS,
        options: &*empty_dict,
        error: &mut error
    ];
    if !ok.as_bool() {
        return Err(MetalError::AneLoadFailed(unsafe { error_message(error) }));
    }
    Ok(())
}

/// The description of `error`, which may be null.
///
/// # Safety
/// `error` must be null or a valid NSError pointer.
unsafe fn error_message(error: *mut NSError) -> String {
    if error.is_null() {
        "unknown error".to_string()
    } else {
        unsafe { ns_error_description(error) }
    }
}

/// The directory a kernel compiles in: its MIL, its weights, and the
/// framework's own copy of them, so about twice the kernel's weight bytes.
/// Removed when dropped. A loaded model no longer reads it (it evaluates,
/// unloads and reloads without it), so it's dropped as soon as `compile`
/// returns. Keeping it for the model's lifetime put two copies of every
/// layer's weights on the boot volume, which filled it partway through a 4B
/// model.
struct CompileDir(String);

impl Drop for CompileDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Whether a request may carry an `_ANEPerformanceStats`. On macOS 27,
/// `-[_ANERequest validate]` sends `-count` to the `perfStats` argument, so the
/// single stats object earlier releases take raises an Objective-C exception
/// Rust can't catch, and ANE training aborted on its first step (#34). The
/// shape it now expects isn't known, so stats aren't requested there;
/// evaluation works without them and hardware time reads as 0.
fn perf_stats_request_supported() -> bool {
    !macos_27_or_later()
}

/// macOS 27 changed the private ANE API in several places (#34). Behavior
/// keyed to it is limited to 27+, because earlier releases can't be tested
/// here and their existing path is known to work.
fn macos_27_or_later() -> bool {
    static AT_LEAST_27: OnceLock<bool> = OnceLock::new();
    *AT_LEAST_27.get_or_init(|| {
        objc2_foundation::NSProcessInfo::processInfo()
            .operatingSystemVersion()
            .majorVersion
            >= 27
    })
}

fn empty_options_dict() -> objc2::rc::Retained<NSDictionary<NSString, AnyObject>> {
    NSDictionary::<NSString, AnyObject>::new()
}

/// Build an NSArray from raw AnyObject pointers.
///
/// # Safety
/// All pointers in `items` must be valid ObjC objects.
unsafe fn ns_array_from_raw(items: &[*mut AnyObject]) -> objc2::rc::Retained<NSArray<AnyObject>> {
    let cls = objc2::runtime::AnyClass::get(c"NSMutableArray").unwrap();
    let arr: *mut AnyObject = msg_send![cls, arrayWithCapacity: items.len()];
    for &item in items {
        let _: () = msg_send![arr, addObject: item];
    }
    unsafe { objc2::rc::Retained::retain(arr as *mut NSArray<AnyObject>).unwrap() }
}

/// Build an NSArray from NSNumber references.
///
/// # Safety
/// This function performs ObjC message sends.
unsafe fn ns_array_from_numbers(
    items: &[objc2::rc::Retained<NSNumber>],
) -> objc2::rc::Retained<NSArray<AnyObject>> {
    let cls = objc2::runtime::AnyClass::get(c"NSMutableArray").unwrap();
    let arr: *mut AnyObject = msg_send![cls, arrayWithCapacity: items.len()];
    for item in items {
        let _: () = msg_send![arr, addObject: &**item];
    }
    unsafe { objc2::rc::Retained::retain(arr as *mut NSArray<AnyObject>).unwrap() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ane::{iosurface::IoSurface, mil::MilProgram};

    fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn test_ane_runtime_global() {
        // On M1+ hardware this should succeed; on CI/non-Apple it returns AneNotAvailable
        let result = AneRuntime::global();
        match result {
            Ok(rt) => {
                // Successfully loaded — verify classes are non-null
                assert!(!std::ptr::eq(
                    rt.descriptor_class as *const _,
                    std::ptr::null()
                ));
            }
            Err(MetalError::AneNotAvailable) => {
                // Expected on non-Apple hardware or missing framework
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_weight_dict() {
        let mut wd = WeightDict::new();
        wd.add("@model_path/weights/test.bin", vec![0u8; 256]);
        assert_eq!(wd.entries.len(), 1);
        assert_eq!(wd.entries[0].0, "@model_path/weights/test.bin");
    }

    #[test]
    fn test_runtime_capability_flags_are_consistent() {
        let result = AneRuntime::global();
        match result {
            Ok(rt) => {
                assert_eq!(
                    rt.perf_stats_available(),
                    rt.perf_stats_class.is_some() && perf_stats_request_supported()
                );
            }
            Err(MetalError::AneNotAvailable) => {}
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    fn two_conv_kernel(w1: &[f32], w2: &[f32], c: usize, sp: usize) -> (String, WeightDict) {
        use crate::ane::kernel::WeightBlob;
        let mut p = MilProgram::new(c, sp);
        p.emit_conv_constants();
        let mut wd = WeightDict::new();
        for (name, w) in [("w1", w1), ("w2", w2)] {
            let path = format!("@model_path/weights/{name}.bin");
            p.emit_weight_const(name, &[c, c, 1, 1], &path);
            wd.add(&path, WeightBlob::from_f32(w, c, c));
        }
        p.emit_conv("y", &[1, c, 1, sp], "w1", "x");
        p.emit_conv("z", &[1, c, 1, sp], "w2", "y");
        (p.finalize("z"), wd)
    }

    #[test]
    fn pack_weights_lays_blobs_out_like_coreml_weight_bin() {
        let (c, sp) = (8, 32);
        let (mil, wd) = two_conv_kernel(&[0.5; 64], &[0.25; 64], c, sp);
        let (packed_mil, packed) = pack_weights(&mil, &wd).expect("two blobs pack");

        assert_eq!(packed.entries.len(), 1);
        let (path, file) = &packed.entries[0];
        assert_eq!(path, PACKED_WEIGHTS_PATH);
        let u32_at = |at: usize| u32::from_le_bytes(file[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(file[at..at + 8].try_into().unwrap());
        assert_eq!((u32_at(0), u32_at(4)), (2, 2), "count, version");

        // Each record: magic, size, data offset just past the record, data
        // copied, and the MIL pointing at it.
        let data_bytes = c * c * 2;
        for (record_at, name) in [(64usize, "w1"), (64 + 64 + data_bytes, "w2")] {
            assert_eq!(record_at % 64, 0);
            assert_eq!(u32_at(record_at), 0xDEAD_BEEF);
            assert_eq!(u64_at(record_at + 8) as usize, data_bytes);
            assert_eq!(u64_at(record_at + 16) as usize, record_at + 64);
            let original = &wd.entries.iter().find(|(p, _)| p.contains(name)).unwrap().1;
            assert_eq!(
                &file[record_at + 64..record_at + 64 + data_bytes],
                &original[128..]
            );
            assert!(packed_mil.contains(&format!(
                "BLOBFILE(path=string(\"{PACKED_WEIGHTS_PATH}\"), offset=uint64({record_at}))"
            )));
        }
        assert!(!packed_mil.contains("w1.bin") && !packed_mil.contains("w2.bin"));
    }

    #[test]
    fn pack_weights_leaves_what_it_cant_handle() {
        let (mil, wd) = two_conv_kernel(&[0.5; 64], &[0.25; 64], 8, 32);
        let mut one = WeightDict::new();
        one.add(&wd.entries[0].0, wd.entries[0].1.clone());
        assert!(
            pack_weights(&mil, &one).is_none(),
            "a single file needs no packing"
        );
        assert!(
            pack_weights("no references here", &wd).is_none(),
            "a blob the MIL doesn't reference at offset 64"
        );
        let mut bad = WeightDict::new();
        bad.add(&wd.entries[0].0, vec![0u8; 200]);
        bad.add(&wd.entries[1].0, wd.entries[1].1.clone());
        assert!(
            pack_weights(&mil, &bad).is_none(),
            "a blob without the header"
        );
    }

    /// Packed weights must evaluate to the same function: two layers with
    /// distinct weights against a CPU reference. At width 32 the fp16 rows
    /// are exactly 64 bytes; at 12 they're 24, which macOS 27 pads to 64, so
    /// the surfaces go through AneModel's restriding both ways.
    #[test]
    #[ignore = "requires ANE hardware and private AppleNeuralEngine.framework"]
    fn test_multi_weight_kernel_evaluates_correctly() {
        for sp in [32usize, 12] {
            multi_weight_kernel_case(sp);
        }
    }

    fn multi_weight_kernel_case(sp: usize) {
        let rt = match AneRuntime::global() {
            Ok(rt) => rt,
            Err(MetalError::AneNotAvailable) => return,
            Err(e) => panic!("Unexpected error: {e}"),
        };
        let c = 8usize;
        let w1: Vec<f32> = (0..c * c).map(|k| ((k % 7) as f32 - 3.0) * 0.05).collect();
        let w2: Vec<f32> = (0..c * c).map(|k| ((k % 5) as f32 - 2.0) * 0.07).collect();
        let (mil, wd) = two_conv_kernel(&w1, &w2, c, sp);
        let model = rt
            .compile(mil.as_bytes(), Some(&wd))
            .expect("a kernel with several weight files compiles");

        let x: Vec<f32> = (0..c * sp).map(|k| ((k % 11) as f32 - 5.0) * 0.1).collect();
        let input = IoSurface::for_tensor(c, sp).unwrap();
        let output = IoSurface::for_tensor(c, sp).unwrap();
        input.write_f32_as_fp16(&x, c, sp);
        model
            .evaluate(&[input.as_ptr()], &[output.as_ptr()])
            .expect("ANE evaluation");
        let mut got = vec![0.0f32; c * sp];
        output.read_fp16_as_f32(&mut got, 0, c, sp);

        let conv = |w: &[f32], x: &[f32]| {
            let mut y = vec![0.0f32; c * sp];
            for o in 0..c {
                for t in 0..sp {
                    y[o * sp + t] = (0..c).map(|i| w[o * c + i] * x[i * sp + t]).sum();
                }
            }
            y
        };
        let expected = conv(&w2, &conv(&w1, &x));
        let diff = max_abs_diff(&got, &expected);
        assert!(diff < 1e-3, "width {sp}: max |ane - cpu| = {diff}");
    }

    #[test]
    #[ignore = "requires ANE hardware and private AppleNeuralEngine.framework"]
    fn test_second_compile_of_a_kernel_loads_from_cache() {
        let rt = match AneRuntime::global() {
            Ok(rt) => rt,
            Err(MetalError::AneNotAvailable) => return,
            Err(e) => panic!("Unexpected error: {e}"),
        };
        let (c, sp) = (8usize, 32usize);
        // Weights no earlier run compiled, so the first compile can't hit
        // the cache.
        let mut stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let mut w1: Vec<f32> = (0..c * c).map(|k| ((k % 7) as f32 - 3.0) * 0.05).collect();
        for w in &mut w1[..8] {
            *w = (stamp % 100) as f32 * 0.01;
            stamp /= 100;
        }
        let w2: Vec<f32> = (0..c * c).map(|k| ((k % 5) as f32 - 2.0) * 0.07).collect();
        let (mil, wd) = two_conv_kernel(&w1, &w2, c, sp);

        let first = rt.compile(mil.as_bytes(), Some(&wd)).unwrap();
        let second = rt.compile(mil.as_bytes(), Some(&wd)).unwrap();
        assert!(!first.from_cache(), "a new kernel was compiled");
        assert!(second.from_cache(), "the same kernel again was loaded");

        let x: Vec<f32> = (0..c * sp).map(|k| ((k % 11) as f32 - 5.0) * 0.1).collect();
        let input = IoSurface::for_tensor(c, sp).unwrap();
        input.write_f32_as_fp16(&x, c, sp);
        let run = |model: &AneModel| {
            let output = IoSurface::for_tensor(c, sp).unwrap();
            model
                .evaluate(&[input.as_ptr()], &[output.as_ptr()])
                .unwrap();
            let mut y = vec![0.0f32; c * sp];
            output.read_fp16_as_f32(&mut y, 0, c, sp);
            y
        };
        let y = run(&first);
        assert!(y.iter().any(|v| *v != 0.0));
        assert_eq!(y, run(&second));
    }

    #[test]
    #[ignore = "requires ANE hardware and private AppleNeuralEngine.framework"]
    fn test_f32_evaluation_round_trips() {
        let rt = match AneRuntime::global() {
            Ok(rt) => rt,
            Err(MetalError::AneNotAvailable) => return,
            Err(e) => panic!("Unexpected error: {e}"),
        };

        let mut program = MilProgram::new_fp32(1, 4);
        program.emit_cast("x16", &[1, 1, 1, 4], "x", "fp16");
        program.emit_cast("out", &[1, 1, 1, 4], "x16", "fp32");
        let mil_text = program.finalize("out");

        let model = match rt.compile(mil_text.as_bytes(), None) {
            Ok(model) => model,
            Err(e) => panic!("ANE is present but the test program failed to compile or load: {e}"),
        };

        // Sized to the bare 16-byte row. On macOS 27 the model pads rows to
        // 64 bytes, so this also exercises AneModel's restriding.
        let input = IoSurface::for_tensor_f32(1, 4).unwrap();
        let output = IoSurface::for_tensor_f32(1, 4).unwrap();
        let input_values = [1.5f32, -2.0, 0.25, 7.0];
        input.write_f32_at(0, &input_values, 1, 4);

        // An ANE is present, so a failed evaluation is a failure, not a skip:
        // skipping here is how Code=42 on macOS 27 passed unnoticed (#34).
        model
            .evaluate(&[input.as_ptr()], &[output.as_ptr()])
            .expect("ANE evaluation");
        let mut values = [0.0f32; 4];
        output.read_f32(&mut values, 0, 1, 4);
        // The program casts through fp16, which represents these exactly.
        assert_eq!(values, input_values, "evaluation round-trips");
    }
}
