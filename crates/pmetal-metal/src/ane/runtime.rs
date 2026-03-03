//! ANE private API FFI via dlopen + objc2.
//!
//! Wraps the four private ObjC classes from `AppleNeuralEngine.framework`:
//! - `_ANEInMemoryModelDescriptor`
//! - `_ANEInMemoryModel`
//! - `_ANERequest`
//! - `_ANEIOSurfaceObject`
//!
//! The framework is loaded at runtime via `dlopen` (not linked at build time).
//! Classes are resolved via `NSClassFromString`. Returns `AneNotAvailable`
//! gracefully if the framework is missing.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;

use objc2::msg_send;
use objc2::runtime::{AnyClass, AnyObject, Bool};
use objc2_foundation::{NSArray, NSData, NSDictionary, NSError, NSFileManager, NSNumber, NSString};

use crate::error::{MetalError, Result};

// dlopen FFI — avoid libc dependency
const RTLD_NOW: c_int = 0x2;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
}

/// QoS constant used for all ANE operations. Value 21 = userInteractive.
/// The ANE reference confirmed this has no latency impact vs other values.
const ANE_QOS: u32 = 21;

/// Global ANE runtime singleton.
static ANE_RUNTIME: OnceLock<std::result::Result<AneRuntime, MetalError>> = OnceLock::new();

/// Safe wrapper around the ANE private API runtime.
///
/// Holds references to the four private ObjC classes needed for
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
    /// `_ANEChainingRequest` — probe-only, None if unavailable.
    /// Available on M4/M5 hardware but no known working invocation exists.
    chaining_class: Option<&'static AnyClass>,
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

        // Probe for chaining API (M4/M5 only, no working invocation known)
        let chaining_class = AnyClass::get(c"_ANEChainingRequest");
        if chaining_class.is_some() {
            tracing::info!("ANE chaining API (_ANEChainingRequest) detected — available for future research");
        } else {
            tracing::debug!("ANE chaining API not available on this hardware");
        }

        Ok(AneRuntime {
            descriptor_class,
            model_class,
            request_class,
            io_surface_class,
            chaining_class,
        })
    }

    /// Compile a MIL program with weights into an ANE model.
    ///
    /// This performs the full pipeline: descriptor → model → compile → load.
    /// The returned `AneModel` implements `Drop` for RAII cleanup.
    pub fn compile(&self, mil_text: &[u8], weight_dict: Option<&WeightDict>) -> Result<AneModel> {
        // SAFETY: All ObjC message sends use valid class/object pointers obtained
        // from the framework. Memory management follows ObjC retain/release rules.
        unsafe {
            let mil_data = NSData::with_bytes(mil_text);

            // Build weight dictionary (or nil)
            let wdict_obj: Option<objc2::rc::Retained<NSDictionary<NSString, AnyObject>>> =
                weight_dict.map(|wd| wd.to_ns_dict());

            // Create descriptor: modelWithMILText:weights:optionsPlist:
            let wdict_ptr: *const AnyObject = match &wdict_obj {
                Some(d) => d.as_ref() as *const _,
                None => std::ptr::null(),
            };

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

            // Pre-populate temp directory with MIL + weights
            let fm = NSFileManager::defaultManager();
            let weights_dir = format!("{}/weights", tmp_dir_str);
            let weights_dir_ns = NSString::from_str(&weights_dir);
            let _: Bool = msg_send![
                &*fm,
                createDirectoryAtPath: &*weights_dir_ns,
                withIntermediateDirectories: Bool::YES,
                attributes: std::ptr::null::<AnyObject>(),
                error: std::ptr::null_mut::<*mut NSError>()
            ];

            // Write MIL text
            let mil_path = format!("{}/model.mil", tmp_dir_str);
            let mil_path_ns = NSString::from_str(&mil_path);
            let _: Bool = msg_send![&*mil_data, writeToFile: &*mil_path_ns, atomically: Bool::YES];

            // Write weight files
            if let Some(wd) = weight_dict {
                for (name, data) in &wd.entries {
                    let rel = name.replace("@model_path/", "");
                    let path = format!("{}/{}", tmp_dir_str, rel);
                    let path_ns = NSString::from_str(&path);
                    let ns_data = NSData::with_bytes(data);
                    let _: Bool =
                        msg_send![&*ns_data, writeToFile: &*path_ns, atomically: Bool::YES];
                }
            }

            // Compile: compileWithQoS:options:error:
            let mut error: *mut NSError = std::ptr::null_mut();
            let empty_dict = NSDictionary::<NSString, AnyObject>::new();
            let ok: Bool = msg_send![
                model,
                compileWithQoS: ANE_QOS,
                options: &*empty_dict,
                error: &mut error
            ];
            if !ok.as_bool() {
                let msg = if !error.is_null() {
                    ns_error_description(error)
                } else {
                    "unknown error".to_string()
                };
                cleanup_tmp(&tmp_dir_str);
                return Err(MetalError::AneCompileFailed(msg));
            }

            // Load: loadWithQoS:options:error:
            error = std::ptr::null_mut();
            let ok: Bool = msg_send![
                model,
                loadWithQoS: ANE_QOS,
                options: &*empty_dict,
                error: &mut error
            ];
            if !ok.as_bool() {
                let msg = if !error.is_null() {
                    ns_error_description(error)
                } else {
                    "unknown error".to_string()
                };
                cleanup_tmp(&tmp_dir_str);
                return Err(MetalError::AneLoadFailed(msg));
            }

            // Retain the model object
            let _: *mut AnyObject = msg_send![model, retain];

            Ok(AneModel {
                model,
                request_class: self.request_class,
                io_surface_class: self.io_surface_class,
                tmp_dir: PathBuf::from(&tmp_dir_str),
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

    /// Check if the ANE chaining API is available (M4/M5 hardware).
    ///
    /// Returns `Some` if `_ANEChainingRequest` was resolved. No working
    /// invocation pattern exists yet — this is telemetry for future research.
    pub fn chaining_available(&self) -> bool {
        self.chaining_class.is_some()
    }
}

/// A compiled ANE model ready for evaluation.
///
/// Implements `Drop` for RAII: unloads from ANE hardware and cleans up temp directory.
pub struct AneModel {
    model: *mut AnyObject,
    request_class: &'static AnyClass,
    io_surface_class: &'static AnyClass,
    tmp_dir: PathBuf,
}

// SAFETY: ANE model objects are thread-safe for evaluation dispatch.
unsafe impl Send for AneModel {}
unsafe impl Sync for AneModel {}

impl AneModel {
    /// Build a request and evaluate the model.
    ///
    /// `inputs` and `outputs` are IOSurface references for data transfer.
    pub fn evaluate(&self, inputs: &[*mut c_void], outputs: &[*mut c_void]) -> Result<()> {
        // SAFETY: IOSurface pointers are valid kernel objects; ObjC message sends
        // target classes resolved during runtime init.
        unsafe {
            // Wrap IOSurfaces as _ANEIOSurfaceObject instances
            let mut wrapped_inputs: Vec<*mut AnyObject> = Vec::with_capacity(inputs.len());
            let mut input_indices: Vec<objc2::rc::Retained<NSNumber>> =
                Vec::with_capacity(inputs.len());
            for (i, &surface) in inputs.iter().enumerate() {
                let wrapped: *mut AnyObject = msg_send![
                    self.io_surface_class,
                    objectWithIOSurface: surface
                ];
                wrapped_inputs.push(wrapped);
                input_indices.push(NSNumber::new_usize(i));
            }

            let mut wrapped_outputs: Vec<*mut AnyObject> = Vec::with_capacity(outputs.len());
            let mut output_indices: Vec<objc2::rc::Retained<NSNumber>> =
                Vec::with_capacity(outputs.len());
            for (i, &surface) in outputs.iter().enumerate() {
                let wrapped: *mut AnyObject = msg_send![
                    self.io_surface_class,
                    objectWithIOSurface: surface
                ];
                wrapped_outputs.push(wrapped);
                output_indices.push(NSNumber::new_usize(i));
            }

            // Build NSArrays
            let ns_inputs = ns_array_from_raw(&wrapped_inputs);
            let ns_input_idx = ns_array_from_numbers(&input_indices);
            let ns_outputs = ns_array_from_raw(&wrapped_outputs);
            let ns_output_idx = ns_array_from_numbers(&output_indices);
            let zero = NSNumber::new_usize(0);

            // Build request
            let request: *mut AnyObject = msg_send![
                self.request_class,
                requestWithInputs: &*ns_inputs,
                inputIndices: &*ns_input_idx,
                outputs: &*ns_outputs,
                outputIndices: &*ns_output_idx,
                weightsBuffer: std::ptr::null::<AnyObject>(),
                perfStats: std::ptr::null::<AnyObject>(),
                procedureIndex: &*zero
            ];

            // Evaluate
            let mut error: *mut NSError = std::ptr::null_mut();
            let empty_dict = NSDictionary::<NSString, AnyObject>::new();
            let ok: Bool = msg_send![
                self.model,
                evaluateWithQoS: ANE_QOS,
                options: &*empty_dict,
                request: request,
                error: &mut error
            ];

            if !ok.as_bool() {
                let msg = if !error.is_null() {
                    ns_error_description(error)
                } else {
                    "unknown error".to_string()
                };
                return Err(MetalError::AneEvalFailed(msg));
            }

            Ok(())
        }
    }

    /// Unload the model from ANE hardware.
    fn unload(&self) {
        unsafe {
            let mut error: *mut NSError = std::ptr::null_mut();
            let _: Bool = msg_send![
                self.model,
                unloadWithQoS: ANE_QOS,
                error: &mut error
            ];
        }
    }
}

impl Drop for AneModel {
    fn drop(&mut self) {
        self.unload();
        cleanup_tmp(&self.tmp_dir.to_string_lossy());
        unsafe {
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
        unsafe {
            let dict: *mut AnyObject = msg_send![
                objc2::runtime::AnyClass::get(c"NSMutableDictionary").unwrap(),
                new
            ];

            for (path, data) in &self.entries {
                let key = NSString::from_str(path);
                let ns_data = NSData::with_bytes(data);

                // Build inner dict: { "offset": @0, "data": ns_data }
                let inner: *mut AnyObject = msg_send![
                    objc2::runtime::AnyClass::get(c"NSMutableDictionary").unwrap(),
                    new
                ];
                let offset_key = NSString::from_str("offset");
                let data_key = NSString::from_str("data");
                let zero = NSNumber::new_i32(0);

                let _: () = msg_send![inner, setObject: &*zero, forKey: &*offset_key];
                let _: () = msg_send![inner, setObject: &*ns_data, forKey: &*data_key];
                let _: () = msg_send![dict, setObject: inner, forKey: &*key];
            }

            objc2::rc::Retained::retain(dict as *mut NSDictionary<NSString, AnyObject>).unwrap()
        }
    }
}

impl Default for WeightDict {
    fn default() -> Self {
        Self::new()
    }
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

/// Clean up a temp directory.
fn cleanup_tmp(path: &str) {
    let _ = std::fs::remove_dir_all(path);
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
}
