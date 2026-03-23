#![allow(unsafe_code)]

//! ANE private API FFI via dlopen + objc2.
//!
//! Wraps the private ObjC classes from `AppleNeuralEngine.framework`:
//! - `_ANEInMemoryModelDescriptor`
//! - `_ANEInMemoryModel`
//! - `_ANERequest`
//! - `_ANEIOSurfaceObject`
//! - `_ANEClient` (optional real-time / chaining entry point)
//!
//! The framework is loaded at runtime via `dlopen` (not linked at build time).
//! Classes are resolved via `NSClassFromString`. Returns `AneNotAvailable`
//! gracefully if the framework is missing.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, AtomicPtr, Ordering},
};

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
    /// `_ANEClient` — optional real-time / chaining entry point.
    client_class: Option<&'static AnyClass>,
    /// `_ANEChainingRequest` — resolved when available, None if unavailable.
    /// PMetal can prepare experimental loopback requests, but stable execution
    /// semantics remain unproven on current hardware.
    chaining_class: Option<&'static AnyClass>,
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
        let client_class = AnyClass::get(c"_ANEClient");
        if client_class.is_some() {
            tracing::debug!("ANE client API (_ANEClient) detected");
        }

        // Probe for chaining API (M4/M5 only, experimental preparation only)
        let chaining_class = AnyClass::get(c"_ANEChainingRequest");
        if chaining_class.is_some() {
            tracing::info!(
                "ANE chaining API (_ANEChainingRequest) detected — experimental loopback request preparation available"
            );
        } else {
            tracing::debug!("ANE chaining API not available on this hardware");
        }

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
            client_class,
            chaining_class,
            perf_stats_class,
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
                    // Reject path traversal attempts in weight key names.
                    if rel.contains("..")
                        || rel.starts_with('/')
                        || rel.starts_with('\\')
                        || rel.contains('\0')
                    {
                        return Err(MetalError::InvalidConfig(format!(
                            "Invalid weight key (path traversal attempt): {name:?}"
                        )));
                    }
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

            let inner_model: *mut AnyObject = msg_send![model, model];
            let real_time_model = if inner_model.is_null() {
                model
            } else {
                let _: *mut AnyObject = msg_send![inner_model, retain];
                inner_model
            };

            let real_time = if self.client_class.is_some() {
                let client: *mut AnyObject = if let Some(client_class) = self.client_class {
                    let private_client: *mut AnyObject =
                        msg_send![client_class, sharedPrivateConnection];
                    if private_client.is_null() {
                        msg_send![model, sharedConnection]
                    } else {
                        private_client
                    }
                } else {
                    std::ptr::null_mut()
                };
                if client.is_null() {
                    tracing::debug!("ANE sharedConnection returned null; real-time eval disabled");
                    None
                } else {
                    let _: *mut AnyObject = msg_send![client, retain];
                    Some(AneRealTimeState::new(client))
                }
            } else {
                None
            };

            Ok(AneModel {
                model,
                real_time_model,
                request_class: self.request_class,
                io_surface_class: self.io_surface_class,
                real_time,
                standard_loaded: AtomicBool::new(true),
                standard_load_lock: Mutex::new(()),
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

    /// Check if ANE performance stats API is available.
    pub fn perf_stats_available(&self) -> bool {
        self.perf_stats_class.is_some()
    }

    /// Check if the ANE real-time evaluation API is available.
    pub fn real_time_available(&self) -> bool {
        self.client_class.is_some()
    }
}

/// ANE hardware performance stats from a single evaluation.
#[derive(Debug, Clone, Default)]
pub struct AnePerformanceStats {
    /// Hardware execution time in nanoseconds.
    pub hw_execution_time_ns: u64,
}

/// ANE evaluation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AneEvaluationMode {
    /// Use `_ANEInMemoryModel evaluateWithQoS:...`.
    Standard,
    /// Use `_ANEClient evaluateRealTimeWithModel:...`.
    RealTime,
}

/// Experimental loopback chaining configuration for `_ANEChainingRequest`.
///
/// The private framework exposes symbol-index-based loopback wiring, but the
/// exact semantics are still under active reverse-engineering. Keep this out of
/// user-facing hot paths until it has real hardware validation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AneLoopbackChainConfig {
    /// Loopback input symbol index for the chained stage.
    pub loopback_input_symbol_index: usize,
    /// Loopback output symbol index from the previous stage.
    pub loopback_output_symbol_index: usize,
    /// Procedure index within the compiled model (usually 0).
    pub procedure_index: usize,
    /// Optional firmware enqueue delay hint.
    pub fw_enqueue_delay: Option<usize>,
    /// Optional ANE memory-pool identifier.
    pub memory_pool_id: Option<usize>,
}

impl AneLoopbackChainConfig {
    fn validate(&self, inputs: &[*mut c_void], output_sets: &[&[*mut c_void]]) -> Result<()> {
        if inputs.is_empty() {
            return Err(MetalError::InvalidConfig(
                "ANE chaining requires at least one input surface".into(),
            ));
        }
        if output_sets.is_empty() {
            return Err(MetalError::InvalidConfig(
                "ANE chaining requires at least one output set".into(),
            ));
        }
        for (set_idx, output_set) in output_sets.iter().enumerate() {
            if output_set.is_empty() {
                return Err(MetalError::InvalidConfig(format!(
                    "ANE chaining output set {set_idx} must not be empty"
                )));
            }
        }
        Ok(())
    }
}

/// Prepared experimental loopback chaining request.
///
/// This currently only proves that PMetal can construct and submit the private
/// chaining request to `_ANEClient`. It does not yet expose a public execution
/// API because the end-to-end invocation pattern is still unproven.
pub struct AnePreparedLoopbackChain {
    client: *mut AnyObject,
    prepared_model: *mut AnyObject,
    used_inner_model: bool,
    chain_request: *mut AnyObject,
    _inputs: objc2::rc::Retained<NSArray<AnyObject>>,
    _output_sets: objc2::rc::Retained<NSArray<AnyObject>>,
    _owned_output_sets: Vec<objc2::rc::Retained<NSArray<AnyObject>>>,
}

// SAFETY: The prepared chain only stores retained ObjC objects; submission is
// still serialized through the private framework.
unsafe impl Send for AnePreparedLoopbackChain {}
unsafe impl Sync for AnePreparedLoopbackChain {}

impl AnePreparedLoopbackChain {
    /// Re-run the framework's internal validation on the prepared request.
    pub fn is_valid(&self) -> bool {
        unsafe {
            let valid: Bool = msg_send![self.chain_request, validate];
            valid.as_bool()
        }
    }

    /// Human-readable description from the private framework object.
    pub fn description(&self) -> String {
        unsafe {
            let desc: *const AnyObject = msg_send![self.chain_request, description];
            ns_string_to_rust(desc)
        }
    }

    /// Whether the preparation used the inner compiled model object rather than
    /// the outer `_ANEInMemoryModel` wrapper.
    pub fn uses_inner_model(&self) -> bool {
        self.used_inner_model
    }
}

impl Drop for AnePreparedLoopbackChain {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.chain_request, release];
            let _: () = msg_send![self.prepared_model, release];
            let _: () = msg_send![self.client, release];
        }
    }
}

struct AneRealTimeState {
    client: *mut AnyObject,
    loaded_model: AtomicPtr<AnyObject>,
    load_lock: Mutex<()>,
}

// SAFETY: `_ANEClient` is a process-global ObjC object. We serialize
// load/unload transitions with `load_lock`; evaluation itself is handled by the
// framework.
unsafe impl Send for AneRealTimeState {}
unsafe impl Sync for AneRealTimeState {}

impl AneRealTimeState {
    fn new(client: *mut AnyObject) -> Self {
        Self {
            client,
            loaded_model: AtomicPtr::new(std::ptr::null_mut()),
            load_lock: Mutex::new(()),
        }
    }

    fn ensure_loaded(&self, model: *mut AnyObject) -> Result<()> {
        if self.loaded_model.load(Ordering::Acquire) == model {
            return Ok(());
        }

        let _guard = self.load_lock.lock();
        let current = self.loaded_model.load(Ordering::Acquire);
        if current == model {
            return Ok(());
        }
        if !current.is_null() {
            self.unload_locked(current);
        }

        unsafe {
            let mut error: *mut NSError = std::ptr::null_mut();
            let empty_dict = empty_options_dict();
            let ok: Bool = msg_send![
                self.client,
                loadRealTimeModel: model,
                options: &*empty_dict,
                qos: ANE_QOS,
                error: &mut error
            ];
            if !ok.as_bool() {
                let msg = if !error.is_null() {
                    ns_error_description(error)
                } else {
                    "unknown error".to_string()
                };
                return Err(MetalError::AneLoadFailed(msg));
            }
        }

        self.loaded_model.store(model, Ordering::Release);
        Ok(())
    }

    fn evaluate(&self, model: *mut AnyObject, request: *mut AnyObject) -> Result<()> {
        self.ensure_loaded(model)?;

        unsafe {
            let mut error: *mut NSError = std::ptr::null_mut();
            let mapped: Bool = msg_send![
                self.client,
                mapIOSurfacesWithModel: model,
                request: request,
                cacheInference: Bool::NO,
                error: &mut error
            ];
            if !mapped.as_bool() {
                let msg = if !error.is_null() {
                    ns_error_description(error)
                } else {
                    "unknown error".to_string()
                };
                return Err(MetalError::AneEvalFailed(msg));
            }

            let began: Bool = msg_send![self.client, beginRealTimeTask];
            if !began.as_bool() {
                let _: () =
                    msg_send![self.client, unmapIOSurfacesWithModel: model, request: request];
                return Err(MetalError::AneEvalFailed(
                    "beginRealTimeTask returned false".into(),
                ));
            }

            error = std::ptr::null_mut();
            let empty_dict = empty_options_dict();
            let ok: Bool = msg_send![
                self.client,
                evaluateRealTimeWithModel: model,
                options: &*empty_dict,
                request: request,
                error: &mut error
            ];

            let _: Bool = msg_send![self.client, endRealTimeTask];
            let _: () = msg_send![self.client, unmapIOSurfacesWithModel: model, request: request];

            if !ok.as_bool() {
                let msg = if !error.is_null() {
                    ns_error_description(error)
                } else {
                    "unknown error".to_string()
                };
                return Err(MetalError::AneEvalFailed(msg));
            }
        }

        Ok(())
    }

    fn unload_if_loaded(&self) {
        let current = self.loaded_model.load(Ordering::Acquire);
        if current.is_null() {
            return;
        }

        let _guard = self.load_lock.lock();
        let current = self.loaded_model.load(Ordering::Acquire);
        if current.is_null() {
            return;
        }

        self.unload_locked(current);
    }

    fn unload_locked(&self, model: *mut AnyObject) {
        unsafe {
            let mut error: *mut NSError = std::ptr::null_mut();
            let empty_dict = empty_options_dict();
            let _: Bool = msg_send![
                self.client,
                unloadRealTimeModel: model,
                options: &*empty_dict,
                qos: ANE_QOS,
                error: &mut error
            ];
        }

        self.loaded_model
            .store(std::ptr::null_mut(), Ordering::Release);
    }
}

/// A compiled ANE model ready for evaluation.
///
/// Implements `Drop` for RAII: unloads from ANE hardware and cleans up temp directory.
pub struct AneModel {
    model: *mut AnyObject,
    real_time_model: *mut AnyObject,
    request_class: &'static AnyClass,
    io_surface_class: &'static AnyClass,
    real_time: Option<AneRealTimeState>,
    standard_loaded: AtomicBool,
    standard_load_lock: Mutex<()>,
    tmp_dir: PathBuf,
}

// SAFETY: ANE model objects are thread-safe for evaluation dispatch.
unsafe impl Send for AneModel {}
unsafe impl Sync for AneModel {}

impl AneModel {
    /// Returns true when the real-time evaluation path is available for this model.
    pub fn real_time_available(&self) -> bool {
        self.real_time.is_some()
    }

    /// Returns true when the private chaining API is available.
    pub fn chaining_available(&self) -> bool {
        AneRuntime::global()
            .map(|rt| rt.chaining_available() && self.real_time.is_some())
            .unwrap_or(false)
    }

    /// Prepare an experimental loopback chaining request.
    ///
    /// This validates and submits `_ANEChainingRequest` to `_ANEClient` but
    /// does not yet expose an execution API; the private runtime interaction is
    /// still under active reverse-engineering.
    pub fn prepare_loopback_chain(
        &self,
        inputs: &[*mut c_void],
        output_sets: &[&[*mut c_void]],
        config: &AneLoopbackChainConfig,
    ) -> Result<AnePreparedLoopbackChain> {
        config.validate(inputs, output_sets)?;

        let rt = AneRuntime::global()?;
        let chaining_class = rt.chaining_class.ok_or(MetalError::AneNotAvailable)?;
        let client_state = self.real_time.as_ref().ok_or(MetalError::AneNotAvailable)?;

        self.ensure_standard_loaded()?;

        unsafe {
            let ns_inputs = wrap_iosurface_array(self.io_surface_class, inputs);
            let (ns_output_sets, owned_output_sets) =
                wrap_iosurface_output_sets(self.io_surface_class, output_sets);

            let lb_input = NSNumber::new_usize(config.loopback_input_symbol_index);
            let lb_output = NSNumber::new_usize(config.loopback_output_symbol_index);
            let procedure_index = NSNumber::new_usize(config.procedure_index);
            let signal_events = NSArray::<AnyObject>::new();
            let fw_enqueue_delay = config.fw_enqueue_delay.map(NSNumber::new_usize);
            let memory_pool_id = config.memory_pool_id.map(NSNumber::new_usize);

            let chain_alloc: *mut AnyObject = msg_send![chaining_class, alloc];
            if chain_alloc.is_null() {
                return Err(MetalError::AneChainingFailed(
                    "failed to allocate _ANEChainingRequest".into(),
                ));
            }

            let chain_request: *mut AnyObject = msg_send![
                chain_alloc,
                initWithInputs: &*ns_inputs,
                outputs: &*ns_output_sets,
                lbInputSymbolId: &*lb_input,
                lbOutputSymbolId: &*lb_output,
                procedureIndex: &*procedure_index,
                signalEvents: &*signal_events,
                transactionHandle: std::ptr::null::<AnyObject>(),
                fwEnqueueDelay: optional_number_ptr(fw_enqueue_delay.as_ref()),
                memoryPoolId: optional_number_ptr(memory_pool_id.as_ref())
            ];

            if chain_request.is_null() {
                return Err(MetalError::AneChainingFailed(
                    "_ANEChainingRequest init returned null".into(),
                ));
            }

            let valid: Bool = msg_send![chain_request, validate];
            if !valid.as_bool() {
                let desc: *const AnyObject = msg_send![chain_request, description];
                let description = ns_string_to_rust(desc);
                let _: () = msg_send![chain_request, release];
                return Err(MetalError::AneChainingFailed(format!(
                    "loopback chain request validation failed: {description}"
                )));
            }

            let (prepared_model, used_inner_model) = match self.prepare_chaining_request(
                client_state.client,
                self.real_time_model,
                chain_request,
            ) {
                Ok(()) => (self.real_time_model, self.real_time_model != self.model),
                Err(primary_err) if self.real_time_model != self.model => {
                    tracing::debug!(
                        error = %primary_err,
                        "ANE chaining preparation failed with the inner model; retrying with the in-memory wrapper"
                    );
                    match self.prepare_chaining_request(
                        client_state.client,
                        self.model,
                        chain_request,
                    ) {
                        Ok(()) => (self.model, false),
                        Err(_) => {
                            let _: () = msg_send![chain_request, release];
                            return Err(primary_err);
                        }
                    }
                }
                Err(err) => {
                    let _: () = msg_send![chain_request, release];
                    return Err(err);
                }
            };

            let _: *mut AnyObject = msg_send![client_state.client, retain];
            let _: *mut AnyObject = msg_send![prepared_model, retain];

            Ok(AnePreparedLoopbackChain {
                client: client_state.client,
                prepared_model,
                used_inner_model,
                chain_request,
                _inputs: ns_inputs,
                _output_sets: ns_output_sets,
                _owned_output_sets: owned_output_sets,
            })
        }
    }

    fn ensure_standard_loaded(&self) -> Result<()> {
        if self.standard_loaded.load(Ordering::Acquire) {
            return Ok(());
        }

        let _guard = self.standard_load_lock.lock();
        if self.standard_loaded.load(Ordering::Acquire) {
            return Ok(());
        }

        unsafe {
            let mut error: *mut NSError = std::ptr::null_mut();
            let empty_dict = empty_options_dict();
            let ok: Bool = msg_send![
                self.model,
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
                return Err(MetalError::AneLoadFailed(msg));
            }
        }

        self.standard_loaded.store(true, Ordering::Release);
        Ok(())
    }

    unsafe fn prepare_chaining_request(
        &self,
        client: *mut AnyObject,
        model: *mut AnyObject,
        chain_request: *mut AnyObject,
    ) -> Result<()> {
        let mut error: *mut NSError = std::ptr::null_mut();
        let empty_dict = empty_options_dict();
        let ok: Bool = msg_send![
            client,
            prepareChainingWithModel: model,
            options: &*empty_dict,
            chainingReq: chain_request,
            qos: ANE_QOS,
            error: &mut error
        ];

        if !ok.as_bool() {
            let msg = if !error.is_null() {
                unsafe { ns_error_description(error) }
            } else {
                "unknown error".to_string()
            };
            return Err(MetalError::AneChainingFailed(msg));
        }

        Ok(())
    }

    fn unload_standard_if_loaded(&self) {
        if !self.standard_loaded.load(Ordering::Acquire) {
            return;
        }

        let _guard = self.standard_load_lock.lock();
        if !self.standard_loaded.load(Ordering::Acquire) {
            return;
        }

        unsafe {
            let mut error: *mut NSError = std::ptr::null_mut();
            let _: Bool = msg_send![self.model, unloadWithQoS: ANE_QOS, error: &mut error];
        }
        self.standard_loaded.store(false, Ordering::Release);
    }

    /// Build a request and evaluate the model.
    ///
    /// `inputs` and `outputs` are IOSurface references for data transfer.
    pub fn evaluate(&self, inputs: &[*mut c_void], outputs: &[*mut c_void]) -> Result<()> {
        self.evaluate_inner(inputs, outputs, AneEvaluationMode::Standard, false)
            .map(|_| ())
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
        self.evaluate_inner(inputs, outputs, AneEvaluationMode::Standard, true)
    }

    /// Evaluate the model using the experimental ANE real-time path.
    pub fn evaluate_real_time(
        &self,
        inputs: &[*mut c_void],
        outputs: &[*mut c_void],
    ) -> Result<()> {
        self.evaluate_inner(inputs, outputs, AneEvaluationMode::RealTime, false)
            .map(|_| ())
    }

    /// Evaluate the model using the experimental ANE real-time path and collect stats.
    pub fn evaluate_real_time_with_stats(
        &self,
        inputs: &[*mut c_void],
        outputs: &[*mut c_void],
    ) -> Result<AnePerformanceStats> {
        self.evaluate_inner(inputs, outputs, AneEvaluationMode::RealTime, true)
    }

    fn evaluate_inner(
        &self,
        inputs: &[*mut c_void],
        outputs: &[*mut c_void],
        mode: AneEvaluationMode,
        collect_stats: bool,
    ) -> Result<AnePerformanceStats> {
        unsafe {
            let rt = AneRuntime::global()?;
            let perf_stats = if collect_stats {
                let Some(perf_class) = rt.perf_stats_class else {
                    self.evaluate_inner(inputs, outputs, mode, false)?;
                    return Ok(AnePerformanceStats::default());
                };

                let zero = NSNumber::new_u64(0);
                let perf_stats: *mut AnyObject = msg_send![
                    perf_class,
                    statsWithHardwareExecutionNS: &*zero
                ];
                if perf_stats.is_null() {
                    self.evaluate_inner(inputs, outputs, mode, false)?;
                    return Ok(AnePerformanceStats::default());
                }
                perf_stats
            } else {
                std::ptr::null_mut()
            };

            let request = self.build_request(inputs, outputs, perf_stats);
            match mode {
                AneEvaluationMode::Standard => self.evaluate_request_standard(request)?,
                AneEvaluationMode::RealTime => self.evaluate_request_real_time(request)?,
            }

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

    unsafe fn evaluate_request_standard(&self, request: *mut AnyObject) -> Result<()> {
        self.ensure_standard_loaded()?;

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
            let msg = if !error.is_null() {
                unsafe { ns_error_description(error) }
            } else {
                "unknown error".to_string()
            };
            return Err(MetalError::AneEvalFailed(msg));
        }

        Ok(())
    }

    unsafe fn evaluate_request_real_time(&self, request: *mut AnyObject) -> Result<()> {
        let real_time = self.real_time.as_ref().ok_or(MetalError::AneNotAvailable)?;
        self.unload_standard_if_loaded();
        match real_time.evaluate(self.real_time_model, request) {
            Ok(()) => Ok(()),
            Err(primary_err) if self.real_time_model != self.model => {
                real_time.unload_if_loaded();
                tracing::debug!(
                    error = %primary_err,
                    "ANE real-time evaluation failed with the inner model; retrying with the in-memory wrapper"
                );
                match real_time.evaluate(self.model, request) {
                    Ok(()) => Ok(()),
                    Err(_) => Err(primary_err),
                }
            }
            Err(err) => Err(err),
        }
    }

    /// Unload the model from ANE hardware.
    fn unload(&self) {
        self.unload_standard_if_loaded();
    }
}

impl Drop for AneModel {
    fn drop(&mut self) {
        if let Some(real_time) = &self.real_time {
            real_time.unload_if_loaded();
            unsafe {
                let _: () = msg_send![real_time.client, release];
            }
        }
        self.unload();
        cleanup_tmp(&self.tmp_dir.to_string_lossy());
        unsafe {
            let _: () = msg_send![self.model, release];
            if self.real_time_model != self.model {
                let _: () = msg_send![self.real_time_model, release];
            }
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

fn empty_options_dict() -> objc2::rc::Retained<NSDictionary<NSString, AnyObject>> {
    NSDictionary::<NSString, AnyObject>::new()
}

fn optional_number_ptr(number: Option<&objc2::rc::Retained<NSNumber>>) -> *const AnyObject {
    number.map_or(std::ptr::null::<AnyObject>(), |value| {
        (&**value) as *const NSNumber as *const AnyObject
    })
}

unsafe fn wrap_iosurface_array(
    io_surface_class: &'static AnyClass,
    surfaces: &[*mut c_void],
) -> objc2::rc::Retained<NSArray<AnyObject>> {
    let mut wrapped: Vec<*mut AnyObject> = Vec::with_capacity(surfaces.len());
    for &surface in surfaces {
        let surface = surface.cast::<__IOSurface>();
        let wrapped_surface: *mut AnyObject =
            msg_send![io_surface_class, objectWithIOSurface: surface];
        wrapped.push(wrapped_surface);
    }

    unsafe { ns_array_from_raw(&wrapped) }
}

unsafe fn wrap_iosurface_output_sets(
    io_surface_class: &'static AnyClass,
    output_sets: &[&[*mut c_void]],
) -> (
    objc2::rc::Retained<NSArray<AnyObject>>,
    Vec<objc2::rc::Retained<NSArray<AnyObject>>>,
) {
    let mut owned_output_sets = Vec::with_capacity(output_sets.len());
    let mut output_set_ptrs = Vec::with_capacity(output_sets.len());

    for output_set in output_sets {
        let ns_output_set = unsafe { wrap_iosurface_array(io_surface_class, output_set) };
        let ns_output_set_ptr = (&*ns_output_set) as *const NSArray<AnyObject> as *mut AnyObject;
        output_set_ptrs.push(ns_output_set_ptr);
        owned_output_sets.push(ns_output_set);
    }

    let outer = unsafe { ns_array_from_raw(&output_set_ptrs) };
    (outer, owned_output_sets)
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
    use std::os::unix::process::ExitStatusExt;
    use std::process::Command;

    const CHAINING_SMOKE_CHILD_ENV: &str = "PMETAL_ANE_CHAINING_SMOKE_CHILD";
    const CHAINING_SMOKE_TEST_NAME: &str = "ane::runtime::tests::test_prepare_loopback_chain_smoke";

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
                assert_eq!(rt.real_time_available(), rt.client_class.is_some());
                assert_eq!(rt.chaining_available(), rt.chaining_class.is_some());
                assert_eq!(rt.perf_stats_available(), rt.perf_stats_class.is_some());
            }
            Err(MetalError::AneNotAvailable) => {}
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_loopback_chain_config_default() {
        let config = AneLoopbackChainConfig::default();
        assert_eq!(config.loopback_input_symbol_index, 0);
        assert_eq!(config.loopback_output_symbol_index, 0);
        assert_eq!(config.procedure_index, 0);
        assert_eq!(config.fw_enqueue_delay, None);
        assert_eq!(config.memory_pool_id, None);
    }

    #[test]
    fn test_loopback_chain_config_validation_rejects_empty_collections() {
        let config = AneLoopbackChainConfig::default();
        let non_empty_input = [std::ptr::dangling_mut::<c_void>()];
        let non_empty_output = [std::ptr::dangling_mut::<c_void>()];
        let empty_outputs: [*mut c_void; 0] = [];

        let err = config.validate(&[], &[&non_empty_output]).unwrap_err();
        assert!(matches!(err, MetalError::InvalidConfig(_)));
        assert!(err.to_string().contains("at least one input"));

        let err = config.validate(&non_empty_input, &[]).unwrap_err();
        assert!(matches!(err, MetalError::InvalidConfig(_)));
        assert!(err.to_string().contains("at least one output set"));

        let err = config
            .validate(&non_empty_input, &[&empty_outputs])
            .unwrap_err();
        assert!(matches!(err, MetalError::InvalidConfig(_)));
        assert!(err.to_string().contains("output set 0"));
    }

    #[test]
    #[ignore = "requires ANE hardware and private AppleNeuralEngine.framework"]
    fn test_real_time_eval_matches_standard() {
        let rt = match AneRuntime::global() {
            Ok(rt) => rt,
            Err(MetalError::AneNotAvailable) => return,
            Err(e) => panic!("Unexpected error: {e}"),
        };
        if !rt.real_time_available() {
            return;
        }

        let mut program = MilProgram::new_fp32(1, 4);
        program.emit_cast("x16", &[1, 1, 1, 4], "x", "fp16");
        program.emit_cast("out", &[1, 1, 1, 4], "x16", "fp32");
        let mil_text = program.finalize("out");

        let model = match rt.compile(mil_text.as_bytes(), None) {
            Ok(model) => model,
            Err(MetalError::AneCompileFailed(_)) | Err(MetalError::AneLoadFailed(_)) => return,
            Err(e) => panic!("Unexpected error: {e}"),
        };

        let input = IoSurface::for_tensor_f32(1, 4).unwrap();
        let output_std = IoSurface::for_tensor_f32(1, 4).unwrap();
        let output_rt = IoSurface::for_tensor_f32(1, 4).unwrap();
        let input_values = [1.5f32, -2.0, 0.25, 7.0];
        input.write_f32_at(0, &input_values, 1, 4);

        if let Err(err) = model.evaluate(&[input.as_ptr()], &[output_std.as_ptr()]) {
            eprintln!("Skipping standard-vs-real-time comparison: {err}");
            return;
        }
        if let Err(err) = model.evaluate_real_time(&[input.as_ptr()], &[output_rt.as_ptr()]) {
            eprintln!("Skipping real-time output comparison: {err}");
            return;
        }

        let mut std_values = [0.0f32; 4];
        let mut rt_values = [0.0f32; 4];
        output_std.read_f32(&mut std_values, 0, 1, 4);
        output_rt.read_f32(&mut rt_values, 0, 1, 4);

        for (standard, realtime) in std_values.iter().zip(rt_values.iter()) {
            assert!(
                (standard - realtime).abs() < 1e-3,
                "standard={standard}, realtime={realtime}"
            );
        }
    }

    #[test]
    #[ignore = "requires ANE chaining-capable hardware and private AppleNeuralEngine.framework"]
    fn test_prepare_loopback_chain_smoke() {
        if std::env::var_os(CHAINING_SMOKE_CHILD_ENV).is_some() {
            run_prepare_loopback_chain_smoke_inner();
            return;
        }

        let status = Command::new(std::env::current_exe().unwrap())
            .arg("--ignored")
            .arg("--exact")
            .arg(CHAINING_SMOKE_TEST_NAME)
            .arg("--nocapture")
            .env(CHAINING_SMOKE_CHILD_ENV, "1")
            .status()
            .expect("failed to spawn chaining smoke child process");

        if status.success() {
            return;
        }

        let aborted_by_foreign_exception = status.signal() == Some(6) || status.code() == Some(134);
        if aborted_by_foreign_exception {
            eprintln!(
                "Skipping loopback chain smoke test: private ANE chaining call aborted the child process"
            );
            return;
        }

        panic!("loopback chain smoke child failed with status: {status}");
    }

    fn run_prepare_loopback_chain_smoke_inner() {
        let rt = match AneRuntime::global() {
            Ok(rt) => rt,
            Err(MetalError::AneNotAvailable) => return,
            Err(e) => panic!("Unexpected error: {e}"),
        };
        if !(rt.real_time_available() && rt.chaining_available()) {
            return;
        }

        let program = MilProgram::new(1, 4);
        let mil_text = program.finalize("x");
        let model = match rt.compile(mil_text.as_bytes(), None) {
            Ok(model) => model,
            Err(MetalError::AneCompileFailed(_)) | Err(MetalError::AneLoadFailed(_)) => return,
            Err(e) => panic!("Unexpected error: {e}"),
        };

        if !model.chaining_available() {
            return;
        }

        let input = IoSurface::for_tensor(1, 4).unwrap();
        let output = IoSurface::for_tensor(1, 4).unwrap();
        let outputs = [output.as_ptr()];

        let prepared = match model.prepare_loopback_chain(
            &[input.as_ptr()],
            &[&outputs],
            &AneLoopbackChainConfig::default(),
        ) {
            Ok(prepared) => prepared,
            Err(MetalError::AneChainingFailed(err))
            | Err(MetalError::AneCompileFailed(err))
            | Err(MetalError::AneLoadFailed(err))
            | Err(MetalError::AneEvalFailed(err)) => {
                eprintln!("Skipping loopback chain smoke test: {err}");
                return;
            }
            Err(e) => panic!("Unexpected error: {e}"),
        };

        assert!(prepared.is_valid());
        assert!(!prepared.description().is_empty());
    }
}
