//! The ANE models and the one thread they live on.
//!
//! A model compiled for the ANE takes seconds to load and holds its weights
//! on the ANE, so there is one per checkpoint, kept between requests. They
//! live on a single long-lived thread and every request runs there, one at a
//! time: callers on any thread (a server's blocking pool, the CLI's main
//! thread) find the model already loaded, and the models and their GPU
//! drafters never cross threads. Tokens come back to the caller as they're
//! generated, and the caller's answer (go on or stop) goes back.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::mpsc;

use pmetal_metal::ane::lm::{
    AneLm, AneLmOptions, Drafter, GenerateOptions, GenerationStats, PromptLookup,
};
use pmetal_metal::error::{MetalError, Result};

use crate::dflash_drafter::DFlashDrafter;

/// How the DFlash drafter's weights are packed. At 8 bits its guesses are as
/// good as at bf16 (Qwen3-4B: 7.1, 4.7 and 3.1 tokens per pass on three chat
/// prompts against 6.9, 4.7 and 3.0) and a draft takes ~2.5 ms less, with
/// half the GPU memory.
const DRAFT_QUANT: pmetal_bridge::native_weight::QuantParams =
    pmetal_bridge::native_weight::QuantParams {
        group_size: 64,
        bits: 8,
        mode: pmetal_bridge::QuantizedMode::Affine,
    };

/// A model on the ANE, and the DFlash drafter (on the GPU) reading its
/// hidden states, if it has one.
struct AneEngine {
    lm: AneLm,
    drafter: Option<(PathBuf, DFlashDrafter)>,
}

/// The ANE thread's models, by checkpoint.
type Engines = HashMap<PathBuf, AneEngine>;

type Job = Box<dyn FnOnce(&mut Engines) + Send>;

/// One generation on the ANE.
pub(crate) struct AneRequest {
    pub model_path: PathBuf,
    /// DFlash draft model to draft with; prompt lookup without one.
    pub draft_path: Option<PathBuf>,
    /// KV cache slots the request needs.
    pub context: usize,
    pub input_ids: Vec<u32>,
    pub opts: GenerateOptions,
}

enum Event {
    Token(u32),
    Done(Result<(Vec<u32>, GenerationStats)>),
}

/// Run `request` on the ANE thread, calling `on_token` on this thread for
/// each token generated; returning false stops generation. Returns the
/// generated tokens.
pub(crate) fn generate(
    request: AneRequest,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<(Vec<u32>, GenerationStats)> {
    let (events, from_worker) = mpsc::channel::<Event>();
    let (answers, from_caller) = mpsc::channel::<bool>();
    submit(Box::new(move |engines| {
        let result = run(engines, &request, |token| {
            events.send(Event::Token(token)).is_ok() && from_caller.recv().unwrap_or(false)
        });
        let _ = events.send(Event::Done(result));
    }))?;
    for event in from_worker {
        match event {
            Event::Token(token) => {
                // The worker is waiting on the answer; if it has gone, the
                // next receive says so.
                let _ = answers.send(on_token(token));
            }
            Event::Done(result) => return result,
        }
    }
    Err(MetalError::Internal(
        "the ANE thread stopped during the request".into(),
    ))
}

/// Queue `job` on the ANE thread, starting the thread on first use.
fn submit(job: Job) -> Result<()> {
    static JOBS: OnceLock<std::result::Result<mpsc::Sender<Job>, String>> = OnceLock::new();
    let jobs = JOBS.get_or_init(|| {
        let (jobs, queue) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("pmetal-ane".into())
            .spawn(move || {
                let mut engines = Engines::new();
                for job in queue {
                    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        job(&mut engines)
                    }));
                    if ran.is_err() {
                        // A model may be half built; start over.
                        tracing::error!("an ANE request panicked; dropping the loaded models");
                        engines.clear();
                    }
                }
            })
            .map(|_| jobs)
            .map_err(|e| e.to_string())
    });
    jobs.as_ref()
        .map_err(|e| MetalError::Internal(format!("can't start the ANE thread: {e}")))?
        .send(job)
        .map_err(|_| MetalError::Internal("the ANE thread has exited".into()))
}

/// Generate for `request` with its model, loading or rebuilding the model
/// first if it isn't loaded with room for the request and the same drafter.
fn run(
    engines: &mut Engines,
    request: &AneRequest,
    on_token: impl FnMut(u32) -> bool,
) -> Result<(Vec<u32>, GenerationStats)> {
    let engine = engine_for(
        engines,
        &request.model_path,
        request.draft_path.as_deref(),
        request.context,
    )?;
    let mut lookup = PromptLookup::default();
    let drafter: &mut dyn Drafter = match &mut engine.drafter {
        Some((_, dflash)) => dflash,
        None => &mut lookup,
    };
    engine
        .lm
        .generate(&request.input_ids, &request.opts, drafter, on_token)
}

fn engine_for<'a>(
    engines: &'a mut Engines,
    model_path: &Path,
    draft_path: Option<&Path>,
    context: usize,
) -> Result<&'a mut AneEngine> {
    let fits = engines.get(model_path).is_some_and(|engine| {
        engine.lm.capacity() >= context
            && engine.drafter.as_ref().map(|(path, _)| path.as_path()) == draft_path
    });
    if !fits {
        // Free the old model's ANE memory before compiling the new one.
        engines.remove(model_path);
        // The drafter decides which layers the ANE programs output.
        let drafter = draft_path
            .map(|path| {
                tracing::info!(draft = %path.display(), "Loading the DFlash drafter");
                DFlashDrafter::load(path, model_path, context, Some(DRAFT_QUANT))
                    .map(|drafter| (path.to_path_buf(), drafter))
                    .map_err(|e| MetalError::InvalidConfig(e.to_string()))
            })
            .transpose()?;
        tracing::info!(
            model = %model_path.display(),
            context,
            "Compiling the model for the ANE (cached by the system after the first run)"
        );
        let opts = AneLmOptions {
            capacity: context,
            taps: drafter
                .as_ref()
                .map(|(_, d)| d.target_layer_ids().to_vec())
                .unwrap_or_default(),
            ..AneLmOptions::default()
        };
        let lm = AneLm::load(model_path, &opts)?;
        engines.insert(model_path.to_path_buf(), AneEngine { lm, drafter });
    }
    Ok(engines.get_mut(model_path).expect("inserted above"))
}
