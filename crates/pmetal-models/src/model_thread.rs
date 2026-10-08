//! A thread that owns a model, and the one place its arrays are touched.
//!
//! MLX streams belong to the thread that created them. An array built lazily
//! on one thread records that thread's stream, and evaluating it anywhere else
//! fails with "There is no Stream(gpu, N) in current thread". A model is full
//! of such arrays: whatever its loader left unevaluated, tables it builds on
//! first use, KV snapshots kept between requests. Moving the model between
//! threads, as a pool of worker threads does, breaks it the first time a
//! request lands on a different thread, and again whenever the pool retires
//! the thread that did the work.
//!
//! [`ModelThread`] keeps the state on one long-lived thread instead: it is
//! built there, every job runs there one at a time, and it is dropped there.
//! Callers on any thread queue closures and get results back over channels
//! they pass in. The state itself never moves, so it need not be `Send`.

use std::sync::mpsc;

type Job<S> = Box<dyn FnOnce(&mut S) + Send>;
type BackgroundFn<S> = Box<dyn FnMut(&mut S) -> Background + Send>;

/// What a [`ModelThread`]'s background work reports after a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Background {
    /// It did some and has more: check for jobs, then call it again.
    Busy,
    /// It has nothing to do until a job gives it some: sleep until one comes.
    Idle,
}

/// The owning thread has exited: its state failed to build, or every handle
/// to it was dropped. Jobs queued after that are dropped unrun.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelThreadExited;

impl std::fmt::Display for ModelThreadExited {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the model thread has exited")
    }
}

impl std::error::Error for ModelThreadExited {}

/// How a [`ModelThread`] failed to start.
#[derive(Debug)]
pub enum ModelThreadStartError<E> {
    /// The OS refused to create the thread.
    Spawn(std::io::Error),
    /// The state's constructor returned this error.
    Init(E),
    /// The state's constructor panicked.
    InitPanicked,
}

impl<E: std::fmt::Display> std::fmt::Display for ModelThreadStartError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(e) => write!(f, "cannot start the model thread: {e}"),
            Self::Init(e) => write!(f, "{e}"),
            Self::InitPanicked => f.write_str("loading the model panicked"),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for ModelThreadStartError<E> {}

/// State of type `S` living on its own thread. See the module docs.
///
/// Handles are cheap to clone. When the last one is dropped the thread
/// finishes the jobs already queued, drops the state and exits.
pub struct ModelThread<S> {
    jobs: mpsc::Sender<Job<S>>,
}

impl<S> Clone for ModelThread<S> {
    fn clone(&self) -> Self {
        Self {
            jobs: self.jobs.clone(),
        }
    }
}

impl<S: 'static> ModelThread<S> {
    /// Start a thread called `name`, build the state on it with `init`, and
    /// return once it is built, or with the reason it could not be.
    pub fn spawn<E: Send + 'static>(
        name: &str,
        init: impl FnOnce() -> Result<S, E> + Send + 'static,
    ) -> Result<Self, ModelThreadStartError<E>> {
        Self::start(name, init, None)
    }

    /// Like [`spawn`](Self::spawn), with work the thread does between jobs:
    /// while the queue is empty it calls `background` for as long as that
    /// reports [`Background::Busy`], and once it reports [`Background::Idle`]
    /// it sleeps until the next job. Jobs always go first, so a long run of
    /// background steps never starves them, and work only a job can create
    /// (a new request, say) is picked up without polling. It first runs after
    /// the first job.
    pub fn spawn_with_background<E: Send + 'static>(
        name: &str,
        init: impl FnOnce() -> Result<S, E> + Send + 'static,
        background: impl FnMut(&mut S) -> Background + Send + 'static,
    ) -> Result<Self, ModelThreadStartError<E>> {
        Self::start(name, init, Some(Box::new(background)))
    }

    fn start<E: Send + 'static>(
        name: &str,
        init: impl FnOnce() -> Result<S, E> + Send + 'static,
        background: Option<BackgroundFn<S>>,
    ) -> Result<Self, ModelThreadStartError<E>> {
        let (jobs, queue) = mpsc::channel::<Job<S>>();
        let (ready, built) = mpsc::sync_channel::<Result<(), E>>(1);
        std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                let mut state = match init() {
                    Ok(state) => state,
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                let _ = ready.send(Ok(()));
                drop(ready);
                run(&mut state, &queue, background);
            })
            .map_err(ModelThreadStartError::Spawn)?;
        match built.recv() {
            Ok(Ok(())) => Ok(Self { jobs }),
            Ok(Err(e)) => Err(ModelThreadStartError::Init(e)),
            // The sender went away without a word: `init` panicked.
            Err(_) => Err(ModelThreadStartError::InitPanicked),
        }
    }

    /// Queue `job` to run on the thread. Results go back through whatever
    /// channel the job captures; a job that panics drops it unsent, so its
    /// caller sees a closed channel instead of waiting forever.
    pub fn submit(
        &self,
        job: impl FnOnce(&mut S) + Send + 'static,
    ) -> Result<(), ModelThreadExited> {
        self.jobs.send(Box::new(job)).map_err(|_| ModelThreadExited)
    }

    /// Run `job` on the thread and block this one until it returns. Not for
    /// use from the model thread itself, which would wait on itself.
    pub fn call<R: Send + 'static>(
        &self,
        job: impl FnOnce(&mut S) -> R + Send + 'static,
    ) -> Result<R, ModelThreadExited> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.submit(move |state| {
            let _ = reply.send(job(state));
        })?;
        answer.recv().map_err(|_| ModelThreadExited)
    }
}

fn run<S>(state: &mut S, queue: &mpsc::Receiver<Job<S>>, mut background: Option<BackgroundFn<S>>) {
    // A panic in one request must not take the others down with it. The
    // panicking job drops its reply channel, which tells its caller.
    fn guarded<T>(what: &str, f: impl FnOnce() -> T) -> Option<T> {
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        if out.is_err() {
            tracing::error!(
                thread = std::thread::current().name().unwrap_or("model"),
                "{what} panicked"
            );
        }
        out.ok()
    }
    let mut idle = true;
    loop {
        let job = match background.as_mut() {
            Some(work) if !idle => match queue.try_recv() {
                Ok(job) => job,
                Err(mpsc::TryRecvError::Disconnected) => return,
                Err(mpsc::TryRecvError::Empty) => {
                    // A panicking step counts as idle, or it would spin.
                    idle = guarded("background model work", || work(state))
                        .is_none_or(|b| b == Background::Idle);
                    continue;
                }
            },
            _ => match queue.recv() {
                Ok(job) => job,
                Err(_) => return,
            },
        };
        guarded("a model job", || job(state));
        // The job may have given the background work something to do.
        idle = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::Array;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The reason this exists: an array built lazily on the owning thread is
    /// evaluated by later jobs, queued from other threads, after the threads
    /// that queued the earlier jobs are gone.
    #[test]
    fn lazy_state_survives_callers_on_other_threads() {
        let thread = ModelThread::spawn("test-model", || {
            // Lazy: no eval here, so it carries this thread's stream.
            let x = Array::from_f32_slice(&[1.0, 2.0, 3.0], &[3]);
            Ok::<_, ()>(x.multiply(&Array::from_f32(2.0)))
        })
        .unwrap();
        for round in 0..3 {
            let handle = thread.clone();
            let sum = std::thread::spawn(move || {
                handle
                    .call(move |x: &mut Array| {
                        let s = x.sum(None);
                        s.eval();
                        pmetal_bridge::check_last_error().map(|()| s.item_f32() + round as f32)
                    })
                    .unwrap()
            })
            .join()
            .unwrap();
            assert_eq!(sum.unwrap(), 12.0 + round as f32);
        }
    }

    /// The defect it fixes, so the test above can't pass for another reason:
    /// the same lazy array evaluated on a thread that didn't build it fails.
    #[test]
    fn lazy_array_from_another_thread_does_not_evaluate() {
        struct Smuggled(Array);
        // SAFETY: moved to exactly one other thread and used only there; the
        // point of the test is that this is wrong for MLX streams.
        #[allow(unsafe_code)]
        unsafe impl Send for Smuggled {}
        let x = std::thread::spawn(|| {
            let x = Array::from_f32_slice(&[1.0, 2.0, 3.0], &[3]);
            Smuggled(x.multiply(&Array::from_f32(2.0)))
        })
        .join()
        .unwrap();
        let s = x.0.sum(None);
        s.eval();
        let err = pmetal_bridge::check_last_error().expect_err("cross-thread eval must fail");
        assert!(err.to_string().contains("Stream"), "{err}");
    }

    #[test]
    fn init_error_is_returned() {
        let err = ModelThread::<()>::spawn("test-model", || Err::<(), _>("no weights"))
            .err()
            .unwrap();
        assert!(matches!(err, ModelThreadStartError::Init("no weights")));
    }

    #[test]
    fn panicking_job_drops_its_reply_and_the_thread_goes_on() {
        let thread = ModelThread::spawn("test-model", || Ok::<_, ()>(5usize)).unwrap();
        assert_eq!(
            thread.call(|_: &mut usize| -> usize { panic!("boom") }),
            Err(ModelThreadExited)
        );
        assert_eq!(thread.call(|n: &mut usize| *n), Ok(5));
    }

    /// Background work runs until it reports idle, sleeps, and wakes for the
    /// work a job hands it.
    #[test]
    fn background_drains_and_wakes_on_jobs() {
        let steps = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&steps);
        let thread = ModelThread::spawn_with_background(
            "test-model",
            || Ok::<_, ()>(3usize),
            move |left: &mut usize| {
                if *left == 0 {
                    return Background::Idle;
                }
                *left -= 1;
                seen.fetch_add(1, Ordering::SeqCst);
                Background::Busy
            },
        )
        .unwrap();
        // Jobs go first, but the queue is empty between these calls, so the
        // background drains whatever each one leaves.
        thread.call(|left: &mut usize| *left += 2).unwrap();
        let mut drained = false;
        for _ in 0..1000 {
            if thread.call(|left: &mut usize| *left).unwrap() == 0 {
                drained = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(drained);
        assert_eq!(steps.load(Ordering::SeqCst), 5);
    }
}
