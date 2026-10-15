//! An array over caller-owned memory: no copy, the owner lives as long as MLX
//! can read the memory, and is dropped once.

use std::alloc::{Layout, alloc, dealloc};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use pmetal_runtime::InlineArray;
use pmetal_runtime::compat::Dtype;

const PAGE: usize = 16384;

/// Page-aligned memory that counts its releases.
struct Pages {
    ptr: *mut u8,
    layout: Layout,
    released: Arc<AtomicUsize>,
}

// SAFETY: plain heap memory, handed to one reader at a time.
unsafe impl Send for Pages {}

impl Pages {
    fn new(bytes: usize, released: Arc<AtomicUsize>) -> Self {
        let layout = Layout::from_size_align(bytes, PAGE).unwrap();
        // SAFETY: non-zero size.
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null());
        Self {
            ptr,
            layout,
            released,
        }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        self.released.fetch_add(1, Ordering::SeqCst);
        // SAFETY: allocated in `new` with this layout.
        unsafe { dealloc(self.ptr, self.layout) };
    }
}

/// MLX drops an array's buffer when the work that read it completes, which can
/// be a moment after the last handle goes: flush what it retains, then wait.
fn wait_for_releases(released: &AtomicUsize, count: usize) {
    for _ in 0..200 {
        InlineArray::from_f32(0.0).eval();
        pmetal_runtime::inline_array::clear_cache();
        if released.load(Ordering::SeqCst) >= count {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn an_external_array_reads_the_memory_in_place_and_releases_it_once() {
    let released = Arc::new(AtomicUsize::new(0));
    let n = PAGE / 4;
    let pages = Pages::new(PAGE, released.clone());
    let ptr = pages.ptr;
    // SAFETY: `n` u32s fit in the allocation.
    unsafe {
        for i in 0..n {
            ptr.cast::<u32>().add(i).write(i as u32);
        }
    }

    // SAFETY: the memory is `n` readable u32s, kept alive by `pages`, and
    // nothing writes to it until the arrays are gone.
    let words = unsafe { InlineArray::from_external(ptr, &[n as i32], Dtype::Uint32, pages) };
    let sum = words.as_dtype(Dtype::Float32.as_i32()).sum_axis(0, false);
    let expected: f64 = (0..n as u64).map(|v| v as f64).sum();
    assert_eq!(f64::from(sum.item_f32()), expected);

    // The array reads the caller's memory, not a copy of it: a view of it
    // starts at the same address.
    assert_eq!(words.as_slice::<u32>().as_ptr() as usize, ptr as usize);

    assert_eq!(released.load(Ordering::SeqCst), 0, "alive while referenced");
    drop(sum);
    drop(words);
    wait_for_releases(&released, 1);
    assert_eq!(released.load(Ordering::SeqCst), 1, "released once, on drop");
    pmetal_runtime::check_unobserved_error().unwrap();
}

#[test]
fn a_region_that_is_not_whole_pages_reads_correctly_and_is_released_once() {
    let released = Arc::new(AtomicUsize::new(0));
    // Forty-eight bytes of a page: not a whole number of pages, so MLX may
    // wrap it or copy it; either way it reads right and the owner goes once.
    let pages = Pages::new(PAGE, released.clone());
    let ptr = pages.ptr;
    // SAFETY: 12 u32s fit in the allocation.
    unsafe {
        for i in 0..12 {
            ptr.cast::<u32>().add(i).write(i as u32 + 1);
        }
    }
    // SAFETY: as above, twelve readable u32s.
    let words = unsafe { InlineArray::from_external(ptr, &[12], Dtype::Uint32, pages) };
    let sum = words.as_dtype(Dtype::Float32.as_i32()).sum_axis(0, false);
    assert_eq!(sum.item_f32(), 78.0);
    pmetal_runtime::check_unobserved_error().unwrap();
    drop(sum);
    drop(words);
    wait_for_releases(&released, 1);
    assert_eq!(released.load(Ordering::SeqCst), 1);
}
