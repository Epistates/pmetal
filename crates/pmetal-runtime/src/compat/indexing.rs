use super::Array;
pub use super::ops_ext::{argmax, argmax_axis, put_along_axis, scatter_add_single, topk_axis};
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Array {
    a.take_along_axis(indices, axis)
}

fn normalize_axis(axis: i32, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as i32 + axis) as usize
    } else {
        axis as usize
    }
}

fn normalize_bound(bound: i32, dim: i32) -> i32 {
    if bound < 0 {
        (dim + bound).clamp(0, dim)
    } else {
        bound.clamp(0, dim)
    }
}

fn slice_axis_range(a: &Array, axis: usize, start: i32, end: i32) -> Array {
    let shape = a.shape();
    let dim = shape[axis];
    let ndim = shape.len();
    let mut starts = vec![0; ndim];
    let mut stops: Vec<i32> = shape.to_vec();
    starts[axis] = normalize_bound(start, dim);
    stops[axis] = normalize_bound(end, dim);
    a.slice(&starts, &stops)
}

fn slice_axis_from(a: &Array, axis: usize, start: i32) -> Array {
    let end = a.shape()[axis];
    slice_axis_range(a, axis, start, end)
}

fn select_axis_idx(a: &Array, axis: usize, idx: i32) -> Array {
    let _ndim = a.ndim();
    let axis_i32 = axis as i32;
    let dim = a.dim(axis_i32);
    let normalized = if idx < 0 { dim + idx } else { idx };
    let indices = Array::from_i32_slice_shaped(&[normalized], &[1]);
    let out = a.take_axis(&indices, axis_i32);
    out.squeeze(axis_i32)
}

/// Thin shim: `IndexOp` replacement for simple array-backed indexing.
pub trait IndexOp<Idx> {
    fn index(&self, idx: Idx) -> Self;
}

impl IndexOp<&Array> for Array {
    fn index(&self, idx: &Array) -> Self {
        self.index_array(idx)
    }
}

impl IndexOp<Array> for Array {
    fn index(&self, idx: Array) -> Self {
        IndexOp::<&Array>::index(self, &idx)
    }
}

// Integer index (e.g. `arr.index(5)`) — squeeze that axis.
impl IndexOp<i32> for Array {
    fn index(&self, idx: i32) -> Self {
        let n = self.ndim();
        assert!(n >= 1, "index(i32): array must have at least 1 dim");
        // Take at position `idx` along axis 0, then remove that axis.
        let i = Array::from_i32_slice_shaped(&[idx], &[1]);
        let out = self.take_axis(&i, 0);
        out.squeeze(0)
    }
}

// usize index
impl IndexOp<usize> for Array {
    fn index(&self, idx: usize) -> Self {
        IndexOp::<i32>::index(self, idx as i32)
    }
}

impl IndexOp<(RangeTo<i32>, RangeFull)> for Array {
    fn index(&self, idx: (RangeTo<i32>, RangeFull)) -> Self {
        let axis = normalize_axis(0, self.ndim() as usize);
        slice_axis_range(self, axis, 0, idx.0.end)
    }
}

impl IndexOp<(RangeTo<usize>, RangeFull)> for Array {
    fn index(&self, idx: (RangeTo<usize>, RangeFull)) -> Self {
        IndexOp::<(RangeTo<i32>, RangeFull)>::index(self, (..(idx.0.end as i32), ..))
    }
}

impl IndexOp<(Range<i32>, RangeFull)> for Array {
    fn index(&self, idx: (Range<i32>, RangeFull)) -> Self {
        let axis = normalize_axis(0, self.ndim() as usize);
        slice_axis_range(self, axis, idx.0.start, idx.0.end)
    }
}

impl IndexOp<(Range<usize>, RangeFull)> for Array {
    fn index(&self, idx: (Range<usize>, RangeFull)) -> Self {
        IndexOp::<(Range<i32>, RangeFull)>::index(
            self,
            ((idx.0.start as i32)..(idx.0.end as i32), ..),
        )
    }
}

impl IndexOp<(RangeFrom<i32>, RangeFull)> for Array {
    fn index(&self, idx: (RangeFrom<i32>, RangeFull)) -> Self {
        let axis = normalize_axis(0, self.ndim() as usize);
        slice_axis_from(self, axis, idx.0.start)
    }
}

impl IndexOp<(RangeFrom<usize>, RangeFull)> for Array {
    fn index(&self, idx: (RangeFrom<usize>, RangeFull)) -> Self {
        IndexOp::<(RangeFrom<i32>, RangeFull)>::index(self, ((idx.0.start as i32).., ..))
    }
}

impl IndexOp<(RangeFull, RangeTo<i32>)> for Array {
    fn index(&self, idx: (RangeFull, RangeTo<i32>)) -> Self {
        let axis = normalize_axis(1, self.ndim() as usize);
        slice_axis_range(self, axis, 0, idx.1.end)
    }
}

impl IndexOp<(RangeFull, RangeTo<usize>)> for Array {
    fn index(&self, idx: (RangeFull, RangeTo<usize>)) -> Self {
        IndexOp::<(RangeFull, RangeTo<i32>)>::index(self, (.., ..(idx.1.end as i32)))
    }
}

impl IndexOp<(RangeFull, Range<i32>)> for Array {
    fn index(&self, idx: (RangeFull, Range<i32>)) -> Self {
        let axis = normalize_axis(1, self.ndim() as usize);
        slice_axis_range(self, axis, idx.1.start, idx.1.end)
    }
}

impl IndexOp<(RangeFull, Range<usize>)> for Array {
    fn index(&self, idx: (RangeFull, Range<usize>)) -> Self {
        IndexOp::<(RangeFull, Range<i32>)>::index(
            self,
            (.., (idx.1.start as i32)..(idx.1.end as i32)),
        )
    }
}

impl IndexOp<(RangeFull, RangeFrom<i32>)> for Array {
    fn index(&self, idx: (RangeFull, RangeFrom<i32>)) -> Self {
        let axis = normalize_axis(1, self.ndim() as usize);
        slice_axis_from(self, axis, idx.1.start)
    }
}

impl IndexOp<(RangeFull, RangeFrom<usize>)> for Array {
    fn index(&self, idx: (RangeFull, RangeFrom<usize>)) -> Self {
        IndexOp::<(RangeFull, RangeFrom<i32>)>::index(self, (.., (idx.1.start as i32)..))
    }
}

impl IndexOp<(RangeFull, i32)> for Array {
    fn index(&self, idx: (RangeFull, i32)) -> Self {
        let axis = normalize_axis(1, self.ndim() as usize);
        select_axis_idx(self, axis, idx.1)
    }
}

impl IndexOp<(RangeFull, RangeTo<i32>, RangeFull)> for Array {
    fn index(&self, idx: (RangeFull, RangeTo<i32>, RangeFull)) -> Self {
        let axis = normalize_axis(1, self.ndim() as usize);
        slice_axis_range(self, axis, 0, idx.1.end)
    }
}

impl IndexOp<(RangeFull, RangeTo<usize>, RangeFull)> for Array {
    fn index(&self, idx: (RangeFull, RangeTo<usize>, RangeFull)) -> Self {
        IndexOp::<(RangeFull, RangeTo<i32>, RangeFull)>::index(self, (.., ..(idx.1.end as i32), ..))
    }
}

impl IndexOp<(RangeFull, Range<i32>, RangeFull)> for Array {
    fn index(&self, idx: (RangeFull, Range<i32>, RangeFull)) -> Self {
        let axis = normalize_axis(1, self.ndim() as usize);
        slice_axis_range(self, axis, idx.1.start, idx.1.end)
    }
}

impl IndexOp<(RangeFull, Range<usize>, RangeFull)> for Array {
    fn index(&self, idx: (RangeFull, Range<usize>, RangeFull)) -> Self {
        IndexOp::<(RangeFull, Range<i32>, RangeFull)>::index(
            self,
            (.., (idx.1.start as i32)..(idx.1.end as i32), ..),
        )
    }
}

impl IndexOp<(RangeFull, RangeFrom<i32>, RangeFull)> for Array {
    fn index(&self, idx: (RangeFull, RangeFrom<i32>, RangeFull)) -> Self {
        let axis = normalize_axis(1, self.ndim() as usize);
        slice_axis_from(self, axis, idx.1.start)
    }
}

impl IndexOp<(RangeFull, RangeFrom<usize>, RangeFull)> for Array {
    fn index(&self, idx: (RangeFull, RangeFrom<usize>, RangeFull)) -> Self {
        IndexOp::<(RangeFull, RangeFrom<i32>, RangeFull)>::index(
            self,
            (.., (idx.1.start as i32).., ..),
        )
    }
}

impl IndexOp<(RangeFull, i32, RangeFull)> for Array {
    fn index(&self, idx: (RangeFull, i32, RangeFull)) -> Self {
        let axis = normalize_axis(1, self.ndim() as usize);
        select_axis_idx(self, axis, idx.1)
    }
}

impl IndexOp<(RangeFull, RangeFull, RangeFull, RangeTo<i32>)> for Array {
    fn index(&self, idx: (RangeFull, RangeFull, RangeFull, RangeTo<i32>)) -> Self {
        let axis = normalize_axis(3, self.ndim() as usize);
        slice_axis_range(self, axis, 0, idx.3.end)
    }
}

impl IndexOp<(RangeFull, RangeFull, RangeFull, RangeTo<usize>)> for Array {
    fn index(&self, idx: (RangeFull, RangeFull, RangeFull, RangeTo<usize>)) -> Self {
        IndexOp::<(RangeFull, RangeFull, RangeFull, RangeTo<i32>)>::index(
            self,
            (.., .., .., ..(idx.3.end as i32)),
        )
    }
}

impl IndexOp<(RangeFull, RangeFull, RangeFull, RangeFrom<i32>)> for Array {
    fn index(&self, idx: (RangeFull, RangeFull, RangeFull, RangeFrom<i32>)) -> Self {
        let axis = normalize_axis(3, self.ndim() as usize);
        slice_axis_from(self, axis, idx.3.start)
    }
}

impl IndexOp<(RangeFull, RangeFull, RangeFull, RangeFrom<usize>)> for Array {
    fn index(&self, idx: (RangeFull, RangeFull, RangeFull, RangeFrom<usize>)) -> Self {
        IndexOp::<(RangeFull, RangeFull, RangeFull, RangeFrom<i32>)>::index(
            self,
            (.., .., .., (idx.3.start as i32)..),
        )
    }
}

impl IndexOp<(RangeFull, RangeFull, RangeFull, Range<i32>)> for Array {
    fn index(&self, idx: (RangeFull, RangeFull, RangeFull, Range<i32>)) -> Self {
        let axis = normalize_axis(3, self.ndim() as usize);
        slice_axis_range(self, axis, idx.3.start, idx.3.end)
    }
}

impl IndexOp<(RangeFull, RangeFull, RangeFull, Range<usize>)> for Array {
    fn index(&self, idx: (RangeFull, RangeFull, RangeFull, Range<usize>)) -> Self {
        IndexOp::<(RangeFull, RangeFull, RangeFull, Range<i32>)>::index(
            self,
            (.., .., .., (idx.3.start as i32)..(idx.3.end as i32)),
        )
    }
}
