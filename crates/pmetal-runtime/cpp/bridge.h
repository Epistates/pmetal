// Inline array bridge — zero heap allocation per op.
// mlx::core::array stored directly in a stack buffer managed by Rust.
//
// Umbrella header. Declarations live in focused `bridge/*.h` sub-headers,
// mirroring the Rust `inline_array/*.rs` submodule split. Existing
// `#include "bridge.h"` call sites continue to pull in every symbol.

#ifndef MLX_INLINE_BRIDGE_H
#define MLX_INLINE_BRIDGE_H

#include "bridge/common.h"        // typedefs, lifecycle, interop, error
#include "bridge/ops.h"           // binary/unary ops, softmax, FFT, leaky_relu
#include "bridge/reductions.h"    // argmax/sum/mean/topk/logsumexp + abs
#include "bridge/eval.h"          // eval/async_eval/detach/item/layout queries
#include "bridge/factory.h"       // zeros/ones/full/eye/arange/random/slice loaders
#include "bridge/shape_ops.h"     // shape queries, slice/concat/squeeze/transpose
#include "bridge/gather.h"        // gather_mm/take_*/kv_cache_append/scatter_add
#include "bridge/fast.h"          // rms_norm/rope/sdpa/conv/tri_inv/svd/layer_norm
#include "bridge/compiled.h"      // fused compiled ops + generic compile wrapper
#include "bridge/quantized.h"     // dequantize/quantized_matmul/gather_qmm/quantize
#include "bridge/gdn.h"           // gdn_update / gdn_metal_step / state_update
#include "bridge/turboquant.h"    // TurboQuant encode/decode/score/pack/attention
#include "bridge/diagnostics.h"   // graph/metal capture/memory/stream helpers
#include "bridge/safetensors.h"   // load/save safetensors, random_seed
#include "bridge/autograd.h"      // value_and_grad + checkpoint

#endif
