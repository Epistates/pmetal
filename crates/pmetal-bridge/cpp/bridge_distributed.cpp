// MLX distributed: process groups and collectives over whichever backends
// MLX was built with (ring, JACCL, MPI). `pmetal-distributed` reaches them
// through `pmetal_bridge::distributed`.
//
// A group is handed to Rust as an opaque heap object owning an
// `mlx::core::distributed::Group` (a shared_ptr to the backend's group);
// a null group means MLX's default, the one `init` made. Each collective is
// lazy like any other op and uses the calling thread's default stream.

#include "bridge_internal.h"

#include <mlx/distributed/distributed.h>
#include <mlx/distributed/ops.h>

namespace dist = mlx::core::distributed;

struct pmetal_dist_group {
    dist::Group group;
};

static std::optional<dist::Group> group_or_default(const pmetal_dist_group* g) {
    if (g) {
        return g->group;
    }
    return std::nullopt;
}

// Run `make` for a new group, reporting a throw through the error channel
// and returning null.
template <typename F>
static pmetal_dist_group* new_group(const char* op, F make) {
    try {
        auto* g = new pmetal_dist_group{make()};
        pmetal_bridge_clear_error_internal();
        return g;
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error(op, e.what());
    } catch (...) {
        pmetal_bridge_set_last_error(op, "unknown C++ exception");
    }
    return nullptr;
}

extern "C" {

bool mlx_inline_distributed_is_available(void) {
    try {
        return dist::is_available();
    } catch (...) {
        return false;
    }
}

pmetal_dist_group* mlx_inline_distributed_init(bool strict) {
    return new_group("distributed_init", [&] { return dist::init(strict); });
}

pmetal_dist_group* mlx_inline_distributed_group_split(
    const pmetal_dist_group* g, int color, int key) {
    return new_group("distributed_group_split", [&] { return g->group.split(color, key); });
}

int mlx_inline_distributed_group_rank(const pmetal_dist_group* g) {
    int rank = 0;
    BRIDGE_TRY_VOID("distributed_group_rank", rank = g->group.rank());
    return rank;
}

int mlx_inline_distributed_group_size(const pmetal_dist_group* g) {
    int size = 1;
    BRIDGE_TRY_VOID("distributed_group_size", size = g->group.size());
    return size;
}

void mlx_inline_distributed_group_free(pmetal_dist_group* g) {
    delete g;
}

void mlx_inline_distributed_all_sum(
    mlx_inline_array* dst, const mlx_inline_array* x, const pmetal_dist_group* g) {
    BRIDGE_TRY_DST("distributed_all_sum", dst,
        new (dst->buf) array(dist::all_sum(as_arr(x), group_or_default(g))));
}

void mlx_inline_distributed_all_gather(
    mlx_inline_array* dst, const mlx_inline_array* x, const pmetal_dist_group* g) {
    BRIDGE_TRY_DST("distributed_all_gather", dst,
        new (dst->buf) array(dist::all_gather(as_arr(x), group_or_default(g))));
}

void mlx_inline_distributed_all_max(
    mlx_inline_array* dst, const mlx_inline_array* x, const pmetal_dist_group* g) {
    BRIDGE_TRY_DST("distributed_all_max", dst,
        new (dst->buf) array(dist::all_max(as_arr(x), group_or_default(g))));
}

void mlx_inline_distributed_all_min(
    mlx_inline_array* dst, const mlx_inline_array* x, const pmetal_dist_group* g) {
    BRIDGE_TRY_DST("distributed_all_min", dst,
        new (dst->buf) array(dist::all_min(as_arr(x), group_or_default(g))));
}

void mlx_inline_distributed_sum_scatter(
    mlx_inline_array* dst, const mlx_inline_array* x, const pmetal_dist_group* g) {
    BRIDGE_TRY_DST("distributed_sum_scatter", dst,
        new (dst->buf) array(dist::sum_scatter(as_arr(x), group_or_default(g))));
}

void mlx_inline_distributed_send(
    mlx_inline_array* dst, const mlx_inline_array* x, int to, const pmetal_dist_group* g) {
    BRIDGE_TRY_DST("distributed_send", dst,
        new (dst->buf) array(dist::send(as_arr(x), to, group_or_default(g))));
}

void mlx_inline_distributed_recv(
    mlx_inline_array* dst, const int* shape, size_t ndim, int dtype, int from,
    const pmetal_dist_group* g) {
    BRIDGE_TRY_DST("distributed_recv", dst,
        new (dst->buf) array(dist::recv(
            mlx::core::Shape(shape, shape + ndim), dtype_from_int(dtype), from,
            group_or_default(g))));
}

void mlx_inline_distributed_recv_like(
    mlx_inline_array* dst, const mlx_inline_array* x, int from, const pmetal_dist_group* g) {
    BRIDGE_TRY_DST("distributed_recv_like", dst,
        new (dst->buf) array(dist::recv_like(as_arr(x), from, group_or_default(g))));
}

}  // extern "C"
