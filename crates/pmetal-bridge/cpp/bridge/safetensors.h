// Safetensors load/save. Matches inline_array/safetensors.rs.

#ifndef MLX_INLINE_BRIDGE_SAFETENSORS_H
#define MLX_INLINE_BRIDGE_SAFETENSORS_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Load a single array from a safetensors file by key name.
// Returns 0 on success, 1 on error (key not found, file not found, etc.)
int mlx_inline_load_safetensors_key(mlx_inline_array* dst, const char* path, const char* key);

// Load ALL arrays from a safetensors file in one parse.
// key_buf[i] receives a strdup'd key string (caller must free() it).
// arr_buf[i] receives a placement-new'd array (caller must destroy via mlx_inline_destroy).
// Returns number of entries loaded, or -1 on error.
int mlx_inline_load_safetensors_all(
    const char* path,
    char** key_buf,
    mlx_inline_array* arr_buf,
    int max_entries);

// Save a collection of arrays to a safetensors file.
void mlx_inline_save_safetensors(const char* path, const char** keys, const mlx_inline_array* arrays, int count);

#ifdef __cplusplus
}
#endif

#endif
