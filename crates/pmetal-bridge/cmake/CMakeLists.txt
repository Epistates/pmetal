cmake_minimum_required(VERSION 3.16)
project(pmetal_bridge_mlx LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)

# Avoid DOWNLOAD_EXTRACT_TIMESTAMP warning on CMake >= 3.24
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

# ── Fetch MLX ──
set(MLX_BUILD_TESTS OFF)
set(MLX_BUILD_EXAMPLES OFF)
set(MLX_BUILD_BENCHMARKS OFF)
set(MLX_BUILD_PYTHON_BINDINGS OFF)

# Pinned to a commit rather than a tag. v0.32.2 (2026-08-25) predates the
# Metal kernels Gemma 4 decodes through: D512 vector attention (#4459) covers
# its full-attention layers, and the fused NAX head-dim-256 prefill path
# (#4416) covers its sliding ones. It also predates several quantized-matmul
# correctness fixes, including the non-transposed affine qmm dispatch (#4392)
# that the packed-weight path depends on. Move this back to a tag at the next
# release that contains them.
FetchContent_Declare(
  mlx
  GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
  GIT_TAG 09e67c686f627ca371ebcce82f8f79110d8e186e)
FetchContent_MakeAvailable(mlx)

# ── Expose include dir and libmlx.a for the cc::Build step ──
# We build a tiny sentinel library so cmake --install gives us the lib dir.
add_library(pmetal_bridge_sentinel INTERFACE)
target_link_libraries(pmetal_bridge_sentinel INTERFACE mlx)

install(TARGETS pmetal_bridge_sentinel EXPORT PBridgeTargets)

# Install libmlx.a so build.rs can find it
install(
  TARGETS mlx
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
