cmake_minimum_required(VERSION 3.16)
project(pmetal_runtime_mlx LANGUAGES CXX C)
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

# Pinned to a commit on upstream main, which we track: the latest release
# (v0.32.3, 2026-09-28) trails main by the fused Metal SDPA backward
# (#4563), the expert-offset gather_qmm rework (#4572), f32 split-K qmm
# accumulation (#4641), the M1/M2 SDPA threadgroup fix (#4643) and the
# macOS 27 build fix for the gated-delta VJP kernels (#4594), among others.
# `just deps-check` reports how far this is behind main and the latest tag.
# build.rs's BUNDLED_MLX_GIT_TAG must name the same commit.
FetchContent_Declare(
  mlx
  GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
  GIT_TAG bebd6422f8db88d56e6220520a35bcd146f823f8)
FetchContent_MakeAvailable(mlx)

# ── Expose include dir and libmlx.a for the cc::Build step ──
# We build a tiny sentinel library so cmake --install gives us the lib dir.
add_library(pmetal_runtime_sentinel INTERFACE)
target_link_libraries(pmetal_runtime_sentinel INTERFACE mlx)

install(TARGETS pmetal_runtime_sentinel EXPORT PRuntimeTargets)

# Install libmlx.a so build.rs can find it
install(
  TARGETS mlx
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
