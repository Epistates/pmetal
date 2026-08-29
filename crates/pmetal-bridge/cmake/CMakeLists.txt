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

FetchContent_Declare(
  mlx
  GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
  GIT_TAG v0.32.2)
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
