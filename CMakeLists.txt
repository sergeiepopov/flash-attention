cmake_minimum_required(VERSION 3.20)
project(flash-attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Force CMake to use release Python libraries even in Debug builds
# This avoids the "cannot open file 'python314_d.lib'" error on Windows
if(WIN32)
  set(Python3_USE_DEBUG_LIBS OFF CACHE BOOL "" FORCE)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[1])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python site-packages path: ${PYTHON_SITE_PACKAGES_PATH}")
set(CMAKE_PREFIX_PATH "${PYTHON_SITE_PACKAGES_PATH}/torch/share/cmake" CACHE PATH "Path to CUTLASS root directory")

# Options to speed up compilation by disabling features (all disabled by default for faster builds)
option(FLASHATTENTION_DISABLE_BACKWARD "Disable backward pass" OFF)
option(FLASHATTENTION_DISABLE_SPLIT "Disable split kernel" ON)
option(FLASHATTENTION_DISABLE_PAGEDKV "Disable paged KV cache" ON)
option(FLASHATTENTION_DISABLE_APPENDKV "Disable append KV" ON)
option(FLASHATTENTION_DISABLE_LOCAL "Disable local/windowed attention" ON)
option(FLASHATTENTION_DISABLE_SOFTCAP "Disable softcap" ON)
option(FLASHATTENTION_DISABLE_PACKGQA "Disable packed GQA" ON)
option(FLASHATTENTION_DISABLE_BF16 "Disable BF16 support" ON)
option(FLASHATTENTION_DISABLE_FP16 "Disable FP16 support" OFF)
option(FLASHATTENTION_DISABLE_FP8 "Disable FP8 (E4M3) support" ON)
option(FLASHATTENTION_DISABLE_VARLEN "Disable variable length sequences" ON)
option(FLASHATTENTION_DISABLE_CLUSTER "Disable cluster support" ON)
option(FLASHATTENTION_DISABLE_HDIMDIFF64 "" ON)
option(FLASHATTENTION_DISABLE_HDIM64 "Disable head dimension 64" OFF)
option(FLASHATTENTION_DISABLE_HDIM96 "Disable head dimension 96" ON)
option(FLASHATTENTION_DISABLE_HDIM128 "Disable head dimension 128" ON)
option(FLASHATTENTION_DISABLE_HDIM192 "Disable head dimension 192" ON)
option(FLASHATTENTION_DISABLE_HDIM256 "Disable head dimension 256" ON)

# CUTLASS core headers
find_path(CUTLASS_INCLUDE_DIR
  NAMES cutlass/cutlass.h
  PATHS "csrc/cutlass/include"
  NO_DEFAULT_PATH
)

# Optional: CUTLASS util headers (used by many examples)
find_path(CUTLASS_UTIL_INCLUDE_DIR
  NAMES cutlass/util/device_memory.h
  PATHS "csrc/cutlass/tools/util/include"
  NO_DEFAULT_PATH
)

if(NOT CUTLASS_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find cutlass/cutlass.h under csrc/cutlass/include")
endif()

# Force use of release Python libraries on Windows for all build configurations
if(WIN32 AND TARGET Python3::Module)
  # Get the release library location
  get_target_property(PYTHON_RELEASE_LIB Python3::Module IMPORTED_IMPLIB_RELEASE)
  if(NOT PYTHON_RELEASE_LIB)
    get_target_property(PYTHON_RELEASE_LIB Python3::Module IMPORTED_IMPLIB)
  endif()
  
  get_target_property(PYTHON_RELEASE_LOC Python3::Module IMPORTED_LOCATION_RELEASE)
  if(NOT PYTHON_RELEASE_LOC)
    get_target_property(PYTHON_RELEASE_LOC Python3::Module IMPORTED_LOCATION)
  endif()
  
  # Set the same release library for all configurations
  if(PYTHON_RELEASE_LIB)
    set_target_properties(Python3::Module PROPERTIES
      IMPORTED_IMPLIB_DEBUG "${PYTHON_RELEASE_LIB}"
      IMPORTED_IMPLIB_RELWITHDEBINFO "${PYTHON_RELEASE_LIB}"
      IMPORTED_IMPLIB_MINSIZEREL "${PYTHON_RELEASE_LIB}"
    )
  endif()
  
  if(PYTHON_RELEASE_LOC)
    set_target_properties(Python3::Module PROPERTIES
      IMPORTED_LOCATION_DEBUG "${PYTHON_RELEASE_LOC}"
      IMPORTED_LOCATION_RELWITHDEBINFO "${PYTHON_RELEASE_LOC}"
      IMPORTED_LOCATION_MINSIZEREL "${PYTHON_RELEASE_LOC}"
    )
  endif()
endif()

find_package(Torch REQUIRED)

add_executable(flash-attention
    hopper/main.cpp hopper/flash_prepare_scheduler.cu
    hopper/flash_fwd_combine.cu
    hopper/instantiations/flash_bwd_hdim64_fp16_sm80.cu
    hopper/instantiations/flash_fwd_hdim64_fp16_sm80.cu
    hopper/flash_fwd_launch_template.h
)

#file(GLOB CU_SOURCES CONFIGURE_DEPENDS
#  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/src/*.cu"
#)

target_sources(flash-attention PRIVATE ${CU_SOURCES})

target_link_libraries(flash-attention PRIVATE
  Python3::Module
  ${TORCH_LIBRARIES}
)

target_include_directories(flash-attention PRIVATE
  "hopper"
  "${CUTLASS_INCLUDE_DIR}"
  "${CUTLASS_UTIL_INCLUDE_DIR}"
  "${TORCH_INCLUDE_DIRS}"
)

# Apply feature disable flags to speed up compilation
if(FLASHATTENTION_DISABLE_BACKWARD)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_BACKWARD)
  message(STATUS "FlashAttention: Backward pass disabled")
endif()

if(FLASHATTENTION_DISABLE_SPLIT)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_SPLIT)
  message(STATUS "FlashAttention: Split kernel disabled")
endif()

if(FLASHATTENTION_DISABLE_PAGEDKV)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_PAGEDKV)
  message(STATUS "FlashAttention: Paged KV disabled")
endif()

if(FLASHATTENTION_DISABLE_APPENDKV)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_APPENDKV)
  message(STATUS "FlashAttention: Append KV disabled")
endif()

if(FLASHATTENTION_DISABLE_LOCAL)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_LOCAL)
  message(STATUS "FlashAttention: Local attention disabled")
endif()

if(FLASHATTENTION_DISABLE_SOFTCAP)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_SOFTCAP)
  message(STATUS "FlashAttention: Softcap disabled")
endif()

if(FLASHATTENTION_DISABLE_PACKGQA)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_PACKGQA)
  message(STATUS "FlashAttention: Packed GQA disabled")
endif()

if(FLASHATTENTION_DISABLE_FP16)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_FP16)
  message(STATUS "FlashAttention: FP16 disabled")
endif()

if(FLASHATTENTION_DISABLE_BF16)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_BF16)
  message(STATUS "FlashAttention: BF16 disabled")
endif()

if(FLASHATTENTION_DISABLE_FP8)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_FP8)
  message(STATUS "FlashAttention: FP8 disabled")
endif()

if(FLASHATTENTION_DISABLE_VARLEN)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_VARLEN)
  message(STATUS "FlashAttention: Variable length disabled")
endif()

if(FLASHATTENTION_DISABLE_CLUSTER)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_CLUSTER)
  message(STATUS "FlashAttention: Cluster disabled")
endif()

if(FLASHATTENTION_DISABLE_HDIMDIFF64)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_HDIMDIFF64)
  message(STATUS "FLASHATTENTION_DISABLE_HDIMDIFF64")
endif()

if(FLASHATTENTION_DISABLE_HDIM64)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_HDIM64)
  message(STATUS "FlashAttention: Head dimension 64 disabled")
endif()

if(FLASHATTENTION_DISABLE_HDIM96)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_HDIM96)
  message(STATUS "FlashAttention: Head dimension 96 disabled")
endif()

if(FLASHATTENTION_DISABLE_HDIM128)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_HDIM128)
  message(STATUS "FlashAttention: Head dimension 128 disabled")
endif()

if(FLASHATTENTION_DISABLE_HDIM192)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_HDIM192)
  message(STATUS "FlashAttention: Head dimension 192 disabled")
endif()

if(FLASHATTENTION_DISABLE_HDIM256)
  target_compile_definitions(flash-attention PRIVATE FLASHATTENTION_DISABLE_HDIM256)
  message(STATUS "FlashAttention: Head dimension 256 disabled")
endif()

# Copy PyTorch DLLs to output directory on Windows
if(WIN32)
  # Get the directory containing torch DLLs
  get_filename_component(TORCH_INSTALL_PREFIX "${TORCH_INCLUDE_DIRS}/.." ABSOLUTE)
  set(TORCH_DLL_DIR "${TORCH_INSTALL_PREFIX}/lib")
  
  # List of essential PyTorch DLLs
  set(TORCH_DLLS
    c10.dll
    c10_cuda.dll
    torch.dll
    torch_cpu.dll
    torch_cuda.dll
    fbgemm.dll
    asmjit.dll
    uv.dll
  )
  
  # Copy each DLL if it exists
  foreach(DLL_NAME ${TORCH_DLLS})
    if(EXISTS "${TORCH_DLL_DIR}/${DLL_NAME}")
      add_custom_command(TARGET flash-attention POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${TORCH_DLL_DIR}/${DLL_NAME}"
          "$<TARGET_FILE_DIR:flash-attention>"
        COMMENT "Copying ${DLL_NAME}"
      )
    endif()
  endforeach()
  
  message(STATUS "PyTorch library path: ${TORCH_DLL_DIR}")
  message(STATUS "Will copy PyTorch DLLs to executable directory after build")
endif()

# On Windows: Configure MSVC compiler options
if(WIN32)
  target_compile_definitions(flash-attention PRIVATE
    $<$<CONFIG:Debug>:SHIBOKEN_PYTHON_LIMITED_API>
  )
  
  # Use standard-conforming preprocessor (required for CUTLASS)
  # Remove _DEBUG define for Python compatibility in Debug builds
  target_compile_options(flash-attention PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/U_DEBUG>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/U_DEBUG>
    #$<$<COMPILE_LANGUAGE:CUDA>:-g>           # Generate debug info
    #$<$<COMPILE_LANGUAGE:CUDA>:-G>           # Generate device debug info
    #$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>    # Generate line info
    #$<$<COMPILE_LANGUAGE:CUDA>:-O0>          # Disable optimizations for debug
  )
  # Force linker to ignore debug Python library and use release version
  target_link_options(flash-attention PRIVATE
    $<$<CONFIG:Debug>:/NODEFAULTLIB:python314_d.lib>
    $<$<CONFIG:Debug>:python314.lib>
  )
  # Add the Python library directory to the linker search path
  if(Python3_LIBRARY_DIRS)
    target_link_directories(flash-attention PRIVATE ${Python3_LIBRARY_DIRS})
  endif()
  if(Python3_RUNTIME_LIBRARY_DIRS)
    target_link_directories(flash-attention PRIVATE ${Python3_RUNTIME_LIBRARY_DIRS})
  endif()
endif()