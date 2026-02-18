/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <type_traits>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "cute/tensor.hpp"
#include "cute/arch/util.hpp"  // cute::detail::explode for FLASH_RAW_MMA

#include "seqlen.h"
#include "block.h"
#include "mask.h"
#include "pack_gqa.h"
#include "paged_kv.h"
#include "rotary.h"
#include "utils.h"

#define FLASH_USE_CUTLASS_TENSOR 0
#define FLASH_MANUAL_GEMM
#define FLASH_RAW_MMA

namespace flash {

using namespace cute;

// ============================================================================
// EDUCATIONAL: Understanding ldmatrix and Shared Memory → Register Transfer
// ============================================================================
//
// This section explains how cute::copy works when loading matrix fragments
// from shared memory to registers for Tensor Core operations.
//
// The key instruction is: ldmatrix.sync.aligned.x4.m8n8.shared.b16
//
// ============================================================================
// LDMATRIX INSTRUCTION OVERVIEW
// ============================================================================
//
// ldmatrix is a WARP-COOPERATIVE instruction. All 32 threads in a warp 
// participate together to load matrix fragments optimized for tensor cores.
//
// Instruction: ldmatrix.sync.aligned.x4.m8n8.shared.b16 {d0,d1,d2,d3}, [addr]
//
//   - sync:    All threads must execute together (warp-synchronous)
//   - aligned: Source address must be 16-byte aligned
//   - x4:      Load 4 matrices simultaneously  
//   - m8n8:    Each matrix is 8 rows × 8 columns
//   - shared:  Source is in shared memory
//   - b16:     Each element is 16 bits (half/bf16)
//
// ============================================================================
// THREAD-TO-DATA MAPPING (for m8n8 with x4, loading 4 8×8 matrices)
// ============================================================================
//
//   Shared Memory Layout (4 matrices, each 8×8 of fp16):
//   ┌─────────────────────────────────────────────────────────────────┐
//   │ Matrix 0 (8×8)  │ Matrix 1 (8×8)  │ Matrix 2 (8×8)  │ Matrix 3  │
//   │ rows 0-7        │ rows 0-7        │ rows 0-7        │ rows 0-7  │
//   └─────────────────────────────────────────────────────────────────┘
//
//   Thread Assignment:
//   - Threads 0-7   provide addresses for rows 0-7 of Matrix 0
//   - Threads 8-15  provide addresses for rows 0-7 of Matrix 1  
//   - Threads 16-23 provide addresses for rows 0-7 of Matrix 2
//   - Threads 24-31 provide addresses for rows 0-7 of Matrix 3
//
//   Each thread provides ONE address pointing to the START of ONE ROW (8 fp16 = 16 bytes)
//
//   After ldmatrix, each thread holds 4 registers (d0,d1,d2,d3):
//   - d0: 2 fp16 values from matrix 0
//   - d1: 2 fp16 values from matrix 1
//   - d2: 2 fp16 values from matrix 2
//   - d3: 2 fp16 values from matrix 3
//
// ============================================================================
// VISUALIZATION: How 32 threads cooperatively load 4 matrices
// ============================================================================
//
//   BEFORE (in Shared Memory):                AFTER (in Registers):
//   
//   Matrix 0 (smem):                          Thread 0: d0=M0[0,0:1], d1=M1[0,0:1], d2=M2[0,0:1], d3=M3[0,0:1]
//   Row 0: [a0 a1 a2 a3 a4 a5 a6 a7] ←T0     Thread 1: d0=M0[1,0:1], d1=M1[1,0:1], d2=M2[1,0:1], d3=M3[1,0:1]
//   Row 1: [b0 b1 b2 b3 b4 b5 b6 b7] ←T1     ...
//   ...                                       Thread 7: d0=M0[7,0:1], d1=M1[7,0:1], d2=M2[7,0:1], d3=M3[7,0:1]
//   Row 7: [h0 h1 h2 h3 h4 h5 h6 h7] ←T7     
//                                             Thread 8:  d0=M0[0,2:3], d1=M1[0,2:3], ...
//   Matrix 1 (smem):                          Thread 9:  d0=M0[1,2:3], d1=M1[1,2:3], ...
//   Row 0: [i0 i1 i2 i3 i4 i5 i6 i7] ←T8     ...
//   ...                                       Thread 31: d0=M0[7,6:7], d1=M1[7,6:7], ...
//
//   The data gets REDISTRIBUTED across threads in a pattern optimized for MMA!
//
// ============================================================================

// ============================================================================
// EDUCATIONAL COPY: Direct ldmatrix implementation with full explanation
// ============================================================================
// This function implements the smem→register copy WITHOUT calling cute::copy,
// showing exactly how the ldmatrix instruction works.
//
// Key insight from CuTe's copy_unpack:
//   1. RECAST source tensor to uint128_t (16 bytes = what ldmatrix loads)
//   2. RECAST destination tensor to uint32_t (4 registers = what ldmatrix produces)
//   3. Call the ldmatrix PTX with individual tensor elements
//
// The CuTe tensor layout ALREADY encodes swizzle patterns and thread mappings.
// ============================================================================
template<typename TiledCopy, typename TensorS, typename TensorD>
CUTLASS_DEVICE void educational_copy_smem_to_regs(
    TiledCopy const& tiled_copy,  // Copy descriptor (unused, kept for API compat)
    TensorS const& src,           // Source: shared memory tensor slice
    TensorD& dst)                 // Destination: register tensor slice
{
    // ========================================================================
    // STEP 1: RECAST tensors to match ldmatrix register types
    // ========================================================================
    // This is exactly what CuTe's copy_unpack does internally!
    //
    // For SM75_U32x4_LDSM_N:
    //   - SRegisters = uint128_t[1]  →  source is 1 × 16-byte chunk
    //   - DRegisters = uint32_t[4]   →  destination is 4 × 4-byte registers
    //
    // recast<uint128_t>(src) reinterprets the fp16 tensor as uint128_t:
    //   - Original: 8 × fp16 elements = 16 bytes
    //   - Recast:   1 × uint128_t element = 16 bytes
    //
    // recast<uint32_t>(dst) reinterprets the fp16 tensor as uint32_t:
    //   - Original: 8 × fp16 elements = 16 bytes  
    //   - Recast:   4 × uint32_t elements = 16 bytes
    //
    Tensor src_u128 = recast<uint128_t const>(src);  // View as uint128_t
    Tensor dst_u32  = recast<uint32_t>(dst);         // View as uint32_t
    
    // ========================================================================
    // STEP 2: Verify sizes match ldmatrix expectations
    // ========================================================================
    // After recast:
    //   - src_u128 should have 1 element per ldmatrix call
    //   - dst_u32 should have 4 elements per ldmatrix call
    //
    // The tensor shapes are now:
    //   src_u128: (1, NUM_REPEATS) or (NUM_REPEATS,) if 1D
    //   dst_u32:  (4, NUM_REPEATS) or (4 * NUM_REPEATS,) if 1D
    
    // ========================================================================
    // STEP 3: Issue ldmatrix for each atom
    // ========================================================================
    // CuTe's detail::explode calls CopyOp::copy(src(0), dst(0), dst(1), dst(2), dst(3))
    // We replicate this pattern directly.
    
    // 2D tensor: loop over repeats
    // src_u128 has shape (1, NUM_REPEATS), dst_u32 has shape (4, NUM_REPEATS)
        
    #pragma unroll
    for (int m = 0; m < size<1>(src_u128); ++m) {
        // Get source address for this repeat
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&src_u128(_0{}, m));
            
        // Get destination registers for this repeat
        uint32_t& r0 = dst_u32(_0{}, m);
        uint32_t& r1 = dst_u32(_1{}, m);
        uint32_t& r2 = dst_u32(_2{}, m);
        uint32_t& r3 = dst_u32(_3{}, m);
            
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(smem_addr)
        );
    }
}

// ============================================================================
// SM75 x2 VARIANTS: ldmatrix.sync.aligned.x2.m8n8.shared.b16 (2 regs per load)
// Used when MMA is SM75_16x8x8 (smaller fragment → copy atom is U32x2 / U16x4).
// ============================================================================
template<typename TiledCopy, typename TensorS, typename TensorD>
CUTLASS_DEVICE void educational_copy_smem_to_regs_x2(
    TiledCopy const& tiled_copy,
    TensorS const& src,
    TensorD& dst)
{
    Tensor src_u128 = recast<uint128_t const>(src);
    Tensor dst_u32  = recast<uint32_t>(dst);
    #pragma unroll
    for (int m = 0; m < size<1>(src_u128); ++m) {
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&src_u128(_0{}, m));
        uint32_t& r0 = dst_u32(_0{}, m);
        uint32_t& r1 = dst_u32(_1{}, m);
        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(r0), "=r"(r1)
            : "r"(smem_addr)
        );
    }
}

template<typename TiledCopy, typename TensorS, typename TensorD>
CUTLASS_DEVICE void educational_copy_smem_to_regs_transposed_x2(
    TiledCopy const& tiled_copy,
    TensorS const& src,
    TensorD& dst)
{
    Tensor src_u128 = recast<uint128_t const>(src);
    Tensor dst_u32  = recast<uint32_t>(dst);
    #pragma unroll
    for (int m = 0; m < size<1>(src_u128); ++m) {
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&src_u128(_0{}, m));
        uint32_t& r0 = dst_u32(_0{}, m);
        uint32_t& r1 = dst_u32(_1{}, m);
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(r0), "=r"(r1)
            : "r"(smem_addr)
        );
    }
}

// ============================================================================
// TRANSPOSED VERSION: For loading V matrix with .trans modifier
// ============================================================================
// V is stored transposed in smem for coalesced global→smem loads.
// ldmatrix.trans loads AND transposes in a single instruction.
// ============================================================================
template<typename TiledCopy, typename TensorS, typename TensorD>
CUTLASS_DEVICE void educational_copy_smem_to_regs_transposed(
    TiledCopy const& tiled_copy,
    TensorS const& src,
    TensorD& dst)
{
    // Same recast pattern as non-transposed version
    Tensor src_u128 = recast<uint128_t const>(src);
    Tensor dst_u32  = recast<uint32_t>(dst);

    #pragma unroll
    for (int m = 0; m < size<1>(src_u128); ++m) {
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&src_u128(_0{}, m));
        uint32_t& r0 = dst_u32(_0{}, m);
        uint32_t& r1 = dst_u32(_1{}, m);
        uint32_t& r2 = dst_u32(_2{}, m);
        uint32_t& r3 = dst_u32(_3{}, m);
        
        asm volatile(
            "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(smem_addr)
        );
    }
}

// ============================================================================
// MANUAL ADDRESS CALCULATION: smem→register copy for Q matrix
// ============================================================================
//
// These functions replace the CuTe shared memory tensor (tSsQ) with manual
// index calculation for the Q matrix smem→register copy.  The destination is
// either a CuTe register tensor (tCrQ_copy_view) or a raw uint32_t array,
// depending on the variant used (see _raw_regs variants below).
//
// Address derivation (row-major Q in smem, no swizzle):
//
//   smem_Q layout: (kBlockM, kHeadDim), stride (kHeadDim, 1)
//
//   For a given k_block and MMA-atom repeat m:
//     atom_row_start = m * MmaTileM + warp_id * 16
//
//   Each warp's 32 lanes map to 16 rows of the MMA atom (16 × MmaTileK block):
//     SM75 (x2, MmaTileK=8):
//       sub_matrix  = (lane_id / 8) % 2        — 0 = top 8×8, 1 = bottom 8×8
//       row_in_sub  = lane_id % 8
//       row_offset  = sub_matrix * 8 + row_in_sub
//       col         = k_block * 8
//
//     SM80 (x4, MmaTileK=16):
//       group       = lane_id / 8               — 0..3
//       row_offset  = (group % 2) * 8 + (lane_id % 8)
//       col         = k_block * 16 + (group / 2) * 8
//
//   smem address = smem_Q_base + (atom_row_start + row_offset) * kHeadDim + col
//
// ============================================================================

// SM80 variant (ldmatrix.x4): loads 16×16 block as 4 × 8×8 sub-matrices.
template<int kBlockM_, int kHeadDim_, int kNWarps_, typename Element_, typename TensorD>
CUTLASS_DEVICE void copy_Q_smem_to_regs_manual(
    Element_ const* smem_Q_base,
    int thread_idx,
    int k_block,
    TensorD& dst)
{
    // K-dim of SM80 MMA atom mma.sync.aligned.m16n8k16 (PTX ISA).
    static constexpr int MmaTileK = 16;
    // M-dim of MMA atom (16 rows); all MMA atoms use m16 for fp16/bf16.
    static constexpr int MmaAtomM = 16;
    // Total M rows handled by one MMA tile = MmaAtomM * kNWarps (one atom per warp).
    static constexpr int MmaTileM = 16 * kNWarps_;

    // 32 threads per warp (CUDA warp size).
    int const warp_id = thread_idx / 32;
    int const lane_id = thread_idx % 32;
    // ldmatrix.x4 uses 4 groups of 8 threads; each group provides 8 addresses
    // for one 8×8 sub-matrix (m8n8 — the fundamental ldmatrix unit, PTX ISA).
    int const group = lane_id / 8;          // 0..3 → selects which 8×8 sub-matrix
    int const row_in_group = lane_id % 8;   // 0..7 → row within that 8×8 sub-matrix

    // The 16×16 A-operand is a 2×2 grid of 8×8 sub-matrices (PTX m16n8k16 spec):
    //   group 0 → (rows 0-7,  cols 0-7)    group 2 → (rows 0-7,  cols 8-15)
    //   group 1 → (rows 8-15, cols 0-7)    group 3 → (rows 8-15, cols 8-15)
    // Vertical position: groups 0,2 → top half; groups 1,3 → bottom half.
    int const row_offset = (group % 2) * 8 + row_in_group;
    // Horizontal position: groups 0,1 → left k-half; groups 2,3 → right k-half.
    int const col_offset = (group / 2) * 8;

    Tensor dst_u32 = recast<uint32_t>(dst);

    #pragma unroll
    for (int m = 0; m < size<1>(dst_u32); ++m) {
        int const atom_row_start = m * MmaTileM + warp_id * MmaAtomM;
        int const smem_row = atom_row_start + row_offset;
        int const smem_col = k_block * MmaTileK + col_offset;

        Element_ const* addr = smem_Q_base + smem_row * kHeadDim_ + smem_col;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(addr);

        uint32_t& r0 = dst_u32(_0{}, m);
        uint32_t& r1 = dst_u32(_1{}, m);
        uint32_t& r2 = dst_u32(_2{}, m);
        uint32_t& r3 = dst_u32(_3{}, m);

        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(smem_addr)
        );
    }
}

// SM75 variant (ldmatrix.x2): loads 16×8 block as 2 × 8×8 sub-matrices.
template<int kBlockM_, int kHeadDim_, int kNWarps_, typename Element_, typename TensorD>
CUTLASS_DEVICE void copy_Q_smem_to_regs_manual_x2(
    Element_ const* smem_Q_base,
    int thread_idx,
    int k_block,
    TensorD& dst)
{
    // K-dim of SM75 MMA atom mma.sync.aligned.m16n8k8 (PTX ISA).
    static constexpr int MmaTileK = 8;
    // M-dim of MMA atom (16 rows); all MMA atoms use m16 for fp16/bf16.
    static constexpr int MmaAtomM = 16;
    // Total M rows handled by one MMA tile = MmaAtomM * kNWarps (one atom per warp).
    static constexpr int MmaTileM = 16 * kNWarps_;

    // 32 threads per warp (CUDA warp size).
    int const warp_id = thread_idx / 32;
    int const lane_id = thread_idx % 32;

    // ldmatrix.x2 loads 2 × m8n8 sub-matrices (PTX ISA), stacked vertically
    // in a 16-row × 8-col block.  The 32 lanes divide into 4 groups of 8;
    // only 2 groups contribute unique addresses (groups 2,3 mirror groups 0,1).
    //   groups 0,2 → sub-matrix 0 (rows 0-7)
    //   groups 1,3 → sub-matrix 1 (rows 8-15)
    int const sub_matrix = (lane_id / 8) % 2;  // 0 = top 8×8, 1 = bottom 8×8
    int const row_in_sub = lane_id % 8;         // 0..7 → row within that 8×8
    int const row_offset = sub_matrix * 8 + row_in_sub;

    Tensor dst_u32 = recast<uint32_t>(dst);

    #pragma unroll
    for (int m = 0; m < size<1>(dst_u32); ++m) {
        int const atom_row_start = m * MmaTileM + warp_id * MmaAtomM;
        int const smem_row = atom_row_start + row_offset;
        int const smem_col = k_block * MmaTileK;

        Element_ const* addr = smem_Q_base + smem_row * kHeadDim_ + smem_col;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(addr);

        uint32_t& r0 = dst_u32(_0{}, m);
        uint32_t& r1 = dst_u32(_1{}, m);

        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(r0), "=r"(r1)
            : "r"(smem_addr)
        );
    }
}

// ============================================================================
// MANUAL ADDRESS CALCULATION: smem→register copy for Q matrix — RAW REGISTER variant
// ============================================================================
//
// These functions are identical to copy_Q_smem_to_regs_manual[_x2] in the
// shared-memory address calculation, but write directly to a raw uint32_t
// array instead of a CuTe register tensor.
//
// Flat array layout:
//   Q_regs_k[m * VRegsPerAtom + v]
//   m = MMA atom index in the M direction (per warp), v = uint32 sub-register.
//   SM75 (x2): VRegsPerAtom = 2, so Q_regs_k[m * 2 + v]    (v ∈ {0,1})
//   SM80 (x4): VRegsPerAtom = 4, so Q_regs_k[m * 4 + v]    (v ∈ {0,1,2,3})
//
// The caller passes a pointer to the slice for one k_block:
//   &Q_regs[k_block * RegsPerKBlockQ]
//
// ============================================================================

// SM80 variant (ldmatrix.x4): writes to raw uint32_t array.
template<int kBlockM_, int kHeadDim_, int kNWarps_, typename Element_>
CUTLASS_DEVICE void copy_Q_smem_to_raw_regs(
    Element_ const* smem_Q_base,
    int thread_idx,
    int k_block,
    uint32_t* Q_regs_k)   // output: NAtomsM * 4 uint32 for this k_block
{
    // K-dim of SM80 MMA atom mma.sync.aligned.m16n8k16 (PTX ISA).
    static constexpr int MmaTileK = 16;
    // M-dim of MMA atom (16 rows).
    static constexpr int MmaAtomM = 16;
    // Total M rows per MMA tile = MmaAtomM * kNWarps.
    static constexpr int MmaTileM = 16 * kNWarps_;
    // A-operand registers per atom: 4 uint32 (8 fp16).
    static constexpr int VRegsPerAtom = 4;
    // Number of M-atom steps per warp.
    static constexpr int NAtomsM = kBlockM_ / MmaTileM;

    // 32 threads per warp (CUDA warp size).
    int const warp_id = thread_idx / 32;
    int const lane_id = thread_idx % 32;
    // 4 groups of 8 lanes; each group addresses one 8×8 sub-matrix (PTX ISA).
    int const group = lane_id / 8;          // 0..3
    int const row_in_group = lane_id % 8;   // 0..7

    // Same 2×2 sub-matrix layout as the non-raw Q x4 variant.
    int const row_offset = (group % 2) * 8 + row_in_group;
    int const col_offset = (group / 2) * 8;

    #pragma unroll
    for (int m = 0; m < NAtomsM; ++m) {
        int const atom_row_start = m * MmaTileM + warp_id * MmaAtomM;
        int const smem_row = atom_row_start + row_offset;
        int const smem_col = k_block * MmaTileK + col_offset;

        Element_ const* addr = smem_Q_base + smem_row * kHeadDim_ + smem_col;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(addr);

        // ldmatrix.x4: 4 sequential registers per atom (A-operand is row-major).
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(Q_regs_k[m * VRegsPerAtom + 0]),
              "=r"(Q_regs_k[m * VRegsPerAtom + 1]),
              "=r"(Q_regs_k[m * VRegsPerAtom + 2]),
              "=r"(Q_regs_k[m * VRegsPerAtom + 3])
            : "r"(smem_addr)
        );
    }
}

// SM75 variant (ldmatrix.x2): writes to raw uint32_t array.
template<int kBlockM_, int kHeadDim_, int kNWarps_, typename Element_>
CUTLASS_DEVICE void copy_Q_smem_to_raw_regs_x2(
    Element_ const* smem_Q_base,
    int thread_idx,
    int k_block,
    uint32_t* Q_regs_k)   // output: NAtomsM * 2 uint32 for this k_block
{
    // K-dim of SM75 MMA atom mma.sync.aligned.m16n8k8 (PTX ISA).
    static constexpr int MmaTileK = 8;
    // M-dim of MMA atom (16 rows).
    static constexpr int MmaAtomM = 16;
    // Total M rows per MMA tile = MmaAtomM * kNWarps.
    static constexpr int MmaTileM = 16 * kNWarps_;
    // A-operand registers per atom: 2 uint32 (4 fp16).
    static constexpr int VRegsPerAtom = 2;
    // Number of M-atom steps per warp.
    static constexpr int NAtomsM = kBlockM_ / MmaTileM;

    // 32 threads per warp (CUDA warp size).
    int const warp_id = thread_idx / 32;
    int const lane_id = thread_idx % 32;
    // Same vertical sub-matrix mapping as the non-raw Q x2 variant.
    int const sub_matrix = (lane_id / 8) % 2;  // 0 = top 8×8, 1 = bottom 8×8
    int const row_in_sub = lane_id % 8;         // 0..7
    int const row_offset = sub_matrix * 8 + row_in_sub;

    #pragma unroll
    for (int m = 0; m < NAtomsM; ++m) {
        int const atom_row_start = m * MmaTileM + warp_id * MmaAtomM;
        int const smem_row = atom_row_start + row_offset;
        int const smem_col = k_block * MmaTileK;

        Element_ const* addr = smem_Q_base + smem_row * kHeadDim_ + smem_col;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(addr);

        // ldmatrix.x2: 2 sequential registers per atom.
        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(Q_regs_k[m * VRegsPerAtom + 0]),
              "=r"(Q_regs_k[m * VRegsPerAtom + 1])
            : "r"(smem_addr)
        );
    }
}

// ============================================================================
// MANUAL ADDRESS CALCULATION: smem→register copy for K matrix (B-operand)
// ============================================================================
//
// K is the B-operand of the MMA.  All warps share the same K data (warps are
// distributed along M, not N), so there is no warp-based row offset.
//
//   smem_K layout per stage: (kBlockN, kHeadDim), stride (kHeadDim, 1)
//
//   For a given k_block and ldmatrix repeat n:
//     base_n_row = n * 16
//
//   Thread address within the 16-row block is identical to Q:
//     SM75 (x2):  row_offset = ((lane_id/8)%2)*8 + lane_id%8
//                 col = k_block * 8
//     SM80 (x4):  row_offset = ((lane_id/8)%2)*8 + lane_id%8
//                 col = k_block * 16 + (lane_id/8/2)*8
//
//   smem address = smem_K_stage_base + (base_n_row + row_offset) * kHeadDim + col
//
// The caller must pass a base pointer already adjusted for the pipeline stage:
//   smem_K_stage_base = smem_K_ptr + stage * kBlockN * kHeadDim
//
// ============================================================================

// SM80 variant (ldmatrix.x4): loads 16×16 block as 4 × 8×8 sub-matrices.
template<int kHeadDim_, typename Element_, typename TensorD>
CUTLASS_DEVICE void copy_K_smem_to_regs_manual(
    Element_ const* smem_K_stage_base,
    int thread_idx,
    int k_block,
    TensorD& dst)
{
    // K-dim of SM80 MMA atom mma.sync.aligned.m16n8k16 (PTX ISA).
    static constexpr int MmaTileK = 16;
    // ldmatrix.x4 loads 4 × m8n8 sub-matrices arranged as a 2×2 grid, covering
    // 16 rows (2 vertical × 8) × 16 cols (2 horizontal × 8).  One call therefore
    // advances 16 rows in the N dimension of K.
    static constexpr int NPerLdmatrix = 16;

    // 32 threads per warp (CUDA warp size).
    int const lane_id = thread_idx % 32;
    // 4 groups of 8 lanes; each group addresses one 8×8 sub-matrix (PTX ISA).
    int const group = lane_id / 8;          // 0..3
    int const row_in_group = lane_id % 8;   // 0..7

    // Same 2×2 sub-matrix layout as the Q x4 variant (see comments there).
    int const row_offset = (group % 2) * 8 + row_in_group;
    int const col_offset = (group / 2) * 8;

    Tensor dst_u32 = recast<uint32_t>(dst);

    #pragma unroll
    for (int n = 0; n < size<1>(dst_u32); ++n) {
        int const smem_row = n * NPerLdmatrix + row_offset;
        int const smem_col = k_block * MmaTileK + col_offset;

        Element_ const* addr = smem_K_stage_base + smem_row * kHeadDim_ + smem_col;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(addr);

        uint32_t& r0 = dst_u32(_0{}, n);
        uint32_t& r1 = dst_u32(_1{}, n);
        uint32_t& r2 = dst_u32(_2{}, n);
        uint32_t& r3 = dst_u32(_3{}, n);

        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(smem_addr)
        );
    }
}

// SM75 variant (ldmatrix.x2): loads 16×8 block as 2 × 8×8 sub-matrices.
template<int kHeadDim_, typename Element_, typename TensorD>
CUTLASS_DEVICE void copy_K_smem_to_regs_manual_x2(
    Element_ const* smem_K_stage_base,
    int thread_idx,
    int k_block,
    TensorD& dst)
{
    // K-dim of SM75 MMA atom mma.sync.aligned.m16n8k8 (PTX ISA).
    static constexpr int MmaTileK = 8;
    // ldmatrix.x2 loads 2 × m8n8 sub-matrices stacked vertically, covering
    // 16 rows (2 × 8) × 8 cols.  One call advances 16 N-rows of K.
    static constexpr int NPerLdmatrix = 16;

    // 32 threads per warp (CUDA warp size).
    int const lane_id = thread_idx % 32;
    // Same vertical sub-matrix mapping as the Q x2 variant (see comments there).
    int const sub_matrix = (lane_id / 8) % 2;  // 0 = top 8×8, 1 = bottom 8×8
    int const row_in_sub = lane_id % 8;         // 0..7 → row within that 8×8
    int const row_offset = sub_matrix * 8 + row_in_sub;

    Tensor dst_u32 = recast<uint32_t>(dst);

    #pragma unroll
    for (int n = 0; n < size<1>(dst_u32); ++n) {
        int const smem_row = n * NPerLdmatrix + row_offset;
        int const smem_col = k_block * MmaTileK;

        Element_ const* addr = smem_K_stage_base + smem_row * kHeadDim_ + smem_col;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(addr);

        uint32_t& r0 = dst_u32(_0{}, n);
        uint32_t& r1 = dst_u32(_1{}, n);

        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(r0), "=r"(r1)
            : "r"(smem_addr)
        );
    }
}

// ============================================================================
// MANUAL ADDRESS CALCULATION: smem→register copy for K matrix — RAW REGISTER variant
// ============================================================================
//
// These functions are identical to copy_K_smem_to_regs_manual[_x2] in the
// shared-memory address calculation, but write directly to a raw uint32_t
// array instead of a CuTe register tensor.
//
// Flat array layout:
//   K_regs_k[ns * VRegsPerAtom + v]
//   ns = MMA atom index in the N direction, v = uint32 sub-register index.
//   SM75 (x2): VRegsPerAtom = 1, so K_regs_k[ns]
//   SM80 (x4): VRegsPerAtom = 2, so K_regs_k[ns * 2 + v]
//
// The caller passes a pointer to the slice for one k_block:
//   &K_regs[k_block * RegsPerKBlock]
//
// ============================================================================

// SM80 variant (ldmatrix.x4): writes to raw uint32_t array.
template<int kHeadDim_, int kBlockN_, typename Element_>
CUTLASS_DEVICE void copy_K_smem_to_raw_regs(
    Element_ const* smem_K_stage_base,
    int thread_idx,
    int k_block,
    uint32_t* K_regs_k)   // output: (kBlockN_/8)*2 uint32 for this k_block
{
    // K-dim of SM80 MMA atom mma.sync.aligned.m16n8k16 (PTX ISA).
    static constexpr int MmaTileK = 16;
    // ldmatrix.x4 covers 16 N-rows per call (2 vertical × 8 rows per m8n8).
    static constexpr int NPerLdmatrix = 16;
    // Each MMA atom (8 N-rows) has 2 uint32 B-operand registers (4 fp16).
    static constexpr int VRegsPerAtom = 2;
    // Number of ldmatrix calls to cover all kBlockN_ rows.
    static constexpr int N_CPY = kBlockN_ / NPerLdmatrix;

    // 32 threads per warp (CUDA warp size).
    int const lane_id = thread_idx % 32;
    // 4 groups of 8 lanes; each group addresses one 8×8 sub-matrix (PTX ISA).
    int const group = lane_id / 8;          // 0..3
    int const row_in_group = lane_id % 8;   // 0..7

    // Same 2×2 sub-matrix layout as the Q x4 variant (see comments there).
    int const row_offset = (group % 2) * 8 + row_in_group;
    int const col_offset = (group / 2) * 8;

    #pragma unroll
    for (int n = 0; n < N_CPY; ++n) {
        int const smem_row = n * NPerLdmatrix + row_offset;
        int const smem_col = k_block * MmaTileK + col_offset;

        Element_ const* addr = smem_K_stage_base + smem_row * kHeadDim_ + smem_col;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(addr);

        uint32_t r0, r1, r2, r3;
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(smem_addr)
        );
        // ldmatrix.x4 sub-matrix → MMA atom register mapping:
        //   r0: (rows 0-7,  cols 0-7)  → atom ns=2*n+0, v=0
        //   r1: (rows 8-15, cols 0-7)  → atom ns=2*n+1, v=0
        //   r2: (rows 0-7,  cols 8-15) → atom ns=2*n+0, v=1
        //   r3: (rows 8-15, cols 8-15) → atom ns=2*n+1, v=1
        K_regs_k[(2*n+0) * VRegsPerAtom + 0] = r0;
        K_regs_k[(2*n+1) * VRegsPerAtom + 0] = r1;
        K_regs_k[(2*n+0) * VRegsPerAtom + 1] = r2;
        K_regs_k[(2*n+1) * VRegsPerAtom + 1] = r3;
    }
}

// SM75 variant (ldmatrix.x2): writes to raw uint32_t array.
template<int kHeadDim_, int kBlockN_, typename Element_>
CUTLASS_DEVICE void copy_K_smem_to_raw_regs_x2(
    Element_ const* smem_K_stage_base,
    int thread_idx,
    int k_block,
    uint32_t* K_regs_k)   // output: kBlockN_/8 uint32 for this k_block
{
    // K-dim of SM75 MMA atom mma.sync.aligned.m16n8k8 (PTX ISA).
    static constexpr int MmaTileK = 8;
    // ldmatrix.x2 covers 16 N-rows per call (2 vertical × 8 rows per m8n8).
    static constexpr int NPerLdmatrix = 16;
    // Number of ldmatrix calls to cover all kBlockN_ rows.
    static constexpr int N_CPY = kBlockN_ / NPerLdmatrix;

    // 32 threads per warp (CUDA warp size).
    int const lane_id = thread_idx % 32;
    // Same vertical sub-matrix mapping as the Q x2 variant (see comments there).
    int const sub_matrix = (lane_id / 8) % 2;  // 0 = top 8×8, 1 = bottom 8×8
    int const row_in_sub = lane_id % 8;         // 0..7 → row within that 8×8
    int const row_offset = sub_matrix * 8 + row_in_sub;

    #pragma unroll
    for (int n = 0; n < N_CPY; ++n) {
        int const smem_row = n * NPerLdmatrix + row_offset;
        int const smem_col = k_block * MmaTileK;

        Element_ const* addr = smem_K_stage_base + smem_row * kHeadDim_ + smem_col;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(addr);

        // ldmatrix.x2: r0 → atom ns=2*n+0, r1 → atom ns=2*n+1
        // For SM75 (VRegsPerAtom=1), flat layout: K_regs_k[ns]
        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(K_regs_k[2*n + 0]), "=r"(K_regs_k[2*n + 1])
            : "r"(smem_addr)
        );
    }
}

// ============================================================================
// RawRegs: minimal wrapper around a raw register pointer for use with
// cute::detail::explode.  Provides operator[] (used by explode to access
// register elements) indexing into a contiguous array of uint32_t (or other
// register type) values.
// ============================================================================
template<typename T, int N>
struct RawRegs {
    T* data;
    CUTE_HOST_DEVICE constexpr T& operator[](int i) { return data[i]; }
    CUTE_HOST_DEVICE constexpr T const& operator[](int i) const { return data[i]; }
};

// ============================================================================
// MANUAL ADDRESS CALCULATION: smem→register TRANSPOSED copy for V matrix
// ============================================================================
//
// V is the B-operand of the P×V GEMM.  V is stored in smem in the SAME physical
// layout as K — (kBlockN, kHeadDim) row-major — but accessed through a transposed
// view sVt with logical shape (kHeadDim, kBlockN).  The ldmatrix.trans instruction
// loads rows from the physical layout and transposes each 8×8 sub-matrix on the fly.
//
// This SWAPS the role of row/col in the sub-matrix mapping vs. non-transposed:
//
//   NON-TRANSPOSED (Q, K):                    TRANSPOSED (V):
//     sub-matrix selects ROW blocks             sub-matrix selects COLUMN blocks
//     all sub-matrices share same cols          all sub-matrices share same rows
//
//   Physical smem_V layout: (kBlockN, kHeadDim), stride (kHeadDim, 1)
//
//   SM75 (x2.trans):
//     Loads 8 token-rows × 16 headdim-cols (2 col-adjacent 8×8 sub-matrices).
//     After transpose: 16 headdim × 8 tokens = (MmaTileN, MmaTileK) = (16, 8).
//       smem_row = k_block * 8 + lane_id % 8
//       smem_col = n * 16 + ((lane_id/8) % 2) * 8
//
//   SM80 (x4.trans):
//     Loads 16 token-rows × 16 headdim-cols (2×2 grid of 8×8 sub-matrices).
//     After transpose: 16 headdim × 16 tokens = (MmaTileN, MmaTileK) = (16, 16).
//       group = lane_id / 8
//       smem_row = k_block * 16 + (group / 2) * 8 + lane_id % 8
//       smem_col = n * 16 + (group % 2) * 8
//
//   smem address = smem_V_base + smem_row * kHeadDim + smem_col
//
// ============================================================================

// SM80 variant (ldmatrix.x4.trans): loads+transposes a 16×16 block.
template<int kHeadDim_, int kBlockN_, typename Element_, typename TensorD>
CUTLASS_DEVICE void copy_V_smem_to_regs_manual_transposed(
    Element_ const* smem_V_base,
    int thread_idx,
    int k_block,
    TensorD& dst)
{
    // K-dim of SM80 MMA atom mma.sync.aligned.m16n8k16 (PTX ISA).
    static constexpr int MmaTileK = 16;
    // ldmatrix.x4.trans loads a 2×2 grid of m8n8 sub-matrices (16 rows × 16 cols
    // in physical smem) and transposes each 8×8 independently.
    static constexpr int NPerLdmatrix = 16;

    // 32 threads per warp (CUDA warp size).
    int const lane_id = thread_idx % 32;
    // 4 groups of 8 lanes (PTX ISA: each group addresses one 8×8 sub-matrix).
    int const group = lane_id / 8;
    int const row_in_group = lane_id % 8;

    // TRANSPOSED mapping (group / 2 selects token-ROW block, group % 2 selects
    // headdim-COLUMN block — swapped vs. non-transposed Q/K):
    //   group 0 → token rows 0-7,  headdim cols 0-7
    //   group 1 → token rows 0-7,  headdim cols 8-15
    //   group 2 → token rows 8-15, headdim cols 0-7
    //   group 3 → token rows 8-15, headdim cols 8-15
    int const row_offset = (group / 2) * 8 + row_in_group;   // token rows
    int const col_offset = (group % 2) * 8;                   // headdim cols

    Tensor dst_u32 = recast<uint32_t>(dst);

    #pragma unroll
    for (int n = 0; n < size<1>(dst_u32); ++n) {
        int const smem_row = k_block * MmaTileK + row_offset;
        int const smem_col = n * NPerLdmatrix + col_offset;

        Element_ const* addr = smem_V_base + smem_row * kHeadDim_ + smem_col;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(addr);

        uint32_t& r0 = dst_u32(_0{}, n);
        uint32_t& r1 = dst_u32(_1{}, n);
        uint32_t& r2 = dst_u32(_2{}, n);
        uint32_t& r3 = dst_u32(_3{}, n);

        asm volatile(
            "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(smem_addr)
        );
    }
}

// SM75 variant (ldmatrix.x2.trans): loads+transposes a 8×16 block.
template<int kHeadDim_, int kBlockN_, typename Element_, typename TensorD>
CUTLASS_DEVICE void copy_V_smem_to_regs_manual_transposed_x2(
    Element_ const* smem_V_base,
    int thread_idx,
    int k_block,
    TensorD& dst)
{
    // K-dim of SM75 MMA atom mma.sync.aligned.m16n8k8 (PTX ISA).
    static constexpr int MmaTileK = 8;
    // ldmatrix.x2.trans loads 2 column-adjacent m8n8 sub-matrices (8 rows × 16 cols
    // in physical smem) and transposes each 8×8.
    static constexpr int NPerLdmatrix = 16;

    // 32 threads per warp (CUDA warp size).
    int const lane_id = thread_idx % 32;

    // TRANSPOSED mapping (sub_matrix selects headdim-COLUMN block, not row block):
    //   groups 0,2 → headdim cols 0-7    (sub_matrix 0)
    //   groups 1,3 → headdim cols 8-15   (sub_matrix 1)
    // All sub-matrices share the same 8 token rows.
    int const sub_matrix = (lane_id / 8) % 2;  // selects headdim column block
    int const row_in_sub = lane_id % 8;         // 0..7 → token row within the block

    Tensor dst_u32 = recast<uint32_t>(dst);

    #pragma unroll
    for (int n = 0; n < size<1>(dst_u32); ++n) {
        int const smem_row = k_block * MmaTileK + row_in_sub;
        int const smem_col = n * NPerLdmatrix + sub_matrix * 8;

        Element_ const* addr = smem_V_base + smem_row * kHeadDim_ + smem_col;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(addr);

        uint32_t& r0 = dst_u32(_0{}, n);
        uint32_t& r1 = dst_u32(_1{}, n);

        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(r0), "=r"(r1)
            : "r"(smem_addr)
        );
    }
}

// ============================================================================
// MANUAL ADDRESS CALCULATION: smem→register TRANSPOSED copy for V — RAW REGISTER variant
// ============================================================================
//
// These functions are identical to copy_V_smem_to_regs_manual_transposed[_x2]
// in the shared-memory address calculation, but write directly to a raw
// uint32_t array instead of a CuTe register tensor.
//
// Flat array layout (same as K raw copy — see copy_K_smem_to_raw_regs):
//   V_regs_k[ns * VRegsPerAtom + v]
//   ns = MMA atom index in the N (headdim) direction, v = uint32 sub-register.
//   SM75 (x2.trans): VRegsPerAtom = 1, so V_regs_k[ns]
//   SM80 (x4.trans): VRegsPerAtom = 2, so V_regs_k[ns * 2 + v]
//
// The caller passes a pointer to the slice for one k_block:
//   &V_regs[k_block * RegsPerKBlockV]
//
// ============================================================================

// SM80 variant (ldmatrix.x4.trans): writes to raw uint32_t array.
// kSmemStride_ = physical row stride of V in smem (= kHeadDim, may differ from kHeadDimV_).
// kHeadDimV_   = V head dimension (number of headdim cols to load).
template<int kSmemStride_, int kHeadDimV_, typename Element_>
CUTLASS_DEVICE void copy_V_smem_to_raw_regs_transposed(
    Element_ const* smem_V_base,
    int thread_idx,
    int k_block,
    uint32_t* V_regs_k)   // output: (kHeadDimV_/8)*2 uint32 for this k_block
{
    // K-dim of SM80 MMA atom mma.sync.aligned.m16n8k16 (PTX ISA).
    // For the P×V GEMM, K = kBlockN (token reduction), stepped in MmaTileK.
    static constexpr int MmaTileK = 16;
    // ldmatrix.x4.trans covers 16 headdim-cols per call (2 horizontal × 8 per m8n8).
    static constexpr int NPerLdmatrix = 16;
    // Each MMA atom (8 headdim-rows) has 2 uint32 B-operand registers (4 fp16).
    static constexpr int VRegsPerAtom = 2;
    // Number of ldmatrix calls to cover all kHeadDimV_ columns.
    static constexpr int N_CPY = kHeadDimV_ / NPerLdmatrix;

    // 32 threads per warp (CUDA warp size).
    int const lane_id = thread_idx % 32;
    // 4 groups of 8 lanes (PTX ISA: each group addresses one 8×8 sub-matrix).
    int const group = lane_id / 8;
    int const row_in_group = lane_id % 8;

    // TRANSPOSED mapping (see copy_V_smem_to_regs_manual_transposed):
    //   group/2 selects token-ROW block, group%2 selects headdim-COLUMN block.
    int const row_offset = (group / 2) * 8 + row_in_group;   // token rows
    int const col_offset = (group % 2) * 8;                   // headdim cols

    #pragma unroll
    for (int n = 0; n < N_CPY; ++n) {
        int const smem_row = k_block * MmaTileK + row_offset;
        int const smem_col = n * NPerLdmatrix + col_offset;

        Element_ const* addr = smem_V_base + smem_row * kSmemStride_ + smem_col;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(addr);

        uint32_t r0, r1, r2, r3;
        asm volatile(
            "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(smem_addr)
        );
        // ldmatrix.x4.trans sub-matrix → MMA atom register mapping
        // (same pattern as non-transposed K — see copy_K_smem_to_raw_regs):
        //   r0: headdim 0-7,  tokens 0-7  → atom ns=2*n+0, v=0
        //   r1: headdim 8-15, tokens 0-7  → atom ns=2*n+1, v=0
        //   r2: headdim 0-7,  tokens 8-15 → atom ns=2*n+0, v=1
        //   r3: headdim 8-15, tokens 8-15 → atom ns=2*n+1, v=1
        V_regs_k[(2*n+0) * VRegsPerAtom + 0] = r0;
        V_regs_k[(2*n+1) * VRegsPerAtom + 0] = r1;
        V_regs_k[(2*n+0) * VRegsPerAtom + 1] = r2;
        V_regs_k[(2*n+1) * VRegsPerAtom + 1] = r3;
    }
}

// SM75 variant (ldmatrix.x2.trans): writes to raw uint32_t array.
// kSmemStride_ = physical row stride of V in smem (= kHeadDim, may differ from kHeadDimV_).
// kHeadDimV_   = V head dimension (number of headdim cols to load).
template<int kSmemStride_, int kHeadDimV_, typename Element_>
CUTLASS_DEVICE void copy_V_smem_to_raw_regs_transposed_x2(
    Element_ const* smem_V_base,
    int thread_idx,
    int k_block,
    uint32_t* V_regs_k)   // output: kHeadDimV_/8 uint32 for this k_block
{
    // K-dim of SM75 MMA atom mma.sync.aligned.m16n8k8 (PTX ISA).
    static constexpr int MmaTileK = 8;
    // ldmatrix.x2.trans covers 16 headdim-cols per call (2 horizontal × 8).
    static constexpr int NPerLdmatrix = 16;
    // Number of ldmatrix calls to cover all kHeadDimV_ columns.
    static constexpr int N_CPY = kHeadDimV_ / NPerLdmatrix;

    // 32 threads per warp (CUDA warp size).
    int const lane_id = thread_idx % 32;

    // TRANSPOSED mapping (see copy_V_smem_to_regs_manual_transposed_x2):
    //   sub_matrix selects headdim-COLUMN block (not row block).
    int const sub_matrix = (lane_id / 8) % 2;  // selects headdim column block
    int const row_in_sub = lane_id % 8;         // 0..7 → token row

    #pragma unroll
    for (int n = 0; n < N_CPY; ++n) {
        int const smem_row = k_block * MmaTileK + row_in_sub;
        int const smem_col = n * NPerLdmatrix + sub_matrix * 8;

        Element_ const* addr = smem_V_base + smem_row * kSmemStride_ + smem_col;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(addr);

        // ldmatrix.x2.trans: r0 → atom ns=2*n+0, r1 → atom ns=2*n+1
        // For SM75 (VRegsPerAtom=1), flat layout: V_regs_k[ns]
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(V_regs_k[2*n + 0]), "=r"(V_regs_k[2*n + 1])
            : "r"(smem_addr)
        );
    }
}

// ============================================================================
// HOW cute::copy ORCHESTRATES THIS
// ============================================================================
//
// When you call: cute::copy(smem_tiled_copy, tSsQ(_, _, k), tCrQ_copy_view(_, _, k))
//
// CuTe does the following:
//
// 1. PARTITION: The TiledCopy divides the tile among threads
//    - smem_thr_copy_Q.partition_S(sQ) → creates thread's view of source
//    - smem_thr_copy_Q.retile_D(tSrQ) → creates thread's view of destination regs
//
// 2. ADDRESS CALCULATION: For each thread, compute smem address
//    - Uses the Layout to map (thread_id, value_id) → memory offset
//    - Handles swizzling patterns for bank conflict avoidance
//
// 3. DISPATCH TO PTX: Based on Copy_Atom type, select instruction
//    - SM75_U32x4_LDSM_N → ldmatrix.sync.aligned.x4.m8n8.shared.b16
//    - SM75_U16x8_LDSM_T → ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 (transposed)
//
// 4. EXECUTE: All threads in warp execute ldmatrix together
//
// ============================================================================

// ============================================================================
// Raw-register version of Mask::apply for forward pass (non-SwapAB).
//
// Operates directly on the flat S_regs float array without CuTe tensors.
// Replaces the CuTe-based coordinate mapping (partition_C on identity tensor,
// convert_layout_acc_rowcol) with explicit address arithmetic derived from
// the SM80 m16n8 MMA atom's C/D register layout.
//
// S_regs layout (LayoutLeft from partition_fragment_C):
//   flat index = c0 + r0*2 + m*4 + n*4*NAtomsM_
//   where: r0 = row%2 (0=top half rows 0-7, 1=bottom half rows 8-15 of atom)
//          m  = row/2  (M-atom index per warp)
//          c0 = col%2  (0=even column, 1=odd column within atom)
//          n  = col/2  (N-atom index)
//
// Per-thread tile coordinates (SM80 m16n8 atom, non-SwapAB):
//   lane_id     = thread_idx % 32
//   warp_id     = thread_idx / 32
//   groupID     = lane_id / 4    (selects row within atom: rows groupID, groupID+8)
//   threadInGrp = lane_id % 4    (selects column pair: cols threadInGrp*2, threadInGrp*2+1)
//
//   M-position in tile for rowcol row index 'row':
//     (row/2) * 16*kNWarps_ + warp_id*16 + groupID + (row%2)*8
//
//   Thread-0 N-position for rowcol col index 'col':
//     (col/2) * 8 + (col%2)
//
//   thread_col_offset = threadInGrp * 2
// ============================================================================
template <bool Seqlenk_mask, bool Causal_mask, bool Local_mask,
          int kBlockM_, int kBlockN_, int kNWarps_,
          int NAtomsM_, int NAtomsN_, bool PackGQA_>
CUTLASS_DEVICE void apply_mask_raw_regs(
    float* S_regs,
    int thread_idx,
    int m_block, int n_block,
    int seqlen_q, int seqlen_k,
    int window_size_left, int window_size_right,
    int sink_token_length,
    cutlass::FastDivmod const& attention_chunk_divmod,
    cutlass::FastDivmod const& qhead_per_khead_divmod)
{
    static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
    if constexpr (!Seqlenk_mask && !Causal_mask && !Local_mask) { return; }

    static constexpr int kVRegsPerAtom = 4;    // float[4] per C/D atom
    static constexpr int kNRows = 2 * NAtomsM_; // total row indices per thread
    static constexpr int kNCols = 2 * NAtomsN_; // total col indices per thread

    // SM80 m16n8: 4 threads share each row (same groupID, different threadInGrp).
    static constexpr int kMmaThreadsPerRow = 4;
    static_assert(!PackGQA_ || kNRows <= kMmaThreadsPerRow);

    int const lane_id = thread_idx % 32;
    int const warp_id = thread_idx / 32;
    int const groupID = lane_id / 4;
    int const threadInGrp = lane_id % 4;

    // This thread's column offset (= N-position of its first element).
    int const thread_col_offset = threadInGrp * 2;
    int const seqlenk_col_limit = seqlen_k - n_block * kBlockN_ - thread_col_offset;

    // rowcol (row, col) → flat S_regs offset.
    auto s_idx = [](int row, int col) -> int {
        return (col & 1) + (row & 1) * 2
             + (row >> 1) * kVRegsPerAtom
             + (col >> 1) * kVRegsPerAtom * NAtomsM_;
    };

    // Thread-0 N-position for rowcol col index.
    auto t0_col = [](int col) -> int {
        return (col >> 1) * 8 + (col & 1);
    };

    if constexpr (!Causal_mask && !Local_mask) {
        // Seqlenk_mask only: mask entire columns beyond seqlen_k.
        if constexpr (Seqlenk_mask) {
            #pragma unroll
            for (int col = 0; col < kNCols; ++col) {
                if (t0_col(col) >= seqlenk_col_limit) {
                    #pragma unroll
                    for (int row = 0; row < kNRows; ++row) {
                        S_regs[s_idx(row, col)] = -INFINITY;
                    }
                }
            }
        }
    } else {
        // M-position within tile for rowcol row index.
        auto row_M_pos = [&](int row) -> int {
            return (row >> 1) * 16 * kNWarps_ + warp_id * 16 + groupID + (row & 1) * 8;
        };

        // PackGQA: precompute divided row index, then share via warp shuffle.
        int mma_m_idx;
        if constexpr (PackGQA_) {
            int pack_row = thread_idx % kMmaThreadsPerRow;
            mma_m_idx = qhead_per_khead_divmod.divide(
                m_block * kBlockM_ + row_M_pos(pack_row));
        }

        int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN_
                                        - seqlen_q - thread_col_offset;

        if constexpr (Causal_mask) {
            #pragma unroll
            for (int row = 0; row < kNRows; ++row) {
                int const row_idx = !PackGQA_
                    ? row_M_pos(row) + m_block * kBlockM_
                    : __shfl_sync(0xffffffff, mma_m_idx,
                                  row % kMmaThreadsPerRow, kMmaThreadsPerRow);
                int const col_limit_right = !Seqlenk_mask
                    ? row_idx + causal_row_offset
                    : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
                #pragma unroll
                for (int col = 0; col < kNCols; ++col) {
                    if (t0_col(col) >= col_limit_right) {
                        S_regs[s_idx(row, col)] = -INFINITY;
                    }
                }
            }
        } else { // Local_mask
            int const local_row_offset_right = causal_row_offset + window_size_right;
            int const local_row_offset_left  = causal_row_offset - 1 - window_size_left;
            int const col_limit_sink = sink_token_length - n_block * kBlockN_;
            #pragma unroll
            for (int row = 0; row < kNRows; ++row) {
                int const row_idx = !PackGQA_
                    ? row_M_pos(row) + m_block * kBlockM_
                    : __shfl_sync(0xffffffff, mma_m_idx,
                                  row % kMmaThreadsPerRow, kMmaThreadsPerRow);
                int col_limit_right = !Seqlenk_mask
                    ? row_idx + local_row_offset_right
                    : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
                int col_limit_left = row_idx + local_row_offset_left;
                if (attention_chunk_divmod.divisor > 0) {
                    int col_limit_left_chunk =
                        flash::round_down(attention_chunk_divmod,
                                          row_idx + seqlen_k - seqlen_q)
                        - n_block * kBlockN_ - thread_col_offset;
                    col_limit_left  = std::max(col_limit_left,  col_limit_left_chunk);
                    col_limit_right = std::min(col_limit_right,
                                              col_limit_left_chunk + attention_chunk_divmod.divisor);
                }
                #pragma unroll
                for (int col = 0; col < kNCols; ++col) {
                    int const col_idx = t0_col(col);
                    if (col_idx >= col_limit_right ||
                        (col_idx < col_limit_left && col_idx >= col_limit_sink)) {
                        S_regs[s_idx(row, col)] = -INFINITY;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Raw-register version of Softmax::max_get_scale.
//
// Operates directly on the flat S_regs array using the same rowcol index
// mapping as apply_mask_raw_regs.  Row-wise max is reduced across 4 threads
// via __shfl_xor_sync (Allreduce<4> butterfly), matching CuTe's quad_allreduce_.
//
// Parameters:
//   S_regs          – attention scores (read-only here)
//   row_max         – [kNRows] per-row running maximum (updated in-place)
//   row_sum         – [kNRows] per-row running sum (scaled on non-first iter)
//   scores_scale    – [kNRows] output: scale factors for rescale_o
//   softmax_scale_log2 – log2(softmax_scale), pre-multiplied
// ============================================================================
template <bool Is_first, bool Check_inf,
          int NAtomsM_, int NAtomsN_>
CUTLASS_DEVICE void max_get_scale_raw_regs(
    float const* S_regs,
    float* row_max,
    float* row_sum,
    float* scores_scale,
    float softmax_scale_log2)
{
    static constexpr int kVRegsPerAtom = 4;
    static constexpr int kNRows = 2 * NAtomsM_;
    static constexpr int kNCols = 2 * NAtomsN_;

    // rowcol (row, col) → flat S_regs offset  (same mapping as mask).
    auto s_idx = [](int row, int col) -> int {
        return (col & 1) + (row & 1) * 2
             + (row >> 1) * kVRegsPerAtom
             + (col >> 1) * kVRegsPerAtom * NAtomsM_;
    };

    if constexpr (Is_first) {
        // First iteration: compute row_max from scratch, scores_scale = 1.
        #pragma unroll
        for (int row = 0; row < kNRows; ++row) {
            float mx = S_regs[s_idx(row, 0)];
            #pragma unroll
            for (int col = 1; col < kNCols; ++col) {
                mx = fmaxf(mx, S_regs[s_idx(row, col)]);
            }
            // Allreduce<4> butterfly: reduce across 4 threads sharing this row.
            mx = fmaxf(mx, __shfl_xor_sync(uint32_t(-1), mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(uint32_t(-1), mx, 1));
            row_max[row] = mx;
            scores_scale[row] = 1.f;
        }
    } else {
        // Subsequent iterations: update row_max, compute scale, adjust row_sum.
        #pragma unroll
        for (int row = 0; row < kNRows; ++row) {
            float const prev_max = row_max[row];
            float mx = prev_max;                    // start from previous max
            #pragma unroll
            for (int col = 0; col < kNCols; ++col) {
                mx = fmaxf(mx, S_regs[s_idx(row, col)]);
            }
            // Allreduce<4> butterfly.
            mx = fmaxf(mx, __shfl_xor_sync(uint32_t(-1), mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(uint32_t(-1), mx, 1));
            row_max[row] = mx;

            float const cur_max = !Check_inf
                ? mx
                : (mx == -INFINITY ? 0.0f : mx);
            scores_scale[row] = exp2f((prev_max - cur_max) * softmax_scale_log2);
            row_sum[row] *= scores_scale[row];
        }
    }
}

// ============================================================================
// Raw-register version of Softmax::online_softmax.
//
// Applies exp2 scaling to each element of S_regs in-place, then accumulates
// the per-row sum (thread-local only, no warp reduction – that happens in
// finalize).
//
// score = exp2f(score * softmax_scale_log2 - max_scaled)
//
// where max_scaled = row_max * softmax_scale_log2 - Max_offset
//       (with -inf guard when Check_inf is true).
// ============================================================================
template <bool Is_first, bool Check_inf, int Max_offset,
          int NAtomsM_, int NAtomsN_>
CUTLASS_DEVICE void online_softmax_raw_regs(
    float* S_regs,
    float const* row_max,
    float* row_sum,
    float softmax_scale_log2)
{
    static constexpr int kVRegsPerAtom = 4;
    static constexpr int kNRows = 2 * NAtomsM_;
    static constexpr int kNCols = 2 * NAtomsN_;
    static constexpr float max_offset = float(Max_offset);

    auto s_idx = [](int row, int col) -> int {
        return (col & 1) + (row & 1) * 2
             + (row >> 1) * kVRegsPerAtom
             + (col >> 1) * kVRegsPerAtom * NAtomsM_;
    };

    #pragma unroll
    for (int row = 0; row < kNRows; ++row) {
        // Compute scaled max for this row (matches scale_apply_exp2).
        float const max_scaled = Check_inf
            ? (row_max[row] == -INFINITY ? 0.f : row_max[row] * softmax_scale_log2 - max_offset)
            : row_max[row] * softmax_scale_log2 - max_offset;

        // Apply exp2 to each element, accumulate row sum (thread-local).
        float sum = Is_first ? 0.f : row_sum[row];
        #pragma unroll
        for (int col = 0; col < kNCols; ++col) {
            int const idx = s_idx(row, col);
            S_regs[idx] = exp2f(S_regs[idx] * softmax_scale_log2 - max_scaled);
            sum += S_regs[idx];
        }
        row_sum[row] = sum;
    }
}

// ============================================================================
// Raw-register version of Softmax::rescale_o.
//
// Rescales the O accumulator (output registers) by per-row scores_scale
// factors.  Uses the same rowcol index mapping as the S accumulator, but
// parameterized on NAtomsN_O (= kHeadDimV / 8) instead of NAtomsN_S.
//
// O_regs layout (LayoutLeft from partition_fragment_C):
//   flat index = (col & 1) + (row & 1) * 2 + (row / 2) * 4 + (col / 2) * 4 * NAtomsM_
// ============================================================================
template <int NAtomsM_, int NAtomsN_>
CUTLASS_DEVICE void rescale_o_raw_regs(
    float* O_regs,
    float const* scores_scale)
{
    static constexpr int kVRegsPerAtom = 4;
    static constexpr int kNRows = 2 * NAtomsM_;
    static constexpr int kNCols = 2 * NAtomsN_;

    auto o_idx = [](int row, int col) -> int {
        return (col & 1) + (row & 1) * 2
             + (row >> 1) * kVRegsPerAtom
             + (col >> 1) * kVRegsPerAtom * NAtomsM_;
    };

    #pragma unroll
    for (int row = 0; row < kNRows; ++row) {
        float const scale = scores_scale[row];
        #pragma unroll
        for (int col = 0; col < kNCols; ++col) {
            O_regs[o_idx(row, col)] *= scale;
        }
    }
}

template <int kNWarps, int Stages, bool Q_in_regs, class TileShape_MNK_, int kHeadDimV, class Element_, class ElementAccum_, class ArchTag_,
        bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool PagedKV_, bool AppendKV_,
        bool PackGQA_, bool Split_>
struct CollectiveMainloopFwdSm80 {

    static constexpr int kStages = Stages;
    static_assert(kStages > 0, "kStages must be greater than 0");
    using TileShape_MNK = TileShape_MNK_;
    using TileShape_MNK_PV = Shape<decltype(get<0>(TileShape_MNK{})), Int<kHeadDimV>, decltype(get<1>(TileShape_MNK{}))>;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool PagedKV = PagedKV_;
    static constexpr bool AppendKV = AppendKV_;
    static constexpr bool PackGQA = PackGQA_;
    static constexpr bool Split = Split_;
    static constexpr bool Transpose_V = Is_FP8;

    static_assert(ArchTag::kMinComputeCapability >= 75);

    static constexpr bool Has_cp_async = /*ArchTag::kMinComputeCapability >= 80*/false;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    using SeqlenInfo_t = flash::SeqlenInfoQKNewK<Varlen, AppendKV>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local, PackGQA, Split>;

    // SM80: 16x8x16 MMA atom → tile K = 16.  SM75: 16x8x8 MMA atom → tile K = 8.
    static constexpr bool UseSM80MMA = /*ArchTag::kMinComputeCapability >= 80*/ false;
    using MMA_Atom_Arch = std::conditional_t<
        UseSM80MMA,
        std::conditional_t<
            std::is_same_v<Element, cutlass::half_t>,
            MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
            MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
        >,
        MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>
    >;
    using MmaTileK = std::conditional_t<UseSM80MMA, _16, _8>;
    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, MmaTileK>>;

    static constexpr int NumMmaThreads = size(TiledMma{});
    static constexpr int NumProducerThreads = NumMmaThreads;  // For compatibility with TileScheduler

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g. if hdim=128, we want each
    // thread to have 4 loads in the M direction and 2 vectorized load in the K direction.
    static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);

    static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
    static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
    using SmemLayoutAtomQKV = decltype(
        //composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                    Layout<Shape<_8, Int<kBlockKGmem>>,
                           Stride<Int<kBlockKGmem>, _1>>{}/*)*/);
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQKV{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomQKV{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomQKV{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutVt = decltype(
        composition(SmemLayoutV{},
                    make_ordered_layout(make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}),
                                        Step<_2, _1, _3>{})));

    // SM75 16x8x8 MMA yields fewer vals per thread; need x2/x4 LDSM atoms so TiledNumVal % AtomNumVal.
    using SmemCopyAtom = Copy_Atom<std::conditional_t<UseSM80MMA, SM75_U32x4_LDSM_N, SM75_U32x2_LDSM_N>, Element>;
    using SmemCopyAtomTransposed = Copy_Atom<std::conditional_t<UseSM80MMA, SM75_U16x8_LDSM_T, SM75_U16x4_LDSM_T>, Element>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using GmemCopyAtom = Copy_Atom<std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>,
        AutoVectorizingCopyWithAssumedAlignment<128>
    >, Element>;

    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static constexpr int thread_rows = NumMmaThreads / kGmemThreadsPerRow;
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(GmemCopyAtom{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per read
    // So that we don't have to check if we overshot kBlockM when we load Q
    static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);

    // For AppendKV, We want each thread to have at least 2 loads in the K direction since in the case of
    // non-interleaved rotary (combining elements at indices 0 and rotary_dim/2, 1 and rotary_dim/2+1, etc),
    // each thread will load twice from the same row.
    static constexpr int kBytePerHalfRow = kHeadDim / 2 * sizeof(Element);
    static constexpr int kBlockKGmemAppend = (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRowAppend = kBlockKGmemAppend / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRowAppend == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRowAppend");
    // We assume threads loading the same row are in the same warp. This is for an optimization in PagedKV where
    // these threads share the same page table entry and share the work of computing pointers to paged K and paged V.
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRowAppend == 0, "kGmemThreadsPerRowAppend must divide NumThreadsPerWarp");
    using GmemLayoutAtomAppend = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRowAppend>, Int<kGmemThreadsPerRowAppend>>,
                                        Stride<Int<kGmemThreadsPerRowAppend>, _1>>;
    // If AppendKV, we'll be loading Q for rotary, and we assume divisibility to avoid predication
    static_assert(!AppendKV || kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtomAppend{})) == 0, "kBlockM must be a multiple of NumMmaThreads / kGmemThreadsPerRowAppend");
    using GmemTiledCopyAppendKV = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtomAppend{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideV = StrideQK;
    // ((qhead_per_khead, seqlen_q), d, nheads_kv, batch, num_splits)
    using ShapeQPacked = std::conditional_t<!PackGQA, ShapeQKV, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
    using StrideQPacked = std::conditional_t<!PackGQA, StrideQK, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t>>;
    using ShapePageTable = cute::Shape<int32_t, int32_t>;  // (batch, max_num_pages_per_seq)
    using StridePageTable = cute::Stride<int64_t, _1>;
    using ShapeRotary = cute::Shape<int32_t, int32_t>;  // (seqlen_ro, rotary_dim // 2)
    using StrideRotary = cute::Stride<int64_t, _1>;
    using StrideDescale = cute::Stride<int64_t, int64_t>;

    static constexpr bool Share_QV_Smem = Q_in_regs;

    struct TensorStorageSharedQV : cute::aligned_struct<128> {
        union {
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    };

    struct TensorStorageSeparateQV : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    };

    using TensorStorage = /*std::conditional_t<Share_QV_Smem, TensorStorageSharedQV, */TensorStorageSeparateQV/*>*/;

    // Host side kernel arguments
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        Element* const ptr_K;  // Not Element const* since we might append to KV cache in-place
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        int32_t const headdim_v;
        StrideV const stride_V;
        Element const* const ptr_K_new;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        Element const* const ptr_V_new;
        StrideV const stride_V_new;
        Element const* const ptr_Qv;
        StrideQK const stride_Qv;
        Element const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        Element const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        bool const is_rotary_interleaved;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        float const softmax_scale;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        int const window_size_left = -1, window_size_right = -1, attention_chunk = 0;
        float const softcap_val;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
        int const* const seqlens_rotary = nullptr;
    };

    // Device side kernel params
    struct Params {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        ShapeQPacked const shape_Q_packed;
        StrideQPacked const stride_Q_packed;
        Element* const ptr_K;
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        int32_t const headdim_v;
        StrideV const stride_V;
        Element const* const ptr_K_new;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        Element const* const ptr_V_new;
        StrideV const stride_V_new;
        Element const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        Element const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        bool const is_rotary_interleaved;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        cutlass::FastDivmod page_size_divmod;
        cutlass::FastDivmod qhead_per_khead_divmod;
        float const softmax_scale_log2;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        float const softcap_val;
        int const window_size_left, window_size_right;
        cutlass::FastDivmod attention_chunk_divmod;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
        int const* const seqlens_rotary = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        // If PackGQA, reshape Q to be ((qhead_per_khead, seqlen_q), head_size, nhead_k, batch_size)
        int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K));
        auto const shape_Q_packed = cute::conditional_return<!PackGQA>(
            args.shape_Q,
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_Q)), get<1>(args.shape_Q), get<2>(args.shape_K), get<3>(args.shape_Q))
        );
        auto const stride_Q_packed = cute::conditional_return<!PackGQA>(
            args.stride_Q,
            make_stride(make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), get<1>(args.stride_Q), get<2>(args.stride_Q) * qhead_per_khead, get<3>(args.stride_Q))
        );
        if (get<1>(args.shape_rotary) > 0) {
            assert(args.ptr_rotary_cos != nullptr && args.ptr_rotary_sin != nullptr);
        }
        assert(args.num_splits >= 1);
        // Avoid dividing by zero
        cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
        attention_chunk_divmod.divisor = args.attention_chunk;
        // If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        // Right after this, we multiply by log2(e) before applying exp2.
        // To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        // (assigning it to params.softcap_val) and pre-multiply softcap_val * log2(e)
        // (assigning it to params.softmax_scale_log2).
        return {args.ptr_Q, args.shape_Q, args.stride_Q, shape_Q_packed, stride_Q_packed,
                args.ptr_K, args.shape_K, args.stride_K, args.ptr_V, args.headdim_v, args.stride_V,
                args.ptr_K_new, args.shape_K_new, args.stride_K_new, args.ptr_V_new, args.stride_V_new,
                args.ptr_rotary_cos, args.shape_rotary, args.stride_rotary_cos,
                args.ptr_rotary_sin, args.stride_rotary_sin, args.is_rotary_interleaved,
                args.ptr_pagetable, args.shape_pagetable, args.stride_pagetable,
                cutlass::FastDivmod(int(get<0>(args.shape_K))),
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                /*!Has_softcap ? */float(args.softmax_scale * M_LOG2E)/* : float(args.softcap_val * M_LOG2E)*/,
                args.ptr_q_descale, args.ptr_k_descale, args.ptr_v_descale,
                args.stride_q_descale, args.stride_k_descale, args.stride_v_descale,
                /*!Has_softcap ? */0.f/* : args.softmax_scale / args.softcap_val*/,
                args.window_size_left, args.window_size_right, attention_chunk_divmod,
                !Split ? 1 : args.num_splits,
                args.kv_batch_idx,
                args.cu_seqlens_q, args.cu_seqlens_k, args.cu_seqlens_k_new,
                args.seqused_q, args.seqused_k, args.leftpad_k, args.seqlens_rotary};
    }

    template <typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        FrgTensorO& tOrO,
        Softmax& softmax,
        int const thread_idx,
        SeqlenInfo_t const& seqlen_info,
        cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        // can't use auto [m_block, ...] = block_coord since structured binding cannot be captured in lambda
        int const m_block = get<0>(block_coord);
        int const bidh = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        int const split_idx = get<3>(block_coord);
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        auto n_block_min_max = BlockMN_t::get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
        int const n_block_min = get<0>(n_block_min_max);
        int const n_block_max = get<1>(n_block_min_max);
        // It's possible to have n_block_max <= n_block_min. We don't want to load Q or change any barrier
        //if constexpr (Is_causal || Is_local || Varlen || Split) {
            if (n_block_max <= n_block_min)
            {
                return false;
            }
        //}

#if FLASH_USE_CUTLASS_TENSOR
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});

        bool const is_varlen_q = Varlen && params.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;

        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];
        Tensor mQ = make_tensor(make_gmem_ptr(params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q)), params.shape_Q_packed, params.stride_Q_packed)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor gQ = local_tile(mQ, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K + seqlen_info.offset_k * get<0>(params.stride_K)), params.shape_K, params.stride_K)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor gK = local_tile(mK, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V + seqlen_info.offset_k * get<0>(params.stride_V)), params.shape_K, params.stride_V)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor gV = local_tile(mV, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
#else
        static_assert(!PackGQA, "Raw register path does not support PackGQA");
        bool const is_varlen_q = Varlen && params.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];

        // Smem base pointers (CuTe: raw_pointer_cast(sQ/sK/sV.data()))
        Element* smem_Q_base = shared_storage.tensors.mainloop.smem_q.data();
        Element* smem_K_base = shared_storage.tensors.mainloop.smem_k.data();
        Element* smem_V_base = shared_storage.tensors.mainloop.smem_v.data();

        // Gmem base pointers after head/batch indexing
        // CuTe: raw_pointer_cast(gQ.data()) = ptr_Q + varlen_offset + head_offset + batch_offset + m_block_offset
        // CuTe: raw_pointer_cast(gK.data()) = ptr_K + varlen_offset + head_offset + batch_offset
        // CuTe: raw_pointer_cast(gV.data()) = ptr_V + varlen_offset + head_offset + batch_offset
        // Stride layout: ShapeQKV=(seqlen, d, head, batch), StrideQK=(row_stride, _1, head_stride, batch_stride)
        Element const* gmem_Q_base = params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q) + bidh    * get<2>(params.stride_Q) + (!is_varlen_q ? bidb    : 0) * get<3>(params.stride_Q) + m_block * kBlockM * get<0>(params.stride_Q);
        Element const* gmem_K_base = params.ptr_K + seqlen_info.offset_k * get<0>(params.stride_K) + bidh_kv * get<2>(params.stride_K) + (!is_varlen_k ? bidb_kv : 0) * get<3>(params.stride_K);
        Element const* gmem_V_base = params.ptr_V + seqlen_info.offset_k * get<0>(params.stride_V) + bidh_kv * get<2>(params.stride_V) + (!is_varlen_k ? bidb_kv : 0) * get<3>(params.stride_V);
#endif

#if FLASH_USE_CUTLASS_TENSOR
        GmemTiledCopyQKV gmem_tiled_copy_QKV;
        auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(_0{});  // For index calculation

        Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
        Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
        Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
        Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_slice(thread_idx);

        // Allocate "fragments/descriptors"
        Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

        // Copy Atom retiling
        auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
        auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
        auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
        auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);

        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
        Tensor tSsK = smem_thr_copy_K.partition_S(sK);
        Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
#else
        Element const* smem_Q_ptr = smem_Q_base;
        Element const* smem_K_ptr = smem_K_base;
        Element const* smem_V_ptr = smem_V_base;
#endif

        // Predicates
#if FLASH_USE_CUTLASS_TENSOR
        Tensor cKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
        Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
        Tensor t0KVcKV = gmem_thr0_copy_QKV.partition_S(cKV);
        Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k)
        {
            tKVpKV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(params.shape_K);
        }
#else
        // K-dimension predicates for KV: check which K-chunks are within valid head dimension.
        // CuTe: tKVpKV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(params.shape_K)
        //   where get<1>(tKVcKV(_0{}, _0{}, k)) = thread_k_base + k * kBlockKGmem
        // CuTe: size<2>(tKsK) = kHeadDim / kBlockKGmem
        int const thread_k_base_KV = (thread_idx % kGmemThreadsPerRow) * kGmemElemsPerLoad;
        int const headdim_K = get<1>(params.shape_K);
        static constexpr int num_k_preds = kHeadDim / kBlockKGmem;
        bool tKVpKV_raw[num_k_preds];
        #pragma unroll
        for (int k = 0; k < num_k_preds; ++k) {
            tKVpKV_raw[k] = thread_k_base_KV + k * kBlockKGmem < headdim_K;
        }
#endif

        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        int n_block = n_block_max - 1;

        // Prologue: load Q, K, V
        // If persistent, we don't need to wait for the previous work_idx to finish
        // since we assume that all MMA threads sync in the epilogue before writing to smem_o.
        // So any thread gets there, all threads must have finished the previous MMA and at least started
        // writing to smem_o.
        // If persistent, need to sync to make sure all threads have finished with smem_o before writing to smem_v
        //if constexpr (Share_QV_Smem) { __syncthreads(); }
        //if constexpr (!PackGQA) {
#if FLASH_USE_CUTLASS_TENSOR
            Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
            Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
            Tensor cQ = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
            Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
            Tensor t0QcQ = gmem_thr0_copy_QKV.partition_S(cQ);
            Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k)
            {
                tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(params.shape_Q);
            }
#else
            // K-dimension predicates: check which K-chunks are within valid head dimension.
            // CuTe: tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(params.shape_Q)
            //   where get<1>(tQcQ(_0{}, _0{}, k)) = thread_k_base + k * kBlockKGmem
            int const thread_k_base_Q = (thread_idx % kGmemThreadsPerRow) * kGmemElemsPerLoad;
            int const headdim_Q = get<1>(params.shape_Q);
            bool tQpQ_raw[kHeadDim / kBlockKGmem];
            #pragma unroll
            for (int k = 0; k < kHeadDim / kBlockKGmem; ++k) {
                tQpQ_raw[k] = thread_k_base_Q + k * kBlockKGmem < headdim_Q;
            }
#endif
#if 0
            // Instead of passing in tQcQ, we pass in t0QcQ and subtract the offset from the limit
            // (seqlen_q - m_block * kBlockM). This is because the entries of t0QcQ are known at compile time.
            // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
                gmem_tiled_copy_QKV, tQgQ, tQsQ, t0QcQ, tQpQ, seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}))
            );
#else
            // Q gmem→smem copy with bounds checking (Clear_OOB_MN=false, Clear_OOB_K=true)
#if FLASH_USE_CUTLASS_TENSOR
            int const max_valid_m = seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}));
            int const num_vec_copies = size<0>(tQgQ);
            int const num_m_elements = size<1>(tQgQ);
            int const num_k_elements = size<2>(tQgQ);
#else
            // CuTe: get<0>(tQcQ(_0{}, _0{}, _0{})) = thread's first M-row in partition
            int const max_valid_m = seqlen_info.seqlen_q - m_block * kBlockM
                - int(thread_idx / kGmemThreadsPerRow);
            // CuTe: size<0/1/2>(tQgQ)
            static constexpr int num_vec_copies = kGmemElemsPerLoad;
            static constexpr int num_m_elements = kBlockM / thread_rows;
            static constexpr int num_k_elements = kHeadDim / kBlockKGmem;
            // Gmem strides (CuTe: tQgQ.layout()(1,0,0), (0,1,0), (0,0,1))
            int const q_row_stride = static_cast<int>(get<0>(params.stride_Q));
            int const stride_v_Q = 1;
            int const stride_m_Q = thread_rows * thread_rows * q_row_stride;
            int const stride_k_Q = kBlockKGmem;
            // Smem strides (CuTe: tQsQ.layout()(1,0,0), (0,1,0), (0,0,1))
            static constexpr int stride_sv_Q = 1;
            static constexpr int stride_sm_Q = thread_rows * kHeadDim;
            static constexpr int stride_sk_Q = kBlockKGmem;
            // Thread base offsets in gmem/smem partitions
            int const threads_along_k = kGmemThreadsPerRow;
            int const thread_base_row_stride = (kHeadDim / 2) * kGmemThreadsPerRow;
            int const thread_base_offset = (thread_idx / threads_along_k) * thread_base_row_stride
                                         + (thread_idx % threads_along_k) * threads_along_k;
            Element const* gmem_tile_base = gmem_Q_base;
            Element* smem_tile_base = smem_Q_base;
#endif
            #pragma unroll
            for (int vec = 0; vec < num_vec_copies; ++vec) {
                #pragma unroll
                for (int m = 0; m < num_m_elements; ++m) {
#if FLASH_USE_CUTLASS_TENSOR
                    int m_coord = get<0>(t0QcQ(vec, m, _0{}));
#else
                    int m_coord = m * thread_rows;  // CuTe: get<0>(t0QcQ(vec, m, _0{}))
#endif
                    if (m_coord < max_valid_m) {
                        #pragma unroll
                        for (int k = 0; k < num_k_elements; ++k) {
#if FLASH_USE_CUTLASS_TENSOR
                            if (tQpQ(k)) {
                                tQsQ(vec, m, k) = tQgQ(vec, m, k);
                            } else {
                                tQsQ(vec, m, k) = Element(0);
                            }
#else
                            int const smem_idx = thread_idx * threads_along_k
                                + vec * stride_sv_Q + m * stride_sm_Q + k * stride_sk_Q;
                            if (tQpQ_raw[k]) {  // CuTe: tQpQ(k)
                                int const gmem_idx = thread_base_offset
                                    + vec * stride_v_Q + m * stride_m_Q + k * stride_k_Q;
                                smem_tile_base[smem_idx] = gmem_tile_base[gmem_idx];
                            } else {
                                smem_tile_base[smem_idx] = Element(0);
                            }
#endif
                        }
                    }
                }
            }
#endif
        //} else {
        //    using PackGQAt = flash::PackGQAManager<get<0>(TileShape_MNK{}), get<2>(TileShape_MNK{}), NumMmaThreads, Element>;
        //    PackGQAt::load_Q(mQ, sQ, params.qhead_per_khead_divmod, thread_idx, seqlen_q, m_block);
        //}
        cute::cp_async_fence();

        //using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumMmaThreads, Element, true /*KV_Same_Iter*/>;
        //PagedKVManager_t paged_kv_manager(
        //    params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
        //    params.ptr_K, params.shape_K, params.stride_K,
        //    params.ptr_V, params.headdim_v, params.stride_V,
        //    params.page_size_divmod,
        //    params.page_size_divmod /*blockN_per_page_size_divmod, not used since we don't use TMA*/,
        //    bidb_kv, bidh_kv, thread_idx, seqlen_info.seqlen_k, seqlen_info.leftpad_k,
        //    0 /*bidb_kv_idx, not used since we don't use TMA for Sm8x*/
        //);

        auto load_K = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
            static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
            //if constexpr (!PagedKV) {
                // Do we need bound check to make sure the row doesn't go above kBlockN
                static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
#if FLASH_USE_CUTLASS_TENSOR
                Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_write);
                // Instead of passing in tKVcKV, we pass in t0KVcKV and subtract the offset from the limit
                // (seqlen_k - n_block * kBlockN). This is because the entries of t0KVcKV are known at compile time.
                int const seqlenk_row_limit = -int(get<0>(tKVcKV(_0{}, _0{}, _0{}))) + (EvenN
                    ? seqlen_info.seqlen_k - n_block * kBlockN
                    : (!Seqlenk_mask ? kBlockN : std::min(seqlen_info.seqlen_k - n_block * kBlockN, kBlockN)));
                // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
                flash::copy</*Is_even_MN=*/!Seqlenk_mask && EvenN, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
                    gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK_cur, t0KVcKV, tKVpKV, seqlenk_row_limit);
#else
                // ============================================================================
                // MANUAL REWRITE - What it does under the hood:
                // ============================================================================

                // Step 1: Extract current K tile for this n_block and pipeline stage
                // tKsK has shape: (KCPY, KCPY_N, KCPY_K, kStages)
                // tKgK has shape: (KCPY, KCPY_N, KCPY_K, num_blocks)
#if FLASH_USE_CUTLASS_TENSOR
                Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_write);  // Select pipeline stage
                Tensor tKgK_cur = tKgK(_, _, _, n_block);          // Select K block
#endif
                // CuTe: raw_pointer_cast(sK.data()), raw_pointer_cast(gK.data())
                Element * smem_tile_base_K = smem_K_base;
                Element const * gmem_tile_base_K = gmem_K_base;

                // Step 2: Calculate the N dimension limit (similar to Q's M limit)
                // This is more complex than V because it handles the compile-time optimization better

                // First, get the thread's N coordinate offset
                // CuTe: get<0>(tKVcKV(_0{}, _0{}, _0{})) = thread's first N-row in partition
                int const thread0_n_offset = thread_idx / kGmemThreadsPerRow;

                // Calculate the limit based on whether we need masking
                int const seqlenk_row_limit = -thread0_n_offset + (EvenN
                    ? seqlen_info.seqlen_k - n_block * kBlockN           // Simple case: just remaining rows
                    : (!Seqlenk_mask 
                        ? kBlockN                                         // No masking: use full tile
                        : std::min(seqlen_info.seqlen_k - n_block * kBlockN, kBlockN)));  // Masking: min of remaining and tile

                // Step 3: Determine template parameters for flash::copy
                constexpr bool Is_even_MN = !Seqlenk_mask && EvenN;  // Perfect alignment: no N checking needed
                constexpr bool Is_even_K = false;                     // Always check K (head_dim might not align)
                constexpr bool Clear_OOB_MN = false;                  // Don't zero invalid N rows (will mask in scores)
                constexpr bool Clear_OOB_K = true;                    // DO zero invalid K columns (prevent garbage)

                // Step 4: Get tensor dimensions (compile-time constants from GmemTiledCopyQKV partition)
                //   vec:  kGmemElemsPerLoad elements per 128-bit vectorized load
                //   n:    kBlockN / thread_rows rows assigned to this thread
                //   k:    kHeadDim / kBlockKGmem K-chunks across head dimension
                // CuTe tensor equivalent:
                //   int const num_vec_copies = size<0>(tKgK_cur);
                //   int const num_n_elements = size<1>(tKgK_cur);
                //   int const num_k_elements = size<2>(tKgK_cur);
                static constexpr int num_vec_copies = kGmemElemsPerLoad;
                static constexpr int num_n_elements = kBlockN / thread_rows;
                static constexpr int num_k_elements = kHeadDim / kBlockKGmem;
                // CuTe tensor equivalent:
                //   int const stride_v_K = static_cast<int>(tKgK.layout()(1, 0, 0, 0));
                //   int const stride_m_K = static_cast<int>(tKgK.layout()(0, 1, 0, 0));
                //   int const stride_k_K = static_cast<int>(tKgK.layout()(0, 0, 1, 0));
                int const k_row_stride = static_cast<int>(get<0>(params.stride_K));  // from params

                // Gmem strides (CuTe: tKgK.layout()(1,0,0,0), (0,1,0,0), (0,0,1,0)):
                int const stride_v_K  = 1;
                int const stride_m_K  = thread_rows * k_row_stride;
                // stride_k_gmem only matters if num_k_elements > 1:
                int const stride_k_K  = kBlockKGmem;  // = kGmemThreadsPerRow * kGmemElemsPerLoad
                // CuTe: size<1>(tKgK_cur)
                int const num_n_per_thread = kBlockN / thread_rows;
                // Smem strides (CuTe: tKsK_cur.layout()(1,0,0), (0,1,0), (0,0,1)):
                int const stride_sv_K = 1;
                int const stride_sn_K = kGmemThreadsPerRow;
                int const stride_sk_K = kGmemThreadsPerRow * num_n_per_thread;
                int const thread_base_offset_K = (thread_idx / kGmemThreadsPerRow) * (kHeadDim / 2) * kGmemThreadsPerRow + (thread_idx % kGmemThreadsPerRow) * kGmemThreadsPerRow;

                // Step 5: Triple nested loop with bounds checking
                #pragma unroll
                for (int vec = 0; vec < num_vec_copies; ++vec) {
                    
                    #pragma unroll
                    for (int n = 0; n < num_n_elements; ++n) {
                        
                        // Get the N coordinate for this element (relative to block start)
                        // CuTe: get<0>(t0KVcKV(vec, n, _0{})) = thread 0's N coordinate = n * thread_rows
                        int n_coord = n * thread_rows;
                        
                        // Check if this N coordinate is within valid sequence length
                        bool n_is_valid;
                        if (Is_even_MN) {
                            // Perfect alignment: all N coords valid (compile-time known)
                            n_is_valid = true;
                        } else {
                            // Need runtime check
                            n_is_valid = (n_coord < seqlenk_row_limit);
                        }
                        
                        if (n_is_valid) {
                            // This N row is valid, process all K elements
                            
                            #pragma unroll
                            for (int k = 0; k < num_k_elements; ++k) {
                                int const offset_s = vec * stride_sv_K + n * stride_sn_K + k * stride_sk_K;
                                int const smem_idx = thread_idx * threads_along_k + offset_s;
                                
                                // Check if this K coordinate is within valid head dimension
                                // CuTe: tKVpKV(k)
                                bool k_is_valid = tKVpKV_raw[k];
                                
                                if (k_is_valid) {
                                    // Write to shared memory
#if FLASH_USE_CUTLASS_TENSOR
                                    // Both N and K are valid - do the actual copy
                                    // Load from global memory (128-bit vectorized)
                                    Element value = tKgK_cur(vec, n, k);
                                    tKsK_cur(vec, n, k) = value;
#else
                                    int const offset_g = vec * stride_v_K + n * stride_m_K + k * stride_k_K;
                                    int const gmem_idx = thread_base_offset_K + offset_g;
                                    Element value = gmem_tile_base_K[gmem_idx];
                                    smem_tile_base_K[smem_idx] = value;
#endif
                                } else {
                                    // K is out of bounds - write zero to shared memory
                                    // This prevents garbage from affecting Q×K^T computation
#if FLASH_USE_CUTLASS_TENSOR
                                    tKsK_cur(vec, n, k) = Element(0);
#else
                                    smem_tile_base_K[smem_idx] = Element(0);
#endif
                                }
                            }
                            
                        } else {
                            // This N row is out of bounds
                            // With Clear_OOB_MN=false, we do NOTHING (optimization!)
                            // Comment from code: "We don't need to clear the sK smem tiles 
                            // since we'll mask out the scores anyway."
                            //
                            // The attention scores S = Q×K^T will be masked later, so
                            // garbage in out-of-bounds K rows won't affect the final output.
                        }
                    }
                }
#endif
            //} else {
            //    paged_kv_manager.template load_page_table<Seqlenk_mask>(n_block);
            //    paged_kv_manager.template load_K<Seqlenk_mask>(n_block, sK(_, _, smem_pipe_write));
            //}
        };

        auto load_V = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
            static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
            //if constexpr (!PagedKV) {
                // Do we need bound check to make sure the row doesn't go above kBlockN
                static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
#if 0
                Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_write);
                // We don't call flash::copy since it doesn't support bound checking
                // to not overshot kBlockN when writing to smem.
                Tensor tVgV_cur = tVgV(_, _, _, n_block);
                int const seqlenk_row_limit = seqlen_info.seqlen_k - n_block * kBlockN - get<0>(tKVcKV(_0{}, _0{}, _0{}));
                #pragma unroll
                for (int m = 0; m < size<1>(tVsV); ++m) {
                    // If kBlockN doesn't evenly divide the tiled copy, only the last `m` needs to be checked
                    if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
                        bool const predicate_n = !Seqlenk_mask || get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
                        #pragma unroll
                        for (int k = 0; k < size<2>(tVsV); ++k) {
                            cute::copy(gmem_tiled_copy_QKV.with(tKVpKV(k) && predicate_n), tVgV_cur(_, m, k), tVsV_cur(_, m, k));
                        }
                    }
                }
#else
                // ============================================================================
                // MANUAL REWRITE - What it does under the hood:
                // ============================================================================

                // Step 1: Extract current V tile for this n_block and pipeline stage
#if FLASH_USE_CUTLASS_TENSOR
                Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_write);
                Tensor tVgV_cur = tVgV(_, _, _, n_block);
#endif

                // Step 2: Calculate how many valid rows in this V block
                // CuTe: get<0>(tKVcKV(_0{}, _0{}, _0{})) = thread's first N-row in the partition
                int const seqlenk_row_limit = seqlen_info.seqlen_k - n_block * kBlockN
                    - int(thread_idx / kGmemThreadsPerRow);

                // Step 3: Dimensions (same partition as K: same GmemTiledCopyQKV, same tile shape)
                // CuTe: size<0>(tVsV_cur), size<1>(tVsV_cur), size<2>(tVsV_cur)
                static constexpr int num_vec_copies = kGmemElemsPerLoad;
                static constexpr int num_n_elements = kBlockN / thread_rows;
                static constexpr int num_k_elements = kHeadDim / kBlockKGmem;

#if !FLASH_USE_CUTLASS_TENSOR
                // Manual address calculation for V (no cutlass tensor indexing)
                int const v_row_stride = static_cast<int>(get<0>(params.stride_V));
                // Gmem strides (CuTe: tVgV.layout()(1,0,0,0), (0,1,0,0), (0,0,1,0)):
                int const stride_v_V   = 1;
                int const stride_n_V   = thread_rows * v_row_stride;
                int const stride_k_V   = kBlockKGmem;
                // gmem thread base: (thread_row * row_stride_gmem) + (thread_col * kGmemElemsPerLoad)
                int const thread_base_gmem_V = (thread_idx / kGmemThreadsPerRow) * v_row_stride
                                             + (thread_idx % kGmemThreadsPerRow) * kGmemElemsPerLoad;
                // Manual smem index calculation for V.
                // SmemLayoutV is tile_to_shape of row-major atom (8, kBlockKGmem) → (kBlockN, kHeadDim, kStages).
                // Physical offset = row * kHeadDim + col + stage * kBlockN * kHeadDim
                //   row = thread_row + m * thread_rows
                //   col = thread_col * kGmemElemsPerLoad + vec + k * kBlockKGmem
                static constexpr int smem_row_stride_V  = kHeadDim;
                static constexpr int smem_stage_stride_V = kBlockN * kHeadDim;
                int const thread_row_V = thread_idx / kGmemThreadsPerRow;
                int const thread_col_V = thread_idx % kGmemThreadsPerRow;
                int const thread_base_smem_V = thread_row_V * smem_row_stride_V
                                             + thread_col_V * kGmemElemsPerLoad;
                // CuTe: raw_pointer_cast(gV(_, _, n_block).data()), raw_pointer_cast(sV.data())
                Element const* gmem_tile_base_V = gmem_V_base + n_block * kBlockN * get<0>(params.stride_V);
                Element* smem_tile_base_V = smem_V_base;
#endif

                // Step 4: Nested loops with two levels of predicates
                #pragma unroll
                for (int m = 0; m < num_n_elements; ++m) {

                    // First predicate: is this N row within the tile bounds (kBlockN)?
                    bool row_within_tile;
                    if (EvenN) {
                        row_within_tile = true;
                    } else {
                        // CuTe: get<0>(tKVcKV(_0{}, m, _0{})) = thread's N coord for row m
                        row_within_tile = (m < num_n_elements - 1) ||
                                        (int(thread_idx / kGmemThreadsPerRow) + m * thread_rows < kBlockN);
                    }

                    if (row_within_tile) {
                        // Second predicate: is this N row within sequence bounds?
                        // CuTe: get<0>(t0KVcKV(_0{}, m, _0{})) = thread 0's N coord = m * thread_rows
                        int n_coord = m * thread_rows;
                        bool const predicate_n = !Seqlenk_mask || (n_coord < seqlenk_row_limit);

                        #pragma unroll
                        for (int k = 0; k < num_k_elements; ++k) {
#if FLASH_USE_CUTLASS_TENSOR
                            bool const predicate_k = tKVpKV(k);
#else
                            bool const predicate_k = tKVpKV_raw[k];
#endif
                            bool const predicate_both = predicate_k && predicate_n;

                            #pragma unroll
                            for (int vec = 0; vec < num_vec_copies; ++vec) {
#if FLASH_USE_CUTLASS_TENSOR
                                if (predicate_both) {
                                    Element value = tVgV_cur(vec, m, k);
                                    tVsV_cur(vec, m, k) = value;
                                } else {
                                    tVsV_cur(vec, m, k) = Element(0);
                                }
#else
                                int const smem_idx_V = thread_base_smem_V
                                                     + m * thread_rows * smem_row_stride_V
                                                     + vec
                                                     + k * kBlockKGmem
                                                     + smem_pipe_write * smem_stage_stride_V;
                                if (predicate_both) {
                                    int const offset_g_V = vec * stride_v_V + m * stride_n_V + k * stride_k_V;
                                    int const gmem_idx_V = thread_base_gmem_V + offset_g_V;
                                    Element value = gmem_tile_base_V[gmem_idx_V];
                                    smem_tile_base_V[smem_idx_V] = value;
                                } else {
                                    smem_tile_base_V[smem_idx_V] = Element(0);
                                }
#endif
                            }
                        }
                    }
                }
#endif
            //} else {
            //    paged_kv_manager.template load_V<Seqlenk_mask>(n_block, sV(_, _, smem_pipe_write));
            //}
        };

        auto preprocess_Q = [&] {
            //if constexpr (!AppendKV) {
                flash::cp_async_wait</*Share_QV_Smem ? 1 : */kStages * 2 - 1>();
            //} else {
            //    if (get<1>(params.shape_rotary) > 0) {  // Apply rotary to Q
            //        using Rotary_t = Rotary<kBlockM, kHeadDim, NumMmaThreads, Element, !(Is_causal || Is_local) /*FixedPosition*/>;
            //        Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
            //                        params.ptr_rotary_sin, params.stride_rotary_sin,
            //                        params.is_rotary_interleaved, thread_idx, seqlen_q,
            //                        seqlen_info.seqlen_rotary);
            //        int const qhead_per_khead = !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor;
            //        if (params.is_rotary_interleaved) {
            //            auto [tRrCos, tRrSin] = cute::conditional_return<!PackGQA>(
            //                rotary.template load_cos_sin<true /*kInterleaved*/>(m_block),
            //                rotary.template load_cos_sin_packgqa<true /*kInterleaved*/>(m_block, params.qhead_per_khead_divmod)
            //            );
            //            flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
            //            __syncthreads();
            //            rotary.apply_Q_interleaved(sQ, tRrCos, tRrSin, m_block, qhead_per_khead);
            //        } else {
            //            auto [tRrCosCont, tRrSinCont] = cute::conditional_return<!PackGQA>(
            //                rotary.template load_cos_sin<false /*kInterleaved*/>(m_block),
            //                rotary.template load_cos_sin_packgqa<false /*kInterleaved*/>(m_block, params.qhead_per_khead_divmod)
            //            );
            //            flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
            //            __syncthreads();
            //            rotary.apply_Q_contiguous(sQ, tRrCosCont, tRrSinCont, m_block, qhead_per_khead);
            //        }
            //    } else {
            //        flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
            //    }
            //}

            //if constexpr (Q_in_regs) {
            //    __syncthreads();
            //    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
            //    Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(sQ);
            //    cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
            //}
        };

        // If Share_QV_Smem, we load Q, then load 1 stage of K, then (optionally) rotate Q and
        // read from smem_q to registers, then load V.
        // If !Share_QV, Smem, we load Q, load all stages of K & V, then (optionally) rotate Q.

        //if constexpr (Share_QV_Smem) {
        //    load_K(n_block, 0, cute::true_type{} /*Seqlenk_mask*/);
        //    cute::cp_async_fence();
        //    preprocess_Q();
        //    __syncthreads();  // Make sure all threads have read smem_q before loading V
        //}

        // For persistent, make sure all threads have finished reading smem_o
        //if constexpr (!Share_QV_Smem)
        //{
            __syncthreads();
        //}
        // Note, using the for_each() function here to ensure `stage` is of type Int<x>.
        //for_each(make_int_sequence<kStages>{}, [&] (auto stage) {
            Int<0> stage;
            static constexpr bool Is_first_stage = /*CUTE_STATIC_V(stage) == 0*/true;
            //static constexpr bool Is_last_stage = /*CUTE_STATIC_V(stage) == kStages - 1*/true;
            //if constexpr (!Share_QV_Smem || !Is_first_stage) {
                //if (Is_first_stage || n_block - stage >= n_block_min) {
                    load_K(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} /*Seqlenk_mask*/);
                //}
                // We want the fence outside the if statement to have a fixed number of cp.async commits.
                // so that we can wait with the correct number of outstanding commits.
                cute::cp_async_fence();
            //}
            //if constexpr (!Is_last_stage) {
            //    if (Is_first_stage || n_block - stage >= n_block_min) {
            //        load_V(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} /*Seqlenk_mask*/);
            //    }
            //    cute::cp_async_fence();
            //}
        //});

        //if constexpr (!Share_QV_Smem) 
        {
            preprocess_Q();
        }

        flash::Mask<kBlockM, kBlockN, PackGQA, TiledMma> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod
        );

        //float softcap_val = params.softcap_val;
        //if constexpr (Has_softcap && Is_FP8) {
        //    float const q_descale = params.ptr_q_descale == nullptr ? 1.0f : params.ptr_q_descale[bidb * get<0>(params.stride_q_descale) + bidh_kv * get<1>(params.stride_q_descale)];
        //    float const k_descale = params.ptr_k_descale == nullptr ? 1.0f : params.ptr_k_descale[bidb * get<0>(params.stride_k_descale) + bidh_kv * get<1>(params.stride_k_descale)];
        //    softcap_val *= q_descale * k_descale;
        //}
        // Softcapping needs to happen before masking since if we apply after masking, softcapping can turn
        // -inf to e.g. -50.0, which can affect the attention softmax.
        //auto scoremod_premask_fn = [&](auto& tSrS) {
        //    if constexpr (Has_softcap) { flash::apply_softcap(tSrS, softcap_val); }
        //};

        int smem_pipe_read = 0, smem_pipe_write = kStages - 1;

        auto load_K_next = [&] {
            if (n_block - kStages >= n_block_min) {
                load_K(n_block - kStages, kStages > 1 ? smem_pipe_write : 0, cute::false_type{} /*Seqlenk_mask*/);
            }
            cute::cp_async_fence();
        };

        auto sync = [&] {
            flash::cp_async_wait<kStages * 2 - 2>();
            __syncthreads();
        };

#if FLASH_USE_CUTLASS_TENSOR
        clear(tOrO);
#else
        // O accumulator as raw float pointer, aliasing tOrO's internal storage.
        // All raw operations modify tOrO's data directly (no copy needed for epilogue).
        static constexpr int NAtomsM_O = kBlockM / (16 * kNWarps);
        static constexpr int NAtomsN_O = kHeadDimV / 8;
        static constexpr int VRegsPerAtomO = 4;
        static constexpr int TotalORegs = VRegsPerAtomO * NAtomsM_O * NAtomsN_O;
        float* O_regs = tOrO.data();
        #pragma unroll
        for (int i = 0; i < TotalORegs; ++i) O_regs[i] = 0.f;
#endif

        auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
            static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
            static constexpr bool Check_inf = decltype(check_inf_type)::value;
#if FLASH_USE_CUTLASS_TENSOR
            Tensor tSrS = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            clear(tSrS);
#else
            // S accumulator: raw float array backing the MMA D/C registers.
            // A CuTe tensor view is created over this array for mask/softmax/P-conversion
            // (which require the CuTe tensor interface); the MMA D/C access uses RawRegs.
            //
            // S is the C/D accumulator of Q×K^T: shape (kBlockM, kBlockN).
            // Per thread: VRegsPerAtomS floats × NAtomsM_S × NAtomsN_S atoms.
            //
            // partition_fragment_C uses make_layout(shape) which produces LayoutLeft
            // (column-major) strides: first mode varies fastest.
            // For shape (V=4, M=NAtomsM_S, N=NAtomsN_S):
            //   stride_V = 1,  stride_M = 4,  stride_N = 4 * NAtomsM_S
            // => M is inner, N is outer.
            //
            // Flat index: S_regs[m * 4 + ns * 4 * NAtomsM_S + v]
            //   m  = MMA atom index in M per warp (0..NAtomsM_S-1)
            //   ns = MMA atom index in N (0..NAtomsN_S-1)
            //   v  = float sub-register (0..3)
            static constexpr int VRegsPerAtomS = 4;                      // float[4] per C/D atom (SM75 & SM80)
            static constexpr int NAtomsM_S = kBlockM / (16 * kNWarps);   // M-atoms per warp (atom_M=16)
            static constexpr int NAtomsN_S = kBlockN / 8;                // N-atoms (atom_N=8)
            static constexpr int TotalSRegs = NAtomsM_S * NAtomsN_S * VRegsPerAtomS;
            float S_regs[TotalSRegs];
            #pragma unroll
            for (int i = 0; i < TotalSRegs; ++i) S_regs[i] = 0.f;
#endif
            sync();
            auto load_V_next = [&] {
                if (n_block - kStages + 1 >= n_block_min) {
                    load_V(n_block - kStages + 1, kStages > 1 ? smem_pipe_write : 0, cute::bool_constant<Is_first_iter && kStages == 1>{} /*Seqlenk_mask*/);
                }
                cute::cp_async_fence();
            };
#if FLASH_USE_CUTLASS_TENSOR
            Tensor tSrQ_cur = cute::conditional_return<Q_in_regs>(tSrQ, thr_mma.partition_fragment_A(sQ));
#else
            // Raw Q register array: flat array of uint32_t, manually indexed.
            // Layout: Q_regs[k_block * RegsPerKBlockQ + atom_m * VRegsPerAtomQ + v_reg]
            //   atom_m = MMA atom index in M per warp (0..NAtomsM-1)
            //   v_reg  = uint32 sub-register (0..1 for SM75, 0..3 for SM80)
            //
            // Q is the A-operand of Q×K^T: M=kBlockM, N=kBlockN, K=kHeadDim (reduction).
            // Warps are distributed along M: each warp handles kBlockM/kNWarps rows.
            static_assert(!Q_in_regs, "Raw Q registers require Q_in_regs=false");
            static constexpr int MmaTileK_Q = UseSM80MMA ? 16 : 8;          // MMA atom K-dim
            static constexpr int KBlocksQ = kHeadDim / MmaTileK_Q;          // k-steps
            static constexpr int NAtomsM = kBlockM / (16 * kNWarps);         // M-atoms per warp (atom_M=16)
            static constexpr int VRegsPerAtomQ = UseSM80MMA ? 4 : 2;        // uint32 per A atom
            static constexpr int RegsPerKBlockQ = NAtomsM * VRegsPerAtomQ;
            uint32_t Q_regs[KBlocksQ * RegsPerKBlockQ];
#endif
#if FLASH_USE_CUTLASS_TENSOR
            Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));
#else
            // Raw K register array: flat array of uint32_t, manually indexed.
            // Layout: K_regs[k_block * RegsPerKBlock + atom_n * VRegsPerAtomK + v_reg]
            //   atom_n = MMA atom index in the N direction (0..NAtomsK-1)
            //   v_reg  = uint32 sub-register within one atom (0 for SM75, 0..1 for SM80)
            //
            // SM75 (m16n8k8):  VRegsPerAtomK=1, each B atom has 1 uint32 (2 fp16)
            // SM80 (m16n8k16): VRegsPerAtomK=2, each B atom has 2 uint32 (4 fp16)
            static constexpr int MmaTileK_val = UseSM80MMA ? 16 : 8;  // from MMA atom K-dim
            static constexpr int KBlocksK = kHeadDim / MmaTileK_val;  // k-steps in the GEMM
            static constexpr int NAtomsK = kBlockN / 8;               // atom_N = 8 for both SM75/SM80
            static constexpr int VRegsPerAtomK = UseSM80MMA ? 2 : 1;  // uint32 per B atom per thread
            static constexpr int RegsPerKBlock = NAtomsK * VRegsPerAtomK;
            uint32_t K_regs[KBlocksK * RegsPerKBlock];
#endif

// ============================================================================
// SWITCH: Change #if 0 to #if 1 to use FULLY MANUAL version (no CuTe tensors)
// ============================================================================
#if 0
            // ================================================================
            // FULLY MANUAL VERSION: Simple scalar math that ACTUALLY WORKS
            // ================================================================
            // This version:
            // 1. Uses simple for-loops instead of ldmatrix/MMA
            // 2. Writes results directly to tSrS using CuTe's coordinate mapping
            // 3. Produces correct results (but is MUCH slower than tensor cores)
            //
            // KEY INSIGHT: tSrS is a CuTe tensor where each thread owns certain
            // (m,n) output elements. We use thr_mma.partition_C() on an identity
            // tensor to get the coordinate mapping, then compute dot products.
            // ================================================================
            {
                // Get raw pointers to shared memory (row-major layout)
                half_t const* smem_Q_ptr = reinterpret_cast<half_t const*>(shared_storage.tensors.mainloop.smem_q.data());
                half_t const* smem_K_ptr = reinterpret_cast<half_t const*>(shared_storage.tensors.mainloop.smem_k.data());
                
                // Create an identity tensor to get coordinate mapping
                // This maps tSrS element index → (m, n) coordinate in output matrix
                auto cS = thr_mma.partition_C(make_identity_tensor(
                    make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
                
                // ============================================================
                // MAIN COMPUTATION: For each element in tSrS, compute dot product
                // ============================================================
                #pragma unroll
                for (int i = 0; i < size(tSrS); ++i) {
                    // Get the (m, n) output coordinate for element i
                    auto coord = cS(i);
                    int m = get<0>(coord);
                    int n = get<1>(coord);
                    
                    // Compute dot product: S[m,n] = sum_k Q[m,k] * K[n,k]
                    float sum = 0.0f;
                    
                    if (m < kBlockM && n < kBlockN) {
                        #pragma unroll
                        for (int k = 0; k < kHeadDim; ++k) {
                            // Q[m][k] - row m, column k
                            float q_val = float(smem_Q_ptr[m * kHeadDim + k]);
                            // K[n][k] - row n, column k (K^T means we want column n of K^T = row n of K)
                            float k_val = float(smem_K_ptr[n * kHeadDim + k]);
                            sum += q_val * k_val;
                        }
                    }
                    // else: out of bounds, keep sum = 0
                    
                    // Store to register accumulator
                    tSrS(i) = sum;
                }
                
                // Trigger async V loading
                load_V_next();
            }

#elif 0
            flash::gemm_sm80<Q_in_regs>(
                tSrS, tSrQ_cur, tSrK, tSsQ, tSsK(_, _, _, kStages > 1 ? smem_pipe_read : 0),
                tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K, load_V_next
            );
#else
            // ============================================================================
            // INLINED gemm_sm80: Computes S = Q × K^T using Tensor Cores
            // 
            // This performs a tiled matrix multiplication where:
            //   - tSrS:     Accumulator in registers (output: attention scores)
            //   - tSrQ_cur: Q tile fragment in registers (may need loading from smem)
            //   - tSrK:     K tile fragment in registers (needs loading from smem)
            //   - tSsQ:     Q tile in shared memory (source for loading)
            //   - tSsK:     K tile in shared memory (source for loading)
            //
            // The K dimension is split into multiple "k_block" iterations.
            // Each iteration loads a slice and performs MMA (Matrix Multiply-Accumulate).
            // ============================================================================
            {
                // Get the current K tile from shared memory (handles pipeline staging)
#if FLASH_USE_CUTLASS_TENSOR
                auto tCsK_cur = tSsK(_, _, _, kStages > 1 ? smem_pipe_read : 0);
#else
                Element const* smem_K_stage_ptr = smem_K_ptr + (kStages > 1 ? smem_pipe_read : 0) * kBlockN * kHeadDim;
#endif
                
                // Create "retiled" views that match the hardware's expected register layout
                // for the LDSM (Load Shared Memory) instructions.
                // These views remap the logical tensor indices to the physical register layout
                // that Tensor Cores expect for efficient matrix fragment loading.
#if FLASH_USE_CUTLASS_TENSOR
                Tensor tCrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ_cur);  // Retiled view for Q
                Tensor tCrK_copy_view = smem_thr_copy_K.retile_D(tSrK);      // Retiled view for K
#endif
                
                // ========================================================================
                // PHASE 1: Load first K-slice (k=0) from Shared Memory → Registers
                // ========================================================================
                //
                // WHAT cute::copy DOES INTERNALLY (see educational_ldmatrix_x4 above):
                //
                // 1. Get source tensor slice: tSsQ(_, _, _0{}) 
                //    - This is Q's k=0 slice in shared memory
                //    - Shape: (M_tile, K_slice) e.g., (64, 16) for 64 rows, 16 columns
                //
                // 2. Get destination register view: tCrQ_copy_view(_, _, _0{})
                //    - This is the register fragment for k=0, RETILED for ldmatrix layout
                //    - The retile_D() call transformed the MMA register layout to LDSM layout
                //
                // 3. Internally iterates over sub-tiles that fit ldmatrix:
                //    - For M=64, K=16 with ldmatrix loading 8x8 matrices:
                //      Need (64/8) × (16/8) = 8 × 2 = 16 ldmatrix calls (distributed across warps)
                //
                // 4. For each sub-tile, executes:
                //    a) Each thread computes its smem address (with swizzle for bank conflicts)
                //    b) All 32 threads call: ldmatrix.sync.aligned.x4.m8n8.shared.b16
                //    c) Data lands in registers in MMA-ready layout
                //
                // The actual PTX generated (per ldmatrix call):
                //   ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r0,%r1,%r2,%r3}, [%smem_addr];
                //
                // Load Q[*, *, k=0] from shared memory to registers (if not already in regs)
                //if constexpr (!Q_in_regs) {
#if FLASH_USE_CUTLASS_TENSOR
                if constexpr (UseSM80MMA) {
                    flash::educational_copy_smem_to_regs(
                        smem_tiled_copy_Q, tSsQ(_, _, _0{}), tCrQ_copy_view(_, _, _0{}));
                } else {
                    flash::educational_copy_smem_to_regs_x2(
                        smem_tiled_copy_Q, tSsQ(_, _, _0{}), tCrQ_copy_view(_, _, _0{}));
                }
#else
                if constexpr (UseSM80MMA) {
                    flash::copy_Q_smem_to_raw_regs<kBlockM, kHeadDim, kNWarps>(
                        smem_Q_ptr, thread_idx, 0, &Q_regs[0 * RegsPerKBlockQ]);
                } else {
                    flash::copy_Q_smem_to_raw_regs_x2<kBlockM, kHeadDim, kNWarps>(
                        smem_Q_ptr, thread_idx, 0, &Q_regs[0 * RegsPerKBlockQ]);
                }
#endif
                //}
                // Load K[*, *, k=0] from shared memory to registers
#if FLASH_USE_CUTLASS_TENSOR
                if constexpr (UseSM80MMA) {
                    flash::educational_copy_smem_to_regs(
                        smem_tiled_copy_K, tCsK_cur(_, _, _0{}), tCrK_copy_view(_, _, _0{}));
                } else {
                    flash::educational_copy_smem_to_regs_x2(
                        smem_tiled_copy_K, tCsK_cur(_, _, _0{}), tCrK_copy_view(_, _, _0{}));
                }
#else
                if constexpr (UseSM80MMA) {
                    flash::copy_K_smem_to_raw_regs<kHeadDim, kBlockN>(
                        smem_K_stage_ptr, thread_idx, 0, &K_regs[0]);
                } else {
                    flash::copy_K_smem_to_raw_regs_x2<kHeadDim, kBlockN>(
                        smem_K_stage_ptr, thread_idx, 0, &K_regs[0]);
                }
#endif
                
                // ========================================================================
                // PHASE 2: Main K-dimension loop with software pipelining
                // ========================================================================
                // Loop over K dimension tiles. The key optimization is:
                // - While computing MMA for tile k, we prefetch tile k+1 from smem
                // - This hides memory latency behind compute
                #pragma unroll
#if FLASH_USE_CUTLASS_TENSOR
                for (int k_block = 0; k_block < size<2>(tSrQ_cur); ++k_block) {
#else
                for (int k_block = 0; k_block < KBlocksQ; ++k_block) {
#endif
                    
                    // --------------------------------------------------------------------
                    // PREFETCH: Load NEXT k-slice while current MMA is in flight
                    // --------------------------------------------------------------------
                    // SOFTWARE PIPELINING: The key insight is that ldmatrix and mma.sync
                    // can execute concurrently! While the tensor cores compute on k_block,
                    // we load k_block+1 data.
                    //
                    // Timeline (simplified):
                    //   Cycle N:   ldmatrix(k=1) starts | mma(k=0) starts
                    //   Cycle N+1: ldmatrix(k=1) runs   | mma(k=0) runs  
                    //   Cycle N+2: ldmatrix(k=1) done   | mma(k=0) runs
                    //   Cycle N+3:                      | mma(k=0) done
                    //   Cycle N+4: ldmatrix(k=2) starts | mma(k=1) starts (uses data from ldmatrix k=1)
                    //
                    // The registers are double-buffered implicitly by using different
                    // k indices: we write to tCrQ[k+1] while reading from tCrQ[k]
                    //
#if FLASH_USE_CUTLASS_TENSOR
                    if (k_block < size<2>(tSrQ_cur) - 1) {
#else
                    if (k_block < KBlocksQ - 1) {
#endif
                        // Load next Q slice from smem → registers
#if FLASH_USE_CUTLASS_TENSOR
                        if constexpr (UseSM80MMA) {
                            flash::educational_copy_smem_to_regs(
                                smem_tiled_copy_Q, tSsQ(_, _, k_block + 1), tCrQ_copy_view(_, _, k_block + 1));
                        } else {
                            flash::educational_copy_smem_to_regs_x2(
                                smem_tiled_copy_Q, tSsQ(_, _, k_block + 1), tCrQ_copy_view(_, _, k_block + 1));
                        }
#else
                        if constexpr (UseSM80MMA) {
                            flash::copy_Q_smem_to_raw_regs<kBlockM, kHeadDim, kNWarps>(
                                smem_Q_ptr, thread_idx, k_block + 1, &Q_regs[(k_block + 1) * RegsPerKBlockQ]);
                        } else {
                            flash::copy_Q_smem_to_raw_regs_x2<kBlockM, kHeadDim, kNWarps>(
                                smem_Q_ptr, thread_idx, k_block + 1, &Q_regs[(k_block + 1) * RegsPerKBlockQ]);
                        }
#endif
                        // Load next K slice from smem → registers
#if FLASH_USE_CUTLASS_TENSOR
                        if constexpr (UseSM80MMA) {
                            flash::educational_copy_smem_to_regs(
                                smem_tiled_copy_K, tCsK_cur(_, _, k_block + 1), tCrK_copy_view(_, _, k_block + 1));
                        } else {
                            flash::educational_copy_smem_to_regs_x2(
                                smem_tiled_copy_K, tCsK_cur(_, _, k_block + 1), tCrK_copy_view(_, _, k_block + 1));
                        }
#else
                        if constexpr (UseSM80MMA) {
                            flash::copy_K_smem_to_raw_regs<kHeadDim, kBlockN>(
                                smem_K_stage_ptr, thread_idx, k_block + 1, &K_regs[(k_block + 1) * RegsPerKBlock]);
                        } else {
                            flash::copy_K_smem_to_raw_regs_x2<kHeadDim, kBlockN>(
                                smem_K_stage_ptr, thread_idx, k_block + 1, &K_regs[(k_block + 1) * RegsPerKBlock]);
                        }
#endif
                    }
                    
                    // --------------------------------------------------------------------
                    // HOOK: Execute callback on first iteration (loads V asynchronously)
                    // --------------------------------------------------------------------
                    // This is where we kick off async loading of V for the P×V GEMM
                    // while we're still computing S = Q×K^T
                    if (k_block == 0) {
                        load_V_next();  // Initiates cp.async to load V tile into smem
                    }
                    
                    // --------------------------------------------------------------------
                    // TENSOR CORE MMA: The actual matrix multiply-accumulate
                    // --------------------------------------------------------------------
                    // This is where the magic happens! cute::gemm dispatches to:
                    //   PTX: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
                    // 
                    // What this instruction does:
                    //   - Takes 16x16 tile of Q (A matrix) from registers
                    //   - Takes 16x8 tile of K^T (B matrix) from registers  
                    //   - Computes C += A × B using Tensor Cores
                    //   - All 32 threads in warp cooperate; each holds fragment of result
                    //
                    // Register layout (per thread in warp, for m16n8k16 with fp16→fp32):
                    //   A fragment: 8 half values (16 bytes)
                    //   B fragment: 4 half values (8 bytes)
                    //   C fragment: 4 float values (16 bytes)
                    //
                    // The tiled_mma handles multiple MMA instructions to cover the full
                    // tile size (e.g., 64x64 tile = multiple 16x8x16 MMAs)
                    
#ifdef FLASH_MANUAL_GEMM
                    // Manual GEMM: same instruction as cute::gemm (mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 or m16n8k8).
                    // cute::gemm(tiled_mma, A, B, C) with A(V,M,K), B(V,N,K), C(V,M,N) does: for k, then for m,n serpentine, call mma(D, A(_,m,k), B(_,n,k), C).
                    // Here we have a single k slice (k_block), so one (V,M) x (V,N) => (V,M,N) with row-major serpentine over (m,n).
                    {
                        // M = number of MMA-atom steps in the M direction for this warp.
                        // tSrQ_cur is (V, M, K): dim 1 = MmaTileM / (atom_M * kNWarps).
                        // MmaTileM = 16*kNWarps = 64, atom_M = 16, kNWarps = 4: M = 64/(16*4) = 1.
#if FLASH_USE_CUTLASS_TENSOR
                        int const M = size<1>(tSrQ_cur);
#else
                        static constexpr int M = NAtomsM;
#endif
                        // N = number of MMA-atom steps in the N direction.
                        // MmaTileN = 16 (from TiledMma Tile<..., _16, ...>), atom_N = 8.
                        // N = kBlockN / atom_N total atom steps across all tiles.
#if FLASH_USE_CUTLASS_TENSOR
                        int const N = size<1>(tSrK);
#else
                        static constexpr int N = NAtomsK;
#endif
                        #pragma unroll
                        for (int m = 0; m < M; ++m) {
                            #pragma unroll
                            for (int n = 0; n < N; ++n) {
                                int const ns = (m & 1) ? (N - 1 - n) : n;  // serpentine
#ifdef FLASH_RAW_MMA
                                // Expand tiled_mma.call to the actual PTX: recast fragments to register arrays, then call MMA_Op::fma (mma.sync.aligned.m16n8k*...).
                                // Same as cute::mma_unpack(traits, D, A, B, C): recast D,A,B,C to DRegisters/ARegisters/BRegisters/CRegisters, then explode into fma(...).
                                using MMA_Op = typename TiledMma::Atom::MMA_Op;
                                using RegTypeD = typename std::remove_extent<typename MMA_Op::DRegisters>::type;
                                using RegTypeA = typename std::remove_extent<typename MMA_Op::ARegisters>::type;
                                using RegTypeB = typename std::remove_extent<typename MMA_Op::BRegisters>::type;
                                using RegTypeC = typename std::remove_extent<typename MMA_Op::CRegisters>::type;
                                constexpr int RegNumD = std::extent<typename MMA_Op::DRegisters>::value;
                                constexpr int RegNumA = std::extent<typename MMA_Op::ARegisters>::value;
                                constexpr int RegNumB = std::extent<typename MMA_Op::BRegisters>::value;
                                constexpr int RegNumC = std::extent<typename MMA_Op::CRegisters>::value;
#if FLASH_USE_CUTLASS_TENSOR
                                auto D_slice = tSrS(_, m, ns);
                                auto C_slice = tSrS(_, m, ns);
                                auto rD = recast<RegTypeD>(D_slice);
                                auto rC = recast<RegTypeC>(C_slice);
                                auto A_slice = tSrQ_cur(_, m, k_block);
                                auto rA = recast<RegTypeA>(A_slice);
                                auto B_slice = tSrK(_, ns, k_block);
                                auto rB = recast<RegTypeB>(B_slice);
#else
                                // Raw S accumulator D/C: LayoutLeft (M-inner, N-outer).
                                // D (output) and C (input) point to the same location (in-place accumulation).
                                // Offset = m * 4 + ns * 4 * NAtomsM_S  (stride_M=4, stride_N=4*NAtomsM_S).
                                flash::RawRegs<RegTypeD, RegNumD> rD{
                                    &S_regs[m * VRegsPerAtomS + ns * VRegsPerAtomS * NAtomsM_S]};
                                flash::RawRegs<RegTypeC, RegNumC> rC{
                                    &S_regs[m * VRegsPerAtomS + ns * VRegsPerAtomS * NAtomsM_S]};
                                // Raw Q register access.
                                flash::RawRegs<RegTypeA, RegNumA> rA{reinterpret_cast<RegTypeA*>(
                                    &Q_regs[k_block * RegsPerKBlockQ + m * VRegsPerAtomQ])};
                                // Raw K register access.
                                flash::RawRegs<RegTypeB, RegNumB> rB{reinterpret_cast<RegTypeB*>(
                                    &K_regs[k_block * RegsPerKBlock + ns * VRegsPerAtomK])};
#endif
                                cute::detail::explode(MMA_Op::fma,
                                    rD, cute::make_int_sequence<RegNumD>{},
                                    rA, cute::make_int_sequence<RegNumA>{},
                                    rB, cute::make_int_sequence<RegNumB>{},
                                    rC, cute::make_int_sequence<RegNumC>{});
#else
#if FLASH_USE_CUTLASS_TENSOR
                                tiled_mma.call(tSrS(_, m, ns), tSrQ_cur(_, m, k_block), tSrK(_, ns, k_block), tSrS(_, m, ns));
#endif
#endif
                            }
                        }
                    }
#else
#if FLASH_USE_CUTLASS_TENSOR
                    cute::gemm(tiled_mma, tSrQ_cur(_, _, k_block), tSrK(_, _, k_block), tSrS);
#endif
#endif
                    // After this instruction, tSrS accumulates: S += Q_k × K_k^T
                    // The accumulator is in fp32 for numerical precision
                }
            }
            // ============================================================================
            // END INLINED gemm_sm80
            // 
            // At this point:
            //   - tSrS contains the attention scores S = Q × K^T (in registers, fp32)
            //   - V tile loading (cp.async) is in flight to shared memory
            //   - Ready for softmax and then P×V multiplication
            // ============================================================================
#endif
            smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
            //scoremod_premask_fn(tSrS);
            // Faster to load_K before gemm if we only have 1 stage
            //if constexpr (kStages == 1)
            //{
                sync();
                load_K_next();
            //}
            // K-blocks for P×V GEMM = kBlockN / MMA_atom_K.
            // Equals size<2>(tOrP) in CuTe path; defined here in common code for both paths.
            static constexpr int KBlocksPV = kBlockN / (UseSM80MMA ? 16 : 8);
#if FLASH_USE_CUTLASS_TENSOR
            mask_fn(tSrS, n_block);
            Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
            softmax.template online_softmax</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
            //if constexpr (Is_FP8) { flash::permute_Cregs_fp8(tSrS); }
            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMma>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            convert_type_out(tOrP_acc, tOrP);
#else
            // Raw-register mask: lambda calls apply_mask_raw_regs directly on S_regs.
            mask_fn(S_regs, n_block);
            // Raw-register softmax: operates directly on S_regs without CuTe tensors.
            // scores_scale stored as a small float array (kNRows elements).
            float scores_scale_raw[2 * NAtomsM_S];
            flash::max_get_scale_raw_regs<Is_first_iter, Check_inf, NAtomsM_S, NAtomsN_S>(
                S_regs, softmax.row_max.data(), softmax.row_sum.data(),
                scores_scale_raw, softmax.softmax_scale_log2);
            static constexpr int Softmax_max_offset = Is_FP8 ? 8 : 0;
            flash::online_softmax_raw_regs<Is_first_iter, Check_inf, Softmax_max_offset, NAtomsM_S, NAtomsN_S>(
                S_regs, softmax.row_max.data(), softmax.row_sum.data(),
                softmax.softmax_scale_log2);
            //if constexpr (Is_FP8) { flash::permute_Cregs_fp8(tSrS); }
            // Raw P register conversion: convert fp32 S accumulator to fp16 packed as uint32_t.
            // These serve as the A operand for the P×V GEMM.
            //
            // SM80 (m16n8k16): convert_layout_acc_Aregs groups pairs of N-atoms into k-blocks.
            //   Aregs shape: ((4,2), NAtomsM, NAtomsN_S/2)
            //   Strides:     ((1, 4*NAtomsM_S), 4, 8*NAtomsM_S)
            //   Source: S_regs[(v%4) + m*4 + (v/4 + 2*k)*4*NAtomsM_S]
            //   8 fp32 → 8 fp16 → 4 uint32_t per (m, k) MMA A-operand.
            //
            // SM75 (m16n8k8): Aregs layout is unchanged from accumulator layout.
            //   Shape: (4, NAtomsM, NAtomsN_S), Strides: (1, 4, 4*NAtomsM_S)
            //   Source: S_regs[v + m*4 + k*4*NAtomsM_S]
            //   4 fp32 → 4 fp16 → 2 uint32_t per (m, k) MMA A-operand.
            static constexpr int PRegsPerAtom = UseSM80MMA ? 4 : 2;
            uint32_t P_regs[NAtomsM_S * KBlocksPV * PRegsPerAtom];
            {
                if constexpr (UseSM80MMA) {
                    static constexpr int ElemsPerAtom = 8;
                    #pragma unroll
                    for (int m = 0; m < NAtomsM_S; ++m) {
                        #pragma unroll
                        for (int k = 0; k < KBlocksPV; ++k) {
                            cutlass::Array<float, ElemsPerAtom> src;
                            #pragma unroll
                            for (int v = 0; v < ElemsPerAtom; ++v) {
                                src[v] = S_regs[(v % 4) + m * 4 + (v / 4 + 2 * k) * 4 * NAtomsM_S];
                            }
                            cutlass::NumericArrayConverter<Element, float, ElemsPerAtom> cvt;
                            auto dst = cvt(src);
                            memcpy(&P_regs[m * KBlocksPV * PRegsPerAtom + k * PRegsPerAtom],
                                   &dst, PRegsPerAtom * sizeof(uint32_t));
                        }
                    }
                } else {
                    static constexpr int ElemsPerAtom = 4;
                    #pragma unroll
                    for (int m = 0; m < NAtomsM_S; ++m) {
                        #pragma unroll
                        for (int k = 0; k < KBlocksPV; ++k) {
                            cutlass::Array<float, ElemsPerAtom> src;
                            #pragma unroll
                            for (int v = 0; v < ElemsPerAtom; ++v) {
                                src[v] = S_regs[v + m * 4 + k * 4 * NAtomsM_S];
                            }
                            cutlass::NumericArrayConverter<Element, float, ElemsPerAtom> cvt;
                            auto dst = cvt(src);
                            memcpy(&P_regs[m * KBlocksPV * PRegsPerAtom + k * PRegsPerAtom],
                                   &dst, PRegsPerAtom * sizeof(uint32_t));
                        }
                    }
                }
            }
#endif
            if constexpr (!Is_first_iter)
            {
#if FLASH_USE_CUTLASS_TENSOR
                softmax.rescale_o(tOrO, scores_scale);
#else
                flash::rescale_o_raw_regs<NAtomsM_O, NAtomsN_O>(O_regs, scores_scale_raw);
#endif
            }
            //if constexpr (kStages > 1)
            //{
            //    sync();
            //}
#if FLASH_USE_CUTLASS_TENSOR
            Tensor tOrV = thr_mma.partition_fragment_B(sVt(_, _, _0{}));
#else
            // Raw V register array: flat array of uint32_t, manually indexed.
            // Layout: V_regs[k_block * RegsPerKBlockV + atom_n * VRegsPerAtomV + v_reg]
            //   atom_n = MMA atom index in the N (headdim) direction (0..NAtomsV-1)
            //   v_reg  = uint32 sub-register within one atom (0 for SM75, 0..1 for SM80)
            //
            // P×V GEMM dimensions: M=kBlockM, N=kHeadDimV, K=kBlockN (reduction).
            static constexpr int MmaTileK_V = UseSM80MMA ? 16 : 8;     // MMA atom K-dim
            static constexpr int KBlocksV = kBlockN / MmaTileK_V;      // k-steps for P×V
            static constexpr int NAtomsV = kHeadDimV / 8;              // atom_N = 8
            static constexpr int VRegsPerAtomV = UseSM80MMA ? 2 : 1;  // uint32 per B atom
            static constexpr int RegsPerKBlockV = NAtomsV * VRegsPerAtomV;
            uint32_t V_regs[KBlocksV * RegsPerKBlockV];
#endif
#if 0
            flash::gemm_rs_sm80(tOrO, tOrP, tOrV, tOsVt(_, _, _, /*kStages > 1 ? smem_pipe_read : */0), tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
#else
            // ============================================================================
            // INLINED gemm_rs_sm80: Computes O += P × V using Tensor Cores
            // 
            // "RS" = Register-Smem: A matrix (P) is already in registers, only B (V) 
            // needs to be loaded from shared memory.
            //
            // This is more efficient than gemm_sm80 because:
            //   - P (softmax output) is still hot in registers from previous computation
            //   - No need to spill P to shared memory and reload it
            //   - Only V needs smem→register transfer
            //
            // Variables:
            //   - tOrO:  Output accumulator in registers (O += P × V)
            //   - tOrP:  P matrix (softmax output) already in registers
            //   - tOrV:  V tile fragment in registers (loaded from smem)
            //   - tOsVt: V tile in shared memory (transposed layout for coalescing)
            // ============================================================================
            {
                // Get the current V tile from shared memory
#if FLASH_USE_CUTLASS_TENSOR
                auto tCsV_cur = tOsVt(_, _, _, /*kStages > 1 ? smem_pipe_read : */0);
#else
                // V stage is currently hardcoded to 0 (pipeline staging commented out).
                // Physical layout: (kBlockN, kHeadDim) row-major per stage.
                Element const* smem_V_stage_ptr = smem_V_ptr /* + stage * kBlockN * kHeadDim */;
#endif
                
                // Create retiled view for V that matches ldmatrix register layout
#if FLASH_USE_CUTLASS_TENSOR
                Tensor tCrV_copy_view = smem_thr_copy_V.retile_D(tOrV);
#endif
                
                // ========================================================================
                // PHASE 1: Load first K-slice of V from Shared Memory → Registers
                // ========================================================================
                //
                // V is stored TRANSPOSED in smem (as Vt) for two reasons:
                // 1. Better global→smem coalescing when loading V from HBM
                // 2. The ldmatrix.trans instruction can load and transpose in one op
                //
                // For V, we use SM75_U16x8_LDSM_T (the "T" = Transposed variant):
                //   PTX: ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {regs}, [smem]
                //
                // The .trans modifier tells the hardware to transpose the 8×8 matrix
                // during the load, so we get column-major data into row-major registers.
                //
                // Memory layout (V transposed in smem):    Register layout (after load):
                //   Col 0: [v00 v10 v20 ... v70]            Thread 0: gets v00,v01 from row 0
                //   Col 1: [v01 v11 v21 ... v71]            Thread 1: gets v10,v11 from row 1
                //   ...                                     ...
                //
#if FLASH_USE_CUTLASS_TENSOR
                if constexpr (UseSM80MMA) {
                    flash::educational_copy_smem_to_regs_transposed(
                        smem_tiled_copy_V, tCsV_cur(_, _, _0{}), tCrV_copy_view(_, _, _0{}));
                } else {
                    flash::educational_copy_smem_to_regs_transposed_x2(
                        smem_tiled_copy_V, tCsV_cur(_, _, _0{}), tCrV_copy_view(_, _, _0{}));
                }
#else
                if constexpr (UseSM80MMA) {
                    flash::copy_V_smem_to_raw_regs_transposed<kHeadDim, kHeadDimV>(
                        smem_V_stage_ptr, thread_idx, 0, &V_regs[0]);
                } else {
                    flash::copy_V_smem_to_raw_regs_transposed_x2<kHeadDim, kHeadDimV>(
                        smem_V_stage_ptr, thread_idx, 0, &V_regs[0]);
                }
#endif
                
                // ========================================================================
                // PHASE 2: Main K-dimension loop (P × V accumulation)
                // ========================================================================
                // Similar pipelining as gemm_sm80, but simpler since P is already in regs
                #pragma unroll
#if FLASH_USE_CUTLASS_TENSOR
                for (int k_block = 0; k_block < size<2>(tOrP); ++k_block) {
#else
                for (int k_block = 0; k_block < KBlocksPV; ++k_block) {
#endif
                    
                    // --------------------------------------------------------------------
                    // PREFETCH: Load NEXT k-slice of V while current MMA executes
                    // --------------------------------------------------------------------
#if FLASH_USE_CUTLASS_TENSOR
                    if (k_block < size<2>(tOrP) - 1) {
#else
                    if (k_block < KBlocksPV - 1) {
#endif
#if FLASH_USE_CUTLASS_TENSOR
                        if constexpr (UseSM80MMA) {
                            flash::educational_copy_smem_to_regs_transposed(
                                smem_tiled_copy_V, tCsV_cur(_, _, k_block + 1), tCrV_copy_view(_, _, k_block + 1));
                        } else {
                            flash::educational_copy_smem_to_regs_transposed_x2(
                                smem_tiled_copy_V, tCsV_cur(_, _, k_block + 1), tCrV_copy_view(_, _, k_block + 1));
                        }
#else
                        if constexpr (UseSM80MMA) {
                            flash::copy_V_smem_to_raw_regs_transposed<kHeadDim, kHeadDimV>(
                                smem_V_stage_ptr, thread_idx, k_block + 1, &V_regs[(k_block + 1) * RegsPerKBlockV]);
                        } else {
                            flash::copy_V_smem_to_raw_regs_transposed_x2<kHeadDim, kHeadDimV>(
                                smem_V_stage_ptr, thread_idx, k_block + 1, &V_regs[(k_block + 1) * RegsPerKBlockV]);
                        }
#endif
                    }
                    
                    // --------------------------------------------------------------------
                    // TENSOR CORE MMA: O += P_k × V_k
                    // --------------------------------------------------------------------
                    // PTX: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
                    //
                    // P_k is a slice of the softmax probabilities (in registers)
                    // V_k is a slice of the value matrix (just loaded from smem)
                    //
                    // The output O accumulates across all K iterations:
                    //   O = Σ_k (P_k × V_k)
                    //
                    // This is the weighted sum of values, where weights are attention probs
                    
#if FLASH_USE_CUTLASS_TENSOR
                    cute::gemm(tiled_mma, tOrP(_, _, k_block), tOrV(_, _, k_block), tOrO);
#else
                    // Manual P×V GEMM with raw V registers.
                    // Same structure as the Q×K^T manual GEMM (see FLASH_MANUAL_GEMM above):
                    // iterate over M and N atom steps with serpentine ordering, then call MMA.
                    {
                        // M = number of MMA-atom steps in M (same as Q×K^T GEMM).
#if FLASH_USE_CUTLASS_TENSOR
                        int const M_pv = size<1>(tOrP);
#else
                        static constexpr int M_pv = NAtomsM_S;
#endif
                        // N = number of MMA-atom steps in N (headdimV / atom_N).
                        static constexpr int N_pv = NAtomsV;
                        #pragma unroll
                        for (int m = 0; m < M_pv; ++m) {
                            #pragma unroll
                            for (int n = 0; n < N_pv; ++n) {
                                int const ns = n;  // serpentine
                                using MMA_Op = typename TiledMma::Atom::MMA_Op;
                                using RegTypeD = typename std::remove_extent<typename MMA_Op::DRegisters>::type;
                                using RegTypeA = typename std::remove_extent<typename MMA_Op::ARegisters>::type;
                                using RegTypeB = typename std::remove_extent<typename MMA_Op::BRegisters>::type;
                                using RegTypeC = typename std::remove_extent<typename MMA_Op::CRegisters>::type;
                                constexpr int RegNumD = std::extent<typename MMA_Op::DRegisters>::value;
                                constexpr int RegNumA = std::extent<typename MMA_Op::ARegisters>::value;
                                constexpr int RegNumB = std::extent<typename MMA_Op::BRegisters>::value;
                                constexpr int RegNumC = std::extent<typename MMA_Op::CRegisters>::value;
                                // Raw O accumulator D/C: LayoutLeft (M-inner, N-outer), same as S.
                                // Offset = m * 4 + ns * 4 * NAtomsM_O.
                                flash::RawRegs<RegTypeD, RegNumD> rD{
                                    &O_regs[m * VRegsPerAtomO + ns * VRegsPerAtomO * NAtomsM_O]};
                                flash::RawRegs<RegTypeC, RegNumC> rC{
                                    &O_regs[m * VRegsPerAtomO + ns * VRegsPerAtomO * NAtomsM_O]};
                                // Raw P register access: contiguous PRegsPerAtom uint32_t per (m, k_block).
                                flash::RawRegs<RegTypeA, RegNumA> rA{
                                    &P_regs[m * KBlocksPV * PRegsPerAtom + k_block * PRegsPerAtom]};
                                // Raw V register access (same layout as K — see K GEMM above).
                                flash::RawRegs<RegTypeB, RegNumB> rB{reinterpret_cast<RegTypeB*>(
                                    &V_regs[k_block * RegsPerKBlockV + ns * VRegsPerAtomV])};
                                cute::detail::explode(MMA_Op::fma,
                                    rD, cute::make_int_sequence<RegNumD>{},
                                    rA, cute::make_int_sequence<RegNumA>{},
                                    rB, cute::make_int_sequence<RegNumB>{},
                                    rC, cute::make_int_sequence<RegNumC>{});
                            }
                        }
                    }
#endif
                }
            }
            // ============================================================================
            // END INLINED gemm_rs_sm80
            //
            // At this point:
            //   - tOrO contains accumulated output O += P × V (in registers, fp32)
            //   - This completes one block of the flash attention computation
            //   - Loop continues to next KV block until all blocks processed
            // ============================================================================
#endif
            //if constexpr (kStages > 1)
            //{
            //    load_K_next();
            //}
            smem_pipe_read = smem_pipe_read < kStages - 1 ? smem_pipe_read + 1 : 0;
        };

        auto first_iter_mask_fn = [&](auto& tSrS_or_regs, int n_block)
        {
#if FLASH_USE_CUTLASS_TENSOR
            mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS_or_regs, m_block, n_block);
#else
            static constexpr int NAtomsM_mask = kBlockM / (16 * kNWarps);
            static constexpr int NAtomsN_mask = kBlockN / 8;
            flash::apply_mask_raw_regs<true /*Seqlenk_mask*/, Is_causal, Is_local,
                kBlockM, kBlockN, kNWarps, NAtomsM_mask, NAtomsN_mask, PackGQA>(
                tSrS_or_regs, mask.thread_idx, m_block, n_block,
                mask.seqlen_q, mask.seqlen_k,
                mask.window_size_left, mask.window_size_right,
                mask.sink_token_length,
                mask.attention_chunk_divmod, mask.qhead_per_khead_divmod);
#endif
        };
        fwd_step(n_block, first_iter_mask_fn, cute::true_type{} /*is_first_iter*/, cute::true_type{} /*check_inf*/);
        --n_block;
        //if constexpr (Is_causal || Is_local) { // Separate iterations with causal or local masking
            auto mask_fn = [&](auto& tSrS_or_regs, int n_block)
            {
#if FLASH_USE_CUTLASS_TENSOR
                mask.template apply<false /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS_or_regs, m_block, n_block);
#else
                static constexpr int NAtomsM_mask = kBlockM / (16 * kNWarps);
                static constexpr int NAtomsN_mask = kBlockN / 8;
                flash::apply_mask_raw_regs<false /*Seqlenk_mask*/, Is_causal, Is_local,
                    kBlockM, kBlockN, kNWarps, NAtomsM_mask, NAtomsN_mask, PackGQA>(
                    tSrS_or_regs, mask.thread_idx, m_block, n_block,
                    mask.seqlen_q, mask.seqlen_k,
                    mask.window_size_left, mask.window_size_right,
                    mask.sink_token_length,
                    mask.attention_chunk_divmod, mask.qhead_per_khead_divmod);
#endif
            };
            int const n_block_min_causal_local_mask = BlockMN_t::get_n_block_min_causal_local_mask(
                seqlen_info, m_block, n_block_min, params.window_size_right,
                params.attention_chunk_divmod, params.qhead_per_khead_divmod);
            #pragma unroll 1
            for (; n_block >= n_block_min_causal_local_mask; --n_block) {
                fwd_step(n_block, mask_fn, cute::false_type{} /*is_first_iter*/, cute::true_type{} /*check_inf*/);
            }
        //}
        int const n_block_min_before_local_mask = BlockMN_t::get_n_block_min_before_local_mask(
            seqlen_info, m_block, n_block_min, params.window_size_left,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod);
        auto no_mask_fn = [](auto& tSrS, int n_block) { };
        #pragma unroll 1
        for (; n_block >= n_block_min_before_local_mask; --n_block) {
            fwd_step(n_block, no_mask_fn, cute::false_type{} /*is_first_iter*/, cute::false_type{} /*check_inf*/);
        }
        // Separate masking iterations on the left for local attention
        //if constexpr (Is_local) {
        //    auto local_mask_fn = [&](auto& tSrS, int n_block)
        //    {
        //        mask.template apply<false /*Seqlenk_mask*/, false /*Causal_mask*/, Is_local>(tSrS, m_block, n_block);
        //    };
        //    #pragma unroll 1
        //    for (; n_block >= n_block_min; --n_block) {
        //        fwd_step(n_block, local_mask_fn, cute::false_type{} /*is_first_iter*/, cute::bool_constant<Is_local>{} /*check_inf*/);
        //    }
        //}
        float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
        Tensor scores_scale = softmax.finalize(v_descale);
#if FLASH_USE_CUTLASS_TENSOR
        softmax.rescale_o(tOrO, scores_scale);
#else
        flash::rescale_o_raw_regs<NAtomsM_O, NAtomsN_O>(O_regs, scores_scale.data());
#endif
        //if constexpr (Is_FP8)
        //{
        //    flash::permute_output_fp8(tOrO);
        //}
        return true;
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE bool
    store_kv_new(Params const& params,
                 int const thread_idx,
                 SharedStorage &shared_storage,
                 SeqlenInfo_t const& seqlen_info,
                 cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord
    ) {
#if 0
        auto [m_block, bidh, bidb, split_idx] = block_coord;
        auto n_block_new_min_max = BlockMN_t::get_n_block_k_new_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
        int const n_block_new_min = get<0>(n_block_new_min_max);
        int const n_block_new_max = get<1>(n_block_new_min_max);
        if (n_block_new_max <= n_block_new_min) { return false; }

        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});

        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];

        bool const is_varlen_k_new = Varlen && params.cu_seqlens_k_new;
        Tensor mKnew = make_tensor(make_gmem_ptr(params.ptr_K_new), params.shape_K_new, params.stride_K_new)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);
        Tensor mVnew = make_tensor(make_gmem_ptr(params.ptr_V_new), params.shape_K_new, params.stride_V_new)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);

        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V), params.shape_K, params.stride_V)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);

        Tensor gKnew = local_tile(domain_offset(make_coord(seqlen_info.offset_k_new, _0{}), mKnew), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gVnew = local_tile(domain_offset(make_coord(seqlen_info.offset_k_new, _0{}), mVnew), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        int const offset_k = seqlen_info.offset_k + seqlen_info.seqlen_k_og;
        Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)

        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int kHeadDim = get<2>(TileShape_MNK{});
        int const seqlen_k_new = seqlen_info.seqlen_k_new;
        using Rotary_t = Rotary<kBlockN, kHeadDim, NumMmaThreads, Element>;
        Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
                        params.ptr_rotary_sin, params.stride_rotary_sin,
                        params.is_rotary_interleaved, thread_idx, seqlen_k_new,
                        seqlen_info.seqlen_rotary);

        using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumMmaThreads, Element, true /*KV_Same_Iter*/, 2 /*LoadsPerRow_LB*/>;
        PagedKVManager_t paged_kv_manager(
            params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
            params.ptr_K, params.shape_K, params.stride_K,
            params.ptr_V, params.headdim_v, params.stride_V,
            params.page_size_divmod,
            params.page_size_divmod /*blockN_per_page_size_divmod, not used since we don't use TMA*/,
            bidb_kv, bidh_kv, thread_idx, seqlen_k_new, offset_k,
            // passing offset_k instead of leftpad_k will move the PageTable pointer to the right position
            0 /*bidb_kv_idx, not used since we don't use TMA for Sm8x*/
        );

        static_assert(std::is_same_v<GmemLayoutAtomAppend, typename Rotary_t::LayoutAtom>);
        static_assert(!PagedKV || std::is_same_v<GmemLayoutAtomAppend, typename PagedKVManager_t::GmemLayoutAtomKVCpAsync>);
        GmemTiledCopyQKV gmem_tiled_copy_kv_g2s;
        auto gmem_thr_copy_kv_g2s = gmem_tiled_copy_kv_g2s.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_kv_g2s = gmem_tiled_copy_kv_g2s.get_thread_slice(_0{});  // Only for index calculation
        GmemTiledCopyAppendKV gmem_tiled_copy_kv_s2g;
        auto gmem_thr_copy_kv_s2g = gmem_tiled_copy_kv_s2g.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_kv_s2g = gmem_tiled_copy_kv_s2g.get_thread_slice(_0{});  // Only for index calculation
        Tensor tKgKnew = gmem_thr_copy_kv_g2s.partition_S(gKnew);
        Tensor tKsKg2s = gmem_thr_copy_kv_g2s.partition_S(sK);
        Tensor tKsKs2g = gmem_thr_copy_kv_s2g.partition_S(sK);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tKgK = gmem_thr_copy_kv_s2g.partition_D(gK);
        Tensor tVgVnew = gmem_thr_copy_kv_g2s.partition_S(gVnew);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tVsVg2s = gmem_thr_copy_kv_g2s.partition_S(sV);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tVsVs2g = gmem_thr_copy_kv_s2g.partition_S(sV);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tVgV = gmem_thr_copy_kv_s2g.partition_D(gV);
        Tensor cK = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcKg2s = gmem_thr_copy_kv_g2s.partition_D(cK);
        Tensor t0KcKg2s = gmem_thr0_copy_kv_g2s.partition_D(cK);
        Tensor tKpKg2s = make_tensor<bool>(make_shape(size<2>(tKsKg2s)));
        Tensor tKcKs2g = gmem_thr_copy_kv_s2g.partition_D(cK);
        Tensor t0KcKs2g = gmem_thr0_copy_kv_s2g.partition_D(cK);
        Tensor tKpKs2g = make_tensor<bool>(make_shape(size<2>(tKsKs2g)));
        #pragma unroll
        for (int k = 0; k < size(tKpKg2s); ++k) { tKpKg2s(k) = get<1>(tKcKg2s(_0{}, _0{}, k)) < get<1>(params.shape_K); }
        #pragma unroll
        for (int k = 0; k < size(tKpKs2g); ++k) { tKpKs2g(k) = get<1>(tKcKs2g(_0{}, _0{}, k)) < get<1>(params.shape_K); }

        auto load_K_new = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
            static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
            static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
            Tensor tKsK_cur = tKsKg2s(_, _, _, smem_pipe_write);
            int const seqlenk_row_limit = -int(get<0>(tKcKg2s(_0{}, _0{}, _0{}))) + (EvenN
                ? seqlen_k_new - n_block * kBlockN
                : (!Seqlenk_mask ? kBlockN : std::min(seqlen_k_new - n_block * kBlockN, kBlockN)));
            // We don't need to clear the sK smem tiles since we won't write them out
            flash::copy</*Is_even_MN=*/!Seqlenk_mask && EvenN, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
                gmem_tiled_copy_kv_g2s, tKgKnew(_, _, _, n_block), tKsK_cur, t0KcKg2s, tKpKg2s, seqlenk_row_limit);
        };

        auto load_V_new = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
            static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
            static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
            Tensor tVsV_cur = tVsVg2s(_, _, _, smem_pipe_write);
            int const seqlenk_row_limit = -int(get<0>(tKcKg2s(_0{}, _0{}, _0{}))) + (EvenN
                ? seqlen_k_new - n_block * kBlockN
                : (!Seqlenk_mask ? kBlockN : std::min(seqlen_k_new - n_block * kBlockN, kBlockN)));
            // We don't need to clear the sV smem tiles since we won't write them out
            flash::copy</*Is_even_MN=*/!Seqlenk_mask && EvenN, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
                gmem_tiled_copy_kv_g2s, tVgVnew(_, _, _, n_block), tVsV_cur, t0KcKg2s, tKpKg2s, seqlenk_row_limit);
        };

        auto store_K = [&] (int const n_block, int const smem_pipe_read) {
            int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
            if (get<1>(params.shape_rotary) <= 0) {
                Tensor tKsK_cur = tKsKs2g(_, _, _, smem_pipe_read);
                //if constexpr (!PagedKV) {
                    Tensor tKgK_cur = tKgK(_, _, _, n_block);
                    // Clear_OOB_K must be false since we don't want to write zeros to gmem
                    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                        gmem_tiled_copy_kv_s2g, tKsK_cur, tKgK_cur, tKcKs2g, tKpKs2g, std::min(seqlen_k_new - n_block * kBlockN, kBlockN)
                    );
                //} else {
                //    paged_kv_manager.store_K(n_block, tKsK_cur);
                //}
            } else {
                Tensor gK_cur = gK(_, _, n_block);
                auto tPrKPtr = cute::conditional_return<PagedKV>(paged_kv_manager.compute_K_ptr(), nullptr);
                if (params.is_rotary_interleaved) {
                    auto [tRrCos, tRrSin] = rotary.template load_cos_sin<true /*kInterleaved*/>(n_block);
                    rotary.template apply_K_interleaved<PagedKV>(sK(_, _, smem_pipe_read), gK_cur, tKpKs2g, tRrCos, tRrSin, tPrKPtr, n_block);
                } else {
                    auto [tRrCosCont, tRrSinCont] = rotary.template load_cos_sin<false /*kInterleaved*/>(n_block);
                    rotary.template apply_K_contiguous<PagedKV>(sK(_, _, smem_pipe_read), gK_cur, tKpKs2g, tRrCosCont, tRrSinCont, tPrKPtr, n_block, get<1>(params.shape_K));
                }
            }
        };

        auto store_V = [&] (int const n_block, int const smem_pipe_read) {
            int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
            Tensor tVsV_cur = tVsVs2g(_, _, _, smem_pipe_read);
            //if constexpr (!PagedKV) {
                Tensor tVgV_cur = tVgV(_, _, _, n_block);
                // Clear_OOB_K must be false since we don't want to write zeros to gmem
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                    gmem_tiled_copy_kv_s2g, tVsV_cur, tVgV_cur, tKcKs2g, tKpKs2g, n_limit);
            //} else {
            //    paged_kv_manager.store_V(n_block, tVsV_cur);
            //}
        };

        int n_block = n_block_new_max - 1;
        // Note, using the for_each() function here to ensure `stage` is of type Int<x>.
        for_each(make_int_sequence<kStages>{}, [&] (auto stage) {
            static constexpr bool Is_first_stage = CUTE_STATIC_V(stage) == 0;
            static constexpr bool Is_last_stage = CUTE_STATIC_V(stage) == kStages - 1;
            if (Is_first_stage || n_block - stage >= n_block_new_min) {
                load_K_new(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} /*Seqlenk_mask*/);
            }
            cute::cp_async_fence();
            // If persistent, need to sync to make sure all threads have finished with smem_o before writing to smem_v
            if constexpr (Is_first_stage) { __syncthreads(); }
            if constexpr (!Is_last_stage) {
                if (Is_first_stage || n_block - stage >= n_block_new_min) {
                    load_V_new(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} /*Seqlenk_mask*/);
                }
                cute::cp_async_fence();
            }
        });

        int smem_pipe_read = 0, smem_pipe_write = kStages - 1;
        #pragma unroll 1
        for (; n_block >= n_block_new_min; --n_block) {
            //if constexpr (PagedKV) { paged_kv_manager.template load_page_table<true /*Seqlenk_mask*/>(n_block); }
            flash::cp_async_wait<kStages * 2 - 2>();
            __syncthreads();
            store_K(n_block, kStages > 1 ? smem_pipe_read : 0);
            if (n_block - kStages + 1 >= n_block_new_min) {
                load_V_new(n_block - kStages + 1, kStages > 1 ? smem_pipe_write : 0, cute::bool_constant<kStages == 1>{} /*Seqlenk_mask*/);
            }
            cute::cp_async_fence();
            smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
            flash::cp_async_wait<kStages * 2 - 2>();
            __syncthreads();
            store_V(n_block, kStages > 1 ? smem_pipe_read : 0);
            smem_pipe_read = smem_pipe_read < kStages - 1 ? smem_pipe_read + 1 : 0;
            if (n_block - kStages >= n_block_new_min) {
                load_K_new(n_block - kStages, kStages > 1 ? smem_pipe_write : 0, cute::false_type{} /*Seqlenk_mask*/);
            }
            cute::cp_async_fence();
        }
#endif
        return true;

    }

};

} // namespace flash
