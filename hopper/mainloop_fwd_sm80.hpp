/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "cute/tensor.hpp"

#include "seqlen.h"
#include "block.h"
#include "mask.h"
#include "pack_gqa.h"
#include "paged_kv.h"
#include "rotary.h"
#include "utils.h"

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

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});

        Layout s8 = make_layout(Int<8>{});
        Layout s2xs4 = make_layout(make_shape(Int<2>{}, Int<4>{}));
        if (thread_idx == 0 && bidh == 0 && bidb == 0 && m_block == 0)
        {
            //print2D(s2xs4);
        }
        //print2D(s8);

        bool const is_varlen_q = Varlen && params.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;

        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];
        Tensor mQ = make_tensor(make_gmem_ptr(params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q)), params.shape_Q_packed, params.stride_Q_packed)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor gQ = local_tile(mQ, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K + seqlen_info.offset_k * get<0>(params.stride_K)), params.shape_K, params.stride_K)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor gK = local_tile(mK, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V + seqlen_info.offset_k * get<0>(params.stride_V)), params.shape_K, params.stride_V)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor gV = local_tile(mV, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)

        //printf("%d %d %d %d\n", get<0>(params.shape_Q_packed), get<1>(params.shape_Q_packed), get<2>(params.shape_Q_packed), get<3>(params.shape_Q_packed));

        GmemTiledCopyQKV gmem_tiled_copy_QKV;
        auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(_0{});  // For index calculation

        Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
        Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
        Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
        Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

        if (thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        {
            print("tKgK shape: ");
            print(shape(tKgK));
            print("\n");
            print("tKgK stride: ");
            print(stride(tKgK));
            print("\n");
            print("tKgK layout: ");
            print(layout(tKgK));
            print("\n");

            // For a specific slice:
            print("tKgK(_,_,_,0) layout: ");
            print(layout(tKgK(_, _, _, 0)));
            print("\n");
        }

        if (thread_idx == 0) {
            //print_tensor(tKgK(_, _, _, 0));  // Print first block
        }

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

        // Predicates
        Tensor cKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
        Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
        Tensor t0KVcKV = gmem_thr0_copy_QKV.partition_S(cKV);
        Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k)
        {
            tKVpKV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(params.shape_K);
        }

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
            // Print thread's tQsQ base offset from start of sQ (in elements)
            if (bidh == 0 && bidb == 0 && m_block == 0) {
                Element const* sQ_base = raw_pointer_cast(gQ.data());
                Element const* tQsQ_base = raw_pointer_cast(tQgQ.data());
                ptrdiff_t offset_elems = tQsQ_base - sQ_base;
                printf("thread_idx=%d tQsQ_offset_from_sQ=%lld\n", thread_idx, offset_elems);
            }
#if 0
            // Instead of passing in tQcQ, we pass in t0QcQ and subtract the offset from the limit
            // (seqlen_q - m_block * kBlockM). This is because the entries of t0QcQ are known at compile time.
            // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
                gmem_tiled_copy_QKV, tQgQ, tQsQ, t0QcQ, tQpQ, seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}))
            );
#elif 0
            // ============================================================================
            // MANUAL REWRITE - What it does under the hood:
            // ============================================================================

            // Step 1: Calculate the M dimension limit (how many valid rows in this tile)
            int const max_valid_m = seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}));

            // Step 2: Get tensor dimensions
            // tQgQ and tQsQ have shape (VCOPY, MMA_M, MMA_K)
            // - VCOPY: Number of vectorized copies per thread (e.g., 2)
            // - MMA_M: Number of M elements this thread handles (e.g., 4)
            // - MMA_K: Number of K elements this thread handles (e.g., 8)

            int const num_vec_copies = size<0>(tQgQ);  // e.g., 2
            int const num_m_elements = size<1>(tQgQ);  // e.g., 4  
            int const num_k_elements = size<2>(tQgQ);  // e.g., 8

            //printf("%d %d %d %d %d %d %d\n", num_vec_copies, num_m_elements, num_k_elements, (int)size<0>(tKgK), (int)size<1>(tKgK), (int)size<2>(tKgK), (int)size<3>(tKgK));
            if (thread_idx == 0 && bidh == 0 && bidb == 0 && m_block == 0)
            {
                printf("seqlen_k = %d\n", seqlen_info.seqlen_k);
                printf("kHeadDim = %d\n", kHeadDim);
                printf("kGmemElemsPerLoad = %d\n", kGmemElemsPerLoad);
                printf("kBlockN = %d, kBlockK = %d\n", kBlockN, kHeadDim);
                printf("num K blocks = %d\n", (seqlen_info.seqlen_k + kBlockN - 1) / kBlockN);
			}

            // Step 3: Triple nested loop with bounds checking
            #pragma unroll
            for (int vec = 0; vec < num_vec_copies; ++vec) {
                
                #pragma unroll
                for (int m = 0; m < num_m_elements; ++m) {
                    
                    // Check if this M coordinate is within valid sequence length
                    int m_coord = get<0>(t0QcQ(vec, m, _0{}));  // Get the global M coordinate
                    bool m_is_valid = (m_coord < max_valid_m);
                    
                    if (m_is_valid) {
                        // This M row is valid, process all K elements
                        
                        #pragma unroll
                        for (int k = 0; k < num_k_elements; ++k) {
                            
                            // Check if this K coordinate is within valid head dimension
                            bool k_is_valid = tQpQ(k);  // Predicate we computed earlier
                            
                            if (k_is_valid) {
                                // Both M and K are valid - do the actual copy
                                // This is typically a vectorized load (e.g., 128-bit)
                                
                                // Get source element from global memory
                                Element value = tQgQ(vec, m, k);
                                
                                // Write to shared memory
                                tQsQ(vec, m, k) = value;
                                
                            } else {
                                // K is out of bounds - write zero to shared memory
                                // (prevents garbage values from affecting MMA computation)
                                tQsQ(vec, m, k) = Element(0);
                            }
                        }
                        
                    } else {
                        // This M row is out of bounds
                        // With Clear_OOB_MN=false, we do nothing (optimization)
                        // The output will be masked later anyway
                        // (If Clear_OOB_MN were true, we'd zero the entire row here)
                    }
                }
            }
#else
            // ============================================================================
            // MANUAL REWRITE - What it does under the hood:
            // ============================================================================

            // Step 1: Calculate the M dimension limit (how many valid rows in this tile)
            int const max_valid_m = seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}));

            // Step 2: Get tensor dimensions
            // tQgQ and tQsQ have shape (VCOPY, MMA_M, MMA_K)
            // - VCOPY: Number of vectorized copies per thread (e.g., 2)
            // - MMA_M: Number of M elements this thread handles (e.g., 4)
            // - MMA_K: Number of K elements this thread handles (e.g., 8)

            int const num_vec_copies = size<0>(tQgQ);  // e.g., 2
            int const num_m_elements = size<1>(tQgQ);  // e.g., 4  
            int const num_k_elements = size<2>(tQgQ);  // e.g., 8
            // Strides as ints (from layout once); index in loop is manual only
            int const stride_v = static_cast<int>(tQgQ.layout()(1, 0, 0));
            int const stride_m = static_cast<int>(tQgQ.layout()(0, 1, 0));
            int const stride_k = static_cast<int>(tQgQ.layout()(0, 0, 1));
            int const stride_sv = static_cast<int>(tQsQ.layout()(1, 0, 0));
            int const stride_sm = static_cast<int>(tQsQ.layout()(0, 1, 0));
            int const stride_sk = static_cast<int>(tQsQ.layout()(0, 0, 1));
            // Partition follows GmemLayoutAtom: (NumMmaThreads/kGmemThreadsPerRow, kGmemThreadsPerRow) with stride (kGmemThreadsPerRow, 1).
            // Thread base = (thread_row)*row_stride + (thread_col)*threads_along_k; row_stride = (kHeadDim/2)*kGmemThreadsPerRow (smem layout).
            int const threads_along_k = kGmemThreadsPerRow;
            int const thread_base_row_stride = (kHeadDim / 2) * kGmemThreadsPerRow;
            int const thread_base_offset = (thread_idx / threads_along_k) * thread_base_row_stride + (thread_idx % threads_along_k) * threads_along_k;
            Element const* gmem_tile_base = raw_pointer_cast(gQ.data());
            Element* smem_tile_base = raw_pointer_cast(sQ.data());

            //printf("%d %d %d %d %d %d %d\n", num_vec_copies, num_m_elements, num_k_elements, (int)size<0>(tKgK), (int)size<1>(tKgK), (int)size<2>(tKgK), (int)size<3>(tKgK));
            if (thread_idx == 0 && bidh == 0 && bidb == 0 && m_block == 0)
            {
                printf("num_vec_copies = %d\n", num_vec_copies);
                printf("num_m_elements = %d\n", num_m_elements);
                printf("num_k_elements = %d\n", num_k_elements);
                printf("seqlen_k = %d\n", seqlen_info.seqlen_k);
                printf("kHeadDim = %d\n", kHeadDim);
                printf("kGmemElemsPerLoad = %d\n", kGmemElemsPerLoad);
                printf("kBlockN = %d, kBlockK = %d\n", kBlockN, kHeadDim);
                printf("num K blocks = %d\n", (seqlen_info.seqlen_k + kBlockN - 1) / kBlockN);
			}

            // Step 3: Triple nested loop with bounds checking
            #pragma unroll
            for (int vec = 0; vec < num_vec_copies; ++vec) {
                
                #pragma unroll
                for (int m = 0; m < num_m_elements; ++m) {
                    
                    // Check if this M coordinate is within valid sequence length
                    int m_coord = get<0>(t0QcQ(vec, m, _0{}));  // Get the global M coordinate
                    bool m_is_valid = (m_coord < max_valid_m);
                    
                    if (m_is_valid) {
                        // This M row is valid, process all K elements
                        
                        #pragma unroll
                        for (int k = 0; k < num_k_elements; ++k) {
                            int const offset_s = vec * stride_sv + m * stride_sm + k * stride_sk;
                            int const smem_idx = thread_idx * threads_along_k + offset_s;

                            // Check if this K coordinate is within valid head dimension
                            bool k_is_valid = tQpQ(k);  // Predicate we computed earlier

                            if (k_is_valid) {
                                int const offset_g = vec * stride_v + m * stride_m + k * stride_k;
                                int const gmem_idx = thread_base_offset + offset_g;
                                Element value = gmem_tile_base[gmem_idx];
                                smem_tile_base[smem_idx] = value;
                            } else {
                                smem_tile_base[smem_idx] = Element(0);
                            }
                        }
                        
                    } else {
                        // This M row is out of bounds
                        // With Clear_OOB_MN=false, we do nothing (optimization)
                        // The output will be masked later anyway
                        // (If Clear_OOB_MN were true, we'd zero the entire row here)
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

        //printf("debug: %d %d %d %d %d %d %d %d %d %d\n", m_block, bidh, bidb, split_idx, kStages, kBlockM, kBlockN, kHeadDim, n_block_min, n_block_max);

        auto load_K = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
            static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
            //if constexpr (!PagedKV) {
                // Do we need bound check to make sure the row doesn't go above kBlockN
                static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
            #if 0
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
                Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_write);  // Select pipeline stage
                Tensor tKgK_cur = tKgK(_, _, _, n_block);          // Select K block

                // Step 2: Calculate the N dimension limit (similar to Q's M limit)
                // This is more complex than V because it handles the compile-time optimization better

                // First, get the thread 0's N coordinate offset
                int const thread0_n_offset = get<0>(tKVcKV(_0{}, _0{}, _0{}));

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

                // Step 4: Get tensor dimensions
                int const num_vec_copies = size<0>(tKgK_cur);  // e.g., 2
                int const num_n_elements = size<1>(tKgK_cur);  // e.g., 4
                int const num_k_elements = size<2>(tKgK_cur);  // e.g., 8

                // Step 5: Triple nested loop with bounds checking
                #pragma unroll
                for (int vec = 0; vec < num_vec_copies; ++vec) {
                    
                    #pragma unroll
                    for (int n = 0; n < num_n_elements; ++n) {
                        
                        // Get the N coordinate for this element (relative to block start)
                        // Using t0KVcKV (thread 0's coordinates) for compile-time optimization
                        int n_coord = get<0>(t0KVcKV(vec, n, _0{}));
                        
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
                                
                                // Check if this K coordinate is within valid head dimension
                                bool k_is_valid = tKVpKV(k);  // Pre-computed K predicate
                                
                                if (k_is_valid) {
                                    // Both N and K are valid - do the actual copy
                                    // Load from global memory (128-bit vectorized)
                                    Element value = tKgK_cur(vec, n, k);

                                    // Write to shared memory
                                    tKsK_cur(vec, n, k) = value;
                                    
                                } else {
                                    // K is out of bounds - write zero to shared memory
                                    // This prevents garbage from affecting Q×K^T computation
                                    tKsK_cur(vec, n, k) = Element(0);
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
                // tVsV has shape: (VCOPY, MMA_N, MMA_K, kStages)
                // tVgV has shape: (VCOPY, MMA_N, MMA_K, num_blocks)
                Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_write);  // Select pipeline stage
                Tensor tVgV_cur = tVgV(_, _, _, n_block);          // Select K/V block

                // Step 2: Calculate how many valid rows in this K block
                // Similar to Q, but for the K/V sequence dimension
                int const seqlenk_row_limit = seqlen_info.seqlen_k - n_block * kBlockN - get<0>(tKVcKV(_0{}, _0{}, _0{}));

                // Step 3: Get tensor dimensions
                int const num_vec_copies = size<0>(tVsV_cur);  // e.g., 2
                int const num_n_elements = size<1>(tVsV_cur);  // e.g., 4 (N dimension)
                int const num_k_elements = size<2>(tVsV_cur);  // e.g., 8 (K dimension)

                // Step 4: Nested loops with TWO levels of predicates
                #pragma unroll
                for (int m = 0; m < num_n_elements; ++m) {
                    
                    // First predicate: Check if this N row is within tile bounds (kBlockN)
                    // This is for when kBlockN doesn't evenly divide into the thread layout
                    bool row_within_tile;
                    if (EvenN) {
                        // If kBlockN is evenly divisible, all rows are within tile
                        row_within_tile = true;
                    } else {
                        // Check if this is the last row OR if it's within kBlockN
                        row_within_tile = (m < num_n_elements - 1) || 
                                        (get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN);
                    }
                    
                    if (row_within_tile) {
                        // Second predicate: Check if this N row is within sequence bounds
                        int n_coord = get<0>(t0KVcKV(_0{}, m, _0{}));  // Global N coordinate
                        bool row_within_sequence;
                        
                        if (Seqlenk_mask) {
                            // Need to check sequence bounds (last K block might be partial)
                            row_within_sequence = (n_coord < seqlenk_row_limit);
                        } else {
                            // No masking needed (not the last block, or even division)
                            row_within_sequence = true;
                        }
                        
                        // Combined N predicate
                        bool const predicate_n = row_within_sequence;
                        
                        // Now loop over K dimension
                        #pragma unroll
                        for (int k = 0; k < num_k_elements; ++k) {
                            
                            // K predicate (head dimension bounds)
                            bool const predicate_k = tKVpKV(k);  // Pre-computed K validity
                            
                            // Combined predicate: both N and K must be valid
                            bool const predicate_both = predicate_k && predicate_n;
                            
                            // Now copy each vectorized element
                            for (int vec = 0; vec < num_vec_copies; ++vec) {
                                if (predicate_both) {
                                    // Both N and K valid: Copy data
                                    Element value = tVgV_cur(vec, m, k);  // Read from global memory
                                    tVsV_cur(vec, m, k) = value;          // Write to shared memory
                                } else {
                                    // Either N or K invalid: Zero (prevents garbage in MMA)
                                    tVsV_cur(vec, m, k) = Element(0);
                                }
                            }
                        }
                    }
                    // If !row_within_tile, skip this row entirely (out of tile bounds)
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

        clear(tOrO);

        auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
            static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
            static constexpr bool Check_inf = decltype(check_inf_type)::value;
            Tensor tSrS = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            clear(tSrS);
            sync();
            auto load_V_next = [&] {
                if (n_block - kStages + 1 >= n_block_min) {
                    load_V(n_block - kStages + 1, kStages > 1 ? smem_pipe_write : 0, cute::bool_constant<Is_first_iter && kStages == 1>{} /*Seqlenk_mask*/);
                }
                cute::cp_async_fence();
            };
            Tensor tSrQ_cur = cute::conditional_return<Q_in_regs>(tSrQ, thr_mma.partition_fragment_A(sQ));
            Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));

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
                
                // Debug print (only once)
                if (thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0 && n_block == n_block_max - 1) {
                    printf("\n======== MANUAL GEMM (WORKING VERSION) ========\n");
                    printf("Computing S = Q × K^T using scalar FMA\n");
                    printf("Q in smem: [%d × %d]\n", kBlockM, kHeadDim);
                    printf("K in smem: [%d × %d]\n", kBlockN, kHeadDim);
                    printf("Output S:  [%d × %d]\n", kBlockM, kBlockN);
                    printf("Elements per thread (tSrS size): %d\n", (int)size(tSrS));
                    printf("==============================================\n\n");
                }
                
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
                auto tCsK_cur = tSsK(_, _, _, kStages > 1 ? smem_pipe_read : 0);
                
                // Create "retiled" views that match the hardware's expected register layout
                // for the LDSM (Load Shared Memory) instructions.
                // These views remap the logical tensor indices to the physical register layout
                // that Tensor Cores expect for efficient matrix fragment loading.
                Tensor tCrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ_cur);  // Retiled view for Q
                Tensor tCrK_copy_view = smem_thr_copy_K.retile_D(tSrK);      // Retiled view for K
                
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
                if constexpr (UseSM80MMA) {
                    flash::educational_copy_smem_to_regs(
                        smem_tiled_copy_Q, tSsQ(_, _, _0{}), tCrQ_copy_view(_, _, _0{}));
                } else {
                    flash::educational_copy_smem_to_regs_x2(
                        smem_tiled_copy_Q, tSsQ(_, _, _0{}), tCrQ_copy_view(_, _, _0{}));
                }
                //}
                // Load K[*, *, k=0] from shared memory to registers
                if constexpr (UseSM80MMA) {
                    flash::educational_copy_smem_to_regs(
                        smem_tiled_copy_K, tCsK_cur(_, _, _0{}), tCrK_copy_view(_, _, _0{}));
                } else {
                    flash::educational_copy_smem_to_regs_x2(
                        smem_tiled_copy_K, tCsK_cur(_, _, _0{}), tCrK_copy_view(_, _, _0{}));
                }
                
                // ========================================================================
                // PHASE 2: Main K-dimension loop with software pipelining
                // ========================================================================
                // Loop over K dimension tiles. The key optimization is:
                // - While computing MMA for tile k, we prefetch tile k+1 from smem
                // - This hides memory latency behind compute
                #pragma unroll
                for (int k_block = 0; k_block < size<2>(tSrQ_cur); ++k_block) {
                    
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
                    if (k_block < size<2>(tSrQ_cur) - 1) {
                        if constexpr (UseSM80MMA) {
                            flash::educational_copy_smem_to_regs(
                                smem_tiled_copy_Q, tSsQ(_, _, k_block + 1), tCrQ_copy_view(_, _, k_block + 1));
                            flash::educational_copy_smem_to_regs(
                                smem_tiled_copy_K, tCsK_cur(_, _, k_block + 1), tCrK_copy_view(_, _, k_block + 1));
                        } else {
                            flash::educational_copy_smem_to_regs_x2(
                                smem_tiled_copy_Q, tSsQ(_, _, k_block + 1), tCrQ_copy_view(_, _, k_block + 1));
                            flash::educational_copy_smem_to_regs_x2(
                                smem_tiled_copy_K, tCsK_cur(_, _, k_block + 1), tCrK_copy_view(_, _, k_block + 1));
                        }
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
                    
                    // ================================================================
                    // DEBUG: Print MMA sizes (only from thread 0, block 0, first iter)
                    // ================================================================
                    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k_block == 0 && n_block == n_block_max - 1) {
                        // TileShape from template parameters
                        printf("\n========== MMA SIZE DEBUG ==========\n");
                        printf("TileShape_MNK: M=%d, N=%d, K=%d\n", 
                               kBlockM, kBlockN, kHeadDim);
                        
                        // TiledMma shape: Tile<M, N, K> using tile_shape() function
                        auto mma_tile = tile_shape(tiled_mma);
                        printf("TiledMma Tile Shape: M=%d, N=%d, K=%d\n",
                               int(size<0>(mma_tile)),
                               int(size<1>(mma_tile)),
                               int(size<2>(mma_tile)));
                        
                        // Number of threads
                        printf("NumMmaThreads: %d (kNWarps=%d)\n", NumMmaThreads, kNWarps);
                        
                        // Fragment shapes (per thread)
                        printf("tSrQ_cur shape: (%d, %d, %d)\n",
                               int(size<0>(tSrQ_cur)), int(size<1>(tSrQ_cur)), int(size<2>(tSrQ_cur)));
                        printf("tSrK shape: (%d, %d, %d)\n",
                               int(size<0>(tSrK)), int(size<1>(tSrK)), int(size<2>(tSrK)));
                        printf("tSrS shape: (%d, %d, %d)\n",
                               int(size<0>(tSrS)), int(size<1>(tSrS)), int(size<2>(tSrS)));
                        
                        // MMA Atom info
                        printf("MMA Atom: %s\n", UseSM80MMA ? "SM80_16x8x16_F32F16F16F32_TN" : "SM75_16x8x8_F32F16F16F32_TN");
                        printf("PTX: %s\n", UseSM80MMA ? "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" : "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32");
                        printf("====================================\n\n");
                    }
                    
                    cute::gemm(tiled_mma, tSrQ_cur(_, _, k_block), tSrK(_, _, k_block), tSrS);
                    
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
            mask_fn(tSrS, n_block);
            Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
            softmax.template online_softmax</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
            //if constexpr (Is_FP8) { flash::permute_Cregs_fp8(tSrS); }
            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMma>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            convert_type_out(tOrP_acc, tOrP);
            if constexpr (!Is_first_iter)
            {
                softmax.rescale_o(tOrO, scores_scale);
            }
            //if constexpr (kStages > 1)
            //{
            //    sync();
            //}
            Tensor tOrV = thr_mma.partition_fragment_B(sVt(_, _, _0{}));
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
                auto tCsV_cur = tOsVt(_, _, _, /*kStages > 1 ? smem_pipe_read : */0);
                
                // Create retiled view for V that matches ldmatrix register layout
                Tensor tCrV_copy_view = smem_thr_copy_V.retile_D(tOrV);
                
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
                if constexpr (UseSM80MMA) {
                    flash::educational_copy_smem_to_regs_transposed(
                        smem_tiled_copy_V, tCsV_cur(_, _, _0{}), tCrV_copy_view(_, _, _0{}));
                } else {
                    flash::educational_copy_smem_to_regs_transposed_x2(
                        smem_tiled_copy_V, tCsV_cur(_, _, _0{}), tCrV_copy_view(_, _, _0{}));
                }
                
                // ========================================================================
                // PHASE 2: Main K-dimension loop (P × V accumulation)
                // ========================================================================
                // Similar pipelining as gemm_sm80, but simpler since P is already in regs
                #pragma unroll
                for (int k_block = 0; k_block < size<2>(tOrP); ++k_block) {
                    
                    // --------------------------------------------------------------------
                    // PREFETCH: Load NEXT k-slice of V while current MMA executes
                    // --------------------------------------------------------------------
                    if (k_block < size<2>(tOrP) - 1) {
                        if constexpr (UseSM80MMA) {
                            flash::educational_copy_smem_to_regs_transposed(
                                smem_tiled_copy_V, tCsV_cur(_, _, k_block + 1), tCrV_copy_view(_, _, k_block + 1));
                        } else {
                            flash::educational_copy_smem_to_regs_transposed_x2(
                                smem_tiled_copy_V, tCsV_cur(_, _, k_block + 1), tCrV_copy_view(_, _, k_block + 1));
                        }
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
                    
                    // ================================================================
                    // DEBUG: Print P×V GEMM sizes (only once)
                    // ================================================================
                    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k_block == 0 && n_block == n_block_max - 1) {
                        printf("\n========== P×V GEMM DEBUG ==========\n");
                        printf("tOrP shape: (%d, %d, %d)\n",
                               int(size<0>(tOrP)), int(size<1>(tOrP)), int(size<2>(tOrP)));
                        printf("tOrV shape: (%d, %d, %d)\n",
                               int(size<0>(tOrV)), int(size<1>(tOrV)), int(size<2>(tOrV)));
                        printf("tOrO shape: (%d, %d, %d)\n",
                               int(size<0>(tOrO)), int(size<1>(tOrO)), int(size<2>(tOrO)));
                        printf("=====================================\n\n");
                    }
                    
                    cute::gemm(tiled_mma, tOrP(_, _, k_block), tOrV(_, _, k_block), tOrO);
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

        auto first_iter_mask_fn = [&](auto& tSrS, int n_block)
        {
            mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block);
        };
        fwd_step(n_block, first_iter_mask_fn, cute::true_type{} /*is_first_iter*/, cute::true_type{} /*check_inf*/);
        --n_block;
        //if constexpr (Is_causal || Is_local) { // Separate iterations with causal or local masking
            auto mask_fn = [&](auto& tSrS, int n_block)
            {
                mask.template apply<false /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block);
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
        softmax.rescale_o(tOrO, scores_scale);
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
