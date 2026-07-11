#pragma GCC optimize("O3,unroll-loops,fast-math")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include "nnue.hpp"
#include <cstdlib>
#include <cstring>

// read_little_endian: read a value from a stream in little-endian byte order.
template <typename IntType>
static IntType read_little_endian(std::istream& stream) {
    IntType result;
    std::uint8_t u[sizeof(IntType)];
    typename std::make_unsigned<IntType>::type v = 0;
    stream.read(reinterpret_cast<char*>(u), sizeof(IntType));
    for (std::size_t i = 0; i < sizeof(IntType); ++i)
        v = (v << 8) | u[sizeof(IntType) - i - 1];
    std::memcpy(&result, &v, sizeof(IntType));
    return result;
}

NNUE::NNUE() {}
NNUE::~NNUE() {
    alignedFree(ft_weights_); alignedFree(ft_biases_);
    alignedFree(l1_weights_); alignedFree(l1_biases_);
    alignedFree(l2_weights_); alignedFree(l2_biases_);
    alignedFree(l3_weights_); alignedFree(l3_biases_);
}

bool NNUE::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    std::uint32_t version = read_little_endian<std::uint32_t>(file);
    [[maybe_unused]] std::uint32_t net_hash = read_little_endian<std::uint32_t>(file);
    std::uint32_t desc_len = read_little_endian<std::uint32_t>(file);
    if (!file || version != 0x7AF32F16u) return false;

    std::string desc(desc_len, '\0');
    file.read(desc.data(), desc_len);
    if (desc.find("Features=HalfKP") == std::string::npos) {
        std::cerr << "info string NNUE error: unexpected architecture: "
                  << desc.substr(0, 80) << std::endl;
        return false;
    }

    // Skip FeatureTransformer sub-architecture hash (4 bytes)
    read_little_endian<std::uint32_t>(file);

    // Allocate aligned buffers (64-byte aligned for AVX2)
    auto alloc = [&]() -> bool {
        ft_weights_ = alignedAlloc<int16_t>(FT_INPUTS * (size_t)FT_OUTPUTS);
        ft_biases_  = alignedAlloc<int16_t>(FT_OUTPUTS);
        l1_weights_ = alignedAlloc<int8_t>(L1_SIZE * (size_t)FT_TOTAL);
        l1_biases_  = alignedAlloc<int32_t>(L1_SIZE);
        l2_weights_ = alignedAlloc<int8_t>(L2_SIZE * (size_t)L1_SIZE);
        l2_biases_  = alignedAlloc<int32_t>(L2_SIZE);
        l3_weights_ = alignedAlloc<int8_t>(L3_SIZE * (size_t)L2_SIZE);
        l3_biases_  = alignedAlloc<int32_t>(L3_SIZE);
        return ft_weights_ && ft_biases_ && l1_weights_ && l1_biases_
            && l2_weights_ && l2_biases_ && l3_weights_ && l3_biases_;
    };
    if (!alloc()) return false;

    // Read FeatureTransformer parameters (biases then weights)
    for (int i = 0; i < FT_OUTPUTS; ++i)
        ft_biases_[i] = read_little_endian<int16_t>(file);
    for (size_t i = 0; i < (size_t)FT_INPUTS * FT_OUTPUTS; ++i)
        ft_weights_[i] = read_little_endian<int16_t>(file);

    // Skip Network sub-architecture hash (4 bytes)
    read_little_endian<std::uint32_t>(file);

    if (file.fail()) return false;

    // Layer 1: 32 biases (int32) + 32x512 weights (int8)
    for (int i = 0; i < L1_SIZE; ++i)
        l1_biases_[i] = read_little_endian<int32_t>(file);
    for (size_t i = 0; i < (size_t)L1_SIZE * FT_TOTAL; ++i)
        l1_weights_[i] = read_little_endian<int8_t>(file);

    // Layer 2: 32 biases (int32) + 32x32 weights (int8)
    for (int i = 0; i < L2_SIZE; ++i)
        l2_biases_[i] = read_little_endian<int32_t>(file);
    for (size_t i = 0; i < (size_t)L2_SIZE * L1_SIZE; ++i)
        l2_weights_[i] = read_little_endian<int8_t>(file);

    // Output layer: 1 bias (int32) + 32 weights (int8)
    l3_biases_[0] = read_little_endian<int32_t>(file);
    for (size_t i = 0; i < (size_t)L3_SIZE * L2_SIZE; ++i)
        l3_weights_[i] = read_little_endian<int8_t>(file);

    weights_loaded_ = !file.fail();
    return weights_loaded_;
}

// Stockfish 13 HalfKP(Friend) active feature extraction.
void NNUE::extractFeatures(const uint64_t* pieceBB,
                           uint32_t* whiteFeat, int& wCount,
                           uint32_t* blackFeat, int& bCount) {
    wCount = bCount = 0;

    int wKing = pieceBB[5] ? __builtin_ctzll(pieceBB[5]) : -1;
    int bKing = pieceBB[11] ? __builtin_ctzll(pieceBB[11]) : -1;
    if (wKing < 0 || bKing < 0) return;

    int wKsq = orient(0, toSfSq(wKing));
    int bKsq = orient(1, toSfSq(bKing));

    // White perspective
    for (int i = 0; i < 12; ++i) {
        if (kpp_white[i] < 0) continue;
        uint64_t bb = pieceBB[i];
        while (bb) {
            int sq = __builtin_ctzll(bb); bb &= bb - 1;
            int sfSq = toSfSq(sq);
            whiteFeat[wCount++] = PS_END * wKsq + kpp_white[i] + sfSq;
        }
    }

    // Black perspective
    for (int i = 0; i < 12; ++i) {
        if (kpp_black[i] < 0) continue;
        uint64_t bb = pieceBB[i];
        while (bb) {
            int sq = __builtin_ctzll(bb); bb &= bb - 1;
            int sfSq = toSfSq(sq);
            int oriSq = orient(1, sfSq);
            blackFeat[bCount++] = PS_END * bKsq + kpp_black[i] + oriSq;
        }
    }
}

// ============================================================================
// AVX2-accelerated forward pass
// ============================================================================

// Forward pass through the network (512 uint8 input -> int32 score).
// Uses AVX2 for hidden layer dot products.
int NNUE::forward(const uint8_t* values) const {
    // Hidden Layer 1: 512 -> 32
    // Process each output neuron with SIMD dot product across 512 inputs.
    alignas(32) int32_t l1[L1_SIZE];

    constexpr int kL1Chunk = 64;  // Process 64 input pairs per SIMD iteration
    const __m256i kOnes = _mm256_set1_epi16(1);

    for (int i = 0; i < L1_SIZE; ++i) {
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();

        const int8_t* w = &l1_weights_[i * (size_t)FT_TOTAL];

        // Process 64 inputs per iteration (2 × 32-byte vectors)
        // FT_TOTAL=512, so 512/64 = 8 iterations
        #pragma GCC unroll 4
        for (int j = 0; j < FT_TOTAL; j += kL1Chunk) {
            // Load 32 uint8 input values and 32 int8 weights
            __m256i in0 = _mm256_load_si256((const __m256i*)(values + j));
            __m256i w0  = _mm256_load_si256((const __m256i*)(w + j));
            __m256i in1 = _mm256_load_si256((const __m256i*)(values + j + 32));
            __m256i w1  = _mm256_load_si256((const __m256i*)(w + j + 32));

            // _mm256_maddubs_epi16: uint8 * int8 → 16 int16 (pairwise sum of adjacent pairs)
            __m256i m0 = _mm256_maddubs_epi16(in0, w0);  // 16 int16
            __m256i m1 = _mm256_maddubs_epi16(in1, w1);  // 16 int16

            // _mm256_madd_epi16: sum adjacent pairs of int16 → 8 int32
            __m256i h0 = _mm256_madd_epi16(m0, kOnes);  // 8 int32
            __m256i h1 = _mm256_madd_epi16(m1, kOnes);  // 8 int32

            sum0 = _mm256_add_epi32(sum0, h0);
            sum1 = _mm256_add_epi32(sum1, h1);
        }

        // Combine partial sums
        __m256i total = _mm256_add_epi32(sum0, sum1);

        // Horizontal sum of 8 int32 values
        __m128i lo = _mm256_castsi256_si128(total);
        __m128i hi = _mm256_extracti128_si256(total, 1);
        lo = _mm_add_epi32(lo, hi);
        // Shuffle: _MM_SHUFFLE(1,0,3,2) = 0x4E
        lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
        lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));

        int32_t sum = l1_biases_[i] + _mm_cvtsi128_si32(lo);
        l1[i] = std::max(0, std::min(127, sum >> SCALE_BITS));
    }

    // Pack L1 int32 results to uint8 for L2 SIMD input
    alignas(32) uint8_t l1_u8[L1_SIZE];
    for (int i = 0; i < L1_SIZE; ++i)
        l1_u8[i] = (uint8_t)l1[i];

    // Hidden Layer 2: 32 -> 32
    // L2 input is 32 uint8, weights are [32][32] int8.
    // Process with a single AVX2 maddubs + madd per output neuron.
    alignas(32) int32_t l2[L2_SIZE];
    const __m256i l2_in = _mm256_load_si256((const __m256i*)l1_u8);

    for (int i = 0; i < L2_SIZE; ++i) {
        __m256i w = _mm256_load_si256((const __m256i*)&l2_weights_[i * L1_SIZE]);
        // maddubs: 32 uint8 * 32 int8 → 16 int16
        __m256i m = _mm256_maddubs_epi16(l2_in, w);
        // madd: 16 int16 → 8 int32
        __m256i h = _mm256_madd_epi16(m, kOnes);
        // Horizontal sum
        __m128i lo = _mm256_castsi256_si128(h);
        __m128i hi = _mm256_extracti128_si256(h, 1);
        lo = _mm_add_epi32(lo, hi);
        lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
        lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));

        l2[i] = std::max(0, std::min(127, (l2_biases_[i] + _mm_cvtsi128_si32(lo)) >> SCALE_BITS));
    }

    // Output: 32 -> 1 (scalar, 32 operations is negligible)
    int32_t out = l3_biases_[0];
    for (int j = 0; j < L2_SIZE; ++j)
        out += l2[j] * (int32_t)l3_weights_[j];

    return out / FV_SCALE;
}

// Evaluate from pre-computed accumulators (used by incremental updates).
// Skips the FT accumulation step and goes directly to CReLU + forward pass.
int NNUE::evaluateWithAcc(const int16_t* accWhite, const int16_t* accBlack,
                          bool whiteToMove) const {
    if (!weights_loaded_) return 0;

    // Order perspectives: [side_to_move, ~side_to_move]
    const int16_t* first  = whiteToMove ? accWhite : accBlack;
    const int16_t* second = whiteToMove ? accBlack : accWhite;

    // FT CReLU (clamp to [0, 127]) + pack int16 → uint8
    alignas(32) uint8_t combined[FT_TOTAL];
    const __m256i kZero = _mm256_setzero_si256();
    const __m256i kMax127 = _mm256_set1_epi16(127);

    for (int j = 0; j < FT_OUTPUTS; j += SIMD_WIDTH) {
        __m256i f = _mm256_load_si256((const __m256i*)(first + j));
        __m256i s = _mm256_load_si256((const __m256i*)(second + j));
        f = _mm256_min_epi16(_mm256_max_epi16(f, kZero), kMax127);
        s = _mm256_min_epi16(_mm256_max_epi16(s, kZero), kMax127);
        __m128i f_u8 = _mm_packus_epi16(_mm256_castsi256_si128(f), _mm256_extracti128_si256(f, 1));
        __m128i s_u8 = _mm_packus_epi16(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        _mm_store_si128((__m128i*)(combined + j), f_u8);
        _mm_store_si128((__m128i*)(combined + j + FT_OUTPUTS), s_u8);
    }

    return forward(combined);
}

// Store full NNUE accumulator state for incremental caching.
// Builds accumulators from scratch using the extracted features.
void NNUE::storeAccumulator(const uint32_t* whiteFeat, int wCount,
                             const uint32_t* blackFeat, int bCount,
                             bool whiteToMove,
                             int16_t* outAccWhite, int16_t* outAccBlack) const {
    if (!weights_loaded_) return;

    constexpr int kVecCount = FT_OUTPUTS / SIMD_WIDTH;

    __m256i* accW = (__m256i*)outAccWhite;
    __m256i* accB = (__m256i*)outAccBlack;
    const __m256i* biasVec = (const __m256i*)ft_biases_;

    for (int v = 0; v < kVecCount; ++v) {
        __m256i b = _mm256_load_si256(biasVec + v);
        accW[v] = b;
        accB[v] = b;
    }

    for (int i = 0; i < wCount; ++i) {
        const __m256i* col = (const __m256i*)&ft_weights_[whiteFeat[i] * (size_t)FT_OUTPUTS];
        for (int v = 0; v < kVecCount; ++v)
            accW[v] = _mm256_add_epi16(accW[v], col[v]);
    }

    for (int i = 0; i < bCount; ++i) {
        const __m256i* col = (const __m256i*)&ft_weights_[blackFeat[i] * (size_t)FT_OUTPUTS];
        for (int v = 0; v < kVecCount; ++v)
            accB[v] = _mm256_add_epi16(accB[v], col[v]);
    }
}

// Evaluate a position from the side-to-move perspective.
// Uses AVX2 for FeatureTransformer accumulation and CReLU.
int NNUE::evaluate(const uint32_t* whiteFeat, int wCount,
                   const uint32_t* blackFeat, int bCount,
                   bool whiteToMove) const {
    if (!weights_loaded_) return 0;

    // 32-byte aligned stack arrays for AVX2 loads
    alignas(32) int16_t accWhite[FT_OUTPUTS];
    alignas(32) int16_t accBlack[FT_OUTPUTS];

    constexpr int kVecCount = FT_OUTPUTS / SIMD_WIDTH;  // 256/16 = 16

    // Initialize accumulators from FT biases using SIMD
    __m256i* accW = (__m256i*)accWhite;
    __m256i* accB = (__m256i*)accBlack;
    const __m256i* biasVec = (const __m256i*)ft_biases_;

    #pragma GCC unroll 8
    for (int v = 0; v < kVecCount; ++v) {
        __m256i b = _mm256_load_si256(biasVec + v);
        accW[v] = b;
        accB[v] = b;
    }

    // Accumulate white perspective: add weight columns for each active feature.
    // Each column is 256 int16 values (FT_OUTPUTS = 16 AVX2 vectors).
    for (int i = 0; i < wCount; ++i) {
        const __m256i* col = (const __m256i*)&ft_weights_[whiteFeat[i] * (size_t)FT_OUTPUTS];
        #pragma GCC unroll 8
        for (int v = 0; v < kVecCount; ++v)
            accW[v] = _mm256_add_epi16(accW[v], col[v]);
    }

    // Accumulate black perspective
    for (int i = 0; i < bCount; ++i) {
        const __m256i* col = (const __m256i*)&ft_weights_[blackFeat[i] * (size_t)FT_OUTPUTS];
        #pragma GCC unroll 8
        for (int v = 0; v < kVecCount; ++v)
            accB[v] = _mm256_add_epi16(accB[v], col[v]);
    }

    // Order perspectives: [side_to_move, ~side_to_move]
    const int16_t* first  = whiteToMove ? accWhite : accBlack;
    const int16_t* second = whiteToMove ? accBlack : accWhite;

    // Apply FT CReLU (clamp to [0, 127]) and pack int16 → uint8 using SIMD.
    // combined[0..255] = first perspective (CReLU'd)
    // combined[256..511] = second perspective (CReLU'd)
    alignas(32) uint8_t combined[FT_TOTAL];

    const __m256i kZero = _mm256_setzero_si256();
    const __m256i kMax127 = _mm256_set1_epi16(127);

    // Process 16 values at a time with AVX2 int16 clamping, then SSE pack to uint8
    #pragma GCC unroll 8
    for (int j = 0; j < FT_OUTPUTS; j += SIMD_WIDTH) {
        __m256i f = _mm256_load_si256((const __m256i*)(first + j));
        __m256i s = _mm256_load_si256((const __m256i*)(second + j));

        // Clamp int16 to [0, 127]
        f = _mm256_min_epi16(_mm256_max_epi16(f, kZero), kMax127);
        s = _mm256_min_epi16(_mm256_max_epi16(s, kZero), kMax127);

        // Pack int16[16] → uint8[16] using SSE (sequential order, no lane crossing)
        __m128i f_u8 = _mm_packus_epi16(
            _mm256_castsi256_si128(f),
            _mm256_extracti128_si256(f, 1));
        __m128i s_u8 = _mm_packus_epi16(
            _mm256_castsi256_si128(s),
            _mm256_extracti128_si256(s, 1));

        // Store first at combined[j], second at combined[j + 256]
        _mm_store_si128((__m128i*)(combined + j), f_u8);
        _mm_store_si128((__m128i*)(combined + j + FT_OUTPUTS), s_u8);
    }

    return forward(combined);
}
