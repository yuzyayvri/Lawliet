#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>


// NNUE architecture: HalfKP(Friend)[41024->256x2] -> 32 -> 32 -> 1
// The Feature Transformer maps 41024 sparse binary features to 256 outputs
// per king perspective, for a total of 512 accumulated values fed into
// the first hidden layer.
constexpr int NNUE_FT_INPUTS    = 41024;  // HalfKP Friend features: 64*641
constexpr int NNUE_FT_OUTPUTS   = 256;    // Outputs per king perspective
constexpr int NNUE_FT_TOTAL     = 512;    // Combined white + black perspective
constexpr int NNUE_L1_SIZE      = 32;
constexpr int NNUE_L2_SIZE      = 32;
constexpr int NNUE_L3_SIZE      = 1;

// Stockfish 13 NNUE constants (from source):
constexpr int NNUE_FT_OUTPUT_MAX     = 127;  // FT CReLU: clamp(x, 0, 127)
constexpr int NNUE_WEIGHT_SCALE_BITS = 6;    // Hidden CReLU: x >> 6  (= divide by 64)
constexpr int NNUE_FV_SCALE          = 16;   // Output scaling: raw / FV_SCALE

class NNUE {
public:
    NNUE();
    ~NNUE();

    bool loadWeights(const std::string& filename);
    bool isLoaded() const { return weights_loaded_; }

    // Evaluate position (returns centipawn score from side-to-move perspective)
    // Stockfish 13 outputs: [side_to_move_accumulator, ~side_to_move_accumulator]
    // whiteToMove controls which perspective goes first.
    int evaluate(const uint32_t* whiteFeatures, int wCount,
                 const uint32_t* blackFeatures, int bCount,
                 bool whiteToMove) const;

    // Extract HalfKP Friend feature indices from piece bitboards
    static void extractFeatures(const uint64_t* pieceBB,
                                uint32_t* whiteFeat, int& wCount,
                                uint32_t* blackFeat, int& bCount);

private:
    int16_t* ft_weights_ = nullptr;  // [41024][256] int16_t
    int16_t* ft_biases_  = nullptr;  // [256] int16_t
    int8_t*  l1_weights_ = nullptr;  // [32][512] int8_t
    int32_t* l1_biases_  = nullptr;  // [32] int32_t
    int8_t*  l2_weights_ = nullptr;  // [32][32] int8_t
    int32_t* l2_biases_  = nullptr;  // [32] int32_t
    int8_t*  l3_weights_ = nullptr;  // [32] int8_t
    int32_t* l3_biases_  = nullptr;  // [1] int32_t
    bool     weights_loaded_ = false;

    // HalfKP Friend index formula: kingSq * 641 + pieceType * 64 + pieceSq
    // The extra index per king square (kingSq * 641 + 640) is the always-active
    // king feature. Total: 64 * 641 = 41024.
    static int halfKPIndex(int kingSq, int pType, int pSq) {
        return kingSq * 641 + pType * 64 + pSq;
    }

    // FT CReLU (no division): clamp(x, 0, NNUE_FT_OUTPUT_MAX)
    static int creluFT(int x) { return std::max(0, std::min(NNUE_FT_OUTPUT_MAX, x)); }

    // Hidden layer CReLU: clamp(x >> NNUE_WEIGHT_SCALE_BITS, 0, NNUE_FT_OUTPUT_MAX)
    static int creluHidden(int x) { return std::max(0, std::min(NNUE_FT_OUTPUT_MAX, x >> NNUE_WEIGHT_SCALE_BITS)); }

    // Forward pass through the network given a [0..127]-clamped 512-element accumulator
    int forward(const uint8_t* values) const;
};

// Aligned allocation helpers (file-local, shared across nnue.cpp)
template<typename T> static T* alignedAllocNNUE(size_t n) {
    void* p = nullptr;
    return (posix_memalign(&p, 64, n * sizeof(T)) == 0) ? static_cast<T*>(p) : nullptr;
}
template<typename T> static void alignedFreeNNUE(T* p) { free(p); }
