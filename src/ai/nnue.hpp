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

constexpr int NNUE_SCALE = 400;
constexpr int NNUE_QA    = 255;   // Input quantization (CReLU max)
constexpr int NNUE_QB    = 64;    // Hidden layer quantization

class NNUE {
public:
    NNUE();
    ~NNUE();

    bool loadWeights(const std::string& filename);
    bool isLoaded() const { return weights_loaded_; }

    // Evaluate position (returns centipawn score from side-to-move perspective)
    int evaluate(const uint32_t* whiteFeatures, int wCount,
                 const uint32_t* blackFeatures, int bCount) const;

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

    // Clipped ReLU: clamp(x / QA) to [0, QA]
    static int crelu(int x) { return std::max(0, std::min(NNUE_QA, x / NNUE_QA)); }

    // Forward pass through the network given a saturated 512-element accumulator
    int forward(const int16_t* values) const;
};

// Aligned allocation helpers (file-local, shared across nnue.cpp)
template<typename T> static T* alignedAllocNNUE(size_t n) {
    void* p = nullptr;
    return (posix_memalign(&p, 64, n * sizeof(T)) == 0) ? static_cast<T*>(p) : nullptr;
}
template<typename T> static void alignedFreeNNUE(T* p) { free(p); }
