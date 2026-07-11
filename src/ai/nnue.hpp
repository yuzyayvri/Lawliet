#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <immintrin.h>

// NNUE architecture: HalfKP(Friend)[41024->256x2] -> 32 -> 32 -> 1
// Feature Transformer: 41024 inputs -> 512 (256 per perspective)
// Hidden Layer 1: 512 -> 32
// Hidden Layer 2: 32 -> 32
// Output: 32 -> 1

// Feature Transformer dimensions
constexpr int FT_INPUTS   = 41024;  // 64 * PS_END, HalfKP(Friend)
constexpr int FT_OUTPUTS  = 256;    // Per perspective
constexpr int FT_TOTAL    = 512;    // Both perspectives combined

// Hidden layer dimensions
constexpr int L1_SIZE     = 32;
constexpr int L2_SIZE     = 32;
constexpr int L3_SIZE     = 1;

// AVX2 SIMD width
constexpr int SIMD_WIDTH  = 16;     // 256-bit / 16-bit per element = 16

// Stockfish 13 NNUE constants
constexpr int FT_CReLU_MAX   = 127;   // Clamp accumulator to [0, 127]
constexpr int SCALE_BITS     = 6;     // Hidden CReLU: x >> 6
constexpr int FV_SCALE       = 16;    // Output: raw / 16

// HalfKP(Friend) piece-square enum (from Stockfish 13 nnue_common.h)
constexpr int PS_NONE     = 0;
constexpr int PS_W_PAWN   = 1;
constexpr int PS_B_PAWN   = 65;
constexpr int PS_W_KNIGHT = 129;
constexpr int PS_B_KNIGHT = 193;
constexpr int PS_W_BISHOP = 257;
constexpr int PS_B_BISHOP = 321;
constexpr int PS_W_ROOK   = 385;
constexpr int PS_B_ROOK   = 449;
constexpr int PS_W_QUEEN  = 513;
constexpr int PS_B_QUEEN  = 577;
constexpr int PS_W_KING   = 641;
constexpr int PS_END      = PS_W_KING; // Sentinel (= 641)

class NNUE {
public:
    NNUE();
    ~NNUE();

    bool loadWeights(const std::string& filename);
    bool isLoaded() const { return weights_loaded_; }

    // Evaluate position from side-to-move perspective.
    int evaluate(const uint32_t* whiteFeatures, int wCount,
                 const uint32_t* blackFeatures, int bCount,
                 bool whiteToMove) const;

    // Extract HalfKP(Friend) feature indices from piece bitboards.
    static void extractFeatures(const uint64_t* pieceBB,
                                uint32_t* whiteFeat, int& wCount,
                                uint32_t* blackFeat, int& bCount);

    // Evaluate from pre-computed FT accumulators (incremental update path)
    int evaluateWithAcc(const int16_t* accWhite, const int16_t* accBlack, bool whiteToMove) const;

    // Store full accumulator state for incremental caching
    void storeAccumulator(const uint32_t* whiteFeat, int wCount,
                          const uint32_t* blackFeat, int bCount,
                          bool whiteToMove,
                          int16_t* outAccWhite, int16_t* outAccBlack) const;

    // Access FT weights for incremental diff updates
    const int16_t* getFTWeights() const { return ft_weights_; }

    // Coordinate conversion: Lawliet (a8=0, h1=63) <-> Stockfish (A1=0, H8=63)
    static int toSfSq(int sq) { return (7 - sq / 8) * 8 + sq % 8; }

    // Orient square for perspective: 0 = identity, 1 = 180° rotation
    static int orient(int perspective, int sq) { return sq ^ (perspective * 63); }

    // HalfKP feature index for a given piece index and square (Stockfish coords).
    // Piece index: 0-5 white (P,N,B,R,Q,K), 6-11 black (p,n,b,r,q,k).
    // Perspective: 0 = white (stm), 1 = black (~stm).
    static int featureIndex(int pieceIdx, int sfSq, int kingSq, bool blackPerspective) {
        constexpr int kpp_w[12] = {PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, -1,
                                    PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, -1};
        constexpr int kpp_b[12] = {PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, -1,
                                    PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, -1};
        int kpp = blackPerspective ? kpp_b[pieceIdx] : kpp_w[pieceIdx];
        int pieceSq = blackPerspective ? (sfSq ^ 63) : sfSq;
        return PS_END * kingSq + kpp + pieceSq;
    }

private:
    // Network parameters (64-byte aligned from posix_memalign)
    int16_t* ft_weights_ = nullptr;  // [FT_INPUTS][FT_OUTPUTS]
    int16_t* ft_biases_  = nullptr;  // [FT_OUTPUTS]
    int8_t*  l1_weights_ = nullptr;  // [L1_SIZE][FT_TOTAL]
    int32_t* l1_biases_  = nullptr;  // [L1_SIZE]
    int8_t*  l2_weights_ = nullptr;  // [L2_SIZE][L1_SIZE]
    int32_t* l2_biases_  = nullptr;  // [L2_SIZE]
    int8_t*  l3_weights_ = nullptr;  // [L3_SIZE][L2_SIZE]
    int32_t* l3_biases_  = nullptr;  // [L3_SIZE]
    bool     weights_loaded_ = false;

    static int creluFT(int x) { return std::max(0, std::min(FT_CReLU_MAX, x)); }
    static int creluHidden(int x) { return std::max(0, std::min(FT_CReLU_MAX, x >> SCALE_BITS)); }

    // Forward pass: 512 uint8 values -> int32 score
    int forward(const uint8_t* values) const;

    // Aligned allocation helpers
    template<typename T> static T* alignedAlloc(size_t n) {
        void* p = nullptr;
        return (posix_memalign(&p, 64, n * sizeof(T)) == 0) ? static_cast<T*>(p) : nullptr;
    }
    template<typename T> static void alignedFree(T* p) { free(p); }

    // Stockfish 13's kpp_board_index table for HalfKP(Friend)
    static constexpr int kpp_white[12] = {
        PS_W_PAWN,   PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, -1,
        PS_B_PAWN,   PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, -1
    };
    static constexpr int kpp_black[12] = {
        PS_B_PAWN,   PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, -1,
        PS_W_PAWN,   PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, -1
    };
};
