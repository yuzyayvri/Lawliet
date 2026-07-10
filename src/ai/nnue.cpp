#include "nnue.hpp"
#include <cstdlib>
#include <cstring>

NNUE::NNUE() {}
NNUE::~NNUE() {
    alignedFreeNNUE(ft_weights_); alignedFreeNNUE(ft_biases_);
    alignedFreeNNUE(l1_weights_); alignedFreeNNUE(l1_biases_);
    alignedFreeNNUE(l2_weights_); alignedFreeNNUE(l2_biases_);
    alignedFreeNNUE(l3_weights_); alignedFreeNNUE(l3_biases_);
}

bool NNUE::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    // --- Stockfish NNUE binary format (v2) header ---
    // Bytes 0-7:   64-bit network hash identifier
    // Bytes 8-11:  uint32 LE description length
    // Bytes 12+:   description string (not null-terminated)
    // Then:        raw weight data in order: ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b

    char hash[8];
    file.read(hash, 8);  // skip hash

    int32_t descLen;
    file.read(reinterpret_cast<char*>(&descLen), 4);
    if (descLen <= 0 || descLen > 1024) return false;

    // Validate the description string contains expected architecture
    std::string desc(descLen, '\0');
    file.read(desc.data(), descLen);
    if (desc.find("Features=HalfKP") == std::string::npos) {
        std::cerr << "info string NNUE error: unexpected network architecture: "
                  << desc.substr(0, 80) << std::endl;
        return false;
    }

    // --- Allocate aligned buffers ---
    auto alloc = [&]() -> bool {
        ft_weights_ = alignedAllocNNUE<int16_t>(NNUE_FT_INPUTS * (size_t)NNUE_FT_OUTPUTS);
        ft_biases_  = alignedAllocNNUE<int16_t>(NNUE_FT_OUTPUTS);
        l1_weights_ = alignedAllocNNUE<int8_t>(NNUE_L1_SIZE * (size_t)NNUE_FT_TOTAL);
        l1_biases_  = alignedAllocNNUE<int32_t>(NNUE_L1_SIZE);
        l2_weights_ = alignedAllocNNUE<int8_t>(NNUE_L2_SIZE * (size_t)NNUE_L1_SIZE);
        l2_biases_  = alignedAllocNNUE<int32_t>(NNUE_L2_SIZE);
        l3_weights_ = alignedAllocNNUE<int8_t>(NNUE_L3_SIZE * (size_t)NNUE_L2_SIZE);
        l3_biases_  = alignedAllocNNUE<int32_t>(NNUE_L3_SIZE);
        return ft_weights_ && ft_biases_ && l1_weights_ && l1_biases_
            && l2_weights_ && l2_biases_ && l3_weights_ && l3_biases_;
    };
    if (!alloc()) return false;

    // --- Read layers ---
    auto read = [&](void* buf, size_t sz) {
        file.read(static_cast<char*>(buf), sz);
    };

    read(ft_weights_, NNUE_FT_INPUTS * (size_t)NNUE_FT_OUTPUTS * sizeof(int16_t));
    read(ft_biases_,  NNUE_FT_OUTPUTS * sizeof(int16_t));
    read(l1_weights_, (size_t)NNUE_L1_SIZE * NNUE_FT_TOTAL * sizeof(int8_t));
    read(l1_biases_,  NNUE_L1_SIZE * sizeof(int32_t));
    read(l2_weights_, (size_t)NNUE_L2_SIZE * NNUE_L1_SIZE * sizeof(int8_t));
    read(l2_biases_,  NNUE_L2_SIZE * sizeof(int32_t));
    read(l3_weights_, (size_t)NNUE_L3_SIZE * NNUE_L2_SIZE * sizeof(int8_t));
    read(l3_biases_,  NNUE_L3_SIZE * sizeof(int32_t));

    weights_loaded_ = true;
    return true;
}

// HalfKP(Friend) feature extraction
// pieceBB layout: 0=P_w,1=N_w,2=B_w,3=R_w,4=Q_w,5=K_w,6=P_b,7=N_b,8=B_b,9=R_b,10=Q_b,11=K_b
// HalfKP piece types: 0=P_w,1=N_w,2=B_w,3=R_w,4=Q_w,5=P_b,6=N_b,7=B_b,8=R_b,9=Q_b
//
// "Friend" variant: for the white king perspective, only white pieces (types 0-4)
// are included; for the black king perspective, only black pieces (types 5-9).
void NNUE::extractFeatures(const uint64_t* pieceBB,
                           uint32_t* whiteFeat, int& wCount,
                           uint32_t* blackFeat, int& bCount) {
    wCount = bCount = 0;

    int wKing = pieceBB[5] ? __builtin_ctzll(pieceBB[5]) : -1;
    int bKing = pieceBB[11] ? __builtin_ctzll(pieceBB[11]) : -1;
    if (wKing < 0 || bKing < 0) return;

    // Piece type mapping: pieceBB index -> NNUE piece type (0-9)
    static const int pTypeMap[12] = {0,1,2,3,4,-1,5,6,7,8,9,-1};

    for (int i = 0; i < 12; ++i) {
        int pt = pTypeMap[i];
        if (pt < 0) continue; // skip kings
        uint64_t bb = pieceBB[i];
        while (bb) {
            int sq = __builtin_ctzll(bb); bb &= bb - 1;
            // Friend filtering: white pieces only for white perspective,
            // black pieces only for black perspective
            if (pt < 5) {
                whiteFeat[wCount++] = halfKPIndex(wKing, pt, sq);
            } else {
                blackFeat[bCount++] = halfKPIndex(bKing, pt, sq);
            }
        }
    }

    // Always-active king features: each king square gets an extra bias feature
    // at index kingSq * 641 + 640 (the 641st slot per king square).
    whiteFeat[wCount++] = wKing * 641 + 640;
    blackFeat[bCount++] = bKing * 641 + 640;
}

int NNUE::forward(const int16_t* values) const {
    // Hidden Layer 1: 512 -> 32 (ClippedReLU with QB=64)
    int32_t l1[NNUE_L1_SIZE];
    for (int i = 0; i < NNUE_L1_SIZE; ++i) {
        int32_t sum = l1_biases_[i];
        for (int j = 0; j < NNUE_FT_TOTAL; ++j)
            sum += (int32_t)values[j] * (int32_t)l1_weights_[i * NNUE_FT_TOTAL + j];
        l1[i] = creluHidden(sum);
    }

    // Hidden Layer 2: 32 -> 32 (ClippedReLU with QB=64)
    int32_t l2[NNUE_L2_SIZE];
    for (int i = 0; i < NNUE_L2_SIZE; ++i) {
        int32_t sum = l2_biases_[i];
        for (int j = 0; j < NNUE_L1_SIZE; ++j)
            sum += l1[j] * (int32_t)l2_weights_[i * NNUE_L1_SIZE + j];
        l2[i] = creluHidden(sum);
    }

    // Output: 32 -> 1
    int32_t out = l3_biases_[0];
    for (int j = 0; j < NNUE_L2_SIZE; ++j)
        out += l2[j] * (int32_t)l3_weights_[j];

    // Scale to centipawns: includes FV_SCALE factor for the v2 NNUE protocol
    return out * NNUE_SCALE / (NNUE_FV_SCALE * NNUE_QA * NNUE_QB);
}

int NNUE::evaluate(const uint32_t* whiteFeat, int wCount,
                   const uint32_t* blackFeat, int bCount) const {
    if (!weights_loaded_) return 0;

    // Build two separate [256] accumulators, one for each king perspective,
    // then concatenate into a single [512] array for the hidden layer.
    int32_t accWhite[NNUE_FT_OUTPUTS];
    int32_t accBlack[NNUE_FT_OUTPUTS];

    // Start from the shared FT biases
    for (int j = 0; j < NNUE_FT_OUTPUTS; ++j) {
        int32_t b = (int32_t)ft_biases_[j];
        accWhite[j] = b;
        accBlack[j] = b;
    }

    // Accumulate white perspective features
    for (int i = 0; i < wCount; ++i) {
        const int16_t* col = &ft_weights_[whiteFeat[i] * (size_t)NNUE_FT_OUTPUTS];
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            accWhite[j] += (int32_t)col[j];
    }

    // Accumulate black perspective features
    for (int i = 0; i < bCount; ++i) {
        const int16_t* col = &ft_weights_[blackFeat[i] * (size_t)NNUE_FT_OUTPUTS];
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            accBlack[j] += (int32_t)col[j];
    }

    // Saturate to int16 range and interleave into the [512] forward input.
    // The network expects: [white_0..white_255, black_0..black_255]
    int16_t combined[NNUE_FT_TOTAL];
    for (int j = 0; j < NNUE_FT_OUTPUTS; ++j) {
        int32_t wv = accWhite[j];
        int32_t bv = accBlack[j];
        combined[j]                      = (int16_t)std::max(-32768, std::min(32767, wv));
        combined[j + NNUE_FT_OUTPUTS]    = (int16_t)std::max(-32768, std::min(32767, bv));
    }

    return forward(combined);
}
